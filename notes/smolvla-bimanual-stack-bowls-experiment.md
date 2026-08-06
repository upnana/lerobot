# SmolVLA · Bimanual Stack Bowls（黄→棕→白）实验总结

面向 **技术学习笔记 / 求职面试**：含数据规模、训练超参、推理设置、真机 SR、失败模式与可面试讲述点。

相关脚本：
- 采集：`get-data-bimanual-stack-bowls.sh`
- 训练：`train_smolvla_bimanual_stack_bowls.sh`
- 推理：`infer_smolvla_bimanual_stack_bowls_ep20.sh` / `_ep25.sh`

---

## 1. 任务与设置

| 项目 | 内容 |
|------|------|
| 任务指令 | `Stack the yellow bowl, then the brown bowl, and finally the white bowl.` |
| 成功标准 | 自下而上稳定叠好：**黄底 → 棕中 → 白顶**；时限内；**零人工干预** |
| 机器人 | 双臂 SO101 follower（**12 DoF**）+ 双 leader 遥操作采集 |
| 相机 | 4 路：`top_left` / `top_right` / `left` / `right`，MJPG @ 30fps（top_left 硬件 800×480，其余 640×480） |
| 策略 | SmolVLA（~450M；`train_expert_only` 路线，冻结大部分 VLM） |
| 基座 | `lerobot/smolvla_base` + 本地 SmolVLM2 |
| 相机映射 | `top_left→camera1`, `top_right→camera2`, `left→camera3`, `right→camera4` |
| 环境 | `conda: lerobot`，**2×3090**，`BATCH_SIZE=4` → effective **8** |

**为何比单臂 pick-place 难：** 长 horizon、三阶段顺序、双臂协调、碗口对齐容差小、起始位姿随机大。

---

## 2. 数据（Episodes / Frames）

| 指标 | 数值 |
|------|------|
| Dataset | `/home/rxn/datasets/bimanual_stack_bowls_yellow_brown_white` |
| Episodes | **200** |
| Frames | **98,616** |
| FPS | 30 |
| 平均时长 | **~16.4 s/ep**（98616 / 200 / 30） |
| Action / State | 各 **12** 维（左右臂关节） |
| 图像键 | 4 路 `observation.images.{top_left,top_right,left,right}` |

**采集设计动机：** 任务复杂 + 起始位姿随机大 → 目标定为 **200 ep**（原先觉得 100 不够）。

**数据质量现象（重要，会进模型）：**
1. **过早松爪（premature release）**：采集时有时离目标还有一段**竖直距离**就松爪把碗放下；模型也学到了这个动作。
2. 长 horizon 里阶段性误差会放大到后段（白碗最难）。

---

## 3. 训练（Steps / Epoch / Batch）

| 超参 | 数值 |
|------|------|
| Steps | **250,000** |
| Batch / GPU | 4；`NUM_GPUS=2` → effective **8** |
| 按帧估算的 data-epoch | \(250000 \times 8 / 98616 \approx\) **20.3** |
| 日志打印 `epch`（末期） | **≈ 40.5**（LeRobot tracker 按 ≈`steps×16/frames` 计，约为上式的 2 倍；**面试请报 data-epoch ≈20 或直接报 step**） |
| Save freq | 每 10k → `010000` … `250000` + `last` |
| 训练墙钟 | **~11.6 h**（`19:43` → 次日 `07:18`） |
| 末期 loss | ~**0.040** |
| `updt_s` | ~**0.16 s/step**（双卡） |

**Checkpoint 路径：**
```text
outputs/train/smolvla_bimanual_stack_bowls/checkpoints/
  120000/pretrained_model   # ≈ data-epoch 20 最近保存点（日志 epch≈19.4；精确 20.0≈step 123k）
  150000/pretrained_model   # ≈ data-epoch 25 最近保存点（日志 epch≈24.3；精确 25.0≈step 154k）
  last → 250000             # 训满
```

**关键观察：** **~20 epoch（ckpt `120000`）真机能力较好**；并非训得越久越好（长 horizon BC 易过拟合示教风格 / 悬停）。

---

## 4. 推理设置（Infer time / Latency）

| 项 | 默认值 |
|----|--------|
| 脚本 | `infer_smolvla_bimanual_stack_bowls_ep20.sh` |
| Ckpt | `120000`（~20 ep） |
| `EPISODE_TIME_S` | **150 s**（后从 90 加长） |
| `RESET_TIME_S` | 10 s |
| 控制 FPS | 30 |
| `n_action_steps` | **5**（训练 chunk 默认 50；真机缩短更跟手） |
| `max_relative_target` | 50 |

**Latency（同架构参考，蓝块任务上测过；本任务未单独重测）：**

| 指标 | SmolVLA（3090） |
|------|-----------------|
| Policy `select_action` | ~**148 ms**（~6.8 Hz） |
| 控制环预算 @30Hz | ~33 ms/帧 |

→ 必须靠 **action chunk**，不能每帧重推。真机「infer time」若指整集墙钟，则是 **episode 上限 150s**，不是 148ms。

面试一句分清：
- **Policy latency**：一次前向 ~150 ms  
- **Episode duration**：评测时限 150 s  
- **Control rate**：30 Hz 发指令（chunk 内多步）

---

## 5. 真机评测（Success Rate）

**协议：** \(SR = \#成功 / N\)，成功 = 黄→棕→白叠稳 + 时限内 + 零干预。

### 5.1 ~20 epoch（ckpt `120000`）小样本

| Trial | 结果 | 备注 |
|-------|------|------|
| 1 | **成功** | — |
| 2 | **成功** | — |
| 3 | **失败** | 棕色已放到黄色上；白色最后往叠好的碗上放时失败 |
| **SR** | **2/3 ≈ 67%** | \(N=3\)，仅定性趋势；正式表建议 \(N\ge10\) |

**主观结论：** ~20 epoch 附近策略已能完成多阶段叠碗，但末段白碗放置仍脆。

### 5.2 行为现象（可写进面试「failure analysis」）

#### A. 阶段门控 / 顺序依赖（compositional staging）

> 若 **brown 没有放到 yellow bowl 上**，策略往往 **也不会把 white 放到 yellow 上**。

解读：模型学到的更像 **视觉条件化的阶段技能串**（state-dependent skill chaining），而不是自由重排的「任意目标放置」。前序子目标未完成 → 后序子技能不触发或乱触发。这对长 horizon BC 很典型。

#### B. 示教伪影被模仿（demo artifact cloning）

采集时「距目标还有竖直距离就松爪」→ 推理时同样过早释放 → 碗歪、掉、叠不稳（尤其白碗）。

**改进方向：** 清洗/重采末段接近轨迹；成功标准要求接触/对齐后再松；或加接触相关奖励/过滤。

#### C. 末段失败为主

Trial 3：黄←棕已完成，白失败 → 误差累积 + 更高目标面 + 过早松爪叠加。

#### D. 中期 ckpt 优于训满（与双臂 handover 经验一致）

Loss 仍在降/很低，不代表真机 SR 上升；复杂双臂任务优先 **按真机 SR 选 ckpt / 早停**。

---

## 6. 工程踩坑（本实验相关）

| 问题 | 现象 | 处理 |
|------|------|------|
| USB 相机掉线 | `top_left` / `right` 间歇 disconnect；`by-id` 消失 | 拔插 / 换口；推理前 `v4l2-ctl --list-devices` |
| 旧 preview / `find-cameras` 占设备 | OpenCV FAIL / 卡住数分钟 | `fuser` 杀进程；少用扫全 `/dev/video*` |
| Sonix ↔ HBVCAM 枚举名变化 | 右相机 by-id 对不上 | 以当前 `/dev/v4l/by-id` 为准覆盖 `RIGHT_CAM` |
| 训练脚本注释 vs 日志 epch | 注释写 ~20 ep，日志印 ~40 | 面试报 **step** 或 **frames 公式 epoch** |

---

## 7. Offline vs Robot（别混）

| 模式 | 测什么 | 不代表什么 |
|------|--------|------------|
| Offline MAE | pred vs GT 关节 | 真机叠碗成功 |
| Replay | 回放数据集 | 闭环感知纠错 |
| Robot SR | 零干预任务成功 | 需要固定协议 + 足够 \(N\) |

本任务 offline 可作健康检查；**结论以真机 SR + 失败模式为主**。

---

## 8. 面试怎么讲（60–90 秒）

> 我做了双臂 SO101 三色叠碗：200 ep / ~9.9 万帧 / 4 相机 / 12 维动作，SmolVLA 训 25 万 step（约 20 data-epoch）。真机发现中期 ckpt（~20 ep，`120000`）比一味训满更稳。小样本评测 3 次成功 2 次；失败多在白碗末段。还观察到阶段依赖——棕碗没叠上时白碗也不会往黄碗上放——以及示教里过早松爪被策略克隆。说明长 horizon BC 学的是视觉条件技能链，数据伪影会直接进策略；评测要用零干预 SR，并用中期 checkpoint 做早停。

**可能追问 & 答法：**

1. **为何 20 ep 比 40 好？**  
   BC 过拟合示教节奏/绝对关节；长任务误差敏感 → 用真机选 ckpt。

2. **阶段门控说明什么？**  
   策略是条件技能串，不是开放重规划；可用「打断前序」做失败注入验证。

3. **如何提高 SR？**  
   清洗松爪；加对齐示教；增末段多样性；试 chunk/`n_action_steps`；正式 \(N\ge10\) 对比 15/20/25 ep。

4. **Latency 怎么说？**  
   SmolVLA ~150 ms/前向 << 30 Hz 预算 → chunking；episode 150 s 是任务时限。

---

## 9. 数字速查

```text
Task        : yellow → brown → white stack (bimanual)
Episodes    : 200
Frames      : 98,616 @ 30 FPS
DoF / cams  : 12 / 4
Train steps : 250,000  (eff. bs=8, ~11.6 h on 2×3090)
Data-epoch  : ~20.3 (full run); best robot feel ~20 → ckpt 120000
Infer       : 150 s/ep, n_action_steps=5, FPS=30
Latency ref : SmolVLA ~148 ms/select_action (3090, other task)
Robot SR    : 2/3 @ ckpt 120000 (N=3, informal)
Fail mode   : white final place; staging gated on brown; early release from demos
```

---

## 10. 路径速查

| 用途 | 路径 |
|------|------|
| 数据 | `/home/rxn/datasets/bimanual_stack_bowls_yellow_brown_white` |
| 训练输出 | `outputs/train/smolvla_bimanual_stack_bowls` |
| ~20 ep 权重 | `.../checkpoints/120000/pretrained_model` |
| ~25 ep 权重 | `.../checkpoints/150000/pretrained_model` |
| 推理 | `bash infer_smolvla_bimanual_stack_bowls_ep20.sh robot` |
| 日志 | `logs/smolvla_bimanual_stack_bowls.log` |
