# HIL-SERL 续训 → 安全约束 → 换舵机：详细步骤笔记

面向学习笔记 / 求职面试。比总览版更细：**每一步做什么、多少 episode、checkpoint、关键数字、踩坑**。

相关总览：`notes/hilserl-push-black-block-experiment.md`（早期完整实验）  
本文焦点：续训阶段 + EE/关节安全 + id4 更换 + 双校准分流。

---

## 0. 面试 30 秒

> SO101 上 HIL-SERL 推黑块：25 条演示 → 奖励分类器 → Actor–Learner。早期约 70k 更新、日志成功率 ~76%，但零干预仍不稳。续训收到 checkpoint **174000**；用 `find-joint-limits` 收紧 EE 盒、clip `wrist_roll`/`shoulder_pan`；左臂 id4 过载更换后，建立 **infer 旧校准 / collect 新校准** 两套文件，避免旧 VLA 与新硬件坐标系冲突。

---

## 1. 实验设定（固定参数）

| 项 | 数值 |
|----|------|
| 机器人 | SO101 follower + leader（`control_mode=leader`） |
| 任务 | `push the black block` → 白纸 |
| 算法 | HIL-SERL = SAC + 视觉 Reward Classifier |
| 动作 | EE delta：`Δx,Δy,Δz`（步长 8mm）+ gripper |
| 控制频率 | **5 Hz**（配置 `env.fps=5`） |
| 相机 | front + wrist；训练侧 crop → **128×128** |
| Learner batch | **256** |
| Online steps（配置上限） | **100,000**（可 resume 继续存 ckpt） |
| Home（reset 关节角 °） | `[-5.1, -102.5, 97.5, 71.8, 0.0, 3.0]` |
| 输出 | `outputs/train/hilserl_so101_push_black_block` |
| 配置 | `configs/hilserl_so101_push_black_block_train.json` |

### 干预键

| 键 | 作用 |
|----|------|
| Space | 干预开/关（先 leader→follower 对齐） |
| s | 本局成功 |
| q | 本局失败 |
| r | 重录 |

---

## 2. 数据与 Episode 账本

### 2.1 离线演示（Step 2–3）

| 指标 | 数值 |
|------|------|
| Episodes | **25** |
| Frames | **3,531** |
| 数据集 FPS | **10**（演示录制侧） |
| 标注 | `s`/`q`；起点可小幅随机 |
| 路径 | `/home/rxn/datasets/hilserl_push_black_block_demos_cropped_resized` |

### 2.2 奖励分类器（Step 4）

| 指标 | 数值 |
|------|------|
| 打包后 | **9 ep / 570 frames**（64 帧打包，非只用 9 条 demo） |
| 正样本 | 成功尾帧 `success_tail_frames=15` |
| 负样本 | 失败抽样，约 1:1 |
| 路径 | `/home/rxn/datasets/hilserl_push_black_block_reward_classifier` |
| Checkpoint | `outputs/train/reward_classifier_so101_push_black_block/checkpoints/last` |

### 2.3 在线 RL（Step 5，早期成文时）

| 指标 | 数值 |
|------|------|
| 在线 dataset | **~416 ep / ~31,216 frames** |
| Actor 日志局数 | **~430** |
| Offline buffer | 演示 25 ep |

### 2.4 日志成功率（非零干预评测）

| 窗口 | 结果 |
|------|------|
| 总体 | **~328 / 430 ≈ 76%**（reward>0） |
| 最近 30 局 | ~22/30 ≈ 73% |
| 最近 50 局 | ~38/50 ≈ 76% |

> **解读：** reward 含分类器触发 + 人工 `s`；**不能**当成零干预 success rate。当时结论：行为已收敛，但「自己推成功」仍不够。

### 2.5 Checkpoint 进度（续训）

| 阶段 | Checkpoint | 说明 |
|------|------------|------|
| 早期笔记 | **070000** | ~70.2k SAC 更新 |
| 续训中 | **138000** | 停训检查时 |
| 当前 last | **174000** | `checkpoints/last → 174000` |

保存频率约每 2k 一步（可见 `002000…174000`）。

---

## 3. 详细步骤（按时间顺序）

### Phase 0 — 官方流水线（回顾，已完成）

```text
1) 录演示 25 ep（leader 遥操作）
2) crop ROI → cropped_resized 数据集
3) relabel → 训 reward classifier
4) 开 learner + actor 在线 HIL-SERL
5) Space 干预纠偏；s/q 结束 episode
```

脚本参考：`hilserl_push_black_block_steps.sh`、`train_hilserl_so101.sh`。

---

### Phase 1 — 判断要不要续训

| 步骤 | 内容 |
|------|------|
| 1.1 | 观察：策略仍**不能稳定自主推成功** |
| 1.2 | 结论：日志 76% ≠ 可用；**应续训**，但要高质量干预，非空跑 |
| 1.3 | 评测协议（建议）：零干预连续 **10–20 局**，肉眼是否进白纸；目标 ≥70% 再停 |
| 1.4 | 命令：`RESUME=1 bash train_hilserl_so101.sh learner` → 另开 `bash train_hilserl_so101.sh actor` |

**续训操作要点**
- 每局先让策略试；偏了再 Space，尽量纠到成功再 `s`
- 目标再跑 **100–200 局** 或干预率明显下降
- 勿 `FRESH=1`（会删 checkpoint）

---

### Phase 2 — 总线故障（插曲）

| 步骤 | 现象 / 处理 |
|------|-------------|
| 2.1 | Episode 成功后 reset home 时：`Failed to sync read Present_Position … no status packet` |
| 2.2 | Actor 崩溃（core dump）；本局 reward 已记上 |
| 2.3 | 处理：查供电/USB → 重启 actor；learner 可继续 |
| 2.4 | 教训：真机 RL 连续性 ≈ 算法 + **总线健康** |

---

### Phase 3 — 收紧 EE 安全盒

**动机：** 开局大左转/乱甩；旧 bounds 过宽（`z` 最高到 ~0.5 m）。

| 步骤 | 内容 |
|------|------|
| 3.1 | 停 actor（串口占用） |
| 3.2 | 跑 `lerobot-find-joint-limits`（warmup 5s + 录制 ~60s，只走推块任务区） |
| 3.3 | 测得 EE（米）：`min=[0.1328,-0.0975,-0.0238]`，`max=[0.4279,0.0883,0.0466]` |
| 3.4 | 写入配置（约 ±1.5 cm 余量） |

**Bounds 演变**

| 版本 | min | max | 问题 |
|------|-----|-----|------|
| 旧（过宽） | `[-0.031,-0.363,-0.015]` | `[0.415,0.411,0.506]` | 高抬、大侧摆 |
| find-joint 原始 | `[0.133,-0.098,-0.024]` | `[0.428,0.088,0.047]` | `z_min<0` 易戳桌 |
| 余量版 | `[0.118,-0.113,-0.034]` | `[0.443,0.103,0.062]` | 仍砸/戳桌 |
| **最终** | **`[0.118,-0.113,0.028]`** | **`[0.443,0.103,0.062]`** | `z_min=0.028`；`max_ee_step_m=0.02` |

**步长**

| 项 | 值 |
|----|-----|
| `end_effector_step_sizes` | x/y/z = **0.008 m** |
| `max_ee_step_m` | **0.02**（曾从 0.05 降下来，减猛跳） |

**概念澄清**
- `end_effector_bounds` = 末端坐标系原点 **xyz 盒子**，不是舵机 ID 限位
- SO101 舵机 ID 为 **1–6**（无 id 0）；口语「关节 0」= `shoulder_pan` = **id 1**

---

### Phase 4 — 关节 Clip（补 EE 管不到的姿态）

**动机：** 夹爪尖戳桌、开局大左右转；EE 盒不管 `wx/wy/wz` / 腕滚转。

| 步骤 | 内容 |
|------|------|
| 4.1 | 新增 `ClipJointPositions`（IK / leader 出关节角之后） |
| 4.2 | 配置 `joint_position_bounds` |
| 4.3 | 重启 actor 生效 |

| 关节 | 电机 ID | Clip（°） | 作用 |
|------|---------|-----------|------|
| `shoulder_pan` | 1 | **[-25, 15]** | 限制左右大转（home≈-5°） |
| `wrist_roll` | 5 | **[-20, 20]** | 限制腕滚转戳桌 |

`wrist_flex`（id 4）未默认硬 clip——推块需要俯仰；异常时再收。

---

### Phase 5 — 硬件诊断与换 id4

#### 5.1 故障表现

| 现象 | 检测结果 |
|------|----------|
| Leader/Follower 对不齐、抬不起 | 左 follower **id3/id4 Overload** |
| Homing mismatch | 单臂 `so101_follower` vs 双臂 `bimanual_follower_left` 混用；Debug 工具改过 Homing |
| `Missing motor IDs: 4` | 新舵机未以 **ID=4** 出现在总线 |
| calibrate / teleop 失败 | id4 掉线或过载时无法握手 |

#### 5.2 处理步骤（可复现清单）

| # | 步骤 | 通过标准 |
|---|------|----------|
| 1 | 停 actor / record | 串口空闲 |
| 2 | 左臂**断电 20–30 s** | 清 Overload |
| 3 | 手掰肘/腕，确认不卡 | 可活动 |
| 4 | 上电，`check` / ping id1–6 | **6/6 OK** |
| 5 | 若缺 id4：设电机 ID=4 并查线 | ping 到 id4 |
| 6 | 确认换电机 → **完整校准 `c`** | Homing 写入新文件 |
| 7 | 成对校左 leader（采数据时） | 同姿态度数接近 |
| 8 | 再开遥操作 / 采集 | 左右臂跟手 |

#### 5.3 校准文件（换电机后）

**新校准示例（`collect`，2026-07-20 21:59）**

| 关节 | Homing（新） |
|------|----------------|
| shoulder_pan | 425 |
| shoulder_lift | 589 |
| elbow_flex | -61 |
| wrist_flex | **-2005** |
| wrist_roll | -946 |
| gripper | -519 |

**旧校准（`infer`，恢复自换电机前 Homing）**

| 关节 | Homing（旧） |
|------|----------------|
| shoulder_pan | 409 |
| shoulder_lift | 475 |
| elbow_flex | 5 |
| wrist_flex | **-7** |
| wrist_roll | -1090 |
| gripper | -528 |

> 旧 VLA 数据在旧坐标系下采集；新 Homing 改变同一物理姿势的角度读数 → **必须分流**。

#### 5.4 双校准分流（工程规范）

| 文件 | 用途 | wrist_flex Homing |
|------|------|-------------------|
| `bimanual_follower_left.infer.json` | **旧模型推理** | -7 |
| `bimanual_follower_left.collect.json` | **新采数据/遥操作** | -2005 |
| `bimanual_follower_left.json` | 当前生效（由脚本切换） | — |

```bash
bash switch_bimanual_left_calib.sh infer     # 推理
bash switch_bimanual_left_calib.sh collect   # 采集
bash switch_bimanual_left_calib.sh status
```

切换后启动若 mismatch → **Enter**（写回对应文件）；只有要重校时才 `c`。

备份目录：`calibration_backups/`。

---

### Phase 6 — 双臂试采脚本（并行线）

为避开 HIL 占串口、并做电机预检，增加试跑脚本：

```bash
pkill -f 'lerobot.rl.actor' || true
bash get-data-bimanual-try.sh check          # 端口+相机+双臂舵机
NUM_EPISODES=20 bash get-data-bimanual-try.sh
```

默认试跑 **5 ep**；可 `NUM_EPISODES=20`。预检失败（如 id4 OVERLOAD）应先修硬件再采。

---

## 4. 数字总表（方便背）

| 类别 | 数字 |
|------|------|
| 演示 ep / frames | **25 / 3,531** |
| 分类器打包 | **9 ep / 570 frames** |
| 早期在线 | **~416 ep / ~31k frames**；日志局 **~430** |
| 日志 SR | **~76%**（非零干预） |
| SAC 更新（早期） | **~70k** → ckpt **070000** |
| 续训后 last | **174000** |
| Online 配置上限 | **100,000**（可超此继续存盘） |
| Batch | **256** |
| 控制频率 | **5 Hz** |
| EE 步长 | **8 mm**；单步上限 **20 mm** |
| 最终 z_min | **0.028 m** |
| shoulder_pan clip | **[-25°, 15°]** |
| wrist_roll clip | **[-20°, 20°]** |
| 换电机 | 左 follower **id 4 = wrist_flex** |

---

## 5. 结论（学习 / 面试）

1. **算法有效，评测未完：** 行为收敛 + 日志 ~76%；缺严格零干预 N 局 SR。  
2. **安全要分层：** EE 盒管位置；关节 clip 管姿态；`z_min` 不能进桌面。  
3. **续训靠干预质量：** 空跑失败轨迹会强化坏策略。  
4. **真机瓶颈常在硬件：** Overload、丢包、ID 未设置、校准混用。  
5. **换电机 = 换坐标系：** 重校准；旧 checkpoint 用 `infer` 校准；新数据用 `collect`。  
6. **Leader/Follower 成对校：** 只校一侧，遥操作必歪。  

---

## 6. STAR（面试失败案例）

- **S：** 续训时砸桌、对不齐，后 id4 损坏。  
- **T：** 提高安全并恢复可采集/可推理。  
- **A：** find-joint-limits → 抬 z_min → 关节 clip → 诊断 Overload → 换 id4 → 双校准分流。  
- **R：** 约束落地；采集与旧推理可切换；明确日志 SR ≠ 零干预成功。

---

## 7. 相关路径

| 路径 | 说明 |
|------|------|
| `notes/hilserl-push-black-block-experiment.md` | 早期完整结论 |
| `notes/hilserl-to-motor-swap-study-notes.md` | 本文（详细步骤版） |
| `configs/hilserl_so101_push_black_block_train.json` | EE + joint bounds |
| `train_hilserl_so101.sh` | learner / actor |
| `switch_bimanual_left_calib.sh` | infer ↔ collect |
| `get-data-bimanual-try.sh` | 双臂试采 + 预检 |
| `outputs/train/hilserl_so101_push_black_block/checkpoints/174000` | 当前 last |

---

## 8. 一句话收尾

真机 HIL-SERL 的交付物不只是 SAC 权重，还包括：**episode 账本、安全盒参数、关节限位、校准版本、以及换硬件后的坐标系管理**。
