# SO101 全实验复盘（求职面试版）

从最早「桌上物品入盘」到双臂叠碗，按时间线整理：**任务 → 数据 → 训练 → 离线/真机结果 → 失败案例与原因 → 工程问题**。  
定量数字优先；只有观察没有严格 SR 的标为「定性」。

---

## 0. 一句话总览（面试开场 30 秒）

> 我在 LeRobot + SO101 上做了完整端到端闭环：单臂/双臂数据采集、SmolVLA / PI0.5 / GR00T N1.5 行为克隆、HIL-SERL 在线强化学习，以及语言–视觉消融与推理延迟对比。核心结论是：同数据下 **SmolVLA 真机最稳**；**视觉主导、语言弱**；**绝对关节 + 底盘移位会毁掉跨数据合并**；双臂任务更易过拟合；工程上相机占用、DDP/CUDA、舵机电压是高频坑。

---

## 1. 实验时间线（按任务）

### 1.1 早期多物体入盘（~7/1–7/6）

| 项 | Items → 白盘 | Items → 黄盘 |
|----|--------------|--------------|
| Task | Put all items… into the white plate | Pick up items… into the yellow tray |
| 数据 | `20260701_145251`：**121 ep / 108k fr** | `20260706_yellow_tray`：**110 ep / 80k fr** |
| 策略 | PI0.5（远程 A100） | PI0.5（远程） |
| 训练 | ~110k step，~37 epoch | ~200k step，~40 epoch |
| 真机 | 本机 SO101 **未系统验收** | 同左 |
| 后续 | 合并进 `smolvla_multitask_merged`（182 ep / 149k fr） | — |

**问题：** 任务字符串偏泛、物体多，难做干净消融；本地未做严格 success rate。

---

### 1.2 白块 → 黄盘（~7/8–7/14）「精细抓取」主线起步

| 项 | 内容 |
|----|------|
| Task | `pick up the white block and put it into the yellow tray` |
| 数据 v1 | `yellow_white`：**60 ep / 24k fr** |
| 数据 v2/v3 | 换底盘后：`yellow_white_base2*`、`yellow_white_base3`（91 ep）、merged（110 ep） |

#### 训练与结果

| 模型 | Steps / Epoch | Offline | 真机 |
|------|---------------|---------|------|
| GR00T N1.5 | **120k**，~40 ep；**只训 projector**（DiT 冻） | — | **可用**（文档记载） |
| SmolVLA base2 | **100k**，bs=4，~20 ep，loss~0.014 | MAE ~**0.8°** | **明显好于同数据 PI0.5**，爪开合更正常 |
| PI0.5 多版 | 15k–40k 不等 | MAE ~2–3° | 臂能动但 **经常抓不住**；爪角度常偏小 |

#### 关键失败与原因

1. **底盘移位 + 绝对关节**  
   base1 vs base2 shoulder_pan 均值差 ~**5.5°** → 合并训练等于标签冲突 → **合并模型比单底盘更差**。  
   *面试点：绝对关节策略对标定/底座敏感；换底座必须重采或做相对动作。*

2. **PI0.5 本地双卡 OOM** → 依赖远程 A100；本地对比不完整。

3. **GR00T 仅 projector** → 臂会动但对视觉条件弱，易重复套路、抓不准（后续蓝块也类似）。

---

### 1.3 蓝块 → 黄盘（~7/14–7/17）**最完整、面试主打**

| 项 | 内容 |
|----|------|
| Task | `pick up the blue block and put it into the yellow tray` |
| 数据 | `blue_block_yellow_tray`：**90 ep / 28,877 fr**，均长 ~10.7 s |
| 相机 | front(UGREEN) + wrist(Sonix)，640×480@30 |

#### 同数据三模型对比（控变量）

| 模型 | Steps | Epoch | Offline MAE | 真机（分布内） | Latency (select_action) |
|------|-------|-------|-------------|----------------|-------------------------|
| **SmolVLA** | **40k ✅** | 训满~**22**；真机~**20 ep** | **~0.95°** | **SR≈4/5 @~20 ep**；摆正/斜放可抓，竖着不行；位置泛化尚可 | ~**148 ms** (~7 Hz) · Duration **120 s** |
| **PI0.5** | **20k**（计划 50k 的 40%） | ~**11** | ~**2.9°** | **明显弱于训完的 SmolVLA**（定性） | ~**158 ms** |
| **GR00T N1.5** | **80k ✅** | ~**22** | f0~0.42° / avg~**1.16°** | **有接近+多次开合，未干净成功** | ~**73 ms** (~14 Hz) |

训练共性超参：`tolerance_s=0.05`（视频时间戳漂移）；SmolVLA：`train_expert_only`；GR00T：`tune_projector=true`，`tune_diffusion_model=false`。详注：`notes/smolvla-blue-block-experiment.md`。

**失败要点：** 竖着无数据→抓不到；夹持学了短边方向、未学长边；高 SR 因短任务+in-dist，非语言能力。

#### 语言–视觉消融（SmolVLA，面试高光）

| # | 语言 | 实物 | 结果 | 解读 |
|---|------|------|------|------|
| 1 | 正确 blue | **黑块** | 有抓取意图，**抓不准** | OOD 外观失败 |
| 2 | **错误 black** | **蓝块** | **仍能成功** | 忽略错误指令 |
| 3 | black | 黑块 | 仍失败 | 成败跟外观走，不跟语言 |

**结论：** 学到的是 **视觉条件 skill**，不是可靠语言跟随；单任务常数指令 → 语言几乎不提供因果信号。

#### 真机 Success 怎么算（协议）

\[
SR = \#成功 / N,\quad N\ge10
\]
成功需：正确物体入黄盘、时限内、无人工干预。  
本阶段多记 **定性**；HIL-SERL 另有日志成功率，勿混谈。

---

### 1.4 HIL-SERL 强化学习

#### A. 白块抓放（早期）
- Demo ~15 ep（EE 键盘）；在线 ~105 ep；工程坑多（奖励误触发、gRPC、leader 力矩）。
- **无完整正式 SR 表**。

#### B. 推黑块 → 白纸（有笔记）
| 项 | 值 |
|----|-----|
| Task | `push the black block` |
| Demo | 25 ep → crop/resize |
| 在线 | ~416–430 ep；SAC ~70k update |
| 日志成功信号 | ~**76%**（reward>0；含人工 `s`，非零干预标准评测） |
| 行为演变 | 早期乱抬 → 后期桌面高度前推 |

**面试点：** HIL-SERL = 人机协同 + 视觉奖励分类器 + actor–learner；权重与 VLA BC **不互通**。

---

### 1.5 双臂：胶带 handover（~7/12–7/17）

#### 3 相机版
| 项 | 值 |
|----|-----|
| 数据 | `bimanual_handover`：**80 ep / 65k fr**，12 DoF |
| 文本 | 偏泛（Grab and handover…）；真实任务含胶带→交接→黄盘 |
| SmolVLA | **150k**，loss→**0.001**（过拟合） |
| 真机 | 右臂抬起后 **悬停**；应用中期 ckpt 而非 150k |

#### 4 相机版（主库 / 真机主结论）
| 项 | 值 |
|----|-----|
| 数据 | `…20260717_150424`：**89 ep / 47k fr**，均长 ~17.6 s |
| Task | Grasp double-sided tape → hand over → yellow tray |
| SmolVLA | 计划 200k；**停在 ~120k**；甜区 **`060000`–`080000`** |
| Infer | Duration **65 s**，FPS 30，`n_action_steps=5`；latency 参考 ~148 ms |
| 真机 SR | 小样本 **≈2/3（≥50%）**；**第一臂空抓**为主，抓住后 pass 很少挂 |
| 硬件插曲 | 右臂 id=5 电压错误曾打断评测 |
| 详注 | `notes/smolvla-bimanual-handover-tape-experiment.md` |

---

### 1.6 双臂叠碗（完成一版，~7/19–7/20）

| 项 | 值 |
|----|-----|
| Task | `Stack the yellow bowl, then the brown bowl, and finally the white bowl.` |
| 数据 | **200 ep / 98,616 frames** / 4 cam / 12-DoF |
| 训练 | SmolVLA **250k steps**（eff. bs=8，~11.6 h，2×3090） |
| 较好 ckpt | **~20 data-epoch → `120000`**（非训满最好） |
| 真机 | 小样本 **2/3 ≈ 67%**（N=3）；失败多在白碗末段 |
| 现象 | ① 棕未上黄 → 白也不往黄上放（阶段门控）② 示教过早松爪被克隆 |
| 详注 | `notes/smolvla-bimanual-stack-bowls-experiment.md` |

另：`blue_white_stack` 脚本有、数据未采；`double_sided_tape_yellow_tray` 空目录。

---

## 2. 工程问题清单（面试「踩坑」）

| # | 问题 | 现象 | 根因 / 处理 |
|---|------|------|-------------|
| 1 | 视频时间戳 assert | 训练/离线加载炸 | 漂移~数十 ms → **`tolerance_s=0.05`** |
| 2 | 相机被占用 | robot / find-cameras 超时、FAIL | preview / HIL actor / 旧进程占 V4L → `fuser` 杀进程 |
| 3 | GPU 争用 | GR00T 与 HIL/多训并行卡死 | GR00T ~18GB/卡；错峰或单卡 |
| 4 | GR00T DDP 挂起 | 双卡训到前期无 step | 改 **单卡** 或杀进程重开 |
| 5 | CUDA unknown / `is_available=False` | 强杀 GPU 任务后 | 重载 `nvidia_uvm` |
| 6 | `Resume=1` 无效 | 新目录从 ep0 开始 | Bash 大小写 → 必须 **`RESUME=1` + DATA_ROOT** |
| 7 | PI0.5 本地 OOM | bs 稍大即炸 | 远程 A100 / 降 bs |
| 8 | Feetech 电压错误 | 右臂连不上 | 供电/总线/id=5 硬件 |
| 9 | 安全 clamp 过多 | 动作发涩、爪合不紧 | `max_relative_target` 过严或策略步子过大 |
| 10 | 过拟合 | loss 极低但真机悬停/脆 | 中期 ckpt + 早停；双臂尤其明显 |
| 11 | 绝对关节域偏移 | 换底盘合并变差 | 重采或相对动作 |
| 12 | 语言被忽略 | 错指令仍成功 | 单指令 BC 局限；需消融证明 |

---

## 3. Offline vs Replay vs Robot（别混）

| 模式 | 测什么 | 不测什么 |
|------|--------|----------|
| **Offline** | 数据集上 pred vs GT（MAE）、权重/预处理 | 真机成功率 |
| **Replay** | 回放 GT 动作 → 数据与硬件链路 | 策略能力 |
| **Robot** | 闭环任务 SR / 消融 | — |

控制环 30 fps（预算~33 ms）；VLA 单步 70–160 ms → 必须 **action chunk**。

---

## 4. 面试高频 Q&A（精简）

**Q: 为什么同数据 SmolVLA 比 PI0.5@20k 强？**  
A: 训练完成度不同（40k vs 20k）+ 架构/微调设定不同；公平对比要同 step/epoch。离线 MAE 与真机大致同向。

**Q: GR00T 更快为何真机不稳？**  
A: 延迟低≠任务好；projector-only 视觉条件弱 + clamp/试抓；需 DiT 二阶段与真机 SR。

**Q: 如何证明「没真正理解语言」？**  
A: 反事实：错指令+训练外观仍成功；对指令+OOD 外观仍失败。

**Q: 双臂为什么更难？**  
A: 12 DoF、多相机、交接相位误差放大；绝对关节闭环漂移；数据量相对任务复杂度不足时易过拟合。

**Q: 数据量怎么定？**  
A: 单臂单 skill ~90 可验证；双臂复杂叠放建议 **200+**，且位姿要覆盖随机性，质量优先于纯数量。

---

## 5. 数字速查卡

```text
Collect items白盘       : 121 ep / 108k | PI05远程110k | 本机无SR
Collect items黄盘       : 110 ep / 80k  | PI05远程200k | 本机无SR
Blue block dataset     : 90 ep / 28877 fr
SmolVLA                : 40k step ≈22 ep | MAE≈0.95° | robot OK | 148ms
PI0.5                  : 20k/50k ≈11 ep | MAE≈2.9°  | robot弱 | 158ms
GR00T                  : 80k ≈22 ep | MAE≈1.16° | 试抓多  | 73ms
HIL-SERL push (log)    : ~76% reward>0 (非严格零干预 SR)
Bimanual 4cam          : 89 ep / 47k | SmolVLA stop 120k≈41 ep | 甜区~20–27 ep
Stack bowls            : 200 ep / 98.6k | SmolVLA 250k | 甜区~20ep | 真机2/3
```

---

## 6. 相关笔记路径

- `notes/collect-items-pi05-groot-summary.md`（**collect items + PI05/GR00T train·infer 一页纸**）
- `notes/smolvla-blue-block-experiment.md`
- `notes/smolvla-bimanual-stack-bowls-experiment.md`
- `notes/smolvla-bimanual-handover-tape-experiment.md`
- `notes/pi05-blue-block-experiment.md`
- `notes/vla-language-vs-vision.md`
- `notes/inference-latency.md`
- `notes/hilserl-push-black-block-experiment.md`
- `docs/pi05_experiment_summary.md`（白块/底盘）
- `docs/bimanual_smolvla_project_summary.md`

---

## 7. 诚实缺口（面试加分：知道边界）

1. 多数真机结果是 **定性**，缺统一 N≥10 的 SR 表（GR00T / 双臂尤甚）。  
2. GR00T DiT stage-2：**未完成**。叠碗 SmolVLA：**已训完 + 小样本真机**（详见 `notes/smolvla-bimanual-stack-bowls-experiment.md`）。  
3. E2E（曝光→电机）延迟：**未测**（只有 policy latency）。  
4. HIL-SERL 日志成功率 ≠ 标准零干预评测。

---

*整理日期：2026-07-21 · 平台：SO101 + LeRobot · GPU：2×RTX 3090（+ 远程 A100 用于部分 PI0.5）*
