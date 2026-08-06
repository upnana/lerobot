# Collect Items · PI0.5 · GR00T 实验整理

从最早「桌上物品入盘」到蓝块精抓，把 **collect items** 主线，以及 **PI0.5 / GR00T** 的 train / infer 记录收成一页。  
更全时间线见 `notes/interview-all-experiments.md`；白块底盘细节见 `docs/pi05_experiment_summary.md`。

---

## 0. Collect Items 一句话总结

> 早期任务是 **多物体 → 白盘 / 黄盘**（任务串泛、物体多），数据量大（110–121 ep），用远程 **PI0.5** 训到 ~37–40 epoch，但 **本机未做系统真机 SR**。随后收敛到 **单白块 / 单蓝块 → 黄盘**，才形成可对比的三模型闭环。核心教训：多物体入盘适合「先跑通采集–训练链路」，不适合做干净消融；精细抓取上 **训完的 SmolVLA > 欠训 PI0.5**；**GR00T projector-only 延迟最低但真机抓取不稳**。

---

## 1. Collect Items 阶段（按时间）

### Phase A · 桌上物品 → 白盘（~7/1）

| 项 | 内容 |
|----|------|
| Task | `Put all items on the table into the white plate`（表述偏泛） |
| 数据 | `20260701_145251`：**121 ep / 108,405 fr** · front+wrist |
| PI0.5 | 远程 A100 · **~110k steps · ~37 epoch · bs=16** → 权重 `outputs/train/pi05_110000_trainweight` |
| GR00T | 本地计划 **220k**（`logs/groot_pickup_items.log`）；日志早期有 step，**未见 End of training**（未确认训满） |
| 真机 | **未在本机系统验收**（无严格 SR） |
| 后续 | 并入 `smolvla_multitask_merged`（182 ep / 149k fr） |

### Phase B · 桌上物品 → 黄盘（~7/6）

| 项 | 内容 |
|----|------|
| Task | `Pick up items on the table and put them into the yellow tray` |
| 数据 | `20260706_yellow_tray`：**110 ep / 79,973 fr** |
| PI0.5 | 远程 **~200k steps · ~40 epoch** |
| GR00T | 计划 220k；checkpoint 见 `outputs/train/groot_yellow_tray/checkpoints/`（含 `080000`） |
| 真机 | 同 Phase A，**未严格本机评测** |

### Phase C · 任务收敛：白块 → 黄盘（~7/8–7/12）

| 项 | 内容 |
|----|------|
| Task | `pick up the white block and put it into the yellow tray` |
| 数据 v1 | `yellow_white`：**60 ep / 24,055 fr** |
| 数据 v2 | 换底盘 `yellow_white_base2*`：**50 ep / ~19.7k fr** |
| 数据 merge | `yellow_white_merged`：**110 ep**（底盘不一致 → **合并变差**） |

| 模型 | Steps / Epoch | Offline | 真机 |
|------|---------------|---------|------|
| **GR00T** | **120k**，~40 ep；**只训 projector**（DiT 冻） | — | **可用**（文档记载） |
| **SmolVLA** | **100k**，bs=4，~20 ep | MAE ~**0.8°** | **明显好于同数据 PI0.5** |
| **PI0.5** | 15k–40k 不等（远程） | MAE ~2–3° | 臂能动、**经常抓不住**（爪偏小） |

**Collect items 相关关键失败：**

1. 任务字符串太泛 → 难做语言/物体消融。  
2. **底盘移位 + 绝对关节**：base1 vs base2 `shoulder_pan` 差 ~**5.5°** → merge = 标签冲突。  
3. 多物体阶段无本机 SR → 面试只能讲「链路跑通 + 数据规模」，不能吹成功率。

### Phase D · 蓝块 → 黄盘（~7/14–7/17）**可对比主实验**

| 项 | 内容 |
|----|------|
| Task | `pick up the blue block and put it into the yellow tray` |
| 数据 | `blue_block_yellow_tray`：**90 ep / 28,877 fr** · 均长 ~10.7 s |
| 详注 | `notes/smolvla-blue-block-experiment.md` · `notes/pi05-blue-block-experiment.md` |

同数据三模型：

| 模型 | Steps | Epoch | Offline MAE | 真机（分布内） | Latency |
|------|-------|-------|-------------|----------------|---------|
| **SmolVLA** | **40k ✅** | ~**22** | **~0.95°** | **基本成功** | ~**148 ms** |
| **PI0.5** | **20k** / 计划 50k | ~**11** | ~**2.9°** | **弱于训完 SmolVLA** | ~**158 ms** |
| **GR00T** | **80k ✅** | ~**22** | ~**1.16°** | 有接近+多次开合，**未干净成功** | ~**73 ms** |

---

## 2. PI0.5 Train / Infer 速查

### 2.1 训练记录（权重在本机）

| Job / 目录 | 数据 | Steps（约） | 备注 |
|------------|------|-------------|------|
| `pi05_110000_trainweight` | pickup 白盘 `20260701_145251` | **110k** | Collect items Phase A |
| （远程 yellow_tray） | `20260706_yellow_tray` | **~200k** | Phase B；本机未必有完整目录 |
| `pi05_yellow_white` | 白块 v1 | 计划偏大；本地曾试 bs 受限 | 本地双卡易 OOM |
| `pi05_yellow_white_base2_b16_50k_{15000,40000}` | base2 | **15k / 40k** | 真机仍抓不稳 |
| `pi05_yellow_white_merge_30000` | merged | **30k** | 离线 MAE~2.1，真机差 |
| `pi05_yellow_white_base3_from_base2_40k` | base3 | last zip | 续训线 |
| `pi05_blue_block_yellow_tray_{010000,020000}_pretrained_model` | 蓝块 | **10k / 20k** | **主对比 ckpt=020000**；计划 50k 未在本地评满 |

脚本默认（当前仓库）：`train_pi05.sh` → 默认数据已指向 **蓝块**，`STEPS` 默认 60k，本地 `BATCH_SIZE=2`。

### 2.2 推理

| 脚本 | 默认指向 |
|------|----------|
| `infer_pi05.sh` | `blue_block_yellow_tray` + **020000** 权重；`tolerance_s=0.05` |
| `infer_pi05_rtc.sh` | RTC 变体（减开环 chunk 误差用） |

```bash
# 离线 / 真机
bash infer_pi05.sh offline
bash infer_pi05.sh robot
# 换白块 base2 权重示例
CHECKPOINT=outputs/train/pi05_yellow_white_base2_b16_50k_40000/pretrained_model \
  DATA_ROOT=/home/rxn/datasets/yellow_white_base2 bash infer_pi05.sh robot
```

### 2.3 PI0.5 结论（面试用）

- Collect items 早期：远程训满、**本机无 SR**。  
- 白块：爪不开/不开够是主失败；**大模型 + 小数据 + 绝对关节** 难。  
- 蓝块：只评到 **20k≈40% 计划**，不能下「PI0.5 架构更差」的定论；只能说 **欠拟合大模型弱于训完小模型**。

---

## 3. GR00T Train / Infer 速查

环境：`conda: lerobot-groot` · `bash setup_groot_env.sh`  
设定共性：多轮实验为 **`tune_projector=true`，`tune_diffusion_model=false`（DiT 冻）**。

### 3.1 训练记录

| Job / 目录 | 数据 | Steps | 状态 / 真机 |
|------------|------|-------|-------------|
| `groot_pickup_items` | `20260701_145251`（121 ep） | 计划 **220k** | 日志早期有；**未确认 End of training** |
| `groot_yellow_tray` | `20260706_yellow_tray`（110 ep） | 计划 **220k** | 有 `080000` 等 ckpt |
| `groot_yellow_white` | 白块 60 ep | **120k ✅** | 文档：**真机可用** |
| `groot_yellow_white_base2` | base2 50 ep | 计划 120k | 至少见 `010000`；完成度需看目录 |
| `groot_blue_block_yellow_tray` | 蓝块 90 ep | **80k ✅** · ~22 ep | 离线 MAE~1.16°；真机 **未干净成功** |
| `groot_yellow_white_dit` | stage-2 DiT | — | **未完成**（诚实缺口） |

脚本：`train_groot.sh` / `train_groot_blue_block.sh` / `train_groot_dit.sh`。

### 3.2 推理

| 脚本 | 默认 |
|------|------|
| `infer_groot.sh` | 默认数据 **蓝块**；ckpt=`groot_blue_block_yellow_tray/.../last` |

```bash
conda activate lerobot-groot
bash infer_groot.sh offline
bash infer_groot.sh robot

# 白块可用线
CHECKPOINT=outputs/train/groot_yellow_white/checkpoints/last/pretrained_model \
  DATA_ROOT=/home/rxn/datasets/yellow_white bash infer_groot.sh robot
```

### 3.3 GR00T 结论

- **Latency 最好**（蓝块 ~73 ms），但 projector-only → 视觉条件弱，易重复套路 / 试抓。  
- 白块 120k 曾「可用」；蓝块 80k 仍未干净成功 → **快 ≠ 好**。  
- DiT 二阶段未完成 → 面试要主动说边界。

---

## 4. Collect Items → 精抓：叙事链（面试 60 秒）

1. **先多物体入盘**（121 / 110 ep）：把遥操作采集、PI0.5/GR00T 训练链路跑通；任务泛、无严格本机 SR。  
2. **收敛单物体白块**：发现底盘一动，绝对关节策略合并就崩；SmolVLA / 充分训的 GR00T 更实用。  
3. **控变量蓝块三模型**：同 90 ep；SmolVLA 训满真机可用 + 语言消融；PI0.5 只评到 40% steps；GR00T 最快但不稳。  
4. **工程点**：`tolerance_s=0.05`、相机占用、PI0.5 本地 OOM、GR00T DDP 挂起。

---

## 5. 相关路径索引

| 类型 | 路径 |
|------|------|
| 全实验总览 | `notes/interview-all-experiments.md` |
| PI0.5 白块/底盘长文 | `docs/pi05_experiment_summary.md` |
| PI0.5 蓝块 | `notes/pi05-blue-block-experiment.md` |
| SmolVLA 蓝块 | `notes/smolvla-blue-block-experiment.md` |
| Latency | `notes/inference-latency.md` |
| 语言消融 | `notes/vla-language-vs-vision.md` |
| 训练脚本 | `train_pi05.sh` · `train_groot*.sh` |
| 推理脚本 | `infer_pi05.sh` · `infer_groot.sh` |
| Collect 数据 | `/home/rxn/datasets/20260701_145251` · `20260706_yellow_tray` |

---

*整理：2026-07-21 · 来源：既有 notes/docs + train logs + outputs/train 目录*
