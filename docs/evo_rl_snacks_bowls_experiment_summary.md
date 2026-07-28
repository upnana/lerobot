# Evo-RL ACP 实验总结  
## 双臂 SO101 — 零食入盘 + 叠碗（v1）

> 学习 / 求职用。数字来自 `snacks_bowls_v1`（2026 年 7 月）。  
> 含 v2 与新任务的总览见：`docs/evo_rl_study_job_note_jul2026.md`

---

## 电梯演讲（30 秒）

在**双臂 SO101**上跑通端到端 **Evo-RL / ACP**：HITL 采集 → 清洗 → `pistar06` value → value infer（advantage / ACP）→ SmolVLA ACP 策略 → 真机 A/B（`Advantage: positive` vs 无 tag）。

**结果：** 流水线通，但约 **69** 条长程 episode 上策略**学不会任务**，闭环多为 **hold / 微抖**。低 train loss ≠ 真机成功。根因是 **数据规模 × 任务难度**，不是硬件断连，也不是单靠 ACP tag 开关。

---

## 流水线

```text
采集（HITL 遥操作 + s/f）
    → 清洗（丢掉极短脏集）
    → Value 训练（pistar06）
    → Value infer（每帧 value / advantage / acp_indicator）
    → ACP 策略训练（SmolVLA + Advantage 标签）
    → 真机评测（有 ACP vs 无 ACP）
```

---

## 配置

| 项目 | 规格 |
|---|---|
| 框架 | Evo-RL（LeRobot fork），conda `evo-rl` |
| 机器人 | `bi_so_follower`，双臂 **12-DoF** |
| 相机 | 4 路：`left/right_wrist`，`left_top_left`（800×480 全视野），`right_top_right` |
| 控制 | 30 Hz，关节位置 |
| 算力 | 2× GPU（value / policy 均 DDP） |
| 策略 | SmolVLA（`smolvla_base` + SmolVLM2） |
| Value | `pistar06`（SigLIP + Gemma-3-270M + 201-bin 分布头） |

**任务文案**

```text
Put both snack packages into the yellow tray, and stack the yellow, brown,
and white bowls from bottom to top onto the white plate.
```

- **成功（`s`）：** 两袋零食进黄 tray **且** 碗按黄→棕→白叠在白盘上。  
- **失败（`f`）：** 缺物、掉落、顺序错、未完成。

---

## 1. 数据采集

| 项目 | 数值 |
|---|---|
| 脚本 | `get-data-evo-rl-bimanual.sh` |
| 工具 | `lerobot-human-inloop-record` |
| 模式 | 以**纯遥操作 demo**为主（leader→follower）；本轮几乎无策略干预 |
| 标签 | 键盘 `s` / `f` → `episode_success` |
| 单集上限 | `EPISODE_TIME_S=210`，`RESET_TIME_S=20` |
| **原始集** | **74 条 / 76,234 帧**（`*_before_clean`） |
| 成功率（原始） | ~48 / 26 ≈ **65%** |
| 平均时长（原始） | ~34 s |

快捷键：`i` 干预，`s`/`f` 结束，方向键提前停 / 重录。

---

## 2. 清洗

| 项目 | 数值 |
|---|---|
| 规则 | 丢掉 **&lt;10 s**（误触 / 垃圾） |
| 删除 | ep `3, 27, 34, 43, 73`（含 0.6s 却标 success） |
| 方法 | `delete_episodes`；备份 `*_before_clean` |
| **训练集** | **69 条 / 75,693 帧 ≈ 0.70 h** |
| 成功率（清洗后） | **47 / 22 ≈ 68%** |
| 长度 | 最短 767 / 均值 **1097** / 最长 1434 帧 → 均值 **~36.6 s** |

正式路径：`/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate`  
备份：`/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate_before_clean`

---

## 3. Value 训练（`pistar06`）

| 项目 | 数值 |
|---|---|
| 脚本 | `train-value-evo-rl.sh` |
| Steps | **8,000**（4k、8k 存盘） |
| Batch | 每卡 4 × 2 → **有效 8** |
| 约 epoch | \(8000 × 8 / 75693 ≈ **0.85**\) |
| 精度 | bf16 + 梯度检查点 |
| 骨干 | Gemma-3-270M（ModelScope）+ SigLIP（HF 缓存） |
| 墙钟 | ~**2.5–3 h**（~1.2 s/step） |
| Loss | ~5.3 → **~4.26**（201 bins 上 soft CE） |
| 输出 | `Evo-RL/outputs/value_train/snacks_bowls_v1/checkpoints/` |

**监督信号（面试点）：** 不是手标逐步 reward。目标由 `episode_success` + 剩余步数构造，归一到 \([-1,0]\)。接近成功结束 → 目标靠近 0；失败额外惩罚。Loss = 投影直方图上的 soft 交叉熵（HL-Gauss 风格）。

**工程：** 4 相机 batch=32 OOM → batch=4 + grad ckpt；混分辨率；HF 门控 → 本地路径。

---

## 4. Value infer → ACP 标签

| 项目 | 数值 |
|---|---|
| 脚本 | `value-infer-evo-rl.sh` |
| Batch | 每卡 8 × 2 |
| 吞吐 | ~**1.9 it/s**，**4731** batch ≈ **~40 min** |
| `n_step` | **50** |
| `positive_ratio` | **0.3** |
| 写入字段 | `value_snacks_bowls_v1`，`advantage_...`，`acp_indicator_...` |
| 实测正样本率 | **30.0%** |
| Value 均值 | ≈ **-0.35**；advantage 均值 ≈ **0** |

### 逐帧 advantage（是的，每一步）

\[
A_t = \sum_{k=0}^{n-1} r_{t+k} + V_{t+n}\,\mathbf{1}_{\text{同一 episode}} - V_t
\quad (n=50)
\]

若 \(A_t\) 落在该任务 **最高 30%**，则 `acp_indicator=1`。

→ 75,693 帧 → 75,693 个 advantage / indicator。

**训策略时：** indicator → 任务文本 `Advantage: positive/negative`（约 30% dropout 整段 tag）。  
**部署时：** 常整段固定 `Advantage: positive`（不再在线算 \(A_t\)）。

---

## 5. ACP 策略训练（SmolVLA）

| 项目 | 数值 |
|---|---|
| 脚本 | `train-policy-evo-rl-acp.sh` |
| 初始化 | `lerobot/smolvla_base` + 本地 SmolVLM2 |
| Steps | **170,000**；LR 衰减对齐 170k；warmup 1k；峰值 **1e-4** |
| Batch | 每卡 4 × 2 → **有效 8** |
| 约 epoch | \(170000 × 8 / 75693 ≈ **18**\) |
| Chunk（训练） | `n_action_steps = chunk_size = 50` |
| 相机映射 | `left_top_left→camera1`，`right_top_right→camera2`，双腕→`camera3/4` |
| 墙钟 | **~8 h** |
| 单步 | **updt_s ≈ 0.16 s** |
| Loss | **0.94 → ~0.012** |
| Checkpoint | `.../smolvla_acp_snacks_bowls_v1/checkpoints/last` |

**注意：** 极低离线 loss **不代表**闭环能完成任务。

---

## 6. 真机推理

| 项目 | 数值 |
|---|---|
| 脚本 | `infer_smolvla_evo_rl_acp.sh` |
| 模式 | `lerobot-record` + 纯策略（无遥操作；`i` 忽略） |
| 部署 chunk | `n_action_steps=10` |
| 安全 | `max_relative_target=50` |
| 频率 | 30 Hz；约每 10 帧新 chunk（~333 ms 预算） |

### A/B：有 ACP vs 无 ACP

| Preset | 任务 tag | 定性结果 |
|---|---|---|
| `PRESET=n10` | + `Advantage: positive` | 臂收着 / **hold**；无任务进展 |
| `PRESET=n10_acp_none` | 无 ACP tag | 略抖 / 微伸；仍**无**成功抓放/叠碗 |

**定量（`n10_acp_none`，2251 帧 ≈ 75 s）：**

- `|action − state| ≈ 1.13`（动作≈当前关节）  
- 相邻帧状态变化：p50 **0**，p95 **0.16**

→ 近恒等指令。硬件链路活着；**模型没学出有用行为**。

---

## 结论

1. **根因 = 数据 × 任务难度，不是 ACP 开/关。**  
   ~0.7 h、69 条对长程双技能双臂任务太薄。ACP 能重加权好地段，不能发明几乎没覆盖的技能。

2. **训练指标 ≠ 真机成功。**  
   Policy loss ~0.01 仍可闭环 hold。要用 `|a−s|`、状态位移、视频任务完成度。

3. **有/无 ACP 都失败。**  
   无 tag 略抖一点；positive 更僵。瓶颈是行为覆盖 / BC 质量。

4. **工程收获：**  
   端到端 Evo-RL；混分辨率 4 相机；DDP OOM + grad ckpt；`rename_map`；YAML 安全传 `Advantage:`；用 action–state 残差诊断 hold。

5. **后续：**  
   - 用旧 stack-bowls ckpt 验硬件  
   - 扩到 **150–300+** 高质量 demo，或拆子任务  
   - 先强 BC，再 ACP；或 HITL 纠错  
   -（v2 已验证）中期 ckpt 优于过训后的 `last`

---

## 简历一句话

> 在双臂 SO101 上搭建端到端 **Evo-RL ACP**（HITL → value/`pistar06` → 逐帧 advantage → SmolVLA ACP → 真机 A/B）。在低 train loss 下诊断闭环 **hold** 为**数据受限**，并用 action–state 残差与 preset 消融量化。

---

## 可能的面试问答

**Q：advantage 每步都算吗？**  
算。每帧 n-step（\(n=50\)），再取最高 30% → 二值 ACP 标签。

**Q：ACP 算「RL」吗？**  
更接近 **离线 advantage 条件化模仿**：value → n-step advantage → 条件 BC。不是在线 PPO。

**Q：为何 value 不到 1 个 epoch？**  
8k×8/75k≈0.85；够跑标注/ACP。本轮瓶颈不在 value 精度。

**Q：只改一件事？**  
**更多高质量 demo，或更容易的任务。** 换 preset 救不了 hold 策略。

---

## 关键路径

| 角色 | 路径 |
|---|---|
| 训练数据 | `/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate` |
| 清洗前备份 | `/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate_before_clean` |
| Value ckpt | `Evo-RL/outputs/value_train/snacks_bowls_v1/checkpoints/` |
| Policy ckpt | `Evo-RL/outputs/train/smolvla_acp_snacks_bowls_v1/checkpoints/last` |
| 采集 | `get-data-evo-rl-bimanual.sh` |
| Value 训/推 | `train-value-evo-rl.sh`，`value-infer-evo-rl.sh` |
| Policy 训/推 | `train-policy-evo-rl-acp.sh`，`infer_smolvla_evo_rl_acp.sh` |
| 总览（含 v2/新任务） | `docs/evo_rl_study_job_note_jul2026.md` |
