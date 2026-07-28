# Evo-RL 双臂 SO101 实验笔记（学习 / 求职）
## 从「零食+叠碗」(v1/v2) 到「叠碗+口红纸巾」(3 相机)

> 2026 年 7 月实验日志精简版，适合**学习笔记**与**面试讲述**。  
> v1 细目另见：`docs/evo_rl_snacks_bowls_experiment_summary.md`（已同步中文）

---

## 电梯演讲（30–45 秒）

我在**双臂 SO101（12 自由度）**上跑通了端到端 **Evo-RL / ACP** 流水线：人机协同遥操作采集 → episode 清洗 → `pistar06` 价值网络训练 → 逐帧 advantage / ACP 标注 → SmolVLA ACP 策略训练 → 真机闭环评测。

**核心结论：** 流水线真实可用，但**离线 loss ≠ 真机成功**。约 70 条长程 demo 时策略几乎 **hold**；扩到约 200 条后中期 checkpoint（约 4–6.5 epoch，50k–80k）会出现伸手/试抓，**后期 checkpoint（≥约 150k）又塌回 hold**。部署时强行加 `Advantage: positive` 反而可能让开局更僵。随后我改了任务设计，用**全视野顶视相机**采了更干净的约 150 条 / 1.5 小时数据，质量分析后训练时**去掉该全视野相机、改用 3 路相机**（对齐 SmolVLA 原生 3 视角），并重新跑 value → infer → policy。

---

## 时间线

| 阶段 | 做什么 | 结果 |
|---|---|---|
| **A. v1** | 零食入黄盘 + 叠碗；**69 条干净数据 / ~0.7h** | 流水线通；真机 **hold**；有/无 ACP 都失败 |
| **B. 诊断** | `\|a−s\|`、状态位移、旧 stack-bowls 冒烟 | **硬件正常**；问题在策略/数据，不是串口 |
| **C. 扩数据 → v2** | 续采至 **~201 条 / ~197k 帧** | 成功率 ~**71%**；value 25k；policy 目标 250k（停在 ~**220k**） |
| **D. v2 真机扫 ckpt** | 按 step/epoch 评测 | **中期 ~50–80k 会动/试抓**；**后期 ≥~150k 再 hold** |
| **E. 新任务（今天）** | 先叠碗上盘，再口红+纸巾入黄 tray | 固定阶段顺序；采集用 4 相机 |
| **F. 数据质检** | 删短脏集（ep63、ep135） | **150 条 / 158k / 1.46h**，成功 **~77%** |
| **G. 训练改设计** | 训练时去掉全视野 `left_top_left` | **3 相机** → value 25k → infer → policy 200k |

---

## 软硬件配置

| 项目 | 规格 |
|---|---|
| 框架 | Evo-RL（LeRobot fork），conda `evo-rl` |
| 机器人 | `bi_so_follower` / `bi_so_leader`，双臂 **12-DoF**，30 Hz 关节位置 |
| 算力 | 2× RTX 3090，DDP；bf16 + 梯度检查点 |
| 策略 | SmolVLA（`smolvla_base` + SmolVLM2） |
| 价值网络 | `pistar06`：SigLIP + Gemma-3-270M + **201-bin** 分布头 |
| 采集 | `lerobot-human-inloop-record`；键盘 `s`/`f` → `episode_success` |

### 相机（重点）

采集为 **4 路**（Evo-RL 命名 = 臂前缀 + 角色）：

| 数据字段 | 物理角色 | 分辨率 | 备注 |
|---|---|---|---|
| `left_wrist` | 左腕 | 640×480 | 近景抓取 |
| `right_wrist` | 右腕（Sonix USB） | 640×480 | 4 相机负载下易超时 |
| `right_top_right` | 右顶视 | 640×480 | 场景 / tray |
| **`left_top_left`** | **左顶视 / 全视野** | **800×480** | **全视野顶视**；混分辨率麻烦；采集保留，**今日训练去掉** |

**为何谈「全视野」：** `left_top_left` 是唯一非 640 宽的视角，桌面覆盖更好，但带来：(1) pistar06 需先 **resize 再堆叠**；(2) 4 相机抬高显存/IO；(3) SmolVLA 原生只要 **3** 个图像槽。今日训练映射：

```text
right_top_right → camera1
left_wrist      → camera2
right_wrist     → camera3
（不用 left_top_left）
```

**工程修复（USB 相机）：** OpenCV `async_read` 超时 **200ms → 1000ms**；超时后**复用上一帧**，避免整段采集被掐断。

---

## 方法：ACP 到底是什么（面试版）

```text
HITL 演示（s/f）
  → 用 episode 成功与否构造 value 目标（压到 [-1,0] 的 bins）
  → Value infer：每帧 V_t
  → n-step advantage A_t（n=50），最高 30% → acp_indicator
  → 策略 BC，条件文本 "Advantage: positive/negative"（约 30% dropout 掉 tag）
  → 部署：通常固定 tag 或无 tag（不再在线算 A_t）
```

\[
A_t = \sum_{k=0}^{n-1} r_{t+k} + V_{t+n}\,\mathbf{1}_{\text{同一 episode}} - V_t
\]

**不是在线 PPO。** 更贴切的叫法：**离线、advantage 条件化的模仿学习**。

---

## 实验 A — 零食 + 叠碗（v1）

**任务：** 两袋零食进黄 tray，并且黄→棕→白碗叠到白盘上。

| | |
|---|---|
| 原始 → 清洗 | 74 → **69** 条（去掉 &lt;10s 垃圾） |
| 帧数 | **~75.7k（~0.70 h）** |
| 成功率 | ~**68%** |
| Value | 8k steps（~0.85 epoch） |
| Policy | 170k steps（~18 epoch），loss **0.94 → ~0.012** |
| 真机 | 基本 **hold / 微抖** |

**部署 A/B**

| 设定 | 行为 |
|---|---|
| `Advantage: positive` | 更僵的 hold |
| 无 ACP tag | 略微抖 / 微伸；仍完不成任务 |

**定量（无 tag，约 75s）：** `|action−state|≈1.13`，状态变化 p50=0、p95≈0.16 → 近恒等指令。

**v1 结论：** 瓶颈是 **数据量 × 任务难度**，不是硬件断了，也不是「忘加 ACP」。

---

## 实验 B — 扩数据与 v2

| | |
|---|---|
| 数据 | 扩到 **~201 条 / ~197k 帧（~1.8 h）**，成功 ~**71%** |
| Value `snacks_bowls_v2` | **25k** steps，每 5k 存盘 |
| Policy `smolvla_acp_snacks_bowls_v2` | 目标 250k；停在 ~**220–225k**（~18 epoch），loss ~**0.016** |
| 4 相机映射 | `left_top_left→c1`，`right_top_right→c2`，双腕→`c3/c4` |

### 关键真机发现（面试加分点）

| Checkpoint | 约 epoch | 闭环表现 |
|---|---|---|
| ~50–80k | ~4–6.5 | **会动、伸手、试抓** |
| ≥~150k / 后期 | ~12+ | 再次塌成 **hold** |

**解读：** 更多数据带来了「能动」的能力，但 **BC+ACP 过训** 会抹掉前半段大动作，塌成「安全」近恒等输出。部署应优先 **中期 ckpt**，不要迷信 `last` 或最低 train loss。

**ACP tag（相对 v1 的细化）：** 本数据上 **无 tag** 往往开局更好动；`positive` 对应后段/高 advantage 帧，从 t=0 就喂容易冻住。默认部署：**无 ACP tag**（除非专门做对照）。

**硬件对照：** 旧 stack-bowls ckpt 仍能动 → 臂/相机/控制链路在「会动的策略」下正常。

视频：`docs/eval_clips/v2_approx_4epoch_grasp_attempt.mp4`，`v2_80k_approx_6p5epoch.mp4`。

---

## 专项分析：为何 ~200 条后，100k 以后 hold？50k/80k 真机怎样？

**一句话：** ~200 条数据时，**50k–80k（~4–6.5 epoch）真机最好**（抬臂、够物、试抓）；**过 100k、尤其 ≥150k** 闭环塌成 hold。不是硬件坏了，而是 **离线 BC+ACP 过训 + 部署条件错配**。

### 真机对照表

| Checkpoint | 约 epoch | Loss | Grad | 真机表现 |
|---|---|---|---|---|
| **~50k** | **~4.0** | 0.052 | 0.54 | 抬臂、靠近物体、**有抓取意图** |
| **~80k** | **~6.5** | 0.035 | 0.45 | 仍能动、够物/悬停试抓 |
| **~100k** | **~8.1** | 0.032 | 0.41 | 拐点：loss 还降，动作开始发黏 |
| **≥150k / ~220k** | **12–18** | 0.019→0.015 | 0.33→0.28 | **hold / 微抖** |

训练曲线：30k 后 loss 已从 ~2.3 掉到 ~0.06；**50k→100k 只再降约 38%**，之后收益极小，梯度持续萎缩——典型「还在拟合、闭环已变差」。

### Hold 主因

1. **BC 过拟合「平均安全动作」（主因）**  
   长程 demo 大量帧是接近成功时的微调、停顿、对齐、收臂。多训几遍后，对「输出≈当前关节」误差最小 → 闭环像 identity / hold。中期还记得前半段大动作，后期被静态帧「洗掉」。

2. **ACP `positive` 偏向后段稳定帧**  
   最高 30% advantage 常在 episode 后段。部署从 t=0 喂 `Advantage: positive`，像「已经快做完」→ 更僵。无 tag 更接近整段遥操作分布。

3. **学习率衰减 + 低 loss 假象**  
   50k 时 lr~9e-5，220k 时 ~6e-6。Loss 还能磨到 0.015，学的是更贴训练集的「懒动作」，不是更好技能。**Loss↓ ≠ 真机↑**。

4. **任务仍偏难**  
   ~200 条够让中期「会动」，不够扛 15–18 epoch 过训。

5. **不是根因：** 串口/扭矩/相机（50k/80k 与旧 ckpt 都能动）；也不是「200 条完全学不会」（中期已经能动）。

### 实操建议

- 主评测用 **50k–80k**（或扫 40k–100k），别默认 `last`  
- 真机默认 **无 ACP tag**  
- 目标约 **6–8 epoch 早停**（本数据大约 80k–100k），或按真机 sweep 停  
- 今日 3-cam 新任务同样先扫中期 ckpt

---

## 实验 C — 新任务 + 全视野相机 + 数据分析（今天）

### 任务改写

```text
先把黄、棕、白碗自下而上叠到白盘上，
再把口红和蓝色纸巾包放进黄色 tray。
```

- **固定阶段顺序**（先碗后 tray），减少「任意顺序」多模态混乱  
- 口红/纸巾顺序可自由；任务文本不写死左右臂（臂分工只是采集约定）  
- 成功 `s`：碗叠稳 **且** 两物进黄 tray；失败 `f`：掉落/顺序错/半截/翻倒  

脚本：`get-data-bowls-tray.sh`  
数据：`/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue`

### 采集质量（清洗后）

| 指标 | 数值 |
|---|---|
| Episodes | **150**（重编号 0–149） |
| 帧数 | **157,612（约 1.46 h @ 30 Hz）** |
| 成功 / 失败 | **115 / 35 → ~76.7%** |
| 时长 | 均值/中位 **~35 s**，最短 **23.4 s**，最长 **51.4 s** |
| 短脏集 | 已删（ep63 ~4.9s；ep135 ~16.5s 失败） |
| Schema | 盘上 4 路视频；action/state 无 NaN；单一 task 文案 |
| 干预标记 | 全 0（纯遥操作，正常） |

**质量判断：** 强于 v1——成功率更高、最短更长、阶段更清晰、时长约 2×。仍是长程双臂；150 条**有希望但未饱和**。

### 训练选择：3 相机（去掉全视野）

| 保留 | 去掉 |
|---|---|
| `right_top_right`、`left_wrist`、`right_wrist` | `left_top_left`（800×480 全视野） |

理由：对齐 SmolVLA 三视角；省显存/解码；避开混分辨率；仍保留「一顶 + 双腕」。

流水线：`run-bowls-tray-3cam-pipeline.sh`  
- Value `bowls_tray_3cam` — **25k**  
- Infer tag `bowls_tray_3cam`  
- Policy `smolvla_acp_bowls_tray_3cam` — **200k**（约 10 epoch）

---

## 横切工程经验

1. 混分辨率相机会弄崩直接 stack → processor 里先 resize  
2. 4 相机 SigLIP 易 OOM → 每卡 batch=4 + 梯度检查点  
3. HF 门控权重 → 本地 ModelScope / 缓存 + offline  
4. `rename_map` 必须对齐采集名 ↔ 策略 `camera1..N`  
5. USB 多相机超时是运维问题，不是「机器人死了」→ 复用上一帧  
6. **该看的指标：** 任务视频、`|a−s|`、状态位移——不是 train loss  
7. **选 ckpt：** 扫中期 epoch；出现动作后又退化就早停  

---

## 总结论（学习 + 面试）

1. **真机双臂上 E2E Evo-RL/ACP 可行**——采、标、训、部署、度量全链路打通。  
2. **数据规模与任务设计压过 ACP 技巧。** 69 条 → hold；~200 条 → 中期能动；后期又可 hold。  
3. **Advantage 条件不是免费午餐。** Positive 偏后段；部署条件错了会冻住。未会做任务前优先无 tag。  
4. **过拟合在 loss 曲线上长得像成功。** 要用**行为驱动早停**。  
5. **传感器也是超参。** 全视野利于采集质检；训练时去掉是算力/模型对齐的取舍。  
6. **下一步：** 跑完 3-cam → 中期 ckpt 真机扫；仍弱则加 demo 或拆成子任务再组合。

---

## 简历一句话

> 在双臂 SO101 上搭建并压测 **Evo-RL ACP** 全流程：薄数据下诊断 hold、扩到约 200 条后在中期 checkpoint 恢复伸手/试抓、揭示 ACP 部署标签陷阱，并完成叠碗→口红/纸巾的新任务采集（含全视野相机）与 **3 相机**再训练。

---

## 可能的面试问答

**Q：advantage 是每帧算吗？**  
是。每帧 n-step（n=50），再取最高 30% → 二值 ACP 标签。

**Q：这算 RL 吗？**  
离线 value + advantage 条件化 BC，不是在线策略梯度 RL。

**Q：为什么后期 ckpt 会 hold？**  
BC/ACP 过拟合近静态/高 advantage 段；loss 还降，闭环探索死掉。应用中期 ckpt、少过训、加好数据。

**Q：为什么去掉全视野相机？**  
采集用 4 路覆盖；训练用 3 路对齐 SmolVLA、省算力、避开 800 宽混分辨率——传感器集合 ≠ 固定特征集。

**Q：只改一件事会改什么？**  
要么**更多阶段清晰的 demo**，要么**先拆短子任务再组合**；别先卷 loss 或 ACP tag。

---

## 关键路径 / 脚本

| 角色 | 路径 |
|---|---|
| 新数据集 | `/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue` |
| 采集 | `get-data-bowls-tray.sh`，`get-data-evo-rl-bimanual.sh` |
| Value / infer / policy | `train-value-evo-rl.sh`，`value-infer-evo-rl.sh`，`train-policy-evo-rl-acp.sh` |
| 3 相机流水线 | `run-bowls-tray-3cam-pipeline.sh` |
| Value 输出 | `Evo-RL/outputs/value_train/bowls_tray_3cam` |
| Policy 输出（计划） | `Evo-RL/outputs/train/smolvla_acp_bowls_tray_3cam` |
| v2 value | `Evo-RL/outputs/value_train/snacks_bowls_v2` |
| 评测视频 | `docs/eval_clips/` |
| v1 细目（中文） | `docs/evo_rl_snacks_bowls_experiment_summary.md` |
