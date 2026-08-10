# SO101 实验重总结：Infer · Latency · SR · OOD · Failure

口径统一版（2026-08-10）。Latency **只认**蓝块同条件正式测；其它任务未重测的标 **(ref)**。

---

## 0. 三层时间（永远分开报）

| 名词 | 含义 | 本机量级 |
|------|------|----------|
| **Latency** | 一次 `select_action`（含 preprocess/postprocess，GPU sync） | 见下表 |
| **Control** | 发关节指令频率 | **30 Hz**（预算 ~33 ms）→ 必须 **action chunk** |
| **Infer time** | 单次真机评测 **episode 时限**（墙钟上限） | 任务不同：65–180 s |
| **SR** | 零干预成功 / N | 正式建议 \(N\ge10\)；下表多为 informal |

---

## 1. Latency（正式对照 · 唯一标准）

**条件：** RTX 3090 · `blue_block_yellow_tray` frame0 · warmup5 · N=30 · 2026-07-17  
**来源：** `notes/inference-latency.md`

| 模型 | Ckpt | mean | median | p95 | 等效 Hz |
|------|------|------|--------|-----|---------|
| **GR00T** | 80k | **72.7 ms** | 72.6 | 74.1 | ~13.7 |
| **SmolVLA** | 40k | **147.7 ms** | 147.1 | 150.5 | ~6.8 |
| **PI0.5** | 20k（计划50k未满） | **158.1 ms** | 158.1 | 159.2 | ~6.3 |
| **PI0** | — | **未测** | — | — | — |

**另测（非同表，勿与上表混报）：**

| 设置 | 数 | 说明 |
|------|-----|------|
| SmolVLA 3cam 叠积木 | first `select_action` ~**143 ms**；队列内 pop ~**1.2 ms** | 3 相机 + chunk；与蓝块 147.7 同量级 |

**口述统一：** GR00T ~**73 ms** · SmolVLA ~**148 ms** · PI0.5 ~**158 ms**。  
Pass / 叠碗等文档里的 148 = **(ref) 蓝块引用**，不是该任务重测。

---

## 2. 总表：Infer · Latency · SR · 主失败

| 实验 | Infer time | Latency | SR | 主失败 |
|------|------------|---------|-----|--------|
| 多物体→白/黄盘 | — | 未测 | 本机无严格 SR | 任务串泛，难消融 |
| 白块→黄盘 | ~120 s | (ref) 同蓝块量级 | SmolVLA 好于 PI0.5（定性） | 换底盘+绝对关节冲突 |
| **蓝块→黄盘** | **120 s** | **正式表** | SmolVLA **~4/5** @~20ep | 竖放无数据；短边夹持 |
| HIL-SERL 推黑块 | — | — | 日志~76%（含人工 s） | ≠零干预 SR |
| **Pass 胶带** | **65 s** | **(ref) ~148** | informal **~2/3** | **右臂抓不准/空抓**（非交接） |
| **叠碗 黄→棕→白** | **150 s** | **(ref) ~148** | **>70%**（ckpt~120k 甜区） | 白碗末段；阶段门控；过早松爪 |
| **Evo-RL** 叠碗+口红/纸巾 | 对照 ~**110 s** | 未系统测 | **分阶段**（见下） | 纸巾/口红入 tray；假进度；后期 hold |
| 叠积木 2cam（旧日志） | 67–240 s | — | 分析 **0/8** 全成功 | 黑层；撞塔；prompt 错配 |
| **叠积木 3cam** | 常用 **180 s** | ~**143** / queue **1.2** | 会话 **~4/5**；后 **>50%** | 时限/黑层；场景协议 |

GitHub：[`lerobot_lab` 本笔记路径](https://github.com/upnana/lerobot_lab/blob/main/notes/all-experiments-sr-latency-summary.md) · [pass_tape](https://github.com/upnana/pass_tape) · [stack_bowls](https://github.com/upnana/stack_bowls)

### 2.1 叠碗 SR（独立 SmolVLA 任务）

| 项 | 值 |
|----|-----|
| 任务 | 黄→棕→白 叠稳，零干预 |
| 报告 SR | **>70%**（甜区约 data-epoch 20 / ckpt `120000`） |
| 早期小样本 | 笔记曾记 2/3；后续评测按 **>70%** 报 |
| 仍见失败 | 白碗末段、阶段门控、示教过早松爪 |

### 2.2 Evo-RL 分阶段 SR（不要报一个笼统整任务数）

长程任务 = **subtask1 叠碗** + **纸巾/口红入黄 tray**。整任务正式 SR 未做 \(N\ge10\)；按子任务：

| 子任务 | 真机表现 | SR 口径 |
|--------|----------|---------|
| **① 叠碗**（黄→棕→白 → 白盘） | **能成功**，顺序与堆叠大体学会 | 前半程可用 |
| **② 纸巾入黄 tray** | **做不完整** | **SR 很低** |
| **③ 口红入黄 tray** | **做不完整**（够到/试抓多，抓稳放稳少） | **SR 很低** |
| 整任务 | 前半 OK、后半挂 | **不报单一高 SR** |

面试一句：Evo-RL 上 **subtask1 叠碗能成**；**纸巾和口红入盘 SR 很低**——瓶颈在后半精细抓放，不是不会叠碗。

---

## 3. OOD（分布外）

| 类型 | 实验 | 现象 | 含义 |
|------|------|------|------|
| **颜色 OOD** | 蓝块消融 | 黑块：有抓意图但抓不准 | 未见外观 → 抓取崩 |
| **语言 OOD / 弱接地** | 蓝块消融 | 错指令+蓝块仍成功 | 忽略错语言；视觉主导 |
| **姿态 OOD** | 蓝块 | 竖放无示教 → 抓不到 | 接触几何不在分布内 |
| **标定/底座 OOD** | 白块 | base1 vs base2 角差~5.5° | 绝对关节合并=标签冲突 |
| **语言条件 OOD** | 叠积木 | train dict 字符串 ≠ eval 干净句 | prompt 错配当 OOD |
| **相机/爪 OOD** | 叠积木 | 旧2cam+旧爪 ≠ 新3cam+新爪 | 禁止混数据 |

**Zero-shot：** 单任务微调后，几乎无可靠语言遵循；对新颜色迁移弱。

---

## 4. Failure mode（按实验）

### 蓝块 SmolVLA（主打）
1. 竖放 / 未见姿态 → 抓失败  
2. 夹持先验偏短边  
3. OOD 颜色失败；语言救不了  

### Pass 胶带
1. **预期：** 交接会挂  
2. **实际：** **right arm 抓偏/空抓**  
3. 抓住后 pass+入盘很少挂  
4. 3cam 训满 → 悬停（过拟合）；用中期 **60–80k**

### 叠碗
1. 报告 SR **>70%**（甜区 ckpt ~120k）  
2. 仍见：白碗末段放置、阶段门控（棕未上黄 → 白也不放）、过早松爪克隆  

### Evo-RL / ACP
1. **Subtask1 叠碗：能成功**  
2. **纸巾 / 口红入黄 tray：SR 很低**（长程后半瓶颈）  
3. Value **假进度**（未抓住口红时 \(V\) 仍抬）  
4. 数据少时 hold；中期有伸手；**≥150k 易塌回 hold**  
5. HITL 犹豫 → 策略变磨蹭  
6. 部署默认 **without tag**；硬喂 positive 无整任务增益  
7. 勿把 demo 标注 success% 说成 policy SR；勿只报整任务成/败

### 叠积木
1. 第三层黑完不成 / 撞塔  
2. Train↔eval prompt 错配  
3. `max_relative_target` 过紧 → clamp  
4. 人手进画面污染 SR  
5. 3cam 新数据后会话 SR 明显好于旧 2cam 日志

### GR00T / PI0.5
- GR00T：latency 最低，但接近+多次开合、**未干净成功**（projector-only）  
- PI0.5@20k：欠拟合、抓不稳；**勿与训满 SmolVLA 比架构**

---

## 5. 语言–视觉消融（OOD 证据链）

| # | 语言 | 实物 | 结果 |
|---|------|------|------|
| 1 | 正确 blue | **黑块** | 失败 → 外观 OOD |
| 2 | **错误 black** | **蓝块** | **成功** → 忽略错指令 |
| 3 | black | 黑块 | 失败 → 跟外观走 |

---

## 6. 跨实验结论（短）

1. Latency 标准答案：**73 / 148 / 158 ms**（GR00T / SmolVLA / PI0.5）；**无 PI0 测值**。  
2. Infer time = 任务时限，不是 latency。  
3. 失败在**最难子技能**：Pass=右臂抓；独立叠碗虽 **>70%** 仍卡白碗末段；Evo=**纸巾/口红低 SR**（叠碗子任务已会）。  
4. 视觉 > 语言；OOD 颜色杀抓取。  
5. 双臂/长程：**中期 ckpt** 优于训满；Evo：**without ACP tag**；报 **分阶段 SR**。  
6. 诚实边界：多数整任务 SR 仍 informal；E2E latency 未系统测。

---

## 7. 数字速查

```text
Latency (3090, blue_block, formal):
  GR00T 80k     72.7 ms
  SmolVLA 40k  147.7 ms
  PI0.5 20k    158.1 ms
  PI0          NOT MEASURED

SmolVLA 3cam stack (separate): ~143 ms first / ~1.2 ms queued

Infer time:
  Pass 65s | Blue 120s | Bowls 150s | Stack3 180s | Evo A/B ~110s

SR:
  Blue ~4/5 | Pass ~2/3 | Stack bowls >70% @~120k
  Evo: subtask1 bowls OK | tissue+lipstick SR very low | no single full-task SR
  Stack3 3cam ~4/5→>50% | HIL log 76% ≠ zero-intervention

Main fail:
  Pass = right-arm grasp (NOT handover)
  Bowls = white last / staging / early release (still >70% overall)
  Evo = tissue & lipstick into tray (bowls subtask works)
  Blue OOD = color / upright
```

详注：`notes/inference-latency.md` · `notes/vla-language-vs-vision.md` · pass_tape / stack_bowls · `docs/evo_rl_acp_tag_ab_aug01.md`
