# SO101 全实验点子总表（SR / Latency / Infer / Failure / OOD）

从最早多物体入盘到叠积木 3cam（约 2026-07 → 2026-08）的**统一复盘**。  
定量优先；N 小或未严格协议的标 **informal / 定性**。

**三层时间（全实验通用，勿混）：**

| 名词 | 含义 | 典型量级 |
|------|------|----------|
| Policy latency | 一次 `select_action` | SmolVLA ~**148 ms**；GR00T ~**73 ms**；PI0.5@20k ~**158 ms** |
| Control rate | 发关节指令 | **30 Hz**（预算 ~33 ms）→ 必须 **action chunk** |
| Episode / infer time | 单次评测时限 | 任务不同：**65–180 s** |
| SR | 零干预成功 / N | 正式建议 \(N\ge10\)；多数现有数字 informal |

交互总览 Canvas：`all-experiments-sr-latency.canvas.tsx`（Cursor 侧栏打开）。

---

## 0. 一句话总览

> 端到端跑通了单臂/双臂采集、SmolVLA / PI0.5 / GR00T、HIL-SERL、Evo-RL/ACP。同数据下 **SmolVLA 真机最稳**；**视觉主导、语言弱、OOD 颜色失败**；双臂/长程 **中期 ckpt 优于训满**；失败总落在**最难子技能**（右臂抓胶带、白碗末段、口红），而不是「以为难的那一步」（如交接）。Evo-RL 部署默认 **without ACP tag**。

---

## 1. 总表（按时间线）

| 阶段 | 任务 | 数据 | 策略 | Infer 时限 | SR / 结论 | 主失败 / 点子 |
|------|------|------|------|------------|-----------|----------------|
| 早期 | 多物体→白/黄盘 | 121ep/108k；110ep/80k | PI0.5 远程 ~37–40ep | — | 本机**无严格 SR** | 任务串泛，只适合跑通链路 |
| 精抓起步 | 白块→黄盘 | 60→110ep（换底盘） | SmolVLA / PI05 / GR00T | ~120s | SmolVLA 明显更好（定性） | **底盘移位 + 绝对关节** 合并冲突（~5.5°） |
| **主打对比** | 蓝块→黄盘 | **90ep / 28877fr** | 三模型同数据 | **120s** | SmolVLA **~4/5 @~20ep** | 竖放无数据；语言消融见下 |
| RL | 推黑块→白纸 | demo25 + online~420 | HIL-SERL SAC | — | 日志 ~**76%**（含人工 `s`） | ≠ 零干预 SR；与 VLA 权重不互通 |
| **Pass 胶带** | 抓→交接→黄盘 | **89ep / 47k** 4cam | SmolVLA 停~120k | **65s** | informal **~2/3** | **预期交接难 → 实际右臂抓不准** |
| **叠碗** | 黄→棕→白 | **200ep / 98616** 4cam | SmolVLA **250k** | **150s** | **2/3** @ckpt120k | 白碗末段；阶段门控；过早松爪 |
| Evo-RL | 零食+叠碗 / 碗+口红纸巾 | 69→~287 HITL | value + ACP SmolVLA | 对照 ~**110s** | **无正式整任务 SR** | 口红；假进度；后期 hold；**without tag 基线** |
| 叠积木 2cam | 白→蓝→黑 | 旧库（勿混新爪） | SmolVLA | 67–240s | 日志分析 **0/8** 全成功 | 第三层黑；prompt 错配；clamp |
| **叠积木 3cam** | 白→蓝→黑 | **120ep / 79251** | SmolVLA **100k** | 常用 **180s** | 会话 **~4/5**；后 **>50%** | 新爪数据集；黑层仍关键 |

GitHub 归档：

- [stack_bowls](https://github.com/upnana/stack_bowls)
- [pass_tape](https://github.com/upnana/pass_tape)
- [smolvla_stack_blocks_white_blue_black_3cam](https://github.com/upnana/smolvla_stack_blocks_white_blue_black_3cam)

---

## 2. Latency 专表（3090 · 蓝块同条件）

| 模型 | Ckpt | mean | 等效 Hz | 备注 |
|------|------|------|---------|------|
| GR00T | 80k | **~73 ms** | ~14 | 最快；真机抓取不稳 |
| SmolVLA | 40k | **~148 ms** | ~6.8 | 主力 |
| PI0.5 | 20k/50k | **~158 ms** | ~6.3 | **未训满**，勿判架构 |
| SmolVLA 3cam stack | — | chunk ~**143 ms**；queue ~1.2 ms | — | 控制仍 30Hz，`n_action_steps` 可 5–50 |

控制预算 33 ms ≪ policy latency → **必须 chunk**；E2E（曝光→电机）**未系统测**。

---

## 3. SmolVLA 任务族：只看数据分布（不管变体名）

| 任务 | Ep/Fr | Cam | DoF | Train | 甜区 | SR | 分布要点 |
|------|-------|-----|-----|-------|------|-----|----------|
| 蓝块→盘 | 90/29k | 2 | 6 | 40k | ~20ep | ~4/5 | 短 horizon；in-dist 高；竖放 OOD |
| Pass 胶带 | 89/47k | 4 | 12 | ~120k停 | 60–80k | ~2/3 | **抓取段难**；交接段易学、示教一致 |
| 叠碗 | 200/99k | 4 | 12 | 250k | **120k** | 2/3 | 长链；末段少；**过早松爪**进分布 |
| 叠积木 3cam | 120/79k | 3 | 6 | 100k | 50–100k 扫 | ~4/5→>50% | 新爪；**黑在顶层**样本关键 |
| Evo 碗+口红 | ~150–287 | 3–4 | 12 | 至200k | **80–120k** | 分阶段 | HITL 犹豫污染；口红稀缺 |

**统一读法：** 数据量 × 最难子技能覆盖决定上限；伪影（空抓、过早松、HITL 磨蹭）会被 BC 克隆。

---

## 4. Pass-tape 深挖（你强调的点）

| 项 | 值 |
|----|-----|
| Latency | ~148 ms（参考） |
| Infer time | **65 s**/ep |
| SR | ≈ **2/3**（N≈3 informal） |
| 预期难点 | **胶带交接（handover）** |
| **实际主失败** | **right arm 有时不能准确抓住 tape**（抓偏/空抓） |
| 条件规律 | 抓住后 pass + 入黄盘很少再挂 |
| 甜区 ckpt | **060000**（非训满） |
| 视频 | `videos/eval_pass_tape.mp4` |

\[
P(\text{full}) \approx P(\text{right grasp}) \times P(\text{pass+place}\mid\text{grasp})
\]

---

## 5. Evo-RL · without tag

| 条件 | 结论 |
|------|------|
| **without tag**（`n10_acp_none`） | **更干净的闭环基线**；能力来自任务文本+视觉，不依赖开局硬喂 positive |
| with tag（`Advantage: positive`） | 当晚 **无整任务增益**；value 末段抬升更像**假进度** |
| v3 vs v2 | 纸巾偶发进盘等进步，主要来自 **HITL/数据**，不是 tag |
| 早期 ~69ep snacks+bowls | 真机几乎 **hold**；扩数据后中期有伸手，**≥150k 又 hold** |
| 部署默认 | **without tag** |

Demo 标注 success% ≠ policy SR。

---

## 6. OOD / 语言 /「Generation」

### 语言–视觉消融（蓝块）

| 语言 | 实物 | 结果 |
|------|------|------|
| 正确 blue | **黑块** | 有意图但抓不准 → **OOD 外观** |
| **错误 black** | **蓝块** | **仍成功** → 忽略错指令 |
| black | 黑块 | 仍失败 → 成败跟外观走 |

结论：微调后几乎无可靠 **zero-shot 语言遵循**；对新颜色 **OOD 迁移弱**；学的是视觉条件 skill。

### 其它 OOD / 分布问题

- 竖放蓝块、未见布局 → 抓取崩  
- 换底盘合并绝对关节 → 标签冲突  
- Train↔eval **task 字符串不一致** = 语言条件 OOD（叠积木曾中招）  
- 推理时擅自加长 task 文本（训练没见过）也会 OOD  

### 「Generation」实际是什么

BC **不会**开放生成新技能链；常见是：

- **阶段门控**（棕未上黄 → 白也不往黄上放）  
- **示教伪影克隆**（空抓、过早松爪、HITL 犹豫 → 策略磨蹭）  
- **中期甜区 / 后期过拟合 hold**  

不要说成「模型会自己规划生成」；应说 **视觉条件技能串 + 数据分布回放**。

---

## 7. Failure mode 速查

| 实验 | Top failures |
|------|----------------|
| 蓝块 SmolVLA | 竖放无数据；短边夹持；OOD 颜色 |
| 白块/底盘 | 绝对角域偏移；合并变差 |
| Pass tape | **右臂抓取**（非交接）；3cam 训满悬停 |
| 叠碗 | 白碗末段；阶段门控；过早松爪 |
| Evo-RL | 口红；value 假进度；后期 hold；HITL 变慢 |
| 叠积木（旧日志） | 黑层完不成；撞塔；人手污染；`max_relative_target=10`；prompt 错配 |
| 叠积木 3cam | 时限/黑层；仍靠中期 ckpt 与场景协议 |
| GR00T | 多次开合不成功；projector-only 视觉弱 |
| PI0.5 中途 | 欠拟合；爪偏小；本地 OOM |

---

## 8. Offline vs Robot（别混）

| 模式 | 测什么 | 不测什么 |
|------|--------|----------|
| Offline MAE | pred vs GT 关节 | 任务 SR |
| Value overlay | \(V(s)\) 曲线 | 勿当 SR；会假进度 |
| Replay | 数据/硬件链路 | 策略能力 |
| **Robot SR** | 零干预任务成功 | 需固定 ckpt、时限、场景、足够 N |

---

## 9. 跨实验统一点子（面试可背）

1. **同数据训完的 SmolVLA** 真机通常最稳；latency 中等，靠 chunk 吃 30Hz。  
2. **中期 checkpoint** 在双臂 / Evo / 叠放上反复优于训满。  
3. **失败在最难子技能**，不是整条链均匀难——Pass=右臂抓；叠碗=白碗；Evo=口红。  
4. **视觉 > 语言**；OOD 外观杀抓取；**without ACP tag** 是更干净基线。  
5. **数据质量 > 盲目加 ep**：空抓/过早松/HITL 犹豫会进策略。  
6. 诚实边界：多数 SR informal；E2E latency 未系统测；HIL 日志成功 ≠ 零干预 SR。

---

## 10. 数字速查卡

```text
Latency 3090     : GR00T~73 | SmolVLA~148 | PI05@20k~158 ms
Control          : 30 Hz → need action chunks
Blue block SR    : SmolVLA ~4/5 @~20ep · Duration 120s
Pass tape        : SR~2/3 · 65s · fail=right grasp NOT handover · ckpt 60k
Stack bowls      : SR 2/3 · 150s · ckpt 120k · fail=white last / staging
Evo-RL           : no formal full-task SR · default WITHOUT tag · sweet 80–120k
Stack3 3cam      : 120ep/79k · 100k steps · session ~4/5 then >50% · ~143ms chunk
Language         : wrong text OK if vision in-dist · OOD color fails
```

### 相关笔记

- `notes/interview-all-experiments.md`
- `notes/inference-latency.md` · `notes/vla-language-vs-vision.md`
- `notes/smolvla-bimanual-handover-tape-experiment.md`
- `notes/smolvla-bimanual-stack-bowls-experiment.md`
- `docs/evo_rl_acp_tag_ab_aug01.md` · `docs/evo_rl_study_job_note_jul2026.md`
- `notes/smolvla-stack3-eval-analysis.md`
- `notes/github_smolvla_stack_3cam_repo/README.md`

---

*汇总日期：2026-08-10*
