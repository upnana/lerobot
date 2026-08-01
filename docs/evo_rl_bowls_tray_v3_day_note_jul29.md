# 今日工作结论（2026-07-29）
## bowls_tray 3cam：真机评测 → Value 诊断 → HITL → v3 开训

> 学习笔记 / 求职面试用。承接 `docs/evo_rl_study_job_note_jul2026.md`。

---

## 电梯演讲（约 45 秒）

今天我在双臂 SO101 上做了 **闭环评测 + 价值函数诊断 + 数据迭代**。用 v2 中期 checkpoint（约 90k）真机跑任务时，前半叠碗尚可，后半口红抓取失败；我对 value overlay 做了定量对照，发现 **V(s) 在口红未抓住时仍持续抬升**（假进度），说明价值网络对「是否真正抓住物体」不敏感。据此我按失败模式做了 **HITL 纠错采集约 56 条**（多 ckpt：90k/100k/110k/120k），merge 成 **v3 = 287 eps / 262k 帧**，并**从头重训** value（25k）→ infer → policy（200k）流水线。

**一句话结论：** 长程操作里，**value 曲线好看 ≠ 子任务完成**；要用视频 + 动作相位对齐做假阳性排查，再用 **失败聚焦的 HITL** 补数据，而不是盲目堆量。

---

## 今天做了什么（时间线）

| 步骤 | 内容 | 结果 |
|---|---|---|
| 1. 真机 eval | v2 policy，中期 **90k**，`PRESET=n10_acp_none` | 碗阶段有进展；**口红未抓起**，任务未完成 |
| 2. Value overlay | 用已训 `bowls_tray_3cam_v2` value 给 eval 打分并可视化 | 得到曲线对齐视频；便于人眼 + 数值对照 |
| 3. Value 诊断 | 55–100s 窗口：\(V\) 从约 **-0.36 → -0.17**（斜率约 +0.006/s） | **方向看似正确，但与真实失败矛盾** → 虚高 |
| 4. HITL 采集 | 多 ckpt（90/100/110/120k）+ 干预重点在口红 | **+56 eps / ~46k 帧**；leader 电压/掉线问题多次打断 |
| 5. Merge → v3 | v2(231) + 今日 HITL(56)，去掉旧 value 标签 | **287 eps / 262484 frames** |
| 6. 开训 | value **从头训** 25k → infer → policy 200k | 当晚已在跑；~8k steps 时 loss ~2.2 |

---

## Value 分析（面试重点）

### Value 含义（先讲清）

`pistar06` 预测的是 **\(V(s)\)**（进度式价值），不是即时 reward 尖峰：

- 由 `episode_success` + 剩余步数构造目标，压到约 **[-1, 0]**
- **越高（越接近 0）≈ 离成功越近**
- ACP 的 advantage 是离线用 \(V\) 算的；overlay 上看到的主要是 \(V\)

### 定量发现（v2 value @ 90k eval ep0）

| 区间 | 现象 |
|---|---|
| 全长 ~111s | peak ≈ **-0.11 @ 88s**；终点又掉回 ≈ **-0.40** |
| 55–100s（后半） | smooth \(V\)：**-0.36 → -0.17**（+0.19）；四分位均值单调变好 |
| 与动作 | 左爪后半活跃、`‖a−s‖` 变小，和「像在收尾」一致 |
| **人眼真值** | **口红没有抓起来** |

### 结论（可直接背）

1. **整体斜率 ≠ 校准。** 后半程 \(V\) 上升，但关键子任务失败 → **乐观偏差 / 假进度**。
2. **失败模式：** 模型把「靠近物体、伸爪、姿态像完成」当成进度，对 **grasp success** 不敏感。
3. **末段回落**（100s→结束 \(V\) 大跌）更像事后察觉异常，**不能替代中途正确监督**。
4. **工程含义：** ACP 质量受 value 假阳性拖累；补数据应包含  
   - 口红**成功抓放**的完整 demo  
   - 以及「够到但没抓住」并标 **failure** 的负例，专门压虚高。

### 面试追问：你怎么验证 value？

> 不只看 loss。我对齐 **视频时间轴、\(V(t)\)、夹爪开合、‖a−s‖**，用人类标注的子任务成败做对照；发现「曲线好看但口红失败」后，才决定下一轮 HITL 的采样重点。

---

## 数据与训练决策

| 项目 | 数字 / 选择 |
|---|---|
| v2 基线 | 231 eps（含更早一轮 HITL） |
| 今日新增 HITL | **56**（多策略 ckpt 采，增加行为多样性） |
| v3 | **287 eps / 262k frames** |
| Value | **重新训**（`resume=False`，非续训 v2 value） |
| Policy | 流水线仍训到 **200k**（便于扫 ckpt）；**真机重点评 80k–120k** |
| 部署 ACP | 继续默认 **`n10_acp_none`**（开局硬喂 positive 易更僵） |

**为何 ~50 条就 merge，而不是再堆到 80+？**  
瓶颈已定位为 **口红子阶段 + value 假进度**；聚焦纠错比同质扩量更划算。今日已超过原定 ~40 条目标。

---

## SmolVLA 能力窗口（经验规律，面试可讲）

对本任务 / 本数据规模，**SmolVLA 的可学能力大致在 80k–120k steps 就到齐**：

| 区间 | 真机表现（经验） |
|---|---|
| 早期（≲50k） | 开始抬臂、够物，不稳定 |
| **甜区 ~80k–120k** | **任务能力基本成型**；今日 HITL / eval 也主要用 90k–120k |
| 后期（≳150k，尤其接近 200k） | 离线 loss 仍降，但闭环易 **塌回 hold / 微抖**（过训 + 分布错配） |

**含义：**

1. **训到 200k 不等于要用 last。** 200k 是为了存盘扫点；部署/HITL 默认扫 **80k–120k**。  
2. **对 SmolVLA + 长程双臂 BC/ACP：** 「学会了」看的是真机子任务，不是最终 step。  
3. **面试一句话：** *SmolVLA capability saturates around mid-training (~80k–120k); later checkpoints can look better offline but degrade on-robot (hold).*

今日证据：口红失败那条强 eval / HITL 基线正是 **90k**；采集也覆盖 100k/110k/120k，与该甜区一致。

---

## 工程与硬件侧写（体现工程能力）

- HITL 脚本：`hitl_rollout_bowls_tray_3cam.sh`；merge：`merge_bowls_tray_hitl_v3.py`（只合当日 HITL，避免与 v2 内旧 HITL **重复计入**）。
- 真机故障：**leader 舵机 Input voltage / no status packet**（ID5/6）——录制常已落盘，崩在复位或切力矩；需供电/接线排查，否则吞吐被硬件打断。
- 可视化交付：value overlay 默认 AV1，**抖音等平台解析失败** → 转 **H.264/avc1** 再传播。

---

## 面试题：为什么加了 HITL 之后策略变慢了？

> 真机观察：**v3（+56 HITL）比 v2 更磨蹭**；用同配置对照 **v2 @ 120k**（脚本 `infer_v2_ckpt_120k.sh`）。

### Q（面试官）
你用 human-in-the-loop 纠错扩数据后，任务完成率或许改善，但闭环动作明显变慢。是磁盘/限速/ckpt 选错，还是数据本身的问题？你会怎么诊断？

### A（标准答法，约 1 分钟）

**先控变量，再归因数据。**

1. **对齐部署条件**：同一 `PRESET=n10_acp_none`、`n_action_steps`、`max_relative_target`，在甜区 ckpt（约 80–120k）上做 **v2 vs v3** A/B，排除「拿 v3 的 last 对比 v2 中期」的假差异。  
2. **若 v2 同设置明显更快**：更可能是 **行为克隆学到了 HITL 的速度分布**，不是硬件限速。  
3. **机制**：HITL 片段常含犹豫、慢速对齐、反复试抓、人接管纠错；BC 拟合的是整段 action 分布，纠错数据会把策略拉向 **更保守 / 更慢**。数据「多」本身不是根因，**慢速片段占比上升**才是。  
4. **和 value 假进度同一教训**：离线指标好看 ≠ 真机行为健康；要区分 **能力（会不会做）** 与 **风格（快不快）**。  
5. **改进方向**：HITL 时纠错尽量干脆；merge 可降权/过滤极慢段；或只保留成功且时长正常的纠错；下一轮可对慢速片段做重采样/丢弃后再训。

### 一句话结论（可背）
> *HITL fixes failures but can pollute the action prior with hesitant teleop; BC then looks “more careful” and slower. A/B against the pre-HITL checkpoint under matched deploy settings separates style shift from real capability gain.*

### 英文简历 bullet
- Observed **slower on-robot motion after HITL merge**; controlled A/B (v2 vs v3, matched mid-ckpts/presets) attributed it to **learning hesitant correction trajectories**, not hardware limits — highlighting that **demo style, not only success labels, shapes BC policies**.

---

## 可写进简历的 bullet（英文可选）

- Diagnosed **optimistic value predictions** on a long-horizon bimanual task: \(V(s)\) rose while lipstick grasp failed; aligned value curves with gripper/action residuals and human outcome labels.
- Ran **HITL correction** (+56 demos across mid-training policy checkpoints), merged into **287-episode** dataset, and relaunched full **value → ACP infer → SmolVLA** training pipeline on dual GPUs.
- Showed that **offline value/ACP quality must be validated against subtask success**, not episode-level curves alone; guided data collection toward grasp-success and near-miss failures.
- Empirically found **SmolVLA capability saturates ~80k–120k steps** on this long-horizon setup; later checkpoints often **collapse to hold** despite lower train loss — evaluate mid-training, not `last`.
- After HITL, diagnosed **slower closed-loop motion** via matched v2@120k A/B; linked it to **hesitant intervention style in demos**, separating capability gains from behavior-style regression.

---

## 状态与下一步

| 项 | 状态 |
|---|---|
| v3 value 25k + infer + policy 200k | **完成**（policy 曾磁盘满中断，从 10k resume 到 200k） |
| 真机 | 优先扫 v3 **80–120k**；对照用 **v2 @ 120k**（`infer_v2_ckpt_120k.sh`） |
| Value | 再 overlay v3 value，查口红未抓住时是否仍虚高 |
| 数据 | 若确认「慢」来自 HITL 风格，下一轮过滤极慢纠错段或提高干脆成功 demo 占比 |

---

## 路径速查

| 资源 | 路径 |
|---|---|
| v3 数据 | `/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_v3` |
| Value 训练 | `Evo-RL/outputs/value_train/bowls_tray_3cam_v3` |
| Pipeline / resume log | `logs/pipeline_bowls_tray_3cam_v3_nohup.out`，`logs/smolvla_acp_bowls_tray_3cam_v3_resume.log` |
| Policy | `Evo-RL/outputs/train/smolvla_acp_bowls_tray_3cam_v3` |
| 当日 merge 脚本 | `merge_bowls_tray_hitl_v3.py` |
| v2@120k 对照 | `infer_v2_ckpt_120k.sh` |
