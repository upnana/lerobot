# 推理动作实验 Latency 笔记

面向学习笔记 / 求职面试。测量对象：blue_block → yellow tray 同任务上的 **policy `select_action` 延迟**（含 preprocess 后的一步推理 + postprocess；GPU sync）。

**硬件：** RTX 3090 · `CUDA_VISIBLE_DEVICES=0` · bf16  
**数据样本：** `blue_block_yellow_tray` frame 0 · warmup 5 · 统计 30 次

---

## 1. 测到的 Policy Latency

| 模型 | Checkpoint | mean (ms) | median (ms) | p95 (ms) | 等效 Hz |
|------|------------|-----------|-------------|----------|---------|
| **GR00T** | last @80k | **72.7** | 72.6 | 74.1 | **~13.7** |
| **SmolVLA** | last @40k | **147.7** | 147.1 | 150.5 | **~6.8** |
| **PI0.5** | 20k/50k（未训满） | **158.1** | 158.1 | 159.2 | **~6.3** |

**结论（延迟维度）：**
- GR00T 单步推理明显更快（约 SmolVLA / PI0.5 的一半）
- SmolVLA 与 PI0.5（20k）同量级，约 **150–160 ms/step**
- 三者都 **慢于** 控制环 30 Hz（33 ms/帧）→ 真机必须靠 **action chunk / 异步推理**，不能每帧都重推模型

---

## 2. 和真机控制环的关系（实验时该怎么看）

LeRobot `lerobot-record` 控制环目标：

```text
loop_dt ≈ camera + policy + robot send + sleep_to_fps
target fps = 30  →  budget ≈ 33.3 ms / cycle
```

| 层级 | 含义 | 本机大致量级 |
|------|------|----------------|
| Policy latency | 一次 `select_action` | GR00T ~73 ms；SmolVLA/PI05 ~150 ms |
| Control rate | 发关节指令频率 | 脚本设 30 fps（有 sleep 对齐） |
| Effective closed-loop | 视觉→动作更新有多“新” | 取决于 chunk 长度 & 是否 RTC |

**观察实验时建议记：**
1. **Policy latency**（上面表）：模型算力本身  
2. **是否掉帧 / 卡顿**：Rerun 或电机是否一顿一顿  
3. **Chunk 行为**：一次推理输出多步动作时，手感会更顺，但反应更“滞后”  
4. **端到端**（可选）：从相机曝光到电机动起来的墙钟时间（需额外打点）

本笔记 **未** 测完整 robot E2E（含 USB 相机 + Feetech 通信）；真机若要报 E2E，需在 `record` 循环里加 `time.perf_counter()` 打点。

---

## 3. 面试怎么说（30 秒）

> 在同一块 3090、同一份蓝块数据上，我测了单步 `select_action` latency：GR00T 约 73 ms（~14 Hz），SmolVLA 约 148 ms（~7 Hz），PI0.5（20k）约 158 ms。控制环仍按 30 Hz 跑，因此 VLA 不能每帧重推，要靠 action chunking。延迟对比说明：更大/不同架构不一定更慢——本设置下 GR00T 推理更快，但能力还要另做语言–视觉消融，不能只看 latency。

---

## 4. 复现命令

```bash
# GR00T（lerobot-groot）
CUDA_VISIBLE_DEVICES=0 python measure_policy_latency.py  # 或按此前脚本

# 真机时同时观察
# - 控制：FPS=30
# - 日志/手感：是否因推理慢而动作发涩
bash infer_groot.sh robot
bash infer_smolvla_blue_block.sh robot
bash infer_pi05.sh robot
```

---

## 5. 数字速查

```text
Platform     : RTX 3090, single GPU
Metric       : select_action + postprocess, CUDA sync, N=30
GR00T @80k   : ~73 ms  (~14 Hz)
SmolVLA @40k : ~148 ms (~7 Hz)
PI05 @20k    : ~158 ms (~6 Hz)
Control FPS  : 30 (budget 33 ms) → need action chunks
```

---

*Measured 2026-07-17 · SO101 blue_block_yellow_tray*
