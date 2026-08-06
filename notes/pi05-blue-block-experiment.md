# PI0.5 · Blue Block → Yellow Tray 实验总结

面向学习笔记 / 求职面试。数据与 SmolVLA 实验共用；本稿聚焦 **PI0.5** 训练进度、指标与对比结论。

相关笔记：`notes/smolvla-blue-block-experiment.md`（同任务完整 SmolVLA 结果）。

---

## 1. 任务与设置

| 项目 | 内容 |
|------|------|
| 任务指令 | `pick up the blue block and put it into the yellow tray` |
| 机器人 / 相机 | SO101 · front + wrist · 与 SmolVLA 同套数据 |
| 策略 | **PI0.5**（OpenPI 风格 flow / expert；体量远大于 SmolVLA） |
| 基座 | `lerobot/pi05_base` + PaliGemma tokenizer（本机 `paligemma-3b-pt-224`） |
| 精度 | `bfloat16`，`gradient_checkpointing=true` |
| 视觉 | `freeze_vision_encoder=false` |
| Job 名 | `pi05_blue_block_yellow_tray_b16_50k` |

---

## 2. 数据（与 SmolVLA 相同）

| 指标 | 数值 |
|------|------|
| Episodes | **90** |
| Frames | **28,877** |
| 均长 | **~10.7 s/ep** |
| 数据集路径 | `/home/rxn/datasets/blue_block_yellow_tray` |

---

## 3. 训练计划 vs 实际评测进度

| 项 | 计划（train_config） | 实际用于推理的权重 |
|----|----------------------|-------------------|
| Steps | **50,000** | **20,000**（另有 10k） |
| Batch size | **16**（config 全局 bs） | 同上配置下中途导出 |
| Epochs（按 bs=16） | 满训 ≈ **27.7** | @20k ≈ **11.1**（约满训的 **40%**） |
| Scheduler decay | 50,000 | 未跑满 |
| Save | 训练机导出 `010000` / `020000` | 本地目录见下 |

**本地权重：**
```text
outputs/train/pi05_blue_block_yellow_tray_010000_pretrained_model
outputs/train/pi05_blue_block_yellow_tray_020000_pretrained_model   ← 主测
```

**推理脚本：** `infer_pi05.sh`（默认指向 020000 + blue_block 数据，`tolerance_s=0.05`）

> 注意：当前结论基于 **未训完** 的 20k checkpoint，不能当作 PI0.5 最终能力上限。

---

## 4. 实验结果

### 4.1 离线（PI0.5 @ 20k）

| 指标 | 结果 |
|------|------|
| Avg MAE vs GT | **~2.90°** |
| Diagnostic | pred 有跨帧变化（非塌缩）；但对训练数据拟合明显弱于 SmolVLA |
| 加载 | safetensors 全 key 加载成功；tokenizer 路径需从 `/workspace/...` patch 到本机 |

### 4.2 真机（@ 20k）

- 同场景（蓝块 + 黄盘）下，主观观感 **弱于已训完的 SmolVLA（40k）**  
- 更易出现抓不准 / 抖动 / 任务完成度低（与离线 MAE 偏高一致）  
- 根因归类优先：**训练未充分**，而非直接断言「PI0.5 < SmolVLA」

### 4.3 与 SmolVLA 对照（同数据）

| | SmolVLA（完整） | PI0.5（当前） |
|--|-----------------|---------------|
| Steps | **40k / 40k（100%）** | **20k / 50k（40%）** |
| Epochs（bs=16） | ≈ **22.2** | ≈ **11.1** |
| Offline MAE | **~0.95°** | **~2.90°** |
| 真机 in-dist | 可用、基本成功 | 明显更差 |
| 模型规模 | ~450M（多冻结） | 显著更大（全量/近全量可训分量更多） |

**公平性：** 20k PI0.5 vs 40k SmolVLA 不是同阶段对比。等 PI0.5 到 **40k–50k / last** 再比真机与 MAE。

---

## 5. 分析结论

1. **中途权重表现符合欠拟合：** 只完成约 11 epoch，离线误差约为 SmolVLA 的 3 倍量级，真机偏弱是预期内。  
2. **PI0.5 更大、收敛更慢：** 同数据上往往需要更多 steps / 更仔细的 lr schedule；`b16_50k` 设计合理，但必须跑满或接近跑满再下结论。  
3. **大模型 ≠ 立刻更强：** 在窄域模仿学习里，**训完的小策略**可以短时间打赢 **半训的大策略**。  
4. **语言接地预期：** 即便跑满 50k，单任务恒定指令下，仍可能复现 SmolVLA 的「视觉主导、语言弱」——那是数据形态问题，不独属于 PI0.5。要用多颜色多指令消融另证。  
5. **工程：** 需本机 tokenizer patch；大数据集视频解码同样建议 `tolerance_s=0.05`；推理占显存高（~16GB+），避开 HIL-SERL actor 抢相机。

---

## 6. 面试：怎么讲这段

### 30 秒口述

> 同一份 90 条蓝块放盘数据上，我并行微调了 SmolVLA 和 PI0.5。SmolVLA 完整 4 万步约 22 epoch，离线 MAE 约 1°，真机可用。PI0.5 计划 5 万步，但目前只评估到 2 万步约 11 epoch，MAE 约 2.9°，真机明显更差。这让我强调：跨模型对比必须对齐训练完成度；大模型中途 checkpoint 不能用来下「谁架构更好」的结论。

### 高频问答

**Q: 为什么 PI0.5 比 SmolVLA 差？**  
A: 就现有数字而言，主要是 **checkpoint 只训了 40%**。控制变量后更可能接近或超过；现在只能说「欠拟合的大模型弱于拟合好的小模型」。

**Q: 50k / ~28 epoch 对 PI0.5 够吗？**  
A: 对单任务 2.9 万帧是合理量级起点；要用曲线（loss、离线 MAE、真机成功率）看是否 plateau，不够再加 steps 或从 40k resume。

**Q: 你怎么设计对比实验才公平？**  
A: 同数据、同机器人、同评测协议；至少双方都用 **最后/等量 epoch** 的权重；再加语言–外观消融，而不是只报分布内成功率。

**Q: PI0.5 工程难点？**  
A: 显存与 gradient checkpointing、tokenizer 路径跨机器、视频时间戳容差、推理与其它进程抢相机/GPU。

---

## 7. 数字速查卡

```text
Dataset     : 90 ep · 28877 frames（与 SmolVLA 相同）
Plan        : 50k steps · bs=16 · ~27.7 epochs
Evaluated   : 20k steps · ~11.1 epochs（40% of plan）
Offline MAE : ~2.90° @20k   vs SmolVLA ~0.95° @40k
Robot       : weaker than finished SmolVLA（预期内欠拟合）
Takeaway    : align training progress before comparing VLAs
Next        : eval 40k/50k last；再比真机 + 做语言消融
```

---

## 8. 相关文件

| 用途 | 路径 |
|------|------|
| 推理 | `infer_pi05.sh` |
| 权重 10k / 20k | `outputs/train/pi05_blue_block_yellow_tray_{010000,020000}_pretrained_model` |
| 同任务 SmolVLA 总结 | `notes/smolvla-blue-block-experiment.md` |
| 语言-视觉消融（偏 SmolVLA） | `notes/vla-language-vs-vision.md` |

---

*Setup: PI0.5 · SO101 · `blue_block_yellow_tray` · 评测 ckpt @20k/50k · 2026-07.*
