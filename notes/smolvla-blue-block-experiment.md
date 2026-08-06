# SmolVLA · Blue Block → Yellow Tray 实验总结

面向 **技术学习笔记 / 求职面试**。模板对齐叠碗 / handover 笔记（Episodes、Steps、Epochs、Duration、Latency、Hz、SR、失败模式与消融）。

相关脚本：`get-data.sh` · `train_smolvla_blue_block.sh` · `infer_smolvla_blue_block.sh`  
语言消融详注：`notes/vla-language-vs-vision.md`

---

## 1. 任务与设置

| 项目 | 内容 |
|------|------|
| 任务指令 | `pick up the blue block and put it into the yellow tray` |
| 成功标准 | 抓住蓝块并放入黄盘；时限内；**零人工干预** |
| 机器人 | SO101 follower（**6 DoF**）+ leader 遥操作 |
| 相机 | front（UGREEN）+ wrist（Sonix），640×480 @ **30 fps** |
| 策略 | SmolVLA（~450M；`train_expert_only`，冻结大部分 VLM） |
| 基座 | `lerobot/smolvla_base` + 本地 SmolVLM2 |
| 相机映射 | `front→camera1`, `wrist→camera2`，`empty_cameras=1` |
| 环境 | `conda: lerobot`，单卡 3090，`BATCH_SIZE=16` |

相对双臂任务：短 horizon、单物体、外观一致 → 更易拿到高 SR，也更适合做 **语言/外观消融**。

---

## 2. 数据（Episodes / Frames）

| 指标 | 数值 |
|------|------|
| Dataset | `/home/rxn/datasets/blue_block_yellow_tray` |
| Episodes | **90** |
| Frames | **28,877** |
| FPS | 30 |
| 平均时长 | **~10.7 s/ep** |
| 时长范围 | 约 4.4–15.5 s |
| Action / State | 各 **6** 维 |
| 质量 | 成功轨迹较一致；idle ~20%；stats 齐全 |

**覆盖与缺口（真机对上了）：**

| 姿态 / 维度 | 数据里？ | 真机表现 |
|-------------|---------|----------|
| 摆正（平放对齐） | 有 | 基本能抓稳 |
| 对角斜放 | 有（或近似） | 能抓；夹爪会 **调角度去转/对齐** |
| **竖着**（立起） | **无** | **抓不到** |
| 沿物块 **宽度（短边）** 夹 | 示教主流 | 学会了 |
| 沿物块 **长度（长边）** 夹 | 几乎无 | **没学会** |

→ BC 严格受示教支撑集限制：没见过的姿态/抓取轴不会魔法泛化。

---

## 3. 训练（Steps / Epoch / Batch）

| 超参 | 数值 |
|------|------|
| Steps | **40,000** |
| Batch size | **16**（单卡 effective=16） |
| Epochs | **≈ 22.2**（\(40000\times16/28877\)） |
| Save freq | 每 5k → `005000`…`040000`，`last→040000` |
| 墙钟 | **~4.8 h**（`14:24` → `19:11`） |
| `updt_s` | ~**0.42 s/step**（单卡） |
| 末期 loss | ~**0.010** |
| `tolerance_s` | 0.05 |

**Checkpoint：**
```text
outputs/train/smolvla_blue_block_yellow_tray/checkpoints/
  035000/pretrained_model   # ≈19.4 epoch · 真机评测约用这一档（~20 ep）
  040000 / last             # ≈22.2 epoch · 训满
```

---

## 4. 推理设置（Duration / Latency / Hz）

| 项 | 默认 / 真机实际 |
|----|----------------|
| 真机所用 epoch | **约 ~20**（最近保存点 **`035000`≈19.4**；训满 `040000`≈22.2，口语也常称约20） |
| 脚本默认 Ckpt | `last`→`040000`（若复现 ~20 ep：`CHECKPOINT=.../035000/pretrained_model`） |
| **Duration** | **`EPISODE_TIME_S=120`**（任务短，预算很宽裕） |
| Reset | 15 s |
| 控制 **FPS** | **30**（预算 ≈ **33.3 ms/帧**） |
| `max_relative_target` | **10**（较保守，单步限幅小） |
| `n_action_steps` | 脚本未覆盖 → 用策略默认 chunk（SmolVLA 训练默认常为 **50**） |

**Latency / Hz（本机 3090，同任务测过）：**

| 指标 | 数值 |
|------|------|
| Policy `select_action` | **~147.7 ms**（median ~147，p95 ~150） |
| 等效策略刷新 | **~6.8 Hz** |
| 对照 | GR00T ~73 ms（~14 Hz）；PI0.5 ~158 ms（~6.3 Hz） |

三层不要混：

| 名词 | 本实验 |
|------|--------|
| Policy latency | ~**148 ms**（~7 Hz） |
| Control rate | **30 Hz** |
| Episode duration | **120 s** 时限 |

→ 前向仍慢于 33 ms，靠 **action chunk** 维持流畅；任务本身短，体感「推完一轮很快」。

---

## 5. 真机评测（Success Rate）

**协议：** \(SR=\#成功/N\)，蓝块入黄盘，零干预。

### 5.1 分布内小样本（~20 epoch 权重）

| 项 | 值 |
|----|-----|
| 权重 | **~20 epoch**（`035000` 附近） |
| Trials | ≈ **5** |
| 成功 | ≈ **4** |
| **SR** | **≈ 4/5 = 80%** |

### 5.2 姿态 / 抓取轴观察

| 条件 | 结果 |
|------|------|
| 蓝块摆正 | 基本抓住 |
| 对角斜放 | 基本抓住；夹爪会调角度「去转」再夹 |
| 蓝块竖着 | 失败；数据集无竖着示教 |
| 随机平移蓝块位置 | 能对准、可抓（**位置泛化尚可**） |
| 黄盘在适中范围内挪动 | 能跟位置调整放置（**托盘平移泛化尚可**） |
| 抓取轴 | 会按 **短边/宽度方向** 夹；**长边方向** 夹法未学会 |

### 5.3 语言–视觉消融（保留原结论）

| # | 语言 | 实物 | 行为 | 解读 |
|---|------|------|------|------|
| 1 | blue | **黑块** | 有意图但抓不准 | 语言救不了 OOD 外观 |
| 2 | **black** | **蓝块** | **仍成功** | 忽略错误指令 |
| 3 | black | 黑块 | 仍失败 | 成败跟外观走 |

→ 高 SR 是 **视觉 skill**，不是语言理解。

---

## 6. 分析：为何 SR 较高？

1. **任务短、阶段少：** 接近→抓→放，平均 demo ~10 s；无双臂交接/三碗叠放那种误差放大链。  
2. **外观强一致：** 训练与评测都是蓝块+黄盘；颜色/形状 in-distribution。  
3. **示教覆盖了常见平面姿态：** 摆正 + 斜放 → 评测主分布命中支撑集。  
4. **位置只在「适中范围」扰动：** 平移泛化够用，未测极端 OOD 布局。  
5. **中后期权重已够用：** 真机约用 **~20 epoch**（非必须训满 22）；offline MAE≈0.95° 量级已可 clone。  
6. **时限宽松：** Duration 120 s >> 任务需要，很少因超时失败。

**一句话：** 高 SR ≈ 窄域 BC 在支撑集内的成功，不是开放世界能力。

---

## 7. 其他失败原因

| 失败类型 | 原因 |
|----------|------|
| **竖着抓不到** | 数据集无竖着姿态 → 支撑集外 |
| **长边方向不会夹** | 示教几乎只教短边夹持轴 |
| **黑块 / 新外观** | 外观 OOD；消融已证实 |
| 偶发空抓/滑落 | 对位误差、夹爪行程、桌面摩擦（少数 1/5） |
| （非本任务主因） | 过拟合悬停——本任务训完仍能动，与 3cam handover 满训不同 |

---

## 8. 为何体感「推理快」？

相对双臂叠碗 / handover：

| 因素 | 蓝块单臂 | 双臂长任务 |
|------|----------|------------|
| DoF / 相机 | 6 / 2 | 12 / 4 |
| Horizon | ~10 s 级 | 几十秒、多阶段 |
| Duration 预算 | 120 s 很宽 | 65–150 s 仍紧张 |
| 失败链 | 单点抓取 | 阶段门控 + 末段累积 |
| Policy latency | 同量级 ~150 ms | 同量级，但每集重规划次数/阶段更多 |

「快」主要来自 **任务短、阶段少、一次成功就结束**，不是 SmolVLA 前向突然变成 30 Hz。  
单步仍是 ~7 Hz 策略刷新 + 30 Hz 控制（chunk 内插值执行）。

另：`max_relative_target=10` 单步保守，但短任务下完成时间仍远小于 120 s，体感干净利落。

---

## 9. Offline（健康检查）

| 指标 | 结果 |
|------|------|
| Avg MAE vs GT | **~0.95°**（训练分布帧） |
| Pred 关节 std | 非塌缩 |
| vs PI0.5（当时未训满） | PI MAE ~2.9° → 对比需控完成度 |

Offline ≠ 真机 SR；只证明克隆健康。

---

## 10. 工程踩坑

- `tolerance_s=0.05` 防视频时间戳 assert  
- front/wrist → camera1/2 rename  
- 推理勿与 HIL actor / preview 抢相机  
- 消融时改 `DATASET_TASK` 文案；**分布内评测请用 blue 指令**（脚本里若写死 black 仅用于消融）

---

## 11. 面试怎么讲（60–90 秒）

> 我在 SO101 上采了 90 条蓝块放黄盘（2.9 万帧），SmolVLA 单卡 batch16 训满约 4 万步/~22 epoch，真机推理用约 **20 epoch** 权重。控制 30 Hz，策略前向约 148 ms（~7 Hz），episode 时限 120 s。真机大约 5 次成功 4 次。摆正和斜放能抓，斜放时夹爪会调角度；竖着抓不到，因为数据里没有；夹持主要学了短边方向。蓝块和黄盘在适中范围挪位置仍能跟。语言消融显示改成 black 指令仍抓蓝块，换成黑块就失败——所以是高 SR 的视觉 skill，不是语言 VLA。相对双臂任务，快和高 SR 都来自短 horizon 和窄域一致，不是模型真正变快。

**追问：**

1. **SR 为何高？** 短任务 + in-dist 外观 + 示教覆盖主姿态 + 充分训练。  
2. **竖着为何挂？** 支撑集外；BC 不外推未见姿态。  
3. **如何再涨点？** 补竖着/长边夹持 demo；正式 \(N\ge10\)；保持位置随机。  
4. **和双臂比？** 蓝块瓶颈是姿态 OOD；handover 瓶颈是首抓空抓；叠碗是末段与阶段门控。

---

## 12. 数字速查

```text
Task        : blue block → yellow tray (SO101, 6-DoF, 2 cams)
Data        : 90 ep / 28,877 fr @ 30 FPS · ~10.7 s/ep
Train       : 40k steps · bs=16 · ~22.2 ep · ~4.8 h · loss~0.01
Infer ckpt  : ~20 epoch (≈035000 / 19.4; last=040000≈22.2)
Infer       : Duration 120 s · FPS 30 · max_rel_target=10
Latency     : ~148 ms/select_action (~6.8 Hz) on 3090
Robot SR    : ~4/5 = 80% @~20 ep (N≈5, informal)
OK poses    : upright-flat + diagonal tilt (gripper reorients)
FAIL poses  : standing-on-end (no demos); long-axis grasp not learned
Spatial     : block/tray translation in moderate range OK
Ablation    : wrong text + blue → OK; black object → FAIL
Claim       : high-SR vision skill ✓ · language/zero-shot ✗ · support-set limited
```

---

## 13. 路径速查

| 用途 | 路径 |
|------|------|
| 数据 | `/home/rxn/datasets/blue_block_yellow_tray` |
| 训练 | `outputs/train/smolvla_blue_block_yellow_tray` |
| 权重 | `.../checkpoints/last/pretrained_model` |
| 推理 | `bash infer_smolvla_blue_block.sh robot` |
| 日志 | `logs/smolvla_blue_block_yellow_tray.log` |
| Latency 对照 | `notes/inference-latency.md` |
| 叠碗 / 胶带 | `notes/smolvla-bimanual-*-experiment.md` |

---

*Setup: SmolVLA · SO101 · `blue_block_yellow_tray` · 单任务 · 2026-07.*
