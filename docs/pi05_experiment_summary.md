# PI0.5 真机实验总结

**平台：** LeRobot + SO101 单臂（6-DOF + 夹爪）  
**模型：** PI0.5（~3B，PaliGemma + Flow Matching，绝对关节角）  
**周期：** 2026-07-01 ~ 2026-07-12  
**任务演进：** 多物体 → 黄盘 → 白块精抓

---

## 1. 实验演进总览

| 阶段 | 任务 | 数据集 | Episodes | Frames | PI0.5 训练 | 真机效果 |
|------|------|--------|----------|--------|------------|----------|
| **Phase 1** | 桌上物品 → **白盘** | `20260701_145251` | 121 | 108,405 | 远程 ~110k steps (~37 ep) | 未在本机系统验证 |
| **Phase 2** | 桌上物品 → **黄盘** | `20260706_yellow_tray` | 110 | 79,973 | 远程 ~200k steps (~40 ep) | 未在本机系统验证 |
| **Phase 3a** | **白块 → 黄盘** | `yellow_white` | 60 | 24,055 | 远程 50k steps (~33 ep) | 离线尚可，真机未充分验证 |
| **Phase 3b** | 白块 → 黄盘（**新底盘**） | `yellow_white_base2` | 50 | 19,742 | 远程 15k / 40k steps | **仍无法稳定抓取** |
| **Phase 3c** | 合并 3a+3b | `yellow_white_merged` | 110 | 43,797 | merge @ 30k steps (~11 ep) | 离线 MAE ~2.1，真机差 |

**最终结论：** 在 SO101 + 50~60 条 demo 条件下，PI0.5 未能达到可用的真机 pick-and-place 效果；同数据下 SmolVLA（450M）表现更好。核心瓶颈是 **模型体量 vs 数据量**、**绝对关节角对底盘/位姿敏感**、以及 **训练 epoch 不足**。

---

## 2. 任务与数据多样性（Data Diversity）

### 2.1 三阶段任务对比

```
Phase 1: Put all items on the table into the white plate
         → 多物体、多类别、目标容器为白盘

Phase 2: Pick up items on the table and put them into the yellow tray
         → 多物体、目标容器改为黄盘

Phase 3: pick up the white block and put it into the yellow tray
         → 单物体（白块）、精细抓取 + 放置，难度最高
```

### 2.2 数据多样性分析

| 维度 | Phase 1–2（多物体） | Phase 3（白块） |
|------|---------------------|---------------|
| **物体数量** | 多个（桌上若干物品） | 1 个（白块） |
| **视觉复杂度** | 高（多目标、遮挡） | 低（固定白块） |
| **动作精度** | 中等（大致放入盘/盘） | 高（需精准夹取 3cm 方块） |
| **Episode 数** | 110–121 | 50–60 |
| **数据量** | 80k–108k frames | 20k–24k frames |
| **对 VLA 的要求** | 语义理解 + 粗粒度操作 | 细粒度视觉伺服 + 夹爪控制 |

**关键发现：**

- **任务越简单（单白块），对夹爪精细控制要求反而越高**；PI0.5 在 Phase 3 的失败主因是 gripper 不闭合，而非 arm 不会动。
- Phase 1–2 数据量大、多样性高，但任务与 Phase 3 差异大，**不能直接迁移**到白块任务。
- 合并 `yellow_white` + `yellow_white_base2`（110 ep）增加了帧数，但 **底盘位置不一致** 导致绝对关节角标签冲突，merged 模型真机更差。

---

## 3. 模型能力（Model Capability）

### 3.1 PI0.5 架构特点

| 属性 | 值 | 对实验的影响 |
|------|-----|-------------|
| 参数量 | ~3B | 小数据集（50 ep）难以充分微调 |
| 预训练 | `lerobot/pi05_base`（通用 VLA） | 需较多 in-domain 数据适配 SO101 |
| 动作表示 | **绝对关节角**（6-DOF） | 对底盘位置、相机视角、初始姿态 **极度敏感** |
| 动作 chunk | 50 步开环 | 预测错误后恢复慢，真机表现 "犹豫" |
| 推理 | Flow Matching, 10 steps | 比 SmolVLA 更保守、更慢 |
| 显存 | ~16GB+ 推理 / bs=4 训练 OOM | 本地双 3090 无法训 PI0.5，依赖远程 |

### 3.2 与同平台其他模型对比（Phase 3, base2 数据）

| 模型 | 参数量 | 40k/50k step 真机 | 离线 MAE | 夹爪表现 |
|------|--------|-------------------|----------|----------|
| **PI0.5** | ~3B | 臂动、**夹爪弱** | ~2.1 (merge) / ~3.0 (base2) | 3–35°，偏保守 |
| **SmolVLA** | ~450M | 臂动、夹爪稍好 | ~0.8 (75k) | 1–42°，更积极 |
| **GROO T** | ~2B (DiT) | **昨天 yellow_white 120k 可用** | — | 相对动作，鲁棒性更好 |

**PI0.5 的能力边界（本实验）：**

- ✅ 能学会 "臂往哪个方向动" 的粗粒度策略
- ✅ 离线 MAE 在 2–3° 范围，说明 **训练集上拟合尚可**
- ❌ 无法可靠执行 **gripper close → grasp → lift** 序列
- ❌ 绝对关节角导致 **底盘/场景轻微变化即失效**
- ❌ 50 ep / 30k steps 远不足以激活 3B 模型的 fine-grained manipulation

---

## 4. 训练规模（Epoch / Episode / Steps）

### 4.1 各阶段训练配置

| 实验 | Batch | Steps | ~Epoch | Save checkpoint | 训练环境 |
|------|-------|-------|--------|-----------------|----------|
| pickup_items | 16 | 110,000 | 36.9 | 110k | 远程 A100 |
| yellow_tray | 16 | 200,000 | 40.0 | 200k | 远程 |
| yellow_white | 16 | 50,000 | 33.3 | 50k | 远程 |
| merge_30000 | 16 | 30,000 | **11.0** | 30k | 远程 |
| base2_15000 | 16 | 15,000 | **12.2** | 15k | 远程 |
| base2_40000 | 16 | 40,000 | **32.4** | 40k | 远程 |
| 本地尝试 | 4 (eff) | — | — | OOM | 双 3090 24GB |

> Epoch 计算：`steps × batch_size / total_frames`  
> base2: 50 ep × 19,742 frames, bs=16 → 1 epoch ≈ 1,234 steps

### 4.2 Epoch 与效果关系

| Checkpoint | Epoch | 离线 MAE | 真机 |
|------------|-------|----------|------|
| merge_30k | ~11 | ~2.1 | 臂慢、不抓 |
| base2_15k | ~12 | ~3.0 | 臂动、gripper 几乎不动 |
| base2_40k | ~32 | 未系统测 | 臂动、gripper 0–35°，仍不抓 |

**结论：**

- **11–12 epoch 明显不足** — 模型只学到 "靠近" 而非 "抓取"。
- 即使 ~32 epoch（40k steps），PI0.5 仍无法稳定抓取；对比 GROO T 在类似数据上 **120k steps / ~40 epoch** 才可用。
- 本地 OOM 限制了实验迭代速度；所有可用 checkpoint 来自远程，无法快速 ablation。

---

## 5. 机械臂位置 / 底盘（Arm Location）

### 5.1 问题发现

在 Phase 3 中，`yellow_white`（60 ep）与 `yellow_white_base2`（50 ep）采集时 **机器人底盘（dipan）位置不同**，导致：

- 同一 "白块 → 黄盘" 任务，**绝对关节角标签不同**
- 合并数据集（110 ep）后，模型学到的是 **两种位姿的混合分布**

### 5.2 关节统计对比（state mean）

| 关节 | yellow_white | base2 | 差异 |
|------|-------------|-------|------|
| shoulder_pan | +0.6° | **−4.9°** | 底盘旋转 ~5.5° |
| wrist_flex | 65.3° | 68.8° | 腕部姿态偏移 |
| gripper | 0.1–45° | 0.3–48° | 相近 |

### 5.3 影响

- PI0.5 使用 **绝对关节角** → 底盘偏移 = **domain shift** → 真机推理时 initial pose 与训练不匹配
- merged 模型在 **当前底盘（base2）** 上推理时，相当于用混合标签训练、单一分布测试
- **建议：** 推理时必须使用与 **当前底盘一致** 的数据集训练的 checkpoint（base2 训 → base2 测）

---

## 6. 相机视角（Camera View）

### 6.1 硬件配置（全程一致）

| 相机 | 设备 | 分辨率 | 用途 |
|------|------|--------|------|
| **Front** | UGREEN 2K (`by-id` 固定) | 640×480 @ 30fps | 全局场景、臂+盘+块 |
| **Wrist** | Sonix USB (`by-id` 固定) | 640×480 @ 30fps | 夹爪-物体相对位置 |

- 训练与推理 **同一套 by-id 路径**，无 front/wrist 搞反问题
- PI0.5 直接使用 `front` / `wrist` 键名（无 SmolVLA 的 camera1/2/3 对齐）

### 6.2 腕部视角质量（base2 数据集抽检）

| 阶段 | 可见内容 | 评价 |
|------|----------|------|
| Pre-grasp | 夹爪 + 白块 + 黄盘 | ✅ 方块在指间，位置关系清晰 |
| Grasp | 方块在夹爪指间 | ✅ 抓取瞬间有记录 |
| 图像质量 | 640×480，略糊 | ⚠️ 可用，非瓶颈 |

**结论：相机视角不是 PI0.5 失败主因。** 同一套相机下，GROO T（yellow_white 120k）昨天可用，SmolVLA 真机表现也优于 PI0.5。

---

## 7. 真机 Eval 关键数据

### 7.1 典型失败模式

| Checkpoint | 时长 | 臂运动 | 夹爪 | 结果 |
|------------|------|--------|------|------|
| merge_30k | 72s | 慢、犹豫 | 0–35° | 未抓取 |
| base2_15k | 69s | 有较大位移 | **3.8°→3.1°（几乎不动）** | 未抓取 |
| base2_40k | 90s | 有移动 | 0–35° | 未抓取 |

**Demo 中 gripper 范围：** 0.3°–48°  
**Eval 中 gripper 范围：** 3°–35°（偏小、偏保守）

### 7.2 推理参数影响

- `MAX_RELATIVE_TARGET=10`：限制每步最大关节变化 → 动作更 "稳" 但更慢
- 建议尝试 25–30 以加快 gripper 响应（尚未系统 ablation）

---

## 8. 根因归纳（五个维度）

| 维度 | 问题 | 严重程度 |
|------|------|----------|
| **Data diversity** | Phase 1–2 与 Phase 3 任务差异大；merged 数据底盘不一致 | ⭐⭐⭐ |
| **Model capability** | 3B 模型 + 绝对关节角，小数据难以 fine-tune 精细抓取 | ⭐⭐⭐⭐⭐ |
| **Epoch / episodes** | 50 ep / 11–32 epoch 不足；GROO T 需 ~40 epoch 才可用 | ⭐⭐⭐⭐ |
| **Arm location** | 底盘偏移导致绝对关节角 domain shift；merge 加剧 | ⭐⭐⭐⭐ |
| **Camera view** | 视角 OK，非主因 | ⭐ |

---

## 9. 经验教训 & 可复用的 Conclusion

### 9.1 技术结论（适合面试/展示）

1. **VLA 不是越大越好。** 在 50 条 demo 的 SO101 单臂场景，450M 的 SmolVLA 比 3B 的 PI0.5 更实用；模型容量需与数据量匹配。

2. **绝对关节角对 sim-to-real / 采集条件变化极度敏感。** 底盘移动 5° 即可导致 merged 模型失效；相对动作或 online fine-tuning（HIL-SERL）是更鲁棒的路径。

3. **离线指标 ≠ 真机效果。** PI0.5 离线 MAE ~2.1 仍无法抓取；评估必须包含真机 gripper 行程、关节速度等物理指标。

4. **任务简化 ≠ 训练简化。** 从 "多物体入盘" 到 "单白块入盘"，视觉复杂度降低，但对 **夹爪精细控制** 要求升高；小模型 + 足够 epoch 反而更重要。

5. **数据合并需谨慎。** 仅因 "同任务文本" 合并不同底盘/位姿的数据，会引入 label conflict，尤其在使用 absolute action 的 VLA 上。

### 9.2 若继续 PI0.5 的改进方向

| 方向 | 具体措施 |
|------|----------|
| 数据 | 在当前底盘上补录至 100+ ep；不 merge 旧数据 |
| 训练 | 远程训满 50k–80k steps（40+ epoch）；或 HIL-SERL online 微调 |
| 推理 | `MAX_RELATIVE_TARGET=25~30`；RTC 减少 chunk 开环误差 |
| 替代 | 同数据优先 SmolVLA 100k / GROO T base2 120k |

### 9.3 一句话 Summary（面试用）

> 在 SO101 单臂 pick-and-place 任务上，我从多物体入盘逐步收敛到单白块精抓，系统评估了 PI0.5 在不同数据规模、底盘位置、训练 epoch 下的真机表现。核心发现是：**3B VLA 在小样本 + 绝对关节角设定下，难以学会可靠抓取；数据 domain 一致性（底盘/位姿）比单纯增加 episode 数更重要；离线 loss/MAE 不能预测真机成功率。** 最终在同数据上 SmolVLA 表现优于 PI0.5，GROO T 在充分训练（120k steps）后达到可用水平。

---

## 10. 附录：关键路径

| 类型 | 路径 |
|------|------|
| 最佳 PI0.5 checkpoint | `outputs/train/pi05_yellow_white_base2_b16_50k_40000/pretrained_model` |
| base2 数据集 | `/home/rxn/datasets/yellow_white_base2` |
| 推理脚本 | `infer_pi05.sh` |
| 训练脚本 | `train_pi05.sh` |
| Eval 录像 | `outputs/eval/pi05_yellow_white_base2_b16_50k_40000/` |
| 腕部视角抽检 | `outputs/wrist_preview_base2/` |

---

*文档生成：2026-07-12 | 平台：LeRobot + SO101 + PI0.5*
