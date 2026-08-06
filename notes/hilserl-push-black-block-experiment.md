# HIL-SERL · Push Black Block → White Paper 实验结论

面向学习笔记 / Hugging Face Hub / 求职面试。

任务：在 **SO101** 真机上，用 **Human-in-the-Loop Soft Actor-Critic（HIL-SERL）** 学习将黑块推入白纸区域；从少量遥操作演示出发，经奖励分类器与在线 Actor–Learner 交互，策略探索从高空/大转角乱探收敛到更合理的正面推块行为。

---

## 1. 任务与设置

| 项目 | 内容 |
|------|------|
| 任务指令 | `push the black block`（推至白纸目标区） |
| 机器人 | SO101 follower + **leader** 遥操作 / 人工干预 |
| 相机 | front（UGREEN）+ wrist（Sonix），训练侧 crop → **128×128** |
| 算法 | **HIL-SERL（SAC）** + 视觉 **Reward Classifier** |
| 视觉骨干 | `helper2424/resnet10`（冻结 encoder） |
| 控制 | EE delta；在线约 **5 Hz**；`control_mode=leader` |
| 环境 | `conda: lerobot`，Actor–Learner 分进程（gRPC `50051`） |
| 配置 | `configs/hilserl_so101_push_black_block_train.json` |

**与 VLA 微调的关系：** 本实验 **不** 直接用 PPO/SAC 微调 SmolVLA/PI0.5 权重（结构不兼容）。正确范式是 IL/VLA 作先验，HIL-SERL 训练兼容的小策略（SAC）。

---

## 2. 数据

### 2.1 离线演示（Step 2–3）

| 指标 | 数值 |
|------|------|
| Episodes | **25** |
| Frames | **3,531** |
| 标注 | `s`=成功 / `q`=失败；起点可小幅随机，白纸目标固定 |
| Crop 后路径 | `/home/rxn/datasets/hilserl_push_black_block_demos_cropped_resized` |

### 2.2 奖励分类器数据（Step 4）

| 指标 | 数值 |
|------|------|
| 输出片段 | **9** ep / **570** frames（按 64 帧打包，**非**只用了 9 条 demo） |
| 正样本 | 成功轨迹尾帧（`success_tail_frames=15`） |
| 负样本 | 失败轨迹抽样，约 1:1 平衡 |
| Checkpoint | `outputs/train/reward_classifier_so101_push_black_block/checkpoints/last` |

### 2.3 在线 RL 数据（Step 5，截至成文）

| 指标 | 数值 |
|------|------|
| 在线 dataset | **~416** ep / **~31,216** frames |
| Actor 完成局数（日志） | **~430** |
| dataset_offline | 25 ep（来自演示，供离线 buffer） |

---

## 3. 方法流程

```text
录演示 (Leader)
    → Crop ROI
    → Relabel + 训 Reward Classifier
    → Learner（SAC 更新） ⟷ Actor（真机 rollout + 人工干预）
    → Replay Buffer 反复采样；定期 checkpoint
```

### 干预协议（HIL）

| 键 | 作用 |
|----|------|
| **Space** | 开/关人工干预（先对齐 Leader→Follower，再拖动） |
| **s** | 本局成功结束 |
| **q** | 本局失败结束 |
| **r** | 重录 |

平时：策略控 Follower，Leader 不 mirror。干预时：人控 Leader，Follower 跟随。

### 安全

- EE / 相对目标限位  
- 人工 Space 接管  
- Episode 结束回固定 home  
- 可回滚 `checkpoints/*`  

---

## 4. 训练

| 项 | 数值 / 说明 |
|----|-------------|
| 算法 | Soft Actor-Critic（off-policy） |
| Online steps（配置） | 100,000 |
| 已优化步（日志） | **~70,200** |
| Checkpoint | `002000` … **`070000`（last）** |
| Batch | 256（learner） |
| Buffer | online + offline；干预步写入在线数据集并可反复学 |
| 输出目录 | `outputs/train/hilserl_so101_push_black_block` |

**路径：**
```text
outputs/train/hilserl_so101_push_black_block/checkpoints/last/pretrained_model
outputs/train/reward_classifier_so101_push_black_block/checkpoints/last/pretrained_model
```

**脚本：**
```bash
# Learner
bash train_hilserl_so101.sh learner          # 首次
RESUME=1 bash train_hilserl_so101.sh learner # 续训

# Actor（真机）
bash train_hilserl_so101.sh actor
```

---

## 5. 实验结果与结论

### 5.1 定量（Actor 日志，reward>0 计为有成功信号）

| 指标 | 结果 |
|------|------|
| 总体成功 | **~328 / 430 ≈ 76%** |
| 最近 30 局 | **~22 / 30 ≈ 73%**（随阶段波动） |
| 最近 50 局 | **~38 / 50 ≈ 76%** |

> 注：`reward` 含分类器触发与人工 `s`；多帧成功信号时单局 reward 可能 >1。解读时以「是否完成推入白纸」的人工观察为准，日志奖励为辅助。

### 5.2 定性（核心结论）

| 阶段 | 行为 | 解读 |
|------|------|------|
| 训练早期 | 易高抬、左右大转角乱探 | SAC 高熵探索 + 任务尚未约束好 |
| 训练中后期 | 少高空/大甩动；多在 **front 视角**、桌面高度内触块并前推 | 演示 + 奖励 + 干预把探索压到任务相关流形 |

**结论 1 — HIL-SERL 在真机推块上有效：**  
少量演示即可启动；在线交互与人工纠正后，策略从无工作空间探索收敛到任务区推块。

**结论 2 — 人机回路是样本效率与安全的关键：**  
Space 纠正提供高价值 off-policy 数据；无人时虽能继续跑，但易产生大量失败轨迹且有安全风险。

**结论 3 — 工程可靠性与算法同等重要：**  
本实验中影响训练连续性的主要是：奖励误触发、干预对齐 UX、相机被 Rerun 占用、Leader 总线丢包、gRPC / `torch.load(weights_only)` 序列化问题——真机 RL 交付需同时处理这些。

**结论 4 — 与大 VLA 的分工：**  
VLA/IL 适合快速得到可用先验；HIL-SERL 适合在固定技能上用奖励与干预做细化。二者互补，而非互相替代。

---

## 6. 局限与后续

- 目标区与光照变化较大时，仅靠当前分类器可能不够稳。  
- 未做严格的「零干预 N 局」标准化评测报告。  
- 尚未将 SmolVLA 动作蒸馏进 SAC（可选 BC 正则）。  
- 建议后续：画干预率↓ / 成功率↑曲线；补奖励与安全设计短文；固定评测协议上传到 Hub。

---

## 7. Hugging Face Hub 上传建议

可将本文件作为模型卡 / 数据集卡的 **`README.md`**，并附：

```text
# 建议附带产物
- checkpoints/last/pretrained_model/          # SAC 策略
- reward_classifier .../pretrained_model/    # 奖励分类器
- 本实验结论 Markdown
- （可选）若干成功/失败 rollout 短视频
```

**卡片摘要（可直接贴）：**

> Real-robot HIL-SERL (SAC) on SO101 for pushing a black block onto white paper. Pipeline: 25 leader demos → vision reward classifier → distributed actor–learner with leader interventions. After ~70k SAC updates and 400+ online episodes, exploration shifted from high/wide random motion to frontal, task-aligned pushing (~76% episodes with success signal in logs).

---

## 8. 一句话（笔记末尾）

在 SO101 上用 **演示 + 奖励分类器 + HIL-SERL(SAC)** 完成真机推块技能学习：人机干预保障安全与样本效率，策略行为从乱探收敛到正面有效推块。
