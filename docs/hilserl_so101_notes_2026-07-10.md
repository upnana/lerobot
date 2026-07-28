# HIL-SERL SO101 训练笔记（2026-07-10）

任务：单臂 SO101，白块 → 黄盘 pick-and-place，HIL-SERL + Leader 人工干预 + Reward Classifier。

---

## 1. 硬件与路径

| 设备 | 路径 |
|------|------|
| Follower | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00` |
| Leader | `/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00` |
| Front 相机 | `/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0` |
| Wrist 相机 | `/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0` |

| 数据/输出 | 路径 |
|-----------|------|
| 离线 demo（15 ep，键盘 EE） | `/home/rxn/datasets/hilserl_demos` |
| Crop 后 demo | `/home/rxn/datasets/hilserl_demos_cropped_resized` |
| Reward classifier 数据集 | `/home/rxn/datasets/hilserl_reward_classifier` |
| RL 训练输出 | `/home/rxn/lerobot/outputs/train/hilserl_so101_pick_place` |
| Classifier checkpoint | `.../reward_classifier_so101_pick_place/checkpoints/last/pretrained_model` |
| 在线采集数据 | `outputs/train/hilserl_so101_pick_place/dataset/`（~105 ep） |

Conda 环境：`lerobot`（HIL-SERL）；GROOT 等其它模型用 `lerobot-groot`，互不干扰。

---

## 2. 整体流程（Step 1–5）

```
Step 2  录 demo          → record_hilserl_demos.sh
Step 3  Crop             → crop_hilserl_demos.sh
Step 4a Relabel classifier → relabel_hilserl_classifier.sh
Step 4b 训 classifier    → train_reward_classifier_so101.sh
Step 5  HIL-SERL RL      → train_hilserl_so101.sh learner + actor
```

旧数据（键盘 demo、crop 数据集、checkpoint、online 数据）**仍可用**，换 Leader 不必重录。

---

## 3. 关键脚本

```bash
# Step 4：Reward Classifier
OVERWRITE=1 bash relabel_hilserl_classifier.sh   # 重新 relabel 时
bash train_reward_classifier_so101.sh

# Step 5：RL（必须先 learner，后 actor）
RESUME=1 bash train_hilserl_so101.sh learner   # 终端 1
bash train_hilserl_so101.sh actor              # 终端 2
```

默认 GPU：`CUDA_VISIBLE_DEVICES=1`（与 GROOT 训练错开）。

---

## 4. 主配置文件

`configs/hilserl_so101_train.json`

### Leader 控制

- `control_mode: "leader"`
- `mirror_leader_on_follower: false`（默认）— Leader **不**跟 Follower 动，只在干预时用
- **Space**：开/关人工干预（ON = 拖动 Leader 控制 Follower）
- **s / q / r**：成功 / 失败 / 重录

### Reset / 起手位

```json
"fixed_reset_joint_positions": [0.0, 0.0, 0.0, 90.0, 0.0, 5.0]
```

关节顺序：`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`。

- **每局开始 / episode 结束后** 会回到此姿态（约 5 s）
- 成功完成时 **没有单独「结束关节角」**；块进盘后停在当前姿态
- Reset 画面 **不要** 像「已成功」否则 classifier 误判

### Reward Classifier（已接入）

```json
"reward_classifier": {
  "pretrained_path": ".../reward_classifier_so101_pick_place/checkpoints/last/pretrained_model",
  "success_threshold": 0.85,
  "success_reward": 1.0,
  "terminate_on_success": false
}
```

- `terminate_on_success: false`：判成功只给 reward，**不自动结束** episode（避免 reset 后误触发立刻结束）
- 手动 **s** 仍可结束（`reset.terminate_on_success: true`）
- 误判多 → 阈值提到 0.9；几乎不触发 → 降到 0.8

---

## 5. 今天修过的 Bug

| 问题 | 原因 | 处理 |
|------|------|------|
| Classifier 训练 `KeyError: 'names'` | Relabel 数据集 image feature 缺 `names` | 修 `relabel_reward_classifier_dataset.py`；已 patch 现有 `meta/info.json` |
| `Classifier` 无 `normalize_inputs` | 归一化在 preprocessor pipeline | `RewardClassifierProcessorStep` 加载 `classifier_preprocessor.json` |
| Actor `dict has no attribute 'to'` | 把 Leader joint dict 当 action 发给 learner | Actor 存 3D policy action，非 raw teleop |
| Leader 启动 `enable_torque` 失败 | connect 后立即上力矩，motor 3 通信失败 | 默认松力矩；mirror 时再 lazy enable + retry=5 |
| Leader 一直跟 Follower 动 | 默认 mirror 开启 | `mirror_leader_on_follower: false` |
| Reset 后 step 3 reward=1.0 | Classifier 把 reset 姿态当成功 | 提高 threshold + `terminate_on_success: false` |
| Actor 连不上 Learner 50051 | Actor 崩溃后 gRPC 3 线程占满 | 重启 learner；代码改 `MAX_WORKERS` 3→8 |

---

## 6. 新增/改动的代码文件（本地 patch）

- `src/lerobot/teleoperators/so101_leader/so101_leader_hil.py` — Leader HIL 适配
- `src/lerobot/rl/relabel_reward_classifier_dataset.py` — Classifier 数据集 relabel
- `src/lerobot/processor/hil_processor.py` — Leader 干预 + Classifier 推理
- `src/lerobot/rl/actor.py` — Learner action 格式、干预标记
- `src/lerobot/rl/gym_manipulator.py` — Leader 模式、mirror 配置
- `src/lerobot/rl/learner_service.py` — gRPC 线程池 8
- `configs/hilserl_so101_reward_classifier_train.json`
- Shell：`relabel_hilserl_classifier.sh`, `train_reward_classifier_so101.sh`, `train_hilserl_so101.sh`

---

## 7. 位置概念（易混）

| 概念 | 说明 |
|------|------|
| **Reset / 起手位** | `fixed_reset_joint_positions`，每局开始与 episode 结束后回去 |
| **成功姿态** | 块在黄盘、夹爪松开；由 classifier 看图像判断，无单独关节配置 |
| **Episode 结束** | 按 s/q、超时 90s、或（若开启）classifier `terminate_on_success` |
| **Leader 平时** | 松力矩、自由摆放，不 mirror |
| **Leader 干预** | Space ON → 拖动 Leader 控 Follower |

---

## 8. 故障排查

### 相机 hang

Wrist Sonix 易 `select() timeout` → 重插 USB，确认 by-id 路径。

### Actor 崩溃 / gRPC UNAVAILABLE

1. 先只重启 Actor
2. 仍失败 → **Ctrl+C Learner**，`RESUME=1 bash train_hilserl_so101.sh learner`，等 `gRPC server started`，再启 Actor

### Learner OOM

`storage_device: cpu`（已配置）；必要时 `USE_DUAL_GPU=1` 把 buffer 放 cuda:1。

### 录 demo 提示 15/15 已满

旧键盘 demo 在；要重录 Leader demo 才用 `FRESH=1 bash record_hilserl_demos.sh`。

---

## 9. 后续可选

- [ ] 微调 `fixed_reset_joint_positions`，让 reset 画面不像成功
- [ ] Classifier 不准时：加失败样本 / 重训 / 调 threshold
- [ ] Leader 干预时 action 存 EE delta（当前干预步仍存 policy action，可后续改进）
- [ ] 合并官方 PR #3086 Leader 支持（当前为本地 patch）

---

## 10. 快速命令备忘

```bash
# 检查
bash train_hilserl_so101.sh check

# 续训 RL
RESUME=1 bash train_hilserl_so101.sh learner
bash train_hilserl_so101.sh actor

# 查看日志
tail -f outputs/train/hilserl_so101_pick_place/logs/learner_*.log
tail -f outputs/train/hilserl_so101_pick_place/logs/actor_*.log
```
