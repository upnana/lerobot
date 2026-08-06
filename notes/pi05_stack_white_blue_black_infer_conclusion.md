# Conclusion：pi05 stack white→blue→black 真机推理

日期：2026-08-05

## 目标

用本地 weight 在 SO101 上推理三色叠方块任务（自下而上：white → blue → black）。

| 项 | 路径 |
|---|---|
| Weight | `/home/rxn/lerobot_alohamini/pretrained_models/pi05_stack_white_blue_black/` |
| 采集数据 | `/home/rxn/datasets/stack_3blocks_white_blue_black`（199 ep / 122138 frames） |
| 训练步数 | 80,000 steps（batch 16，relative actions，gripper 绝对） |

## 已确认

- 数据可在 `conda` env `lerobot` 中用 `LeRobotDataset` 正常加载（v3.0，front + wrist，`so101_follower`）。
- 本地采集是**抓取叠方块**（约 89% episode 夹爪会张开），不是推方块。
- Weight README / `train_config` 也写明是 stack（`upna/stack_white_blue_black` / `stack_3blocks_white_blue_black_train`），不是 push 模型。
- 真机上若表现像「推」，多半是夹爪未开、姿态/视角不对导致的失败行为，不是训成了 push。

## Offline 诊断（lerobot env）

```bash
cd /home/rxn/lerobot
bash infer_pi05_stack_white_blue_black.sh offline
```

结果概要：

- 模型可加载；`use_relative_actions=True`。
- pred 跨帧有变化（未塌缩）；avg MAE vs gt ≈ 8.2°（脚本阈值偏严，仅作参考）。
- 曾报错 `AbsoluteActionsProcessorStep requires a paired RelativeActionsProcessorStep` —— 已在 `src/lerobot/policies/factory.py` 加载后 reconnect Relative/Absolute。

## 真机推理脚本

| 方式 | 脚本 | 说明 |
|---|---|---|
| lerobot + `lerobot-record` | `/home/rxn/lerobot/infer_pi05_stack_white_blue_black.sh` | offline / robot / viz / cameras |
| alohamini + RTC（推荐） | `/home/rxn/lerobot_alohamini/infer_pi05_stack_white_blue_black.sh` | `lerobot-rollout`，`INFERENCE_TYPE=rtc` |

推荐 RTC：

```bash
cd /home/rxn/lerobot_alohamini
NUM_EPISODES=1 EPISODE_TIME_S=90 bash infer_pi05_stack_white_blue_black.sh

# 快速试跑（可不落盘）
STRATEGY=base DURATION=60 bash infer_pi05_stack_white_blue_black.sh
```

相机预览（对比采集视角）：

```bash
cd /home/rxn/lerobot
bash get-data-stack_3blocks.sh preview
# Rerun 看 observation.front / observation.wrist；Ctrl+C 退出
```

## 踩过的坑与修复

| 问题 | 原因 | 处理 |
|---|---|---|
| Absolute/Relative 配对报错 | `use_relative_actions=True`，磁盘加载后 `relative_step` 丢失 | `lerobot`：`factory.py` 增加 `_reconnect_relative_absolute_steps` |
| `FileExistsError` eval 目录 | 脚本先 `mkdir` 了 `LeRobotDataset.create` 的 root | 只建父目录，不预建时间戳目录 |
| Calibration offset mismatch | 电机 EEPROM ≠ 标定文件 | 提示时**只按 Enter**写回文件；**不要按 `c`** |
| alohamini 直接进完整重标定 | `SOFollower.name=so_follower`，找 `robots/so_follower/so101_follower.json`，文件实际在 `robots/so101_follower/` | symlink + 脚本加 `--robot.calibration_dir=.../so101_follower` |
| `repo_id` 报错 | `lerobot-rollout` 要求名字含 `rollout_` | 改为 `local/rollout_pi05_stack_white_blue_black` |
| 大量 `max_relative_target` clamp | 策略目标远离当前位 / 越出 ±100 | 安全限速；需对齐初始姿态 |
| 开始「往下拧」 | 初始肩姿与 demo 差很多（demo 约 shoulder_lift≈-98） | 开跑前摆到接近采集初始姿 |
| 卡在极限位抖动 | 收臂顶死在 -100/+100，relative 输出越界被 clamp | 略展开离开极限，再开策略 |

## 相机对比（live vs 采集）

### Front

- 视角家族一致：高角度俯视白桌、白纸、三色块、背景纸箱线材。
- 差别不大：构图/亮度/臂姿略有不同。Front **基本可用**。

### Wrist（主要 gap）

| | Live | 采集任务中 |
|---|---|---|
| 内容 | 常看到**地板**，夹爪在画面底部 | **桌面 + 方块 + 白纸** |
| 含义 | 腕相机未对准工作区 | 任务时腕相机应对着桌面 |

腕部视角不一致，是当前部署最需要先修的视觉问题。

## 最终判断

1. **链路已打通**：weight 可加载；标定可用；`lerobot_alohamini` RTC rollout 可启动。
2. **模型本身是 stack，不是 push**；「像推」更像失败 rollout（闭爪扫桌）。
3. **真机仍不稳**：优先对齐 **腕相机对准工作区** + **初始臂姿接近 demo**，再评策略；仍差再查域差/模型质量。

## 建议下一步

1. `bash get-data-stack_3blocks.sh preview`，边调臂边看 wrist 是否看到桌面与三块。
2. 三块按采集习惯摆开（间距够、顺序可随机但任务仍 white→blue→black）。
3. RTC 少跑几条：`NUM_EPISODES=1 EPISODE_TIME_S=90`。
4. 观察夹爪是否先开再合；若一直闭爪扫，再考虑换 ckpt / 补数据 / 查 relative-action 部署。
