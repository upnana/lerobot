#!/bin/bash
# =============================================================================
# GR00T 推理脚本（基于 train_groot.sh 的 OUTPUT_DIR checkpoint）
# Usage:
#   conda activate lerobot-groot
#   bash infer_groot.sh offline          # 数据集离线推理（终端输出，无 GUI）
#   bash infer_groot.sh robot            # 真机推理（display_data → Rerun 窗口）
#   bash infer_groot.sh cameras          # 列出相机（稳定 by-id 路径）
#
#   CHECKPOINT=.../checkpoints/060000/pretrained_model bash infer_groot.sh offline
#   CUDA_VISIBLE_DEVICES=0 bash infer_groot.sh robot   # 默认 GPU 0（SmolVLA/PI0.5 训练占 GPU 1 时）
#   默认使用 OUTPUT_DIR/checkpoints/last/pretrained_model
#
# 相机说明：
#   /dev/videoN 序号会在重启/换 USB 口后变化，请用 /dev/v4l/by-id/... 固定设备。
#   覆盖示例：
#     FRONT_CAM=/dev/v4l/by-id/usb-XXX-video-index0 bash infer_groot.sh robot
#
# display_data 说明：
#   - 只在 robot 模式（lerobot-record）有效，会弹出 Rerun viewer 窗口
#   - offline 模式没有 display_data
#   - nohup/SSH 无 DISPLAY 时窗口不会出现；需在本机桌面终端前台运行
# =============================================================================
set -euo pipefail

MODE="${1:-offline}"

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/blue_block_yellow_tray}"
DATA_REPO="${DATA_REPO:-my_pick_place/blue_block_yellow_tray}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/groot_blue_block_yellow_tray}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"  # 限制每步最大移动，减轻 policy 输出抖动
CONDA_ENV="lerobot-groot"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # 默认 GPU 0；GROOT 训练也在 GPU 0
MIN_FREE_MIB="${MIN_FREE_MIB:-17000}"
# 采集视频时间戳常有亚毫秒漂移；默认 1e-4 会炸
TOLERANCE_S="${TOLERANCE_S:-0.05}"
# follower=ACM0, leader=ACM1
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
ROBOT_PORT="${ROBOT_PORT:-${FOLLOWER_PORT}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"

# 相机（与 get-data.sh 一致：front=0, wrist=2）
FRONT_CAM="${FRONT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"

# 真机推理时长（默认 30s；快速: EPISODE_TIME_S=20；更长: EPISODE_TIME_S=60）
FPS=30
NUM_EPISODES="${NUM_EPISODES:-1}"   # 多次验证: NUM_EPISODES=3 bash infer_groot.sh robot
EPISODE_TIME_S="${EPISODE_TIME_S:-30}"
RESET_TIME_S="${RESET_TIME_S:-5}"   # episode 间手动复位时间
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"  # true=语音提示 "Recording episode" / "Reset"
# SAVE_EVAL=0: 不保存，/tmp 临时目录，跑完即删（快速试动作）
# SAVE_EVAL=1（默认）: 每次新建独立目录保存 eval，不覆盖历史
SAVE_EVAL="${SAVE_EVAL:-1}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/groot_blue_block_yellow_tray}"
EVAL_SCRATCH="/tmp/groot_eval_scratch_${USER}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

# 刷新 Eagle cache
EAGLE_CACHE="${HOME}/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5"
mkdir -p "${EAGLE_CACHE}"
cp -r "${PROJECT_ROOT}/src/lerobot/policies/groot/eagle2_hg_model/." "${EAGLE_CACHE}/"

if [ ! -d "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    echo "Available checkpoints:"
    ls -la "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || true
    echo ""
    echo "Example:"
    echo "  CHECKPOINT=${OUTPUT_DIR}/checkpoints/080000/pretrained_model bash infer_groot.sh offline"
    exit 1
fi

# 解析 checkpoint step（last → 080000 等）
CKPT_DIR="$(dirname "${CHECKPOINT}")"
CHECKPOINT_STEP="$(basename "${CKPT_DIR}")"
if [ "${CHECKPOINT_STEP}" = "last" ] && [ -L "${CKPT_DIR}" ]; then
    CHECKPOINT_STEP="$(basename "$(readlink -f "${CKPT_DIR}")")"
fi

# 从 dataset meta 读取 task（与训练数据一致）
DATASET_TASK=$(python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="${DATA_REPO}", root="${DATA_ROOT}", tolerance_s=${TOLERANCE_S})
print(ds.meta.tasks.index[0])
PY
)

echo "=============================="
echo "GR00T Inference (${MODE})"
echo "Output:     ${OUTPUT_DIR}"
echo "Checkpoint: ${CHECKPOINT} (step ${CHECKPOINT_STEP})"
echo "Dataset task: ${DATASET_TASK}"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "=============================="

if [ "${MODE}" = "offline" ]; then
    python - <<PY
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors

checkpoint = "${CHECKPOINT}"
data_root = "${DATA_ROOT}"
repo_id = "${DATA_REPO}"
dataset_task = """${DATASET_TASK}"""

cfg = PreTrainedConfig.from_pretrained(checkpoint)
cfg.pretrained_path = checkpoint
cfg.device = "cuda"

ds = LeRobotDataset(repo_id=repo_id, root=data_root, tolerance_s=${TOLERANCE_S})
policy = make_policy(cfg=cfg, ds_meta=ds.meta)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=checkpoint,
    dataset_stats=ds.meta.stats,
)
policy.eval()

print(f"Task from dataset: {dataset_task}")
indices = [0, 1000, 5000, 10000, 50000]
print("Running offline inference on dataset samples...")
preds, maes = [], []
for idx in indices:
    if idx >= len(ds):
        continue
    policy.reset()
    sample = ds[idx]
    sample_task = sample.get("task", dataset_task)
    if isinstance(sample_task, torch.Tensor):
        sample_task = dataset_task
    batch = preprocessor({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()})
    with torch.inference_mode():
        action = policy.select_action(batch)
        action = postprocessor(action)
    gt = sample["action"]
    mae = (action.cpu().float() - gt.float()).abs().mean().item()
    p = action.cpu().numpy().round(3).reshape(-1)
    preds.append(p)
    maes.append(mae)
    print(f"  frame {idx:6d}  task={sample_task!r}")
    print(f"             pred={p}  gt={gt.numpy().round(3)}  mae={mae:.4f}")

import numpy as np
if len(preds) >= 2:
    preds = np.array(preds)
    std = preds.std(axis=0)
    span = preds.max(axis=0) - preds.min(axis=0)
    sat = ((np.abs(preds) > 99).mean(axis=0) * 100).round(0)
    avg_mae = float(np.mean(maes))
    print(f"\n[Diagnostic] pred std per joint:  {std.round(2)}")
    print(f"[Diagnostic] pred span per joint: {span.round(2)}")
    print(f"[Diagnostic] % frames near ±100:  {sat.astype(int)}% per joint")
    print(f"[Diagnostic] avg MAE vs gt:       {avg_mae:.3f}")
    if std.max() < 2.0:
        print("[Diagnostic] ⚠ pred barely changes → model ignores observations")
    elif avg_mae < 3.0:
        print("[Diagnostic] ✓ On training data, pred tracks gt reasonably (projector training worked)")
        print("[Diagnostic] ⚠ Real robot may still fail if cameras/pose differ from training")
        if sat.max() >= 40:
            print("[Diagnostic] ⚠ Many joint outputs hit ±100 (saturation) → DiT not fully adapted")
            print("[Diagnostic]   Recommend: bash train_groot_dit.sh")
    else:
        print("[Diagnostic] ⚠ High MAE on training data → model quality insufficient")
        print("[Diagnostic]   Recommend: bash train_groot_dit.sh")

print("Offline inference OK.")
PY

elif [ "${MODE}" = "robot" ]; then
    if [ "${SAVE_EVAL}" = "1" ]; then
        if [ -n "${EVAL_RUN_DIR:-}" ]; then
            EVAL_DATA_ROOT="${EVAL_RUN_DIR}"
            if [ -e "${EVAL_DATA_ROOT}" ]; then
                echo "ERROR: EVAL_RUN_DIR already exists: ${EVAL_DATA_ROOT}"
                echo "Pick another path or remove the old directory."
                exit 1
            fi
        else
            _eval_base="${EVAL_ROOT}/ckpt${CHECKPOINT_STEP}_$(date +%Y%m%d_%H%M%S)"
            EVAL_DATA_ROOT="${_eval_base}"
            _eval_n=0
            while [ -e "${EVAL_DATA_ROOT}" ]; do
                _eval_n=$((_eval_n + 1))
                EVAL_DATA_ROOT="${_eval_base}_${_eval_n}"
            done
        fi
        # 不要预先 mkdir：lerobot-record 会 create，目录已存在会 FileExistsError
        EVAL_VIDEO=true
        echo "Eval save: ON → ${EVAL_DATA_ROOT}"
    else
        EVAL_DATA_ROOT="${EVAL_SCRATCH}"
        EVAL_VIDEO=false
        rm -rf "${EVAL_DATA_ROOT}"
        echo "Eval save: OFF (scratch ${EVAL_DATA_ROOT}, deleted after run)"
    fi

    echo "Robot port: ${ROBOT_PORT}"
    echo "Front cam: ${FRONT_CAM}, Wrist cam: ${WRIST_CAM}"
    echo "Duration: ${NUM_EPISODES} episodes × ${EPISODE_TIME_S}s (+ ${RESET_TIME_S}s reset each) @ ${FPS}fps"
    echo "Task: ${DATASET_TASK}"
    echo ""

    # --- 真机前置检查 ---
    if [ ! -e "${ROBOT_PORT}" ]; then
        echo "ERROR: Robot port not found: ${ROBOT_PORT}"
        echo "Available serial ports:"
        ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || echo "  (none)"
        echo "Set correct port: ROBOT_PORT=/dev/ttyACM1 bash infer_groot.sh robot"
        exit 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"   # 多卡时取第一个
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
        echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB (GR00T inference needs ~${MIN_FREE_MIB} MiB)"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            OTHER_GPU=$([ "${PHYS_GPU}" = "0" ] && echo 1 || echo 0)
            echo "ERROR: Not enough GPU memory on GPU ${PHYS_GPU}."
            echo "  SmolVLA 训练占 GPU 1 时请用: CUDA_VISIBLE_DEVICES=0 bash infer_groot.sh robot"
            echo "  或换卡: CUDA_VISIBLE_DEVICES=${OTHER_GPU} bash infer_groot.sh robot"
            nvidia-smi
            exit 1
        fi
    fi

    python -c "import scservo_sdk" 2>/dev/null || {
        echo "ERROR: scservo_sdk not installed (required for SO101 Feetech motors)."
        echo "  pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
        echo "  or: pip install -e \".[feetech]\""
        exit 1
    }

    echo "Checking follower motors (ping ×3)..."
    if ! python - <<PY; then
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
import time

port = "${ROBOT_PORT}"
motors = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}
bus = FeetechMotorsBus(port=port, motors=motors)
bus.connect(handshake=False)
failed = []
for attempt in range(3):
    failed = []
    for name, m in motors.items():
        try:
            bus.ping(name)
        except Exception as e:
            failed.append(f"{name}(id={m.id}): {e}")
    if not failed:
        break
    time.sleep(0.3)
bus.disconnect()
if failed:
    print("Failed motors after 3 attempts:")
    for line in failed:
        print(" ", line)
    raise SystemExit(1)
print("All 6 motors OK")
PY
        echo ""
        echo "ERROR: Motor bus check failed."
        echo "  1. Power-cycle the arm (off 10s, on)"
        echo "  2. Reseat daisy-chain cables (especially elbow id=3 / wrist id=4)"
        echo "  3. Ensure no other program uses ${ROBOT_PORT} (get-data.sh / calibrate)"
        exit 1
    fi

    echo "Checking cameras..."
    echo "  front: ${FRONT_CAM}"
    echo "  wrist: ${WRIST_CAM}"
    python - <<PY
import sys
import cv2

cams = [
    ("front", "${FRONT_CAM}"),
    ("wrist", "${WRIST_CAM}"),
]

def open_target(target: str):
    # 支持整数 index 或 /dev/... 路径
    try:
        idx = int(target)
    except ValueError:
        idx = target
    return cv2.VideoCapture(idx)

failed = False
for name, target in cams:
    cap = open_target(target)
    ok = cap.isOpened()
    shape = None
    if ok:
        ret, frame = cap.read()
        ok = ok and ret and frame is not None
        shape = frame.shape if ok else None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape if ok else ''}  ({target})")
    if not ok:
        failed = True
if failed:
    print("ERROR: Camera check failed. Run: bash infer_groot.sh cameras", file=sys.stderr)
    sys.exit(1)
PY
    echo ""
    echo "┌─ 阶段说明（如何分辨）──────────────────────────────────────"
    echo "│  听到/看到 'Recording episode N'  →  ${EPISODE_TIME_S}s  policy 驱动，机械臂应动"
    echo "│  听到/看到 'Reset the environment' →  ${RESET_TIME_S}s  手动摆物体，扭矩已关、臂可手扶"
    echo "│  'No policy or teleoperator' 只在 RESET 阶段出现（已过滤刷屏）"
    echo "│  Rerun 窗口里 action 曲线有变化 = policy 在工作"
    echo "└────────────────────────────────────────────────────────────"
    echo ""
    echo ">>> Starting real robot inference in 3s (Ctrl+C to abort)..."
    sleep 3

    if [ -z "${DISPLAY:-}" ]; then
        echo "WARN: DISPLAY is empty. Rerun viewer (display_data) will NOT show."
        echo "      Run in a local desktop terminal, not nohup/SSH without -X."
        echo "      Or after start, connect manually: rerun --connect rerun+http://127.0.0.1:9876/proxy"
        echo ""
    else
        echo "display_data=true → Rerun viewer should open in a separate window."
        echo "If you don't see it, check taskbar / Alt+Tab, or run:"
        echo "  rerun --connect rerun+http://127.0.0.1:9876/proxy"
        echo ""
    fi

    # 输出过滤：阶段横幅 + 隐藏 reset 阶段的刷屏日志
    if ! lerobot-record \
      --robot.type=so101_follower \
      --robot.port="${ROBOT_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.max_relative_target="${MAX_RELATIVE_TARGET}" \
      --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30} }" \
      --display_data=true \
      --dataset.repo_id="upna/eval_groot_yellow_white" \
      --dataset.root="${EVAL_DATA_ROOT}" \
      --dataset.single_task="${DATASET_TASK}" \
      --dataset.fps=${FPS} \
      --dataset.num_episodes=${NUM_EPISODES} \
      --dataset.episode_time_s=${EPISODE_TIME_S} \
      --dataset.reset_time_s=${RESET_TIME_S} \
      --dataset.video=${EVAL_VIDEO} \
      --dataset.push_to_hub=false \
      --play_sounds=${PLAY_SOUNDS} \
      --policy.path="${CHECKPOINT}" \
      --policy.device=cuda \
      --policy.use_bf16=true \
      --policy.use_amp=false \
      2>&1 | python -u -c "
import sys

phase = 'init'
shown_reset_hint = False

for raw in sys.stdin:
    line = raw.rstrip('\n')
    if 'Recording episode' in line:
        phase = 'record'
        shown_reset_hint = False
        print()
        print('=' * 62)
        print('  POLICY 推理中 — 机械臂应该动 (${EPISODE_TIME_S}s)')
        print('=' * 62)
        print(line, flush=True)
    elif 'Reset the environment' in line:
        phase = 'reset'
        shown_reset_hint = False
        print()
        print('-' * 62)
        print('  RESET 复位 — 扭矩已关，可手扶机械臂摆物体 (${RESET_TIME_S}s)')
        print('-' * 62)
        print(line, flush=True)
    elif 'No policy or teleoperator' in line:
        if phase == 'reset':
            if not shown_reset_hint:
                print('  (reset 阶段无 policy 输出，后续同类日志已隐藏)', flush=True)
                shown_reset_hint = True
            continue
        print(line, flush=True)
    else:
        print(line, flush=True)
"
    then
        [ "${SAVE_EVAL}" = "0" ] && rm -rf "${EVAL_DATA_ROOT}"
        exit 1
    fi
    [ "${SAVE_EVAL}" = "0" ] && rm -rf "${EVAL_DATA_ROOT}"

elif [ "${MODE}" = "cameras" ]; then
    echo "Stable camera symlinks (/dev/v4l/by-id):"
    ls -la /dev/v4l/by-id/ 2>/dev/null || echo "  (none — cameras not plugged in?)"
    echo ""
    echo "Current infer_groot.sh defaults:"
    echo "  FRONT_CAM=${FRONT_CAM}"
    echo "  WRIST_CAM=${WRIST_CAM}"
    echo ""
    echo "OpenCV scan (lerobot-find-cameras):"
    lerobot-find-cameras opencv || true
    echo ""
    echo "Tip: always use *-video-index0 (index1 is usually metadata)."
    echo "If front/wrist view is swapped vs training, swap FRONT_CAM and WRIST_CAM."

elif [ "${MODE}" = "viz" ]; then
    echo "Opening dataset viewer (episode 0)..."
    echo "Rerun window should appear. Ctrl+C to quit."
    lerobot-dataset-viz \
      --repo-id="${DATA_REPO}" \
      --root="${DATA_ROOT}" \
      --episode-index=0 \
      --mode=local

else
    echo "Usage: bash infer_groot.sh [offline|robot|viz|cameras]"
    exit 1
fi

echo "=============================="
echo "Inference finished"
echo "=============================="
