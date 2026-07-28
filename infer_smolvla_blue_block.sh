#!/bin/bash
# =============================================================================
# SmolVLA 推理（blue block → yellow tray）
# Usage:
#   bash infer_smolvla_blue_block.sh offline
#   bash infer_smolvla_blue_block.sh robot
#   EPISODE_TIME_S=90 NUM_EPISODES=1 bash infer_smolvla_blue_block.sh robot
# =============================================================================
set -euo pipefail

MODE="${1:-offline}"

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/blue_block_yellow_tray}"
DATA_REPO="${DATA_REPO:-my_pick_place/blue_block_yellow_tray}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/smolvla_blue_block_yellow_tray}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TOLERANCE_S="${TOLERANCE_S:-0.05}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
ROBOT_PORT="${ROBOT_PORT:-${FOLLOWER_PORT}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
FRONT_CAM="${FRONT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"

RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

FPS=30
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-120}"
RESET_TIME_S="${RESET_TIME_S:-15}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/smolvla_blue_block_yellow_tray}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 1
fi
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy" 2>/dev/null || {
    echo "ERROR: SmolVLA not installed. pip install -e \".[smolvla,feetech]\""
    exit 1
}

# allow CHECKPOINT=.../last or .../last/pretrained_model
if [ -d "${CHECKPOINT}/pretrained_model" ] && [ ! -f "${CHECKPOINT}/config.json" ]; then
    CHECKPOINT="${CHECKPOINT}/pretrained_model"
fi
if [ ! -f "${CHECKPOINT}/config.json" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    ls -la "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || true
    exit 1
fi

CKPT_DIR="$(dirname "${CHECKPOINT}")"
CHECKPOINT_STEP="$(basename "${CKPT_DIR}")"
if [ "${CHECKPOINT_STEP}" = "last" ] && [ -L "${CKPT_DIR}" ]; then
    CHECKPOINT_STEP="$(basename "$(readlink -f "${CKPT_DIR}")")"
fi

DATASET_TASK=$(python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="${DATA_REPO}", root="${DATA_ROOT}", tolerance_s=${TOLERANCE_S})
print(ds.meta.tasks.index[0])
PY
)
# DATASET_TASK="pick up the blue block and put it into the yellow tray"
DATASET_TASK="pick up the black block and put it into the yellow tray"
echo "=============================="
echo "SmolVLA blue_block Inference (${MODE})"
echo "Checkpoint: ${CHECKPOINT} (step ${CHECKPOINT_STEP})"
echo "Dataset:    ${DATA_ROOT}"
echo "Eval out:   ${EVAL_ROOT}"
echo "Task:       ${DATASET_TASK}"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Duration:   ${NUM_EPISODES} ep × ${EPISODE_TIME_S}s (+ ${RESET_TIME_S}s reset)"
echo "=============================="

if [ "${MODE}" = "cameras" ]; then
    ls -la /dev/v4l/by-id/ 2>/dev/null || true
    echo "Defaults: FRONT_CAM=${FRONT_CAM}"
    echo "          WRIST_CAM=${WRIST_CAM}"
    lerobot-find-cameras opencv || true
    exit 0
fi

if [ "${MODE}" = "viz" ]; then
    if [ -d "${EVAL_ROOT}" ] && [ -n "$(ls -A "${EVAL_ROOT}" 2>/dev/null)" ]; then
        LATEST_EVAL=$(ls -td "${EVAL_ROOT}"/*/ 2>/dev/null | head -1)
        LATEST_EVAL="${LATEST_EVAL%/}"
        lerobot-dataset-viz \
          --repo-id="upna/eval_smolvla_blue_block_yellow_tray" \
          --root="${LATEST_EVAL}" \
          --episode-index=0 \
          --mode=local
    else
        lerobot-dataset-viz \
          --repo-id="${DATA_REPO}" \
          --root="${DATA_ROOT}" \
          --episode-index=0 \
          --mode=local
    fi
    exit 0
fi

if [ "${MODE}" = "offline" ]; then
    python - <<PY
import json
import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats

checkpoint = "${CHECKPOINT}"
data_root = "${DATA_ROOT}"
repo_id = "${DATA_REPO}"
rename_map = json.loads('''${RENAME_MAP}''')

cfg = PreTrainedConfig.from_pretrained(checkpoint)
cfg.pretrained_path = checkpoint
cfg.device = "cuda"

ds = LeRobotDataset(repo_id=repo_id, root=data_root, tolerance_s=${TOLERANCE_S})
policy = make_policy(cfg=cfg, ds_meta=ds.meta, rename_map=rename_map)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=checkpoint,
    dataset_stats=rename_stats(ds.meta.stats, rename_map),
    preprocessor_overrides={
        "rename_observations_processor": {"rename_map": rename_map},
    },
)
policy.eval()

print("Running offline inference on blue_block dataset...")
preds, maes = [], []
for idx in [0, 1000, 5000, 10000, 20000]:
    if idx >= len(ds):
        continue
    policy.reset()
    sample = ds[idx]
    batch = preprocessor({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()})
    with torch.inference_mode():
        action = postprocessor(policy.select_action(batch))
    mae = (action.cpu().float() - sample["action"].float()).abs().mean().item()
    p = action.cpu().numpy().round(3).reshape(-1)
    preds.append(p)
    maes.append(mae)
    print(f"  frame {idx:6d}  pred={p}  gt={sample['action'].numpy().round(3)}  mae={mae:.4f}")

if len(preds) >= 2:
    std = np.array(preds).std(axis=0)
    print(f"\n[Diagnostic] pred std per joint: {std.round(2)}")
    print(f"[Diagnostic] avg MAE vs gt:      {float(np.mean(maes)):.3f}")
print("Offline inference OK.")
PY

elif [ "${MODE}" = "robot" ]; then
    if [ ! -e "${ROBOT_PORT}" ]; then
        echo "ERROR: Robot port not found: ${ROBOT_PORT}"
        ls /dev/ttyACM* /dev/serial/by-id/ 2>/dev/null || true
        exit 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
        echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB (need ~${MIN_FREE_MIB} MiB)"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo "ERROR: Not enough GPU memory on GPU ${PHYS_GPU}."
            nvidia-smi
            exit 1
        fi
    fi

    python -c "import scservo_sdk" 2>/dev/null || {
        echo "ERROR: pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
        exit 1
    }

    echo "Robot port: ${ROBOT_PORT}"
    echo "Front cam:  ${FRONT_CAM}"
    echo "Wrist cam:  ${WRIST_CAM}"
    echo ""
    echo "Checking cameras..."
    python - <<PY
import sys, cv2
failed = False
for name, target in [("front", "${FRONT_CAM}"), ("wrist", "${WRIST_CAM}")]:
    try:
        target = int(target)
    except ValueError:
        pass
    cap = cv2.VideoCapture(target)
    ok = cap.isOpened()
    shape = None
    if ok:
        ret, frame = cap.read()
        ok = ok and ret and frame is not None
        shape = frame.shape if ok else None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''}  ({target})")
    failed = failed or not ok
if failed:
    sys.exit(1)
PY

    echo ""
    echo ">>> Starting SmolVLA blue_block inference in 3s (Ctrl+C to abort)..."
    sleep 3

    EVAL_DATA_ROOT="${EVAL_ROOT}/$(date +%Y%m%d_%H%M%S)"
    echo "Eval save: ${EVAL_DATA_ROOT}"

    lerobot-record \
      --robot.type=so101_follower \
      --robot.port="${ROBOT_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.max_relative_target="${MAX_RELATIVE_TARGET}" \
      --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAM}, width: 640, height: 480, fps: ${FPS}}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: ${FPS}} }" \
      --display_data=true \
      --play_sounds="${PLAY_SOUNDS}" \
      --dataset.repo_id="upna/eval_smolvla_blue_block_yellow_tray" \
      --dataset.root="${EVAL_DATA_ROOT}" \
      --dataset.single_task="${DATASET_TASK}" \
      --dataset.fps="${FPS}" \
      --dataset.num_episodes="${NUM_EPISODES}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.push_to_hub=false \
      --dataset.rename_map="${RENAME_MAP}" \
      --policy.path="${CHECKPOINT}" \
      --policy.device=cuda \
      2>&1 | python -u -c "
import sys
phase = 'init'
shown = False
for raw in sys.stdin:
    line = raw.rstrip('\n')
    if 'Recording episode' in line:
        phase = 'record'
        shown = False
        print(); print('=' * 62)
        print('  SmolVLA blue_block 推理中 (${EPISODE_TIME_S}s)')
        print('=' * 62); print(line, flush=True)
    elif 'Reset the environment' in line:
        phase = 'reset'
        shown = False
        print(); print('-' * 62)
        print('  RESET 复位 (${RESET_TIME_S}s)')
        print('-' * 62); print(line, flush=True)
    elif 'No policy or teleoperator' in line and phase == 'reset':
        if not shown:
            print('  (reset 阶段日志已隐藏)', flush=True)
            shown = True
    else:
        print(line, flush=True)
"

else
    echo "Usage: bash infer_smolvla_blue_block.sh [offline|robot|cameras|viz]"
    exit 1
fi

echo "=============================="
echo "Inference finished"
echo "=============================="
