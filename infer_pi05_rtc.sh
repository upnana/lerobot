#!/bin/bash
# =============================================================================
# PI0.5 RTC 真机推理（SO101 + front/wrist）
# Real-Time Chunking：比默认 n_action_steps=50 更频繁重规划，动作更连贯
#
# Usage:
#   conda activate lerobot          # pip install -e ".[pi,feetech]"
#   bash infer_pi05_rtc.sh          # RTC 真机推理
#   bash infer_pi05_rtc.sh cameras  # 列出相机
#
# 关闭 RTC 对比:
#   RTC_ENABLED=false bash infer_pi05_rtc.sh
#
# 指定 checkpoint:
#   CHECKPOINT=/path/to/pretrained_model bash infer_pi05_rtc.sh
# =============================================================================
set -euo pipefail

MODE="${1:-robot}"

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/yellow_white_base2}"
DATA_REPO="${DATA_REPO:-my_pick_place/yellow_white_base2}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/train/pi05_yellow_white_base2_b16_50k_15000/pretrained_model}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/rxn/models/paligemma-3b-pt-224}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
ROBOT_PORT="${ROBOT_PORT:-${FOLLOWER_PORT}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
FRONT_CAM="${FRONT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"

FPS="${FPS:-30}"
DURATION="${DURATION:-60}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"

# RTC
RTC_ENABLED="${RTC_ENABLED:-true}"
RTC_EXECUTION_HORIZON="${RTC_EXECUTION_HORIZON:-10}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-10.0}"
ACTION_QUEUE_SIZE="${ACTION_QUEUE_SIZE:-30}"

RTC_SCRIPT="${PROJECT_ROOT}/examples/rtc/eval_with_real_robot.py"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    echo "  conda create -n lerobot python=3.10 -y && conda activate lerobot"
    echo "  cd ${PROJECT_ROOT} && pip install -e \".[pi,feetech]\""
    exit 1
fi
conda activate "${CONDA_ENV}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

if ! python - <<'PY'
try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
    from lerobot.policies.rtc.configuration_rtc import RTCConfig  # noqa: F401
except ImportError as e:
    raise SystemExit(f"PI0.5/RTC import failed: {e}")
print("PI0.5 + RTC import OK")
PY
then
    echo "ERROR: PI0.5/RTC dependencies missing in env '${CONDA_ENV}'."
    echo "  cd ${PROJECT_ROOT} && pip install -e \".[pi,feetech]\""
    exit 1
fi

if [[ "${CHECKPOINT}" = /* ]] || [[ "${CHECKPOINT}" = ./* ]]; then
    if [ ! -f "${CHECKPOINT}/config.json" ]; then
        echo "ERROR: local checkpoint not found: ${CHECKPOINT}"
        ls -la "$(dirname "${CHECKPOINT}")" 2>/dev/null || true
        exit 1
    fi
fi

if [ ! -d "${TOKENIZER_PATH}" ]; then
    echo "ERROR: tokenizer not found: ${TOKENIZER_PATH}"
    exit 1
fi

for f in "${CHECKPOINT}/config.json" "${CHECKPOINT}/policy_preprocessor.json"; do
    if [ -f "${f}" ] && grep -q '/workspace/models/paligemma' "${f}"; then
        sed -i "s|/workspace/models/paligemma-3b-pt-224|${TOKENIZER_PATH}|g" "${f}"
        echo "Patched tokenizer path in ${f} -> ${TOKENIZER_PATH}"
    fi
done

DATASET_TASK=$(python - <<PY
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(repo_id="${DATA_REPO}", root="${DATA_ROOT}")
print(ds.meta.tasks.index[0])
PY
)

echo "=============================="
echo "PI0.5 RTC Inference (${MODE})"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset:    ${DATA_ROOT}"
echo "Task:       ${DATASET_TASK}"
echo "RTC:        enabled=${RTC_ENABLED} horizon=${RTC_EXECUTION_HORIZON} weight=${RTC_MAX_GUIDANCE_WEIGHT}"
echo "Duration:   ${DURATION}s @ ${FPS}fps"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Conda env:  ${CONDA_ENV}"
echo "=============================="

if [ "${MODE}" = "robot" ]; then
    if [ ! -f "${RTC_SCRIPT}" ]; then
        echo "ERROR: RTC script not found: ${RTC_SCRIPT}"
        exit 1
    fi

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
            exit 1
        fi
    fi

    python -c "import scservo_sdk" 2>/dev/null || {
        echo "ERROR: scservo_sdk missing. pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
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
        idx = int(target)
    except ValueError:
        idx = target
    cap = cv2.VideoCapture(idx)
    ok = cap.isOpened()
    shape = None
    if ok:
        ret, frame = cap.read()
        ok = ok and ret and frame is not None
        shape = frame.shape if ok else None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape if ok else ''}  ({target})")
    failed = failed or not ok
if failed:
    print("ERROR: camera check failed. Run: bash infer_pi05_rtc.sh cameras", file=sys.stderr)
    sys.exit(1)
PY

    echo ""
    echo ">>> Starting PI0.5 RTC in 3s (Ctrl+C to abort)..."
    sleep 3

    cd "${PROJECT_ROOT}"
    python "${RTC_SCRIPT}" \
      --policy.path="${CHECKPOINT}" \
      --policy.tokenizer_path="${TOKENIZER_PATH}" \
      --policy.device=cuda \
      --rtc.enabled="${RTC_ENABLED}" \
      --rtc.execution_horizon="${RTC_EXECUTION_HORIZON}" \
      --rtc.max_guidance_weight="${RTC_MAX_GUIDANCE_WEIGHT}" \
      --action_queue_size_to_get_new_actions="${ACTION_QUEUE_SIZE}" \
      --robot.type=so101_follower \
      --robot.port="${ROBOT_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.max_relative_target="${MAX_RELATIVE_TARGET}" \
      --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAM}, width: 640, height: 480, fps: ${FPS}}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: ${FPS}} }" \
      --task="${DATASET_TASK}" \
      --duration="${DURATION}" \
      --fps="${FPS}" \
      --device=cuda

elif [ "${MODE}" = "cameras" ]; then
    echo "Stable camera symlinks (/dev/v4l/by-id):"
    ls -la /dev/v4l/by-id/ 2>/dev/null || echo "  (none)"
    echo ""
    echo "Defaults: FRONT_CAM=${FRONT_CAM}"
    echo "          WRIST_CAM=${WRIST_CAM}"
    echo ""
    lerobot-find-cameras opencv || true

else
    echo "Usage: bash infer_pi05_rtc.sh [robot|cameras]"
    exit 1
fi

echo "=============================="
echo "RTC inference finished"
echo "=============================="
