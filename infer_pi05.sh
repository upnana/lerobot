#!/bin/bash
# =============================================================================
# PI0.5 推理脚本（SO101 + 本地/HF checkpoint）
# Usage:
#   conda activate lerobot          # 需要 pip install -e ".[pi,feetech]"
#   bash infer_pi05.sh offline      # 数据集离线推理
#   bash infer_pi05.sh robot        # 真机推理（Rerun）
#   bash infer_pi05.sh viz          # 可视化训练集
#   EPISODE_TIME_S=90 NUM_EPISODES=1 bash infer_pi05.sh robot
#
# 指定 checkpoint（本地目录或 HF repo id）:
#   CHECKPOINT=/path/to/pretrained_model bash infer_pi05.sh offline
#   CHECKPOINT=lerobot/pi05_base bash infer_pi05.sh offline
# =============================================================================
set -euo pipefail

MODE="${1:-offline}"

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/blue_block_yellow_tray}"
DATA_REPO="${DATA_REPO:-my_pick_place/blue_block_yellow_tray}"
# CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/train/pi05_blue_block_yellow_tray_010000_pretrained_model}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/train/pi05_blue_block_yellow_tray_020000_pretrained_model}"
# config.json 里 tokenizer_path 可能是训练机 /workspace/... 路径，推理时覆盖为本机
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/rxn/models/paligemma-3b-pt-224}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
TOLERANCE_S="${TOLERANCE_S:-0.05}"

# follower=ACM0, leader=ACM1（稳定 by-id 路径，重插 USB 不变）
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
ROBOT_PORT="${ROBOT_PORT:-${FOLLOWER_PORT}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
FRONT_CAM="${FRONT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"

FPS=30
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-120}"
RESET_TIME_S="${RESET_TIME_S:-15}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-10}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-16000}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/pi05_blue_block_yellow_tray_020000}"

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

# --- PI0.5 依赖检查 ---
if ! python - <<'PY'
try:
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: F401
except ImportError as e:
    raise SystemExit(f"PI0.5 import failed: {e}")
print("PI0.5 import OK")
PY
then
    echo "ERROR: PI0.5 dependencies missing in env '${CONDA_ENV}'."
    echo "  cd ${PROJECT_ROOT} && pip install -e \".[pi,feetech]\""
    echo "  Note: do NOT use lerobot-groot env (transformers version conflict with pi)."
    exit 1
fi

# checkpoint: 本地目录 or HF hub id (user/repo)
if [[ "${CHECKPOINT}" = /* ]] || [[ "${CHECKPOINT}" = ./* ]]; then
    if [ ! -f "${CHECKPOINT}/config.json" ]; then
        echo "ERROR: local checkpoint not found: ${CHECKPOINT}"
        echo "Set CHECKPOINT to your pretrained_model dir or HF repo, e.g.:"
        echo "  CHECKPOINT=${CHECKPOINT} bash infer_pi05.sh offline"
        echo "  CHECKPOINT=lerobot/pi05_base bash infer_pi05.sh offline"
        ls -la "$(dirname "${CHECKPOINT}")" 2>/dev/null || true
        exit 1
    fi
fi

# policy_preprocessor.json 里 tokenizer_name 可能是训练机 /workspace/... 路径
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
ds = LeRobotDataset(repo_id="${DATA_REPO}", root="${DATA_ROOT}", tolerance_s=${TOLERANCE_S})
print(ds.meta.tasks.index[0])
PY
)

echo "=============================="
echo "PI0.5 Inference (${MODE})"
echo "Checkpoint: ${CHECKPOINT}"
echo "Dataset:    ${DATA_ROOT}"
echo "Eval out:   ${EVAL_ROOT}"
echo "Dataset task: ${DATASET_TASK}"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Conda env: ${CONDA_ENV}"
echo "=============================="

if [ "${MODE}" = "offline" ]; then
    python - <<PY
import torch
import numpy as np
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
cfg.tokenizer_path = "${TOKENIZER_PATH}"

ds = LeRobotDataset(repo_id=repo_id, root=data_root, tolerance_s=${TOLERANCE_S})
policy = make_policy(cfg=cfg, ds_meta=ds.meta)
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg,
    pretrained_path=checkpoint,
    dataset_stats=ds.meta.stats,
    preprocessor_overrides={
        "tokenizer_processor": {"tokenizer_name": "${TOKENIZER_PATH}"},
    },
)
policy.eval()

print(f"Policy type: {cfg.type}")
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

if len(preds) >= 2:
    preds = np.array(preds)
    std = preds.std(axis=0)
    avg_mae = float(np.mean(maes))
    print(f"\n[Diagnostic] pred std per joint: {std.round(2)}")
    print(f"[Diagnostic] avg MAE vs gt:      {avg_mae:.3f}")
    if std.max() < 2.0:
        print("[Diagnostic] pred barely changes across frames")
    elif avg_mae < 3.0:
        print("[Diagnostic] reasonable on training data")
    else:
        print("[Diagnostic] high MAE — check checkpoint / fine-tune quality")

print("Offline inference OK.")
PY

elif [ "${MODE}" = "robot" ]; then
    echo "Robot port: ${ROBOT_PORT}"
    echo "Front cam: ${FRONT_CAM}"
    echo "Wrist cam: ${WRIST_CAM}"
    echo "Duration: ${NUM_EPISODES} ep × ${EPISODE_TIME_S}s (+ ${RESET_TIME_S}s reset) @ ${FPS}fps"
    echo "Task: ${DATASET_TASK}"
    echo ""

    if [ ! -e "${ROBOT_PORT}" ]; then
        echo "ERROR: Robot port not found: ${ROBOT_PORT}"
        ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
        exit 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
        echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB (need ~${MIN_FREE_MIB} MiB for PI0.5)"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo "ERROR: Not enough GPU memory on GPU ${PHYS_GPU}. Kill other processes: nvidia-smi"
            exit 1
        fi
    fi

    python -c "import scservo_sdk" 2>/dev/null || {
        echo "ERROR: scservo_sdk missing. pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
        exit 1
    }

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
    print("ERROR: camera check failed. Run: bash infer_pi05.sh cameras", file=sys.stderr)
    sys.exit(1)
PY

    echo ""
    echo ">>> Starting PI0.5 robot inference in 3s (Ctrl+C to abort)..."
    sleep 3

    EVAL_DATA_ROOT="${EVAL_ROOT}/$(date +%Y%m%d_%H%M%S)"
    echo "Eval save: ${EVAL_DATA_ROOT}"
    lerobot-record \
      --robot.type=so101_follower \
      --robot.port="${ROBOT_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.max_relative_target="${MAX_RELATIVE_TARGET}" \
      --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAM}, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: 30} }" \
      --display_data=true \
      --dataset.repo_id="upna/eval_pi05_blue_block_yellow_tray" \
      --dataset.root="${EVAL_DATA_ROOT}" \
      --dataset.single_task="${DATASET_TASK}" \
      --dataset.fps=${FPS} \
      --dataset.num_episodes=${NUM_EPISODES} \
      --dataset.episode_time_s=${EPISODE_TIME_S} \
      --dataset.reset_time_s=${RESET_TIME_S} \
      --dataset.push_to_hub=false \
      --play_sounds=${PLAY_SOUNDS} \
      --policy.path="${CHECKPOINT}" \
      --policy.tokenizer_path="${TOKENIZER_PATH}" \
      --policy.device=cuda \
      --policy.dtype=bfloat16 \
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
        print('  PI0.5 POLICY 推理中 (${EPISODE_TIME_S}s)')
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

elif [ "${MODE}" = "cameras" ]; then
    echo "Stable camera symlinks (/dev/v4l/by-id):"
    ls -la /dev/v4l/by-id/ 2>/dev/null || echo "  (none)"
    echo ""
    echo "Defaults: FRONT_CAM=${FRONT_CAM}"
    echo "          WRIST_CAM=${WRIST_CAM}"
    echo ""
    lerobot-find-cameras opencv || true

elif [ "${MODE}" = "viz" ]; then
    if [ -d "${EVAL_ROOT}" ]; then
        LATEST_EVAL=$(ls -td "${EVAL_ROOT}"/*/ 2>/dev/null | head -1)
        LATEST_EVAL="${LATEST_EVAL%/}"
        echo "Opening eval recording (episode 0): ${LATEST_EVAL}"
        lerobot-dataset-viz \
          --repo-id="upna/eval_pi05_blue_block_yellow_tray" \
          --root="${LATEST_EVAL}" \
          --episode-index=0 \
          --mode=local
    else
        echo "Opening training dataset (episode 0)..."
        lerobot-dataset-viz \
          --repo-id="${DATA_REPO}" \
          --root="${DATA_ROOT}" \
          --episode-index=0 \
          --mode=local
    fi

else
    echo "Usage: bash infer_pi05.sh [offline|robot|viz|cameras]"
    exit 1
fi

echo "=============================="
echo "Inference finished"
echo "=============================="
