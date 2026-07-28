#!/bin/bash
# =============================================================================
# SmolVLA 推理脚本（基于 train_smolvla.sh 的 checkpoint）
# Usage:
#   conda activate lerobot
#   bash infer_smolvla.sh robot              # 真机推理（默认 task=blue_block）
#   bash infer_smolvla.sh offline            # 数据集离线推理
#   bash infer_smolvla.sh tasks              # 列出可用 task
#   bash infer_smolvla.sh preview            # 预览 front/wrist 相机
#   bash infer_smolvla.sh cameras            # 扫描相机
#   bash infer_smolvla.sh viz                 # 可视化训练集
#
# 任务切换:
#   TASK=white_plate bash infer_smolvla.sh robot
#   TASK=blue_block  bash infer_smolvla.sh robot
#   TASK=yellow_tray bash infer_smolvla.sh robot   # 未专门训练，效果可能差
#
# 其他覆盖:
#   CHECKPOINT=.../checkpoints/1100000/pretrained_model bash infer_smolvla.sh robot
#   CUDA_VISIBLE_DEVICES=1 EPISODE_TIME_S=90 NUM_EPISODES=3 bash infer_smolvla.sh robot
# =============================================================================
set -euo pipefail

MODE="${1:-robot}"

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="/home/rxn/datasets/smolvla_multitask_merged"
DATA_REPO="my_multitask/pickup_items_and_blocks"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/smolvla_multitask}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# follower=ACM0, leader=ACM1
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
ROBOT_PORT="${ROBOT_PORT:-${FOLLOWER_PORT}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
FRONT_CAM="${FRONT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"

# front/wrist -> camera1/camera2（lerobot-record 用 --dataset.rename_map）
RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# 预置 task（TASK 环境变量: white_plate | blue_block | yellow_tray | 或完整字符串）
TASK="${TASK:-blue_block}"
case "${TASK}" in
  white_plate)
    SINGLE_TASK="Put all items on the table into the white plate"
    EVAL_SUFFIX="white_plate"
    ;;
  blue_block)
    SINGLE_TASK="pick up the blue block and put it into the square tray"
    EVAL_SUFFIX="blue_block"
    ;;
  yellow_tray)
    SINGLE_TASK="Pick up items on the table and put them into the yellow tray"
    EVAL_SUFFIX="yellow_tray"
    ;;
  *)
    SINGLE_TASK="${TASK}"
    EVAL_SUFFIX="custom"
    ;;
esac

FPS=30
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-90}"
RESET_TIME_S="${RESET_TIME_S:-15}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/smolvla_${EVAL_SUFFIX}}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 1
fi
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE=0
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

if [ ! -d "${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint not found: ${CHECKPOINT}"
    ls -la "${OUTPUT_DIR}/checkpoints/" 2>/dev/null || true
    exit 1
fi

CKPT_DIR="$(dirname "${CHECKPOINT}")"
CHECKPOINT_STEP="$(basename "${CKPT_DIR}")"
if [ "${CHECKPOINT_STEP}" = "last" ] && [ -L "${CKPT_DIR}" ]; then
    CHECKPOINT_STEP="$(basename "$(readlink -f "${CKPT_DIR}")")"
fi

echo "=============================="
echo "SmolVLA Inference (${MODE})"
echo "Checkpoint: ${CHECKPOINT} (step ${CHECKPOINT_STEP})"
echo "Task:       ${SINGLE_TASK}"
echo "GPU:        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Duration:   ${NUM_EPISODES} ep × ${EPISODE_TIME_S}s (+ ${RESET_TIME_S}s reset)"
echo "=============================="

if [ "${MODE}" = "tasks" ]; then
    echo "Available TASK presets:"
    echo "  white_plate  → Put all items on the table into the white plate"
    echo "  blue_block   → pick up the blue block and put it into the square tray"
    echo "  yellow_tray  → Pick up items on the table and put them into the yellow tray (not in multitask train)"
    echo ""
    echo "Example: TASK=blue_block EPISODE_TIME_S=90 bash infer_smolvla.sh robot"
    exit 0
fi

if [ "${MODE}" = "cameras" ]; then
    ls -la /dev/v4l/by-id/ 2>/dev/null || true
    echo ""
    echo "Defaults: FRONT_CAM=${FRONT_CAM}  WRIST_CAM=${WRIST_CAM}"
    echo ""
    lerobot-find-cameras opencv || true
    exit 0
fi

if [ "${MODE}" = "preview" ]; then
    echo "Rerun GUI：看 front / wrist，Ctrl+C 退出"
    echo ">>> 3 秒后开始..."
    sleep 3
    python - <<PY
import time
import cv2
import rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

CAMS = [("front", "${FRONT_CAM}"), ("wrist", "${WRIST_CAM}")]

def open_cam(name, src):
    try:
        src = int(src)
    except ValueError:
        pass
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"FAIL: {name} ({src})")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, ${FPS})
    for _ in range(30):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 5:
            print(f"OK: {name} {frame.shape} ({src})")
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError(f"FAIL: {name} no frame ({src})")

init_rerun(session_name="smolvla_cam_preview")
caps = {name: open_cam(name, src) for name, src in CAMS}
print("Rerun 已开。看 observation.front / observation.wrist")
try:
    while True:
        t0 = time.perf_counter()
        obs = {}
        for name, cap in caps.items():
            ok, frame = cap.read()
            if ok and frame is not None:
                obs[name] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        log_rerun_data(observation=obs, action={})
        time.sleep(max(0, 1 / ${FPS} - (time.perf_counter() - t0)))
except KeyboardInterrupt:
    pass
finally:
    for cap in caps.values():
        cap.release()
    rr.rerun_shutdown()
PY
    exit 0
fi

if [ "${MODE}" = "viz" ]; then
    lerobot-dataset-viz \
      --repo-id="${DATA_REPO}" \
      --root="${DATA_ROOT}" \
      --episode-index=0 \
      --mode=local
    exit 0
fi

if [ "${MODE}" = "offline" ]; then
    python - <<PY
import json
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

ds = LeRobotDataset(repo_id=repo_id, root=data_root)
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

print("Running offline inference on training dataset...")
for idx in [0, 1000, 5000, 10000]:
    if idx >= len(ds):
        continue
    policy.reset()
    sample = ds[idx]
    batch = preprocessor({k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()})
    with torch.inference_mode():
        action = postprocessor(policy.select_action(batch))
    mae = (action.cpu().float() - sample["action"].float()).abs().mean().item()
    print(f"  frame {idx:6d}  mae={mae:.4f}")
print("Offline inference OK.")
PY

elif [ "${MODE}" = "robot" ]; then
    if [ ! -e "${ROBOT_PORT}" ]; then
        echo "ERROR: Robot port not found: ${ROBOT_PORT}"
        ls /dev/ttyACM* 2>/dev/null || true
        exit 1
    fi

    if command -v nvidia-smi >/dev/null 2>&1; then
        PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
        echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB (SmolVLA needs ~${MIN_FREE_MIB} MiB)"
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

    echo "Checking cameras..."
    python - <<PY
import sys, cv2

def check(name, target):
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
    return ok

failed = not check("front", "${FRONT_CAM}") or not check("wrist", "${WRIST_CAM}")
if failed:
    print("ERROR: camera check failed. Run: bash infer_smolvla.sh cameras", file=sys.stderr)
    sys.exit(1)
PY

    echo ""
    echo ">>> Starting SmolVLA robot inference in 3s (Ctrl+C to abort)..."
    sleep 3

    lerobot-record \
      --robot.type=so101_follower \
      --robot.port="${ROBOT_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAM}, width: 640, height: 480, fps: ${FPS}}, wrist: {type: opencv, index_or_path: ${WRIST_CAM}, width: 640, height: 480, fps: ${FPS}} }" \
      --display_data=true \
      --play_sounds="${PLAY_SOUNDS}" \
      --dataset.repo_id="upna/eval_smolvla_${EVAL_SUFFIX}" \
      --dataset.root="${EVAL_ROOT}" \
      --dataset.single_task="${SINGLE_TASK}" \
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
        print('  SmolVLA 推理中 (${EPISODE_TIME_S}s)')
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
    echo "Usage: bash infer_smolvla.sh [robot|offline|tasks|preview|cameras|viz]"
    exit 1
fi

echo "=============================="
echo "Inference finished"
echo "=============================="
