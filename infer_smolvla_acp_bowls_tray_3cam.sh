#!/bin/bash
# =============================================================================
# SmolVLA ACP 真机推理：bowls_tray_3cam（去掉 left_top_left）
#
# 训练映射（必须一致）:
#   right_top_right → camera1
#   left_wrist      → camera2
#   right_wrist     → camera3
#
# 真机默认开 4 路（与采集一致；策略仍只用下面 3 路映射，left_top_left 不进模型）:
#   左臂: wrist + top_left（全视野）
#   右臂: wrist + top_right
#
# Usage:
#   bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh              # 默认 80000 + 无 ACP
#   bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh 50000
#   bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh 80000
#   bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh 120000
#   bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh last
#   bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh cameras
#   bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh offline 80000
#
# 常用:
#   PRESET=n10_acp_none bash ... 50000          # 默认：无 ACP，更敢动（推荐）
#   PRESET=n10 bash ... 80000                   # 带 Advantage: positive
#   EPISODE_TIME_S=360 NUM_EPISODES=2 bash ... 80000
#   CUDA_VISIBLE_DEVICES=0 MIN_FREE_MIB=2000 bash ... 80000   # 训练占卡时放宽显存门槛
#
# 按 v2 经验优先试 50k–80k，不要默认 last。
#
# 录完默认叠 value 曲线到视频（OVERLAY_VALUE=1）。
# 一条命令入口: bash /home/rxn/lerobot/infer_overlay_bowls_tray_3cam.sh 80000
# =============================================================================
set -euo pipefail

ARG1="${1:-}"
ARG2="${2:-}"

# Parse: cameras | offline [step] | [step] | last
MODE="robot"
STEP="${STEP:-80000}"
if [ "${ARG1}" = "cameras" ] || [ "${ARG1}" = "offline" ] || [ "${ARG1}" = "robot" ] || [ "${ARG1}" = "help" ]; then
  MODE="${ARG1}"
  STEP="${ARG2:-${STEP}}"
elif [ -n "${ARG1}" ]; then
  STEP="${ARG1}"
fi

if [ "${MODE}" = "help" ] || [ "${MODE}" = "-h" ] || [ "${MODE}" = "--help" ]; then
  sed -n '2,35p' "$0" | sed 's/^# \?//'
  exit 0
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/outputs/train/smolvla_acp_bowls_tray_3cam_v2}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_v2}"
DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_bowls_stack_lipstick_tissue_v2}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/smolvla_acp_bowls_tray_3cam_v2}"
# After record: run value-infer + burn value curve onto videos (1=yes)
OVERLAY_VALUE="${OVERLAY_VALUE:-1}"

# Resolve checkpoint
if [ "${STEP}" = "last" ]; then
  CKPT_DIR="${TRAIN_ROOT}/checkpoints/last"
else
  if [[ "${STEP}" =~ ^[0-9]+[kK]$ ]]; then
    STEP=$((${STEP%[kK]} * 1000))
  fi
  STEP_PAD="$(printf '%06d' "${STEP}")"
  CKPT_DIR="${TRAIN_ROOT}/checkpoints/${STEP_PAD}"
fi
CHECKPOINT="${CHECKPOINT:-${CKPT_DIR}/pretrained_model}"
if [ ! -f "${CHECKPOINT}/config.json" ]; then
  echo "ERROR: checkpoint missing: ${CHECKPOINT}"
  echo "Available:"
  ls "${TRAIN_ROOT}/checkpoints" 2>/dev/null || true
  exit 1
fi

# Hardware
LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065873-if00}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"

TOP_LEFT_CAM="${TOP_LEFT_CAM:-/dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0}"
TOP_RIGHT_CAM="${TOP_RIGHT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
LEFT_WRIST_CAM="${LEFT_WRIST_CAM:-/dev/v4l/by-id/usb-am_camera_wrist_left_am_camera_wrist_left-video-index0}"
RIGHT_WRIST_CAM="${RIGHT_WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
TOP_LEFT_CAM_WIDTH="${TOP_LEFT_CAM_WIDTH:-800}"
TOP_LEFT_CAM_HEIGHT="${TOP_LEFT_CAM_HEIGHT:-480}"
CAM_WARMUP_S="${CAM_WARMUP_S:-12}"
FPS="${FPS:-30}"
# Sonix 右腕在 4-cam USB 总线上最容易挂；单独降 fps 减带宽
RIGHT_WRIST_FPS="${RIGHT_WRIST_FPS:-15}"
# USE_4CAMS=1（默认）真机开 4 路；=0 则只开训练用的 3 路
USE_4CAMS="${USE_4CAMS:-1}"

if [ "${USE_4CAMS}" = "1" ]; then
  LEFT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${LEFT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_left: {type: opencv, index_or_path: \"${TOP_LEFT_CAM}\", width: ${TOP_LEFT_CAM_WIDTH}, height: ${TOP_LEFT_CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
else
  LEFT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${LEFT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
fi
RIGHT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${RIGHT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${RIGHT_WRIST_FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_right: {type: opencv, index_or_path: \"${TOP_RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"

RENAME_MAP='{"observation.images.right_top_right":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'
TOLERANCE_S="${TOLERANCE_S:-0.05}"

BASE_TASK="${BASE_TASK:-First stack the yellow, brown, and white bowls from bottom to top onto the white plate, then put the lipstick and the blue tissue pack into the yellow tray.}"

PRESET="${PRESET:-n10_acp_none}"
ACP_TAG="${ACP_TAG:-none}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-50}"
NUM_EPISODES="${NUM_EPISODES:-1}"
# 长程任务：真机尝试往往比 demo(~35s) 慢很多，默认给足时间
EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
RESET_TIME_S="${RESET_TIME_S:-20}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-3500}"
SKIP_CAMERA_CHECK="${SKIP_CAMERA_CHECK:-0}"
SKIP_SLEEP="${SKIP_SLEEP:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

case "${PRESET}" in
  default)
    ACP_TAG=positive
    N_ACTION_STEPS=5
    ;;
  acp_none)
    ACP_TAG=none
    ;;
  n10)
    N_ACTION_STEPS=10
    ACP_TAG=positive
    ;;
  n10_acp_none)
    N_ACTION_STEPS=10
    ACP_TAG=none
    ;;
  safe)
    N_ACTION_STEPS=3
    MAX_RELATIVE_TARGET=30
    ACP_TAG=none
    ;;
  aggressive)
    N_ACTION_STEPS=10
    MAX_RELATIVE_TARGET=80
    ACP_TAG=none
    ;;
  *)
    echo "ERROR: unknown PRESET='${PRESET}'"
    echo "Valid: default | acp_none | n10 | n10_acp_none | safe | aggressive"
    exit 1
    ;;
esac

if [ "${ACP_TAG}" = "none" ] || [ -z "${ACP_TAG}" ]; then
  SINGLE_TASK="${BASE_TASK}"
else
  SINGLE_TASK="${BASE_TASK}"$'\n'"Advantage: ${ACP_TAG}"
fi
SINGLE_TASK_CLI="$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${SINGLE_TASK}")"

RUN_TAG="${RUN_TAG:-step-$(basename "$(dirname "${CHECKPOINT}")")_preset-${PRESET}_acp-${ACP_TAG}_n${N_ACTION_STEPS}}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

CKPT_DIR_NAME="$(basename "$(dirname "${CHECKPOINT}")")"
if [ "${CKPT_DIR_NAME}" = "last" ] && [ -L "$(dirname "${CHECKPOINT}")" ]; then
  CKPT_DIR_NAME="$(basename "$(readlink -f "$(dirname "${CHECKPOINT}")")")"
fi

echo "=============================="
echo "bowls_tray_3cam infer (${MODE})"
echo "Checkpoint: ${CHECKPOINT} (${CKPT_DIR_NAME})"
if [ "${USE_4CAMS}" = "1" ]; then
  echo "Cams:       4-cam robot (left_wrist+left_top_left+right_wrist+right_top_right)"
  echo "            policy still uses 3-cam rename (left_top_left ignored by model)"
else
  echo "Cams:       3-cam robot (left_wrist+right_wrist+right_top_right)"
fi
echo "Rename:     ${RENAME_MAP}"
echo "PRESET:     ${PRESET}  ACP=${ACP_TAG}  n_action=${N_ACTION_STEPS}  max_rel=${MAX_RELATIVE_TARGET}"
echo "Task:       ${SINGLE_TASK//$'\n'/ | }"
echo "Episodes:   ${NUM_EPISODES} x ${EPISODE_TIME_S}s"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "=============================="

if [ "${MODE}" = "cameras" ]; then
  echo "Serial:"; ls -la /dev/serial/by-id/ 2>/dev/null || true
  echo "Cams:"; ls -la /dev/v4l/by-id/ 2>/dev/null || true
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
rename_map = json.loads('''${RENAME_MAP}''')
ds = LeRobotDataset(repo_id="${DATA_REPO}", root="${DATA_ROOT}", tolerance_s=${TOLERANCE_S})
cfg = PreTrainedConfig.from_pretrained(checkpoint)
cfg.pretrained_path = checkpoint
cfg.device = "cuda"
cfg.n_action_steps = int("${N_ACTION_STEPS}")
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
task = """${SINGLE_TASK}"""
print(f"Offline smoke | ACP_TAG={json.dumps('${ACP_TAG}')} n_action_steps={cfg.n_action_steps}")
for idx in [0, 1000, 5000, 20000, len(ds)//2]:
    if idx >= len(ds):
        continue
    policy.reset()
    sample = ds[idx]
    raw = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else [v] for k, v in sample.items()}
    raw["task"] = [task]
    batch = preprocessor(raw)
    with torch.inference_mode():
        action = postprocessor(policy.select_action(batch))
    gt = sample["action"].float()
    pred = action.cpu().float().view(-1)
    mae = (pred - gt).abs().mean().item()
    print(f"  frame {idx:6d}  action={tuple(pred.shape)}  mae={mae:.4f}")
print("Offline inference OK.")
PY
  exit 0
fi

if [ "${MODE}" != "robot" ]; then
  echo "Usage: bash $0 [robot|offline|cameras|help] [step]"
  echo "   or: bash $0 [step|last]"
  exit 1
fi

# Warn if policy train still running
if pgrep -af 'smolvla_acp_bowls_tray_3cam' | grep -q lerobot-train; then
  echo "WARN: policy train still running — GPU may be tight. Prefer MIN_FREE_MIB=2000 or wait."
fi

for p in "${LEFT_FOLLOWER_PORT}" "${RIGHT_FOLLOWER_PORT}"; do
  [ -e "${p}" ] || { echo "ERROR: missing port ${p}"; ls /dev/serial/by-id/ || true; exit 1; }
done

if command -v nvidia-smi >/dev/null 2>&1; then
  PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
  echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB (need >= ${MIN_FREE_MIB})"
  if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
    echo "ERROR: GPU memory too low. Training may be occupying VRAM."
    echo "  Try: CUDA_VISIBLE_DEVICES=1 MIN_FREE_MIB=2000 bash $0 ${STEP}"
    nvidia-smi
    exit 1
  fi
fi

if [ "${SKIP_CAMERA_CHECK}" != "1" ]; then
  echo "Checking cameras (USE_4CAMS=${USE_4CAMS})..."
  python - <<PY
import sys, cv2, time

def check(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened(); shape=None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                shape = frame.shape; break
            time.sleep(0.05)
        ok = shape is not None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''} ({src})")
    return ok

cams = [
    ("left_wrist","${LEFT_WRIST_CAM}"),
    ("right_wrist","${RIGHT_WRIST_CAM}"),
    ("top_right","${TOP_RIGHT_CAM}"),
]
if "${USE_4CAMS}" == "1":
    cams.insert(1, ("top_left","${TOP_LEFT_CAM}"))
failed = False
for n,s in cams:
    failed = (not check(n,s)) or failed
sys.exit(1 if failed else 0)
PY
else
  echo "SKIP_CAMERA_CHECK=1"
fi

EVAL_DATA_ROOT="${EVAL_ROOT}/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${EVAL_ROOT}"
echo "Eval save: ${EVAL_DATA_ROOT}"
if [ "${SKIP_SLEEP}" != "1" ]; then
  echo ">>> 3s 后开始真机推理 (Ctrl+C 取消)..."
  sleep 3
fi

lerobot-record \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port="${LEFT_FOLLOWER_PORT}" \
  --robot.right_arm_config.port="${RIGHT_FOLLOWER_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.left_arm_config.max_relative_target="${MAX_RELATIVE_TARGET}" \
  --robot.right_arm_config.max_relative_target="${MAX_RELATIVE_TARGET}" \
  --robot.left_arm_config.cameras="${LEFT_ARM_CAMERAS}" \
  --robot.right_arm_config.cameras="${RIGHT_ARM_CAMERAS}" \
  --display_data=true \
  --play_sounds="${PLAY_SOUNDS}" \
  --dataset.repo_id="eval/eval_smolvla_acp_bowls_tray_3cam" \
  --dataset.root="${EVAL_DATA_ROOT}" \
  --dataset.single_task="${SINGLE_TASK_CLI}" \
  --dataset.fps="${FPS}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false \
  --dataset.rename_map="${RENAME_MAP}" \
  --policy.path="${CHECKPOINT}" \
  --policy.device=cuda \
  --policy.n_action_steps="${N_ACTION_STEPS}"

echo "=============================="
echo "Inference finished: ${EVAL_DATA_ROOT}"
echo "再试: bash $0 50000   或   PRESET=n10 bash $0 80000"
echo "=============================="

if [ "${OVERLAY_VALUE}" = "1" ]; then
  echo "Overlaying value curve onto recorded videos..."
  bash /home/rxn/lerobot/overlay_value_on_eval.sh "${EVAL_DATA_ROOT}"
  echo "Value-overlay videos: ${EVAL_DATA_ROOT}_value_viz/ (see */viz/*.mp4)"
fi
