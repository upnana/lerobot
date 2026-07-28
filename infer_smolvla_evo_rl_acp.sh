#!/bin/bash
# =============================================================================
# SmolVLA ACP 推理评测（Evo-RL snacks + stack bowls）
#
# Usage:
#   bash infer_smolvla_evo_rl_acp.sh cameras
#   bash infer_smolvla_evo_rl_acp.sh offline
#   bash infer_smolvla_evo_rl_acp.sh robot      # 默认
#   bash infer_smolvla_evo_rl_acp.sh help
#
# 快速对比预设 PRESET=...
#   default          ACP=positive, n_action_steps=5, max_rel=50  （默认）
#   acp_none         不加 Advantage 标签
#   n10              n_action_steps=10（更跟手/更敢动）
#   n10_acp_none     n10 + 无 ACP 标签
#   safe             n_action_steps=3, max_rel=30（更保守）
#   aggressive       n_action_steps=10, max_rel=80
#
# 示例:
#   PRESET=acp_none bash infer_smolvla_evo_rl_acp.sh robot
#   PRESET=n10 bash infer_smolvla_evo_rl_acp.sh robot
#   PRESET=n10_acp_none NUM_EPISODES=2 bash infer_smolvla_evo_rl_acp.sh robot
#   ACP_TAG=negative N_ACTION_STEPS=10 bash infer_smolvla_evo_rl_acp.sh robot
#   CHECKPOINT=.../checkpoints/100000/pretrained_model bash ... robot
# =============================================================================
set -euo pipefail

MODE="${1:-robot}"

if [ "${MODE}" = "help" ] || [ "${MODE}" = "-h" ] || [ "${MODE}" = "--help" ]; then
  sed -n '2,25p' "$0" | sed 's/^# \?//'
  exit 0
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate}"
DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_snacks_tray_stack_bowls_plate}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/smolvla_acp_snacks_bowls_v1}"
CHECKPOINT="${CHECKPOINT:-${OUTPUT_DIR}/checkpoints/last/pretrained_model}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

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
CAM_WARMUP_S="${CAM_WARMUP_S:-8}"
FPS="${FPS:-30}"

LEFT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${LEFT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_left: {type: opencv, index_or_path: \"${TOP_LEFT_CAM}\", width: ${TOP_LEFT_CAM_WIDTH}, height: ${TOP_LEFT_CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
RIGHT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${RIGHT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_right: {type: opencv, index_or_path: \"${TOP_RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"

# 与 train-policy-evo-rl-acp.sh 一致
RENAME_MAP='{"observation.images.left_top_left":"observation.images.camera1","observation.images.right_top_right":"observation.images.camera2","observation.images.left_wrist":"observation.images.camera3","observation.images.right_wrist":"observation.images.camera4"}'
TOLERANCE_S="${TOLERANCE_S:-0.05}"

BASE_TASK="${BASE_TASK:-Put both snack packages into the yellow tray, and stack the yellow, brown, and white bowls from bottom to top onto the white plate.}"

# ---------- comparison knobs (defaults; PRESET may override) ----------
PRESET="${PRESET:-default}"
ACP_TAG="${ACP_TAG:-positive}"          # positive | negative | none
N_ACTION_STEPS="${N_ACTION_STEPS:-5}"   # train chunk=50; robot often 5~10
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-50}"
NUM_EPISODES="${NUM_EPISODES:-1}"
EPISODE_TIME_S="${EPISODE_TIME_S:-210}"
RESET_TIME_S="${RESET_TIME_S:-20}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
SKIP_CAMERA_CHECK="${SKIP_CAMERA_CHECK:-0}"
SKIP_SLEEP="${SKIP_SLEEP:-0}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/smolvla_acp_snacks_bowls_v1}"

case "${PRESET}" in
  default) ;;
  acp_none)
    ACP_TAG=none
    ;;
  n10)
    N_ACTION_STEPS=10
    ;;
  n10_acp_none)
    N_ACTION_STEPS=10
    ACP_TAG=none
    ;;
  safe)
    N_ACTION_STEPS=3
    MAX_RELATIVE_TARGET=30
    ;;
  aggressive)
    N_ACTION_STEPS=10
    MAX_RELATIVE_TARGET=80
    ;;
  *)
    echo "ERROR: unknown PRESET='${PRESET}'"
    echo "Valid: default | acp_none | n10 | n10_acp_none | safe | aggressive"
    exit 1
    ;;
esac

# Allow explicit env overrides AFTER preset (if user set them on CLI they already
# took effect above before case — so re-apply only when PRESET_* not used.
# Convention: values already in env win if exported before PRESET case for
# ACP_TAG/N_ACTION_STEPS when user passes both. To keep it simple: PRESET sets
# defaults; user can still override by exporting AFTER selecting preset via:
#   PRESET=n10 ACP_TAG=none  — but ACP_TAG is set before case so preset overwrites.
# Fix: apply preset only when corresponding OVERRIDE flags not set.
# Simpler approach used here: document that PRESET wins for its fields; for
# custom mix use explicit vars without PRESET (or PRESET=default).

# NOTE: must not pass raw "Advantage: ..." through CLI unquoted — draccus YAML-parses args.
if [ "${ACP_TAG}" = "none" ] || [ -z "${ACP_TAG}" ]; then
  SINGLE_TASK="${BASE_TASK}"
else
  SINGLE_TASK="${BASE_TASK}"$'\n'"Advantage: ${ACP_TAG}"
fi
SINGLE_TASK_CLI="$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${SINGLE_TASK}")"

# Tag eval folder for A/B comparison
RUN_TAG="${RUN_TAG:-preset-${PRESET}_acp-${ACP_TAG}_n${N_ACTION_STEPS}_rel${MAX_RELATIVE_TARGET}}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

if [ ! -f "${CHECKPOINT}/config.json" ]; then
  echo "ERROR: checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

CKPT_DIR="$(dirname "${CHECKPOINT}")"
CHECKPOINT_STEP="$(basename "${CKPT_DIR}")"
if [ "${CHECKPOINT_STEP}" = "last" ] && [ -L "${CKPT_DIR}" ]; then
  CHECKPOINT_STEP="$(basename "$(readlink -f "${CKPT_DIR}")")"
fi

echo "=============================="
echo "SmolVLA ACP infer (${MODE})"
echo "Checkpoint: ${CHECKPOINT} (step ${CHECKPOINT_STEP})"
echo "PRESET:     ${PRESET}"
echo "RUN_TAG:    ${RUN_TAG}"
echo "Task:       ${SINGLE_TASK//$'\n'/ | }"
echo "ACP tag:    ${ACP_TAG}"
echo "n_action:   ${N_ACTION_STEPS}   max_rel=${MAX_RELATIVE_TARGET}"
echo "GPU:        ${CUDA_VISIBLE_DEVICES}"
echo "Episodes:   ${NUM_EPISODES} x ${EPISODE_TIME_S}s"
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
  echo "Usage: bash infer_smolvla_evo_rl_acp.sh [robot|offline|cameras|help]"
  exit 1
fi

for p in "${LEFT_FOLLOWER_PORT}" "${RIGHT_FOLLOWER_PORT}"; do
  [ -e "${p}" ] || { echo "ERROR: missing port ${p}"; ls /dev/serial/by-id/ || true; exit 1; }
done

if command -v nvidia-smi >/dev/null 2>&1; then
  PHYS_GPU="${CUDA_VISIBLE_DEVICES%%,*}"
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${PHYS_GPU}" | tr -d ' ')
  echo "GPU ${PHYS_GPU} free: ${FREE_MIB} MiB"
  if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
    echo "ERROR: GPU memory too low"; nvidia-smi; exit 1
  fi
fi

if [ "${SKIP_CAMERA_CHECK}" != "1" ]; then
  echo "Checking cameras..."
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

failed = False
for n,s in [
    ("left_wrist","${LEFT_WRIST_CAM}"),
    ("right_wrist","${RIGHT_WRIST_CAM}"),
    ("top_left","${TOP_LEFT_CAM}"),
    ("top_right","${TOP_RIGHT_CAM}"),
]:
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
  --dataset.repo_id="eval/eval_smolvla_acp_snacks_bowls_v1" \
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
echo "Compare next with e.g. PRESET=acp_none or PRESET=n10"
echo "=============================="
