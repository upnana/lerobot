#!/bin/bash
# =============================================================================
# HITL rollout：policy + 双臂 leader 遥操作（按 i 可干预）
# bowls_tray_3cam 策略
#
# 与纯 infer 的区别：
#   - 接 bi_so_leader，policy 动作会镜像到 leader
#   - 按 i：策略 ↔ 人手接管（teleop）
#   - 按 s / f：标记成功/失败并结束本集（可存成下一轮数据）
#
# Usage:
#   bash /home/rxn/lerobot/hitl_rollout_bowls_tray_3cam.sh              # 默认 80000
#   bash /home/rxn/lerobot/hitl_rollout_bowls_tray_3cam.sh 50000
#   bash /home/rxn/lerobot/hitl_rollout_bowls_tray_3cam.sh 80000
#   bash /home/rxn/lerobot/hitl_rollout_bowls_tray_3cam.sh last
#   bash /home/rxn/lerobot/hitl_rollout_bowls_tray_3cam.sh cameras
#
# 常用:
#   PRESET=n10_acp_none bash ... 80000          # 推荐：无 ACP
#   PRESET=n10 bash ... 80000                   # Advantage: positive
#   USE_4CAMS=0 bash ... 80000                  # Sonix 不稳时只用 3 相机
#   NUM_EPISODES=5 EPISODE_TIME_S=300 bash ... 80000
#   SAVE_TO_TRAIN=1 RESUME=1 bash ... 80000     # 直接续写训练集（慎用）
#
# 热键:
#   i           切换 intervention（policy <-> teleop）
#   s / f       成功 / 失败并结束本集
#   → / ←       提前结束 / 重录
#   Esc         停止整次会话
#   Ctrl+C      强制退出（不是干预）
# =============================================================================
set -euo pipefail

ARG1="${1:-}"
MODE="robot"
STEP="${STEP:-80000}"
if [ "${ARG1}" = "cameras" ] || [ "${ARG1}" = "help" ] || [ "${ARG1}" = "-h" ]; then
  MODE="${ARG1}"
elif [ -n "${ARG1}" ]; then
  STEP="${ARG1}"
fi

if [ "${MODE}" = "help" ] || [ "${MODE}" = "-h" ]; then
  sed -n '2,40p' "$0" | sed 's/^# \?//'
  exit 0
fi

PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"
TRAIN_ROOT="${TRAIN_ROOT:-${PROJECT_ROOT}/outputs/train/smolvla_acp_bowls_tray_3cam}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/outputs/eval/hitl_bowls_tray_3cam}"

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
  echo "Available:"; ls "${TRAIN_ROOT}/checkpoints" 2>/dev/null || true
  exit 1
fi

# Ports
LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065873-if00}"
LEFT_LEADER_PORT="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
RIGHT_LEADER_PORT="${RIGHT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065804-if00}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"
TELEOP_ID="${TELEOP_ID:-bimanual_leader}"

# Cameras
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
RIGHT_WRIST_FPS="${RIGHT_WRIST_FPS:-15}"
USE_4CAMS="${USE_4CAMS:-1}"

if [ "${USE_4CAMS}" = "1" ]; then
  LEFT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${LEFT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_left: {type: opencv, index_or_path: \"${TOP_LEFT_CAM}\", width: ${TOP_LEFT_CAM_WIDTH}, height: ${TOP_LEFT_CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
else
  LEFT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${LEFT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
fi
RIGHT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${RIGHT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${RIGHT_WRIST_FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_right: {type: opencv, index_or_path: \"${TOP_RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"

# Policy camera map (trained 3-cam; left_top_left ignored by model)
RENAME_MAP='{"observation.images.right_top_right":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'

BASE_TASK="${BASE_TASK:-First stack the yellow, brown, and white bowls from bottom to top onto the white plate, then put the lipstick and the blue tissue pack into the yellow tray.}"
PRESET="${PRESET:-n10_acp_none}"
ACP_TAG="${ACP_TAG:-none}"
N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-50}"
NUM_EPISODES="${NUM_EPISODES:-3}"
EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
RESET_TIME_S="${RESET_TIME_S:-20}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
MIN_FREE_MIB="${MIN_FREE_MIB:-3500}"
SKIP_CAMERA_CHECK="${SKIP_CAMERA_CHECK:-0}"
SKIP_SLEEP="${SKIP_SLEEP:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
REQUIRE_EPISODE_SUCCESS_LABEL="${REQUIRE_EPISODE_SUCCESS_LABEL:-true}"

case "${PRESET}" in
  default) ACP_TAG=positive; N_ACTION_STEPS=5 ;;
  acp_none) ACP_TAG=none ;;
  n10) N_ACTION_STEPS=10; ACP_TAG=positive ;;
  n10_acp_none) N_ACTION_STEPS=10; ACP_TAG=none ;;
  safe) N_ACTION_STEPS=3; MAX_RELATIVE_TARGET=30; ACP_TAG=none ;;
  aggressive) N_ACTION_STEPS=10; MAX_RELATIVE_TARGET=80; ACP_TAG=none ;;
  *)
    echo "ERROR: unknown PRESET='${PRESET}'"
    exit 1
    ;;
esac

if [ "${ACP_TAG}" = "none" ] || [ -z "${ACP_TAG}" ]; then
  SINGLE_TASK="${BASE_TASK}"
else
  SINGLE_TASK="${BASE_TASK}"$'\n'"Advantage: ${ACP_TAG}"
fi
SINGLE_TASK_CLI="$(python -c 'import json,sys; print(json.dumps(sys.argv[1]))' "${SINGLE_TASK}")"

# Where to save HITL rollouts
CKPT_TAG="$(basename "$(dirname "${CHECKPOINT}")")"
if [ "${CKPT_TAG}" = "last" ] && [ -L "$(dirname "${CHECKPOINT}")" ]; then
  CKPT_TAG="$(basename "$(readlink -f "$(dirname "${CHECKPOINT}")")")"
fi
RUN_TAG="${RUN_TAG:-hitl_step-${CKPT_TAG}_preset-${PRESET}_acp-${ACP_TAG}}"

if [ "${SAVE_TO_TRAIN:-0}" = "1" ]; then
  # Note: with policy loaded, lerobot requires dataset name to start with eval_.
  # Write a sibling eval_ dataset under /home/rxn/datasets (merge later if needed).
  DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/eval_hitl_bowls_tray_3cam_${CKPT_TAG}}"
  DATA_REPO="${DATA_REPO:-my_bimanual/eval_hitl_bowls_tray_3cam_${CKPT_TAG}}"
  if [ -z "${RESUME+x}" ]; then
    if [ -f "${DATA_ROOT}/meta/info.json" ]; then RESUME=1; else RESUME=0; fi
  fi
else
  DATA_ROOT="${DATA_ROOT:-${EVAL_ROOT}/${RUN_TAG}_$(date +%Y%m%d_%H%M%S)}"
  # lerobot sanity: with policy loaded, dataset name (after /) must start with eval_
  DATA_REPO="${DATA_REPO:-eval/eval_hitl_bowls_tray_3cam_${CKPT_TAG}}"
  RESUME="${RESUME:-0}"
fi

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES

# Calibration softlinks (same as collect)
CALIB_ROOT="${HOME}/.cache/huggingface/lerobot/calibration"
mkdir -p "${CALIB_ROOT}/robots/so_follower" "${CALIB_ROOT}/teleoperators/so_leader"
link_if_needed() {
  local src="$1" dst="$2"
  [ -e "${src}" ] || return 0
  [ -e "${dst}" ] && return 0
  ln -sfn "${src}" "${dst}"
}
link_if_needed "${HOME}/.cache/huggingface/lerobot/calibration/robots/so101_follower/bimanual_follower_left.json" \
  "${CALIB_ROOT}/robots/so_follower/bimanual_follower_left.json"
link_if_needed "${HOME}/.cache/huggingface/lerobot/calibration/robots/so101_follower/bimanual_follower_right.json" \
  "${CALIB_ROOT}/robots/so_follower/bimanual_follower_right.json"
link_if_needed "${HOME}/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/bimanual_leader_left.json" \
  "${CALIB_ROOT}/teleoperators/so_leader/bimanual_leader_left.json"
link_if_needed "${HOME}/.cache/huggingface/lerobot/calibration/teleoperators/so101_leader/bimanual_leader_right.json" \
  "${CALIB_ROOT}/teleoperators/so_leader/bimanual_leader_right.json"

echo "=============================="
echo "HITL rollout bowls_tray_3cam"
echo "Checkpoint: ${CHECKPOINT} (${CKPT_TAG})"
echo "PRESET:     ${PRESET}  ACP=${ACP_TAG}  n_action=${N_ACTION_STEPS}"
echo "Task:       ${SINGLE_TASK//$'\n'/ | }"
echo "Cams:       USE_4CAMS=${USE_4CAMS}  right_wrist_fps=${RIGHT_WRIST_FPS}"
echo "Save:       ${DATA_ROOT}  (RESUME=${RESUME})"
echo "Episodes:   ${NUM_EPISODES} x ${EPISODE_TIME_S}s"
echo "Keys:       i intervene | s success | f failure | Esc stop"
echo "=============================="

if [ "${MODE}" = "cameras" ]; then
  echo "Serial:"; ls -la /dev/serial/by-id/ 2>/dev/null || true
  echo "Cams:"; ls -la /dev/v4l/by-id/ 2>/dev/null || true
  exit 0
fi

for p in "${LEFT_FOLLOWER_PORT}" "${RIGHT_FOLLOWER_PORT}" "${LEFT_LEADER_PORT}" "${RIGHT_LEADER_PORT}"; do
  [ -e "${p}" ] || { echo "ERROR: missing port ${p}"; ls /dev/serial/by-id/ || true; exit 1; }
done

if pgrep -f 'lerobot.rl.actor|lerobot-record|lerobot-human-inloop|lerobot-teleoperate' >/dev/null 2>&1; then
  echo "WARN: another robot session may be running; free ports first if connect fails."
fi

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
fi

mkdir -p "$(dirname "${DATA_ROOT}")"
RESUME_FLAG=()
if [ "${RESUME}" = "1" ]; then
  RESUME_FLAG=(--resume=true)
  [ -f "${DATA_ROOT}/meta/info.json" ] || { echo "ERROR: RESUME=1 but no dataset at ${DATA_ROOT}"; exit 1; }
fi

if [ "${SKIP_SLEEP}" != "1" ]; then
  echo ">>> 3s 后开始 HITL（先把 leader 摆到舒服位置；开局是 policy 控臂）..."
  sleep 3
fi

lerobot-human-inloop-record \
  "${RESUME_FLAG[@]}" \
  --robot.type=bi_so_follower \
  --robot.left_arm_config.port="${LEFT_FOLLOWER_PORT}" \
  --robot.right_arm_config.port="${RIGHT_FOLLOWER_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.left_arm_config.max_relative_target="${MAX_RELATIVE_TARGET}" \
  --robot.right_arm_config.max_relative_target="${MAX_RELATIVE_TARGET}" \
  --robot.left_arm_config.cameras="${LEFT_ARM_CAMERAS}" \
  --robot.right_arm_config.cameras="${RIGHT_ARM_CAMERAS}" \
  --teleop.type=bi_so_leader \
  --teleop.left_arm_config.port="${LEFT_LEADER_PORT}" \
  --teleop.right_arm_config.port="${RIGHT_LEADER_PORT}" \
  --teleop.id="${TELEOP_ID}" \
  --dataset.repo_id="${DATA_REPO}" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.single_task="${SINGLE_TASK_CLI}" \
  --dataset.fps="${FPS}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.episode_time_s="${EPISODE_TIME_S}" \
  --dataset.reset_time_s="${RESET_TIME_S}" \
  --dataset.push_to_hub=false \
  --dataset.rename_map="${RENAME_MAP}" \
  --display_data="${DISPLAY_DATA}" \
  --play_sounds="${PLAY_SOUNDS}" \
  --enable_episode_outcome_labeling=true \
  --require_episode_success_label="${REQUIRE_EPISODE_SUCCESS_LABEL}" \
  --episode_success_key=s \
  --episode_failure_key=f \
  --intervention_toggle_key=i \
  --policy.path="${CHECKPOINT}" \
  --policy.device=cuda \
  --policy.n_action_steps="${N_ACTION_STEPS}"

echo "=============================="
echo "HITL finished. Saved: ${DATA_ROOT}"
echo "纯 policy 无干预: bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh ${CKPT_TAG}"
echo "=============================="
