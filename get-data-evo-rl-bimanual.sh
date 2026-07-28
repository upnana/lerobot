#!/bin/bash
# =============================================================================
# Evo-RL 官方风格双臂采集（human-in-the-loop）
#
# 基于 MINT-SJTU/Evo-RL README Data Collection：
#   lerobot-human-inloop-record + bi_so_follower / bi_so_leader
#   左/右臂相机挂在 left_arm_config / right_arm_config 下
#   用 s/f 标记 success/failure
#
# Usage:
#   bash get-data-evo-rl-bimanual.sh check     # 端口 / 相机 / 校准软链
#   bash get-data-evo-rl-bimanual.sh teleop    # 先遥操作确认（无录像）
#   bash get-data-evo-rl-bimanual.sh           # 开始 HITL 采集
#   bash get-data-evo-rl-bimanual.sh record
#   bash get-data-evo-rl-bimanual.sh report    # 采集后质量检查
#
# 默认任务（long-horizon，结果导向，不强制零食/叠碗先后）:
#   1) 两包零食都放进黄 tray
#   2) 碗按黄(底)→棕→白(顶) 叠到白盘上
# 初始建议：三碗分开散落；黄 tray / 白盘位置可小幅随机；其他杂物尽量拿开
# Success(s): 零食都在黄 tray 且碗叠稳顺序正确；否则按 f
#
# 常用覆盖:
#   NUM_EPISODES=50 bash get-data-evo-rl-bimanual.sh
#   RESUME=1 bash get-data-evo-rl-bimanual.sh
#   DISPLAY_DATA=false bash get-data-evo-rl-bimanual.sh
#   EPISODE_TIME_S=180 bash get-data-evo-rl-bimanual.sh
#
# Note: after value-infer wrote ACP columns into this dataset, RESUME still works
# (extra complementary_info.* fields are filled with zeros on new frames).
# Re-run value-infer after the new demos if you need fresh ACP labels.
#
# 热键（官方）:
#   i          切换 intervention（policy <-> teleop；纯演示阶段也可忽略）
#   s          标记成功并结束当前 episode
#   f          标记失败并结束当前 episode
#   Right Arrow 提前结束当前 loop
#   Left Arrow  结束并重录当前 episode
#   Esc         停止整次采集
# =============================================================================
set -euo pipefail

MODE="${1:-record}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"

# ---------- 4 臂（by-id）----------
LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065873-if00}"
LEFT_LEADER_PORT="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
RIGHT_LEADER_PORT="${RIGHT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065804-if00}"
# 与现有校准文件名对齐：bimanual_follower_{left,right}.json
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"
TELEOP_ID="${TELEOP_ID:-bimanual_leader}"

# ---------- 相机（官方：挂在左右臂 config 下）----------
# 默认 3 相机 = 双腕 + 1 顶视（更贴 SmolVLA；可拔掉 UGREEN）
#   TOP_CAMS=1 (default): left=wrist+top_left；right=仅 wrist
#   TOP_CAMS=2: 旧 4-cam（再加 right top_right）
TOP_CAMS="${TOP_CAMS:-1}"
TOP_LEFT_CAM="${TOP_LEFT_CAM:-/dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0}"
TOP_RIGHT_CAM="${TOP_RIGHT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
LEFT_WRIST_CAM="${LEFT_WRIST_CAM:-/dev/v4l/by-id/usb-am_camera_wrist_left_am_camera_wrist_left-video-index0}"
RIGHT_WRIST_CAM="${RIGHT_WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
TOP_LEFT_CAM_WIDTH="${TOP_LEFT_CAM_WIDTH:-800}"
TOP_LEFT_CAM_HEIGHT="${TOP_LEFT_CAM_HEIGHT:-480}"
FPS="${FPS:-30}"
# 4-cam USB: longer warmup reduces Sonix/wrist timeouts during record
CAM_WARMUP_S="${CAM_WARMUP_S:-12}"

LEFT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${LEFT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_left: {type: opencv, index_or_path: \"${TOP_LEFT_CAM}\", width: ${TOP_LEFT_CAM_WIDTH}, height: ${TOP_LEFT_CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
if [ "${TOP_CAMS}" = "2" ]; then
  RIGHT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${RIGHT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}}, top_right: {type: opencv, index_or_path: \"${TOP_RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
else
  RIGHT_ARM_CAMERAS="{ wrist: {type: opencv, index_or_path: \"${RIGHT_WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\", warmup_s: ${CAM_WARMUP_S}} }"
fi

# ---------- 数据集 ----------
# Long-horizon（结果导向）:
#   Put both snack packages into the yellow tray,
#   and stack yellow→brown→white bowls onto the white plate.
# Success(s): 两包零食在黄 tray + 白盘上碗叠稳且黄底棕中白顶
# Failure(f): 缺物/掉落/顺序错/未进 tray 或未上盘
SINGLE_TASK="${SINGLE_TASK:-Put both snack packages into the yellow tray, and stack the yellow, brown, and white bowls from bottom to top onto the white plate.}"
NUM_EPISODES="${NUM_EPISODES:-50}"
EPISODE_TIME_S="${EPISODE_TIME_S:-210}"
RESET_TIME_S="${RESET_TIME_S:-8}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
RESUME="${RESUME:-0}"
# 官方推荐：强制 s/f 标注，方便后续 value / ACP
REQUIRE_EPISODE_SUCCESS_LABEL="${REQUIRE_EPISODE_SUCCESS_LABEL:-true}"

# 3-cam 默认新路径（勿 RESUME 进旧 4-cam 集）
if [ "${TOP_CAMS}" = "2" ]; then
  DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate}"
  DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_snacks_tray_stack_bowls_plate}"
else
  DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate_3cam}"
  DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_snacks_tray_stack_bowls_plate_3cam}"
fi

CALIB_ROOT="${HOME}/.cache/huggingface/lerobot/calibration"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found. Activate after install:"
    echo "  conda activate evo-rl"
    exit 1
fi
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

echo "=============================="
echo "Evo-RL HITL bimanual (${MODE})"
echo "=============================="
echo "Env:        ${CONDA_ENV}"
echo "Project:    ${PROJECT_ROOT}"
echo "Task:       ${SINGLE_TASK}"
echo "Goals:      snacks→yellow tray  AND  bowls yellow→brown→white on white plate"
echo "Success:    both snacks in tray + bowls stacked correctly on plate"
echo "Dataset:    ${DATA_ROOT}  (${DATA_REPO})"
echo "Cameras:    TOP_CAMS=${TOP_CAMS}  (1=3cam dual-wrist+top_left; 2=4cam)"
echo "Episodes:   ${NUM_EPISODES}  (${EPISODE_TIME_S}s + reset ${RESET_TIME_S}s)"
echo "Robot IDs:  ${ROBOT_ID} / ${TELEOP_ID}"
echo "=============================="

ensure_calibration_links() {
    mkdir -p "${CALIB_ROOT}/robots/so_follower" "${CALIB_ROOT}/teleoperators/so_leader"
    local pairs=(
        "robots/so101_follower/bimanual_follower_left.json:robots/so_follower/bimanual_follower_left.json"
        "robots/so101_follower/bimanual_follower_right.json:robots/so_follower/bimanual_follower_right.json"
        "teleoperators/so101_leader/bimanual_leader_left.json:teleoperators/so_leader/bimanual_leader_left.json"
        "teleoperators/so101_leader/bimanual_leader_right.json:teleoperators/so_leader/bimanual_leader_right.json"
    )
    local failed=0
    for pair in "${pairs[@]}"; do
        local src="${CALIB_ROOT}/${pair%%:*}"
        local dst="${CALIB_ROOT}/${pair##*:}"
        if [ ! -f "${src}" ]; then
            echo "  FAIL missing source calib: ${src}"
            failed=1
            continue
        fi
        ln -sfn "${src}" "${dst}"
        echo "  OK  ${dst} -> ${src}"
    done
    return "${failed}"
}

check_ports() {
    echo "[ports]"
    local failed=0
    for p in \
        "${LEFT_FOLLOWER_PORT}" "${RIGHT_FOLLOWER_PORT}" \
        "${LEFT_LEADER_PORT}" "${RIGHT_LEADER_PORT}"; do
        if [ -e "${p}" ]; then
            echo "  OK  ${p}"
        else
            echo "  FAIL ${p}"
            failed=1
        fi
    done
    return "${failed}"
}

check_cameras() {
    echo "[cameras] TOP_CAMS=${TOP_CAMS}"
    python - <<PY
import sys, cv2, time
cams = [
    ("left_wrist", "${LEFT_WRIST_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("right_wrist", "${RIGHT_WRIST_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("top_left", "${TOP_LEFT_CAM}", ${TOP_LEFT_CAM_WIDTH}, ${TOP_LEFT_CAM_HEIGHT}),
]
if "${TOP_CAMS}" == "2":
    cams.append(("top_right", "${TOP_RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}))
failed = False
for name, src, w, h in cams:
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened()
    shape = None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                shape = frame.shape
                break
            time.sleep(0.05)
        ok = shape is not None
    cap.release()
    print(f"  {'OK' if ok else 'FAIL'}  {name} {shape or ''} ({src})")
    failed = failed or (not ok)
sys.exit(1 if failed else 0)
PY
}

warn_if_actor() {
    if pgrep -f 'lerobot.rl.actor' >/dev/null 2>&1; then
        echo "WARNING: HIL-SERL actor 还在跑，会占串口。请先:"
        echo "  pkill -f 'lerobot.rl.actor' || true"
        return 1
    fi
    return 0
}

run_teleop() {
    lerobot-teleoperate \
      --robot.type=bi_so_follower \
      --robot.left_arm_config.port="${LEFT_FOLLOWER_PORT}" \
      --robot.right_arm_config.port="${RIGHT_FOLLOWER_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.left_arm_config.cameras="${LEFT_ARM_CAMERAS}" \
      --robot.right_arm_config.cameras="${RIGHT_ARM_CAMERAS}" \
      --teleop.type=bi_so_leader \
      --teleop.left_arm_config.port="${LEFT_LEADER_PORT}" \
      --teleop.right_arm_config.port="${RIGHT_LEADER_PORT}" \
      --teleop.id="${TELEOP_ID}" \
      --display_data=true
}

run_record() {
    local resume_flag=""
    if [ "${RESUME}" = "1" ]; then
        if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
            echo "ERROR: RESUME=1 but dataset not found: ${DATA_ROOT}"
            exit 1
        fi
        resume_flag="--resume=true"
        echo "Resume into existing dataset: ${DATA_ROOT}"
    fi

    echo "Keys: i intervene | s success | f failure | → end | ← rerecord | Esc stop"
    echo ">>> 3 秒后开始 HITL 采集..."
    sleep 3

    lerobot-human-inloop-record \
      ${resume_flag} \
      --robot.type=bi_so_follower \
      --robot.left_arm_config.port="${LEFT_FOLLOWER_PORT}" \
      --robot.right_arm_config.port="${RIGHT_FOLLOWER_PORT}" \
      --robot.id="${ROBOT_ID}" \
      --robot.left_arm_config.cameras="${LEFT_ARM_CAMERAS}" \
      --robot.right_arm_config.cameras="${RIGHT_ARM_CAMERAS}" \
      --teleop.type=bi_so_leader \
      --teleop.left_arm_config.port="${LEFT_LEADER_PORT}" \
      --teleop.right_arm_config.port="${RIGHT_LEADER_PORT}" \
      --teleop.id="${TELEOP_ID}" \
      --dataset.repo_id="${DATA_REPO}" \
      --dataset.root="${DATA_ROOT}" \
      --dataset.single_task="${SINGLE_TASK}" \
      --dataset.fps="${FPS}" \
      --dataset.num_episodes="${NUM_EPISODES}" \
      --dataset.episode_time_s="${EPISODE_TIME_S}" \
      --dataset.reset_time_s="${RESET_TIME_S}" \
      --dataset.push_to_hub="${PUSH_TO_HUB}" \
      --display_data="${DISPLAY_DATA}" \
      --play_sounds="${PLAY_SOUNDS}" \
      --enable_episode_outcome_labeling=true \
      --require_episode_success_label="${REQUIRE_EPISODE_SUCCESS_LABEL}" \
      --episode_success_key=s \
      --episode_failure_key=f \
      --intervention_toggle_key=i
}

case "${MODE}" in
    check)
        warn_if_actor || true
        echo "[calibration links for Evo-RL so_follower / so_leader]"
        ensure_calibration_links
        check_ports
        check_cameras
        echo "check done."
        ;;
    teleop)
        warn_if_actor
        ensure_calibration_links
        check_ports
        check_cameras
        run_teleop
        ;;
    record)
        warn_if_actor
        ensure_calibration_links
        check_ports
        check_cameras
        run_record
        ;;
    report)
        if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
            echo "ERROR: dataset not found: ${DATA_ROOT}"
            exit 1
        fi
        # 官方质量检查：优先用本地 root；CLI 吃 repo_id 时也可
        if command -v lerobot-dataset-report >/dev/null 2>&1; then
            lerobot-dataset-report --dataset "${DATA_ROOT}" 2>/dev/null \
              || lerobot-dataset-report --dataset "${DATA_REPO}"
        else
            echo "lerobot-dataset-report not found; showing meta only:"
            python - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATA_ROOT}/meta/info.json").read_text())
print("total_episodes:", info.get("total_episodes"))
print("total_frames:", info.get("total_frames"))
print("fps:", info.get("fps"))
print("features:", list(info.get("features", {}).keys()))
PY
        fi
        ;;
    *)
        echo "Usage: bash get-data-evo-rl-bimanual.sh [check|teleop|record|report]"
        exit 1
        ;;
esac
