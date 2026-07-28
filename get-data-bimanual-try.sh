#!/bin/bash
# =============================================================================
# SO101 双臂数据采集 · 试跑版（精简）
#
# Usage:
#   bash get-data-bimanual-try.sh check      # 查端口 / 相机 / 双臂舵机
#   bash get-data-bimanual-try.sh preview4  # 四相机 Rerun 预览
#   bash get-data-bimanual-try.sh           # 开始采集（默认 5 条试跑）
#   bash get-data-bimanual-try.sh record
#
# 常用覆盖:
#   NUM_EPISODES=20 bash get-data-bimanual-try.sh
#   RESUME=1 DATA_ROOT=/path/to/dataset bash get-data-bimanual-try.sh
#   DISPLAY_DATA=false bash get-data-bimanual-try.sh   # 卡顿时关预览
#
# 开跑前务必停掉 HIL-SERL actor（会占左臂串口）:
#   pkill -f 'lerobot.rl.actor' || true
# =============================================================================
set -euo pipefail

MODE="${1:-record}"
PROJECT_ROOT="/home/rxn/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot}"

# ---------- 4 臂（by-id）----------
LEFT_FOLLOWER_PORT="${LEFT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
RIGHT_FOLLOWER_PORT="${RIGHT_FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065873-if00}"
LEFT_LEADER_PORT="${LEFT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
RIGHT_LEADER_PORT="${RIGHT_LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AB9065804-if00}"
ROBOT_ID="${ROBOT_ID:-bimanual_follower}"
TELEOP_ID="${TELEOP_ID:-bimanual_leader}"

# ---------- 4 相机 ----------
TOP_LEFT_CAM="${TOP_LEFT_CAM:-/dev/v4l/by-id/usb-Generic_Web_Camera_20250708V1.000-video-index0}"
TOP_RIGHT_CAM="${TOP_RIGHT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
LEFT_CAM="${LEFT_CAM:-/dev/v4l/by-id/usb-am_camera_wrist_left_am_camera_wrist_left-video-index0}"
RIGHT_CAM="${RIGHT_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
TOP_LEFT_CAM_WIDTH="${TOP_LEFT_CAM_WIDTH:-800}"
TOP_LEFT_CAM_HEIGHT="${TOP_LEFT_CAM_HEIGHT:-480}"
FPS="${FPS:-30}"
CAM_WARMUP_S="${CAM_WARMUP_S:-8}"

ROBOT_CAMERAS="{ top_left: {type: opencv, index_or_path: \"${TOP_LEFT_CAM}\", width: ${TOP_LEFT_CAM_WIDTH}, height: ${TOP_LEFT_CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, top_right: {type: opencv, index_or_path: \"${TOP_RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, left: {type: opencv, index_or_path: \"${LEFT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, right: {type: opencv, index_or_path: \"${RIGHT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}} }"

# ---------- 数据集（试跑默认少量）----------
SINGLE_TASK="${SINGLE_TASK:-Grasp the double-sided tape roll, hand it over to the other arm, and place it in the yellow rectangular plastic tray.}"
NUM_EPISODES="${NUM_EPISODES:-5}"
EPISODE_TIME_S="${EPISODE_TIME_S:-90}"
RESET_TIME_S="${RESET_TIME_S:-8}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
RESUME="${RESUME:-0}"
IMG_WRITER_THREADS_PER_CAMERA="${IMG_WRITER_THREADS_PER_CAMERA:-1}"

if [ "${MODE}" = "record" ] && [ -z "${DATA_ROOT+x}" ]; then
    COLLECTION_DATE="$(date +%Y%m%d_%H%M%S)"
    DATA_ROOT="/home/rxn/datasets/bimanual_try_4cam_${COLLECTION_DATE}"
    DATA_REPO="my_bimanual/try_4cam_${COLLECTION_DATE}"
else
    COLLECTION_DATE="${COLLECTION_DATE:-}"
    DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/bimanual_try_4cam}"
    DATA_REPO="${DATA_REPO:-my_bimanual/try_4cam}"
fi

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

echo "=============================="
echo "Bimanual TRY (${MODE})"
echo "=============================="

# ---------- helpers ----------
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
    echo "[cameras]"
    python - <<PY
import sys, cv2, time
cams = [
    ("top_left", "${TOP_LEFT_CAM}", ${TOP_LEFT_CAM_WIDTH}, ${TOP_LEFT_CAM_HEIGHT}),
    ("top_right", "${TOP_RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("left", "${LEFT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("right", "${RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
]
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

check_follower_motors() {
    # $1=label $2=port
    local label="$1"
    local port="$2"
    echo "[follower ${label}] ${port}"
    python - <<PY
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode
import sys

PORT = "${port}"
names = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
motors = {n: Motor(i, "sts3215", MotorNormMode.DEGREES if i < 6 else MotorNormMode.RANGE_0_100)
          for i, n in names.items()}
bus = FeetechMotorsBus(port=PORT, motors=motors)
try:
    bus.connect(handshake=False)
except Exception as e:
    print(f"  CONNECT FAIL: {e}")
    sys.exit(1)

ok = 0
for i, n in names.items():
    try:
        pos = bus.read("Present_Position", n, normalize=False, num_retry=3)
        volt = bus.read("Present_Voltage", n, normalize=False, num_retry=2) / 10
        temp = bus.read("Present_Temperature", n, normalize=False, num_retry=2)
        print(f"  OK  id{i} {n:16} pos={pos:4} V={volt:4.1f} T={temp}")
        ok += 1
    except Exception as e:
        tag = "OVERLOAD" if "Overload" in str(e) else "FAIL"
        print(f"  {tag} id{i} {n:16} {e}")
try:
    bus.sync_read("Present_Position", normalize=False, num_retry=2)
    print("  SYNC OK")
except Exception as e:
    print(f"  SYNC FAIL: {e}")
    ok = 0
bus.disconnect(disable_torque=False)
sys.exit(0 if ok == 6 else 1)
PY
}

warn_if_actor() {
    if pgrep -f 'lerobot.rl.actor' >/dev/null 2>&1; then
        echo "WARNING: HIL-SERL actor 还在跑，会占左臂串口。请先:"
        echo "  pkill -f 'lerobot.rl.actor' || true"
        return 1
    fi
    return 0
}

# ---------- modes ----------
case "${MODE}" in
    check)
        warn_if_actor || true
        check_ports
        check_cameras
        check_follower_motors left "${LEFT_FOLLOWER_PORT}"
        check_follower_motors right "${RIGHT_FOLLOWER_PORT}"
        echo "=============================="
        echo "check done. 若左臂 id3/id4 OVERLOAD：断电 20s，手掰到中间位再测。"
        ;;
    preview4)
        check_cameras
        echo "Rerun 四合一预览，Ctrl+C 退出"
        sleep 2
        python - <<PY
import time, cv2, rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

CAMS = [
    ("top_left", "${TOP_LEFT_CAM}", ${TOP_LEFT_CAM_WIDTH}, ${TOP_LEFT_CAM_HEIGHT}),
    ("top_right", "${TOP_RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("left", "${LEFT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
    ("right", "${RIGHT_CAM}", ${CAM_WIDTH}, ${CAM_HEIGHT}),
]
W, H = ${CAM_WIDTH}, ${CAM_HEIGHT}

def open_cam(name, src, w, h):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"FAIL open {name}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, ${FPS})
    for _ in range(45):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 5:
            print(f"OK {name} {frame.shape}")
            return cap
        time.sleep(0.05)
    raise RuntimeError(f"FAIL frame {name}")

def label(frame, text):
    out = frame.copy()
    cv2.putText(out, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return out

init_rerun(session_name="bimanual_try_preview4")
caps = {n: open_cam(n, s, w, h) for n, s, w, h in CAMS}
try:
    while True:
        t0 = time.perf_counter()
        obs, labeled = {}, {}
        for n, cap in caps.items():
            ok, frame = cap.read()
            if ok and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if rgb.shape[1] != W or rgb.shape[0] != H:
                    rgb = cv2.resize(rgb, (W, H))
                obs[n] = rgb
                labeled[n] = label(rgb, n)
        if len(labeled) == 4:
            top = cv2.hconcat([labeled["top_left"], labeled["top_right"]])
            bot = cv2.hconcat([labeled["left"], labeled["right"]])
            obs["all"] = cv2.vconcat([top, bot])
        log_rerun_data(observation=obs, action={})
        time.sleep(max(0, 1 / ${FPS} - (time.perf_counter() - t0)))
except KeyboardInterrupt:
    pass
finally:
    for c in caps.values():
        c.release()
    rr.rerun_shutdown()
PY
        ;;
    record)
        warn_if_actor
        check_ports
        echo "Preflight motors..."
        check_follower_motors left "${LEFT_FOLLOWER_PORT}"
        check_follower_motors right "${RIGHT_FOLLOWER_PORT}"
        check_cameras

        RECORD_EPISODES="${NUM_EPISODES}"
        RESUME_FLAG=""
        if [ "${RESUME}" = "1" ]; then
            if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
                echo "ERROR: RESUME=1 but no dataset at ${DATA_ROOT}"
                exit 1
            fi
            EXISTING="$(python3 -c "import json; print(json.load(open('${DATA_ROOT}/meta/info.json'))['total_episodes'])")"
            RECORD_EPISODES=$((NUM_EPISODES - EXISTING))
            if [ "${RECORD_EPISODES}" -le 0 ]; then
                echo "Already have ${EXISTING}/${NUM_EPISODES}. Done."
                exit 0
            fi
            RESUME_FLAG="--resume=true"
            echo "Resume: ${EXISTING}/${NUM_EPISODES} → +${RECORD_EPISODES}"
        fi

        echo "Task:     ${SINGLE_TASK}"
        echo "Output:   ${DATA_ROOT}"
        echo "Episodes: ${RECORD_EPISODES} (${EPISODE_TIME_S}s + reset ${RESET_TIME_S}s)"
        echo "Keys: → end ep | ← rerecord | Esc stop"
        echo "校准提示出现时：已校过的臂按 Enter；新臂才输入 c"
        echo ">>> 3 秒后开始..."
        sleep 3

        lerobot-record \
          ${RESUME_FLAG} \
          --robot.type=bi_so101_follower \
          --robot.left_arm_port="${LEFT_FOLLOWER_PORT}" \
          --robot.right_arm_port="${RIGHT_FOLLOWER_PORT}" \
          --robot.id="${ROBOT_ID}" \
          --robot.cameras="${ROBOT_CAMERAS}" \
          --teleop.type=bi_so101_leader \
          --teleop.left_arm_port="${LEFT_LEADER_PORT}" \
          --teleop.right_arm_port="${RIGHT_LEADER_PORT}" \
          --teleop.id="${TELEOP_ID}" \
          --display_data="${DISPLAY_DATA}" \
          --play_sounds="${PLAY_SOUNDS}" \
          --dataset.repo_id="${DATA_REPO}" \
          --dataset.root="${DATA_ROOT}" \
          --dataset.single_task="${SINGLE_TASK}" \
          --dataset.fps="${FPS}" \
          --dataset.num_episodes="${RECORD_EPISODES}" \
          --dataset.episode_time_s="${EPISODE_TIME_S}" \
          --dataset.reset_time_s="${RESET_TIME_S}" \
          --dataset.push_to_hub="${PUSH_TO_HUB}" \
          --dataset.num_image_writer_threads_per_camera="${IMG_WRITER_THREADS_PER_CAMERA}"
        ;;
    *)
        echo "Usage: bash get-data-bimanual-try.sh [check|preview4|record]"
        exit 1
        ;;
esac
