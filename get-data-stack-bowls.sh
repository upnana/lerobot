#!/bin/bash
# =============================================================================
# NOTE: 若用双臂请改用: bash get-data-bimanual-stack-bowls.sh
# SO101 叠碗采集（单臂）：黄碗 → 棕碗 → 白碗（自下而上）
# Usage:
#   bash get-data-stack-bowls.sh              # 开始采集（默认 100 条）
#   bash get-data-stack-bowls.sh cameras
#   bash get-data-stack-bowls.sh preview
#   bash get-data-stack-bowls.sh viz
#
# 覆盖参数示例:
#   NUM_EPISODES=10 bash get-data-stack-bowls.sh
#   RESUME=1 bash get-data-stack-bowls.sh     # 注意大写 RESUME
#   EPISODE_TIME_S=90 bash get-data-stack-bowls.sh
#
# 成功标准:
#   自下而上稳定叠好：黄碗在底、棕碗居中、白碗在顶，再结束 episode
# =============================================================================
set -euo pipefail

MODE="${1:-record}"

PROJECT_ROOT="/home/rxn/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot}"

# ---------- 机器人 / 遥操作 ----------
# follower=ACM0, leader=ACM1（稳定 by-id）
FOLLOWER_PORT="${FOLLOWER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6083854-if00}"
LEADER_PORT="${LEADER_PORT:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5AE6084864-if00}"
ROBOT_PORT="${ROBOT_PORT:-${FOLLOWER_PORT}}"
ROBOT_ID="${ROBOT_ID:-so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-${LEADER_PORT}}"
TELEOP_ID="${TELEOP_ID:-so101_leader}"

# ---------- 相机（by-id 路径；重启后枚举顺序变化也不影响）----------
FRONT_CAM="${FRONT_CAM:-/dev/v4l/by-id/usb-UGREEN_Camera_2K_UGREEN_Camera_2K_SN0001-video-index0}"
WRIST_CAM="${WRIST_CAM:-/dev/v4l/by-id/usb-Sonix_Technology_Co.__Ltd._USB2.0_CAM1_USB2.0_CAM1-video-index0}"
CAM_WIDTH="${CAM_WIDTH:-640}"
CAM_HEIGHT="${CAM_HEIGHT:-480}"
FPS="${FPS:-30}"
CAM_WARMUP_S="${CAM_WARMUP_S:-5}"
WRIST_WARMUP_S="${WRIST_WARMUP_S:-10}"

# ---------- 数据集 ----------
DATA_REPO="${DATA_REPO:-my_pick_place/stack_bowls_yellow_brown_white}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/stack_bowls_yellow_brown_white}"
SINGLE_TASK="${SINGLE_TASK:-Stack the yellow bowl, then the brown bowl, and finally the white bowl.}"
NUM_EPISODES="${NUM_EPISODES:-100}"
EPISODE_TIME_S="${EPISODE_TIME_S:-75}"
RESET_TIME_S="${RESET_TIME_S:-10}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
RESUME="${RESUME:-0}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    echo "  cd ${PROJECT_ROOT} && pip install -e \".[feetech]\""
    exit 1
fi
conda activate "${CONDA_ENV}"

echo "=============================="
echo "SO101 stack bowls yellow→brown→white (${MODE})"
echo "=============================="
echo "Task:       ${SINGLE_TASK}"
echo "Output:     ${DATA_ROOT}"
echo "Episodes:   ${NUM_EPISODES}  (${EPISODE_TIME_S}s/ep, reset ${RESET_TIME_S}s)"
echo "Scene tip:  shuffle bowl start poses; keep stack area fixed"
echo "Order:      yellow (bottom) → brown → white (top)"
echo "Success:    three bowls stably stacked in that order"
echo "Cameras:    front=${FRONT_CAM}  wrist=${WRIST_CAM}"
echo "Robot:      ${ROBOT_PORT} (${ROBOT_ID})"
echo "Teleop:     ${TELEOP_PORT} (${TELEOP_ID})"
echo "=============================="

if [ "${MODE}" = "cameras" ]; then
    echo "Stable symlinks (/dev/v4l/by-id):"
    ls -la /dev/v4l/by-id/ 2>/dev/null || echo "  (none)"
    echo ""
    lerobot-find-cameras opencv || true
    exit 0
fi

if [ "${MODE}" = "preview" ]; then
    echo "Checking cameras..."
    python - <<PY
import sys, cv2, time

def check(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened()
    shape = None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        for _ in range(30):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                shape = frame.shape
                break
            time.sleep(0.05)
        ok = shape is not None
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''}  ({src})")
    return ok

cams = [("front", "${FRONT_CAM}"), ("wrist", "${WRIST_CAM}")]
if not all(check(n, s) for n, s in cams):
    sys.exit(1)
PY

    echo ""
    echo "Rerun GUI：看 front / wrist，Ctrl+C 退出（颜色已 BGR→RGB 校正）"
    echo ">>> 3 秒后开始..."
    sleep 3

    python - <<PY
import time
import cv2
import rerun as rr
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

CAMS = [("front", "${FRONT_CAM}"), ("wrist", "${WRIST_CAM}")]

def open_cam(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"FAIL: {name} ({src})")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, ${CAM_WIDTH})
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ${CAM_HEIGHT})
    cap.set(cv2.CAP_PROP_FPS, ${FPS})
    for _ in range(45):
        ok, frame = cap.read()
        if ok and frame is not None and frame.mean() > 5:
            print(f"OK: {name} {frame.shape} ({src})")
            return cap
        time.sleep(0.05)
    cap.release()
    raise RuntimeError(f"FAIL: {name} no frame ({src})")

init_rerun(session_name="front_wrist_preview")
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
    if [ ! -d "${DATA_ROOT}" ]; then
        echo "ERROR: dataset not found: ${DATA_ROOT}"
        exit 1
    fi
    lerobot-dataset-viz \
      --repo-id="${DATA_REPO}" \
      --root="${DATA_ROOT}" \
      --episode-index=0 \
      --mode=local
    exit 0
fi

if [ "${MODE}" != "record" ]; then
    echo "Usage: bash get-data-stack-bowls.sh [record|cameras|preview|viz]"
    exit 1
fi

python -c "import scservo_sdk" 2>/dev/null || {
    echo "ERROR: scservo_sdk missing. pip install 'feetech-servo-sdk>=1.0.0,<2.0.0'"
    exit 1
}

echo "Checking cameras..."
python - <<PY
import sys, cv2, time

def check(name, src):
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    ok = cap.isOpened()
    shape = None
    if ok:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, ${CAM_WIDTH})
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ${CAM_HEIGHT})
        cap.set(cv2.CAP_PROP_FPS, ${FPS})
        for _ in range(45):
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 5:
                ok = True
                shape = frame.shape
                break
            time.sleep(0.05)
        else:
            ok = False
    cap.release()
    print(f"  {name}: {'OK' if ok else 'FAIL'} {shape or ''}  ({src})")
    return ok

failed = False
failed = not check("front", "${FRONT_CAM}") or failed
failed = not check("wrist", "${WRIST_CAM}") or failed
if failed:
    print("ERROR: camera check failed. Run: bash get-data-stack-bowls.sh cameras", file=sys.stderr)
    sys.exit(1)
PY

RECORD_EPISODES="${NUM_EPISODES}"
RESUME_FLAG=""
if [ "${RESUME}" = "1" ]; then
    if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
        echo "ERROR: RESUME=1 but dataset not found: ${DATA_ROOT}"
        exit 1
    fi
    EXISTING_EPISODES="$(python3 -c "import json; print(json.load(open('${DATA_ROOT}/meta/info.json'))['total_episodes'])")"
    RECORD_EPISODES=$(( NUM_EPISODES - EXISTING_EPISODES ))
    if [ "${RECORD_EPISODES}" -le 0 ]; then
        echo "Already have ${EXISTING_EPISODES} episodes (target ${NUM_EPISODES}). Nothing to record."
        exit 0
    fi
    RESUME_FLAG="--resume=true"
    echo ""
    echo "Resume: ${EXISTING_EPISODES}/${NUM_EPISODES} done → recording ${RECORD_EPISODES} more"
fi

echo ""
echo "快捷键: → 结束当前条 | ← 重录 | Esc 停止"
echo ">>> 3 秒后开始采集..."
sleep 3

lerobot-record \
  ${RESUME_FLAG} \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: \"${FRONT_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${CAM_WARMUP_S}}, wrist: {type: opencv, index_or_path: \"${WRIST_CAM}\", width: ${CAM_WIDTH}, height: ${CAM_HEIGHT}, fps: ${FPS}, fourcc: MJPG, warmup_s: ${WRIST_WARMUP_S}} }" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT}" \
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
  --dataset.num_image_writer_threads_per_camera=1

echo "=============================="
echo "Recording finished"
echo "Dataset: ${DATA_ROOT}"
echo "Viz:     bash get-data-stack-bowls.sh viz"
echo "Resume:  RESUME=1 NUM_EPISODES=100 bash get-data-stack-bowls.sh"
echo "=============================="
