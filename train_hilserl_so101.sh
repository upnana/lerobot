#!/bin/bash
# =============================================================================
# HIL-SERL Step 5: SO101 Actor-Learner RL 训练
#
# 前置条件:
#   - Step 2 demo 已采集
#   - Step 3 crop 已完成 (hilserl_push_black_block_demos_cropped_resized)
#   - Step 4 reward classifier 已训好
#   - 首次运行需联网下载 helper2424/resnet10（vision encoder）
#
# Usage:
#   bash train_hilserl_so101.sh learner     # 终端 1：先启动 learner（可安全重启，若仅有 logs）
#   bash train_hilserl_so101.sh actor        # 终端 2：再启动 actor（真机）
#   bash train_hilserl_so101.sh check        # 检查配置与数据集
#   bash train_hilserl_so101.sh download     # 仅下载 vision encoder（首次运行）
#   FRESH=1 bash train_hilserl_so101.sh learner   # 清空旧 checkpoint 重训
#   RESUME=1 bash train_hilserl_so101.sh learner  # 从 checkpoint 续训
#
# 旧 pick-place 任务:
#   CONFIG_PATH=configs/hilserl_so101_train.json \
#   DATA_ROOT=/home/rxn/datasets/hilserl_demos_cropped_resized \
#   OUTPUT_DIR=outputs/train/hilserl_so101_pick_place \
#   bash train_hilserl_so101.sh learner
#
# Leader 控制 (actor 终端需有焦点):
#   Space = 切换人工干预（开=leader 控 follower）
#   s = 成功 (reward=1)   q = 失败结束   r = 重录
# 仍可用键盘 EE: TELEOP=keyboard_ee CONTROL_MODE=keyboard 并改 train json
#
# 硬件: follower=5AE6083854  leader=5AE6084864
# =============================================================================
set -euo pipefail

MODE="${1:-help}"

PROJECT_ROOT="/home/rxn/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/hilserl_so101_push_black_block_train.json}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/hilserl_push_black_block_demos_cropped_resized}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/rxn/lerobot/outputs/train/hilserl_so101_push_black_block}"
FRESH="${FRESH:-0}"
RESUME="${RESUME:-0}"
# Default GPU 1 (same as infer_groot.sh); override: CUDA_VISIBLE_DEVICES=0
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES
# Dual GPU: physical GPU0 = SAC train, physical GPU1 = replay buffer
#   CUDA_VISIBLE_DEVICES=0,1 USE_DUAL_GPU=1 bash train_hilserl_so101.sh learner
USE_DUAL_GPU="${USE_DUAL_GPU:-0}"
LEARNER_DEVICE="${LEARNER_DEVICE:-cuda:0}"
STORAGE_DEVICE="${STORAGE_DEVICE:-cpu}"
ACTOR_DEVICE="${ACTOR_DEVICE:-cuda:0}"
if [ "${USE_DUAL_GPU}" = "1" ]; then
    LEARNER_DEVICE="cuda:0"
    STORAGE_DEVICE="cuda:1"
    ACTOR_DEVICE="cuda:0"
fi
POLICY_DEVICE_ARGS=(--policy.device="${LEARNER_DEVICE}" --policy.storage_device="${STORAGE_DEVICE}")
ACTOR_DEVICE_ARGS=(--policy.device="${ACTOR_DEVICE}")

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# First run downloads helper2424/resnet10; offline flags in shell profile block that.
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=0
# Override with HF_ENDPOINT= if you have direct huggingface.co access.
: "${HF_ENDPOINT:=https://hf-mirror.com}"
export HF_ENDPOINT

python -c "import gym_hil, grpc" 2>/dev/null || {
    echo "ERROR: hilserl deps missing. Run: pip install -e \".[hilserl]\""
    exit 1
}

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
    echo "ERROR: cropped dataset not found: ${DATA_ROOT}"
    echo "  Run: bash crop_hilserl_demos.sh"
    exit 1
fi

EXTRA_ARGS=()
CHECKPOINT_LINK="${OUTPUT_DIR}/checkpoints/last"

prepare_output_dir() {
    local mode="${1:-learner}"
    if [ "${mode}" = "actor" ]; then
        EXTRA_ARGS+=(--allow_existing_output_dir=true)
        return
    fi
    if [ "${RESUME}" = "1" ]; then
        EXTRA_ARGS+=(--resume=true)
        return
    fi
    if [ "${FRESH}" = "1" ] && [ -d "${OUTPUT_DIR}" ]; then
        echo "FRESH=1: removing ${OUTPUT_DIR}"
        rm -rf "${OUTPUT_DIR}"
        return
    fi
    if [ -d "${OUTPUT_DIR}" ]; then
        if [ -e "${CHECKPOINT_LINK}" ]; then
            echo "ERROR: checkpoint already exists at ${CHECKPOINT_LINK}"
            echo "  RESUME=1 bash train_hilserl_so101.sh learner   # continue training"
            echo "  FRESH=1 bash train_hilserl_so101.sh learner    # delete and restart"
            exit 1
        fi
        echo "Reusing output dir (logs only, no checkpoint yet): ${OUTPUT_DIR}"
        EXTRA_ARGS+=(--allow_existing_output_dir=true)
    fi
}

check_setup() {
    echo "=============================="
    echo "HIL-SERL SO101 Training Check"
    echo "=============================="
    echo "Config:  ${CONFIG_PATH}"
    echo "Dataset: ${DATA_ROOT}"
    echo "Output:  ${OUTPUT_DIR}"
    python3 <<PY
import draccus
from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.sac.configuration_sac import SACConfig  # noqa: F401
from lerobot.robots import so101_follower  # noqa: F401
from lerobot.teleoperators.keyboard.configuration_keyboard import (  # noqa: F401
    KeyboardEndEffectorTeleopConfig,
)

with draccus.config_type("json"):
    cfg = draccus.parse(TrainRLServerPipelineConfig, "${CONFIG_PATH}")
cfg.validate()
ds = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
print(f"Config OK | dataset: {ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames")
print(f"Policy device: {cfg.policy.device} | online_steps: {cfg.policy.online_steps}")
PY
    echo ""
    echo "Next:"
    echo "  Terminal 1: bash train_hilserl_so101.sh learner"
    echo "  Terminal 2: bash train_hilserl_so101.sh actor"
}

prefetch_vision_encoder() {
    python3 <<'PY'
import sys
from transformers import AutoModel

model_id = "helper2424/resnet10"
try:
    AutoModel.from_pretrained(model_id, trust_remote_code=True)
    print(f"Vision encoder ready: {model_id}")
except OSError as e:
    print(f"ERROR: cannot load {model_id}: {e}", file=sys.stderr)
    print("Need network once to download. Check HF_HUB_OFFLINE / proxy / firewall.", file=sys.stderr)
    sys.exit(1)
PY
}

run_learner() {
    echo "Starting LEARNER (gRPC server on port 50051)..."
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | policy: ${LEARNER_DEVICE} | buffer: ${STORAGE_DEVICE}"
    echo "Logs: ${OUTPUT_DIR}/logs/"
    prefetch_vision_encoder
    python -m lerobot.rl.learner \
        --config_path="${CONFIG_PATH}" \
        "${POLICY_DEVICE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
}

run_actor() {
    echo "Starting ACTOR (real robot)..."
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} | policy: ${ACTOR_DEVICE}"
    echo "Keyboard: move keys = intervene | s = success | q = fail"
    echo "Logs: ${OUTPUT_DIR}/logs/"
    sleep 1
    python -m lerobot.rl.actor \
        --config_path="${CONFIG_PATH}" \
        "${ACTOR_DEVICE_ARGS[@]}" \
        "${EXTRA_ARGS[@]}"
}

case "${MODE}" in
    learner)
        prepare_output_dir learner
        run_learner
        ;;
    actor)
        prepare_output_dir actor
        run_actor
        ;;
    check)
        check_setup
        ;;
    download)
        prefetch_vision_encoder
        ;;
    help|*)
        cat <<EOF
Usage: bash train_hilserl_so101.sh [learner|actor|check|download]

  learner   Start learner server (run first, keep terminal open)
  actor     Start actor on real robot (run in second terminal)
  check     Validate config and cropped dataset
  download  Prefetch helper2424/resnet10 vision encoder (first run)

Environment overrides:
  CONFIG_PATH=...  OUTPUT_DIR=...  DATA_ROOT=...
  HF_ENDPOINT=...  Default https://hf-mirror.com if unset
  CUDA_VISIBLE_DEVICES=1   Default; only use physical GPU 1
  USE_DUAL_GPU=1  Need CUDA_VISIBLE_DEVICES=0,1; GPU0=train GPU1=buffer
  LEARNER_DEVICE=...  STORAGE_DEVICE=...  ACTOR_DEVICE=...
  FRESH=1          Delete old output dir before learner start
  RESUME=1         Resume from checkpoint in OUTPUT_DIR
                   (required once checkpoints/last exists)
EOF
        ;;
esac
