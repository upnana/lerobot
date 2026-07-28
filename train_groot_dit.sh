#!/bin/bash
# =============================================================================
# GR00T 第二阶段训练：在 stage-1 checkpoint 上解冻 DiT（动作生成头）
#
# 背景：仅 tune_projector=true 时，机械臂会动但几乎不看画面/物体位置，
#       表现为重复同一套动作、抓不起来。
#
# Usage:
#   conda activate lerobot-groot
#   bash train_groot_dit.sh
#   CUDA_VISIBLE_DEVICES=1 bash train_groot_dit.sh          # 指定 GPU
#   BACKGROUND=1 bash train_groot_dit.sh                    # nohup 后台
#   tail -f logs/groot_yellow_white_dit.log
#   FRESH=1 bash train_groot_dit.sh                         # 清空 output 重训
#   RESUME=1 bash train_groot_dit.sh                      # 从 last 续训
#   STEPS=100000 bash train_groot_dit.sh                  # 加长训练（~4 epoch @ bs=1）
#
# 完成后推理:
#   CHECKPOINT=outputs/train/groot_yellow_white_dit/checkpoints/last/pretrained_model \
#     bash infer_groot.sh robot
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/yellow_white}"
DATA_REPO="${DATA_REPO:-my_pick_place/yellow_white}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${PROJECT_ROOT}/outputs/train/groot_yellow_white/checkpoints/last/pretrained_model}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/groot_yellow_white_dit}"
CONDA_ENV="lerobot-groot"
JOB_NAME="${JOB_NAME:-groot_yellow_white_dit}"

# DiT 需要反传全动作头，24GB 显存建议 batch=1
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
STEPS="${STEPS:-50000}"          # 24055 frames @ bs=1 → ~24055 steps/epoch (~2 epoch)
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.pid}"
MIN_FREE_MIB=18000

# --- nohup 后台模式（只启动一次）---
if [ "${BACKGROUND:-0}" = "1" ] && [ -z "${TRAIN_GROOT_DIT_BG:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/logs"
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Already running PID $(cat "${PID_FILE}")"
        echo "Log: ${LOG_FILE}"
        exit 1
    fi
    echo "Starting nohup background DiT training..."
    echo "  log: ${LOG_FILE}"
    nohup env TRAIN_GROOT_DIT_BG=1 \
        FRESH="${FRESH:-0}" RESUME="${RESUME:-0}" \
        STEPS="${STEPS}" BATCH_SIZE="${BATCH_SIZE}" \
        OUTPUT_DIR="${OUTPUT_DIR}" JOB_NAME="${JOB_NAME}" \
        BASE_CHECKPOINT="${BASE_CHECKPOINT}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        bash "$0" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "  pid: $(cat "${PID_FILE}")"
    echo "  tail -f ${LOG_FILE}"
    exit 0
fi

echo "=============================="
echo "GR00T DiT fine-tune (stage 2)"
echo "=============================="

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    echo "Run first: bash ${PROJECT_ROOT}/setup_groot_env.sh"
    exit 1
fi
conda activate "${CONDA_ENV}"

unset HF_HUB_OFFLINE
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export ACCELERATE_DISABLE_RICH=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EAGLE_CACHE="${HOME}/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5"
mkdir -p "${EAGLE_CACHE}"
cp -r "${PROJECT_ROOT}/src/lerobot/policies/groot/eagle2_hg_model/." "${EAGLE_CACHE}/"

for path in "${DATA_ROOT}" "${BASE_CHECKPOINT}"; do
    if [ ! -d "${path}" ]; then
        echo "ERROR: not found: ${path}"
        exit 1
    fi
done

# --- GPU 预检（尊重 CUDA_VISIBLE_DEVICES）---
if command -v nvidia-smi >/dev/null 2>&1; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        GPU_IDX="${CUDA_VISIBLE_DEVICES%%,*}"
    else
        GPU_IDX=0
    fi
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
    TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
    echo "GPU ${GPU_IDX}: free/total ${FREE_MIB} / ${TOTAL_MIB} MiB (need >= ${MIN_FREE_MIB} MiB)"
    if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
        echo "ERROR: GPU ${GPU_IDX} has only ${FREE_MIB} MiB free."
        echo "Kill other processes: nvidia-smi"
        exit 1
    fi
    echo ""
fi

python - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATA_ROOT}/meta/info.json").read_text())
print(f"Dataset:  ${DATA_ROOT} (${DATA_REPO})")
print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")
PY

# --- 输出目录 ---
RESUME="${RESUME:-0}"
CONFIG_PATH=""
if [ "${FRESH:-0}" = "1" ] && [ -d "${OUTPUT_DIR}" ]; then
    echo "FRESH=1: removing existing output dir: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
elif [ "${RESUME}" = "1" ]; then
    CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
    if [ ! -f "${CONFIG_PATH}" ]; then
        echo "ERROR: RESUME=1 but missing ${CONFIG_PATH}"
        exit 1
    fi
    echo "Resume from: ${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
    echo "ERROR: ${OUTPUT_DIR} already exists."
    echo "  FRESH=1 bash train_groot_dit.sh    # delete and start over"
    echo "  RESUME=1 bash train_groot_dit.sh   # continue from last checkpoint"
    echo "  OUTPUT_DIR=/new/path bash train_groot_dit.sh"
    exit 1
fi

echo "Base ckpt:  ${BASE_CHECKPOINT}"
echo "Output:     ${OUTPUT_DIR}"
echo "Tune:       projector=true, diffusion_model=true"
echo "Batch:      ${BATCH_SIZE}, steps: ${STEPS}, save_freq: ${SAVE_FREQ}"
echo "=============================="
echo ""

lerobot-train \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id="${DATA_REPO}" \
  --policy.type=groot \
  --policy.path="${BASE_CHECKPOINT}" \
  --policy.device=cuda \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.use_bf16=true \
  --policy.video_backend=decord \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --eval_freq=-1 \
  --seed=1000 \
  $([ "${RESUME}" = "1" ] && echo "--resume=true --config_path=${CONFIG_PATH}") \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "=============================="
echo "DiT training finished: ${OUTPUT_DIR}"
echo "Inference:"
echo "  CHECKPOINT=${OUTPUT_DIR}/checkpoints/last/pretrained_model bash infer_groot.sh robot"
echo "=============================="
