#!/bin/bash
# =============================================================================
# GR00T 训练脚本（环境安装好后直接运行）
# Usage:
#   conda activate lerobot-groot
#   bash train_groot.sh          # 默认 2 卡 DDP
#   NUM_GPUS=1 bash train_groot.sh   # 单卡
#   FRESH=1 bash train_groot.sh      # 删除旧 output 重新训练
#   RESUME=1 bash train_groot.sh     # 从 checkpoint 续训
#   BACKGROUND=1 bash train_groot.sh # nohup 后台运行，日志写入 logs/
#   tail -f logs/groot_yellow_white.log
# =============================================================================
set -euo pipefail

# ---------- 路径配置（按需修改） ----------
PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/yellow_white}"
DATA_REPO="${DATA_REPO:-my_pick_place/yellow_white}"
BASE_MODEL="${PROJECT_ROOT}/src/lerobot/GR00T-basemodel"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/groot_yellow_white}"
CONDA_ENV="lerobot-groot"
JOB_NAME="${JOB_NAME:-groot_yellow_white}"

# ---------- 训练超参（双卡 RTX 3090 24GB） ----------
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"   # per-GPU; effective = BATCH_SIZE * NUM_GPUS (=16, 等同单卡 bs=16)
NUM_WORKERS="${NUM_WORKERS:-0}" # 多卡时保持 0，避免 dataloader fork 问题
STEPS="${STEPS:-120000}"        # 24055 frames @ effective bs=8 → ~3007 steps/epoch (~40 epoch)
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.pid}"

# --- nohup 后台模式（只启动一次）---
if [ "${BACKGROUND:-0}" = "1" ] && [ -z "${TRAIN_GROOT_BG:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/logs"
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Already running PID $(cat "${PID_FILE}")"
        echo "Log: ${LOG_FILE}"
        exit 1
    fi
    echo "Starting nohup background training..."
    echo "  log: ${LOG_FILE}"
    nohup env TRAIN_GROOT_BG=1 \
        FRESH="${FRESH:-0}" RESUME="${RESUME:-0}" \
        STEPS="${STEPS}" NUM_GPUS="${NUM_GPUS}" BATCH_SIZE="${BATCH_SIZE}" \
        OUTPUT_DIR="${OUTPUT_DIR}" JOB_NAME="${JOB_NAME}" DATA_ROOT="${DATA_ROOT}" DATA_REPO="${DATA_REPO}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        bash "$0" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "  pid: $(cat "${PID_FILE}")"
    echo "  tail -f ${LOG_FILE}"
    exit 0
fi

echo "=============================="
echo "GR00T Training"
echo "=============================="

# --- 激活环境 ---
source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    echo "Run first: bash ${PROJECT_ROOT}/setup_groot_env.sh"
    exit 1
fi
conda activate "${CONDA_ENV}"

# --- 环境变量 ---
unset HF_HUB_OFFLINE
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export ACCELERATE_DISABLE_RICH=1
export ACCELERATE_USE_META=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 双卡通信（单机多卡默认即可；若 NCCL 报错可取消下行注释）
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1

# --- 资源检查 ---
for path in "${DATA_ROOT}" "${BASE_MODEL}"; do
    if [ ! -d "${path}" ]; then
        echo "ERROR: not found: ${path}"
        exit 1
    fi
done

# 刷新 Eagle cache（确保使用最新修复的 modeling 代码）
EAGLE_CACHE="${HOME}/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5"
mkdir -p "${EAGLE_CACHE}"
cp -r "${PROJECT_ROOT}/src/lerobot/policies/groot/eagle2_hg_model/." "${EAGLE_CACHE}/"

echo "Dataset:  ${DATA_ROOT} (${DATA_REPO})"
echo "Model:    ${BASE_MODEL}"
echo "Output:   ${OUTPUT_DIR}"
echo "GPUs:     ${NUM_GPUS}"
echo "Batch:    ${BATCH_SIZE} per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo ""

# --- GPU 预检（GR00T ~3B 需要约 18GB 空闲显存 / 卡）---
MIN_FREE_MIB=18000
if command -v nvidia-smi >/dev/null 2>&1; then
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    else
        GPU_IDS=()
        GPU_IDX=0
        while [ "${GPU_IDX}" -lt "${NUM_GPUS}" ]; do
            GPU_IDS+=("${GPU_IDX}")
            GPU_IDX=$((GPU_IDX + 1))
        done
    fi
    if [ "${#GPU_IDS[@]}" -lt "${NUM_GPUS}" ]; then
        echo "ERROR: need ${NUM_GPUS} visible GPUs, found ${#GPU_IDS[@]} (${CUDA_VISIBLE_DEVICES:-all})"
        exit 1
    fi
    echo "GPU status:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
        [ -n "${line}" ] && echo "  other process: ${line}"
    done || true
    CHECK_IDX=0
    while [ "${CHECK_IDX}" -lt "${NUM_GPUS}" ]; do
        GPU_IDX="${GPU_IDS[CHECK_IDX]}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
        TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
        echo "  GPU ${GPU_IDX}: free/total ${FREE_MIB} / ${TOTAL_MIB} MiB"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo ""
            echo "ERROR: GPU ${GPU_IDX} has only ${FREE_MIB} MiB free (need >= ${MIN_FREE_MIB} MiB)."
            echo "Kill stale training processes first, e.g.:"
            echo "  nvidia-smi"
            echo "  kill <pid>"
            exit 1
        fi
        CHECK_IDX=$((CHECK_IDX + 1))
    done
    echo ""
fi

# --- 快速数据检查 ---
python - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATA_ROOT}/meta/info.json").read_text())
print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")
assert info["total_frames"] > 0, "Dataset is empty!"
PY

# --- 输出目录 ---
# FRESH=1     删除旧目录，从头训练
# RESUME=1    从 checkpoints/last 续训（需 train_config.json）
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
    echo "  FRESH=1 bash train_groot.sh    # delete and start over"
    echo "  RESUME=1 bash train_groot.sh   # continue from last checkpoint"
    echo "  OUTPUT_DIR=/new/path bash train_groot.sh"
    exit 1
fi
echo ""
# --- 训练（多卡 DDP / 单卡 via accelerate）---
# 注意: accelerate 控制 mixed precision; policy.use_bf16 保留给 GR00T 内部 autocast
ACCEL_ARGS=(
  --num_processes="${NUM_GPUS}"
  --num_machines=1
  --mixed_precision=bf16
  --dynamo_backend=no
)
if [ "${NUM_GPUS}" -gt 1 ]; then
  ACCEL_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT:-29500}")
fi
accelerate launch "${ACCEL_ARGS[@]}" \
  "$(which lerobot-train)" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id="${DATA_REPO}" \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path="${BASE_MODEL}" \
  --policy.tokenizer_assets_repo="lerobot/eagle2hg-processor-groot-n1p5" \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=false \
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
echo "Training finished: ${OUTPUT_DIR}"
echo "=============================="
