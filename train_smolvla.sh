#!/bin/bash
# =============================================================================
# SmolVLA 多任务训练脚本（merged dataset）
# Usage:
#   conda activate lerobot
#   bash train_smolvla.sh              # 默认 2 卡 DDP
#   NUM_GPUS=1 bash train_smolvla.sh   # 单卡
#   FRESH=1 bash train_smolvla.sh      # 删除旧 output 重新训练
#   RESUME=1 bash train_smolvla.sh     # 从 checkpoint 续训
#   BACKGROUND=1 bash train_smolvla.sh # nohup 后台运行
#   tail -f logs/smolvla_multitask.log
#
# yellow_white_base2 单任务请用: bash train_smolvla_base2.sh
# =============================================================================
set -euo pipefail

# ---------- 路径配置 ----------
PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/smolvla_multitask_merged}"
DATA_REPO="${DATA_REPO:-my_multitask/pickup_items_and_blocks}"
BASE_MODEL="${BASE_MODEL:-/home/rxn/.cache/modelscope/models/lerobot--smolvla_base/snapshots/master}"
# VLM 骨干（ModelScope 本地下载）；用于 tokenizer/processor
VLM_MODEL="${VLM_MODEL:-/home/rxn/.cache/modelscope/models/HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/master}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/smolvla_multitask}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
JOB_NAME="${JOB_NAME:-smolvla_multitask}"

# 数据集相机 front/wrist -> SmolVLA 预训练 camera1/camera2，camera3 用空相机补齐
RENAME_MAP='{"observation.images.front":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# ---------- 训练超参（双卡 RTX 3090 24GB） ----------
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"   # per-GPU; SmolVLA ~450M，3090 可尝试 16~32
NUM_WORKERS="${NUM_WORKERS:-4}"
STEPS="${STEPS:-1100000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.pid}"

# --- nohup 后台模式 ---
if [ "${BACKGROUND:-0}" = "1" ] && [ -z "${TRAIN_SMOLVLA_BG:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/logs"
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Already running PID $(cat "${PID_FILE}")"
        echo "Log: ${LOG_FILE}"
        exit 1
    fi
    echo "Starting nohup background training..."
    echo "  log: ${LOG_FILE}"
    nohup env TRAIN_SMOLVLA_BG=1 \
        FRESH="${FRESH:-0}" RESUME="${RESUME:-0}" \
        STEPS="${STEPS}" NUM_GPUS="${NUM_GPUS}" BATCH_SIZE="${BATCH_SIZE}" \
        SAVE_FREQ="${SAVE_FREQ}" LOG_FREQ="${LOG_FREQ}" NUM_WORKERS="${NUM_WORKERS}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        OUTPUT_DIR="${OUTPUT_DIR}" JOB_NAME="${JOB_NAME}" DATA_ROOT="${DATA_ROOT}" DATA_REPO="${DATA_REPO}" \
        bash "$0" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "  pid: $(cat "${PID_FILE}")"
    echo "  tail -f ${LOG_FILE}"
    exit 0
fi

echo "=============================="
echo "SmolVLA Multitask Training"
echo "=============================="

# --- 激活环境 ---
source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 1
fi
conda activate "${CONDA_ENV}"

# --- 环境变量（离线训练，避免访问 huggingface.co）---
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export ACCELERATE_DISABLE_RICH=1
export ACCELERATE_USE_META=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 资源检查 ---
if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: dataset not found: ${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${BASE_MODEL}/config.json" ]; then
    echo "ERROR: SmolVLA base model not found: ${BASE_MODEL}"
    echo "Download with: modelscope download --model lerobot/smolvla_base"
    exit 1
fi
if [ ! -f "${VLM_MODEL}/config.json" ]; then
    echo "ERROR: VLM model not found: ${VLM_MODEL}"
    echo "Download with: modelscope download --model HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
    exit 1
fi

echo "Dataset:  ${DATA_ROOT} (${DATA_REPO})"
echo "Model:    ${BASE_MODEL}"
echo "VLM:      ${VLM_MODEL}"
echo "Output:   ${OUTPUT_DIR}"
echo "GPUs:     ${NUM_GPUS}"
echo "Batch:    ${BATCH_SIZE} per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Steps:    ${STEPS}"
echo ""

# --- GPU 预检（SmolVLA batch=4 约需 6~8GB/卡；batch=16 约需 12GB+）---
MIN_FREE_MIB="${MIN_FREE_MIB:-6000}"
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    # Respect CUDA_VISIBLE_DEVICES when checking physical GPU memory
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    else
        GPU_IDS=()
        for ((i=0; i<GPU_COUNT; i++)); do GPU_IDS+=("$i"); done
    fi
    if [ "${NUM_GPUS}" -gt "${#GPU_IDS[@]}" ]; then
        echo "ERROR: NUM_GPUS=${NUM_GPUS} but only ${#GPU_IDS[@]} GPU(s) in CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
        exit 1
    fi
    echo "GPU status (checking physical GPU(s): ${GPU_IDS[*]:0:${NUM_GPUS}}):"
    nvidia-smi --query-compute-apps=gpu_bus_id,pid,process_name,used_memory --format=csv,noheader 2>/dev/null | while IFS= read -r line; do
        [ -n "${line}" ] && echo "  other process: ${line}"
    done || true
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_IDX="${GPU_IDS[$i]}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
        TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
        echo "  GPU ${GPU_IDX}: free/total ${FREE_MIB} / ${TOTAL_MIB} MiB"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo ""
            echo "ERROR: GPU ${GPU_IDX} has only ${FREE_MIB} MiB free (need >= ${MIN_FREE_MIB} MiB)."
            echo "Another job is using the GPU (~$((TOTAL_MIB - FREE_MIB)) MiB occupied). Free it first:"
            echo "  nvidia-smi"
            echo "  kill <pid>"
            echo "Or use another GPU: NUM_GPUS=1 CUDA_VISIBLE_DEVICES=1 bash train_smolvla.sh"
            exit 1
        fi
    done
    echo ""
fi

# --- 快速数据检查（只用 info.json，避免 pandas 读 parquet 偶发 segfault）---
python - <<PY
import json
from pathlib import Path

root = Path("${DATA_ROOT}")
info = json.loads((root / "meta/info.json").read_text())
print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}, Tasks: {info['total_tasks']}")
assert info["total_frames"] > 0, "Dataset is empty!"
assert info["total_episodes"] > 0, "Dataset has no episodes!"
PY

# --- 离线修复：policy_preprocessor.json 里 tokenizer 仍指向 HF Hub ---
python - <<PY
import json
from pathlib import Path

vlm = "${VLM_MODEL}"
prep_path = Path("${BASE_MODEL}") / "policy_preprocessor.json"
cfg = json.loads(prep_path.read_text())
for step in cfg.get("steps", []):
    if step.get("registry_name") == "tokenizer_processor":
        old = step["config"].get("tokenizer_name")
        if old != vlm:
            step["config"]["tokenizer_name"] = vlm
            prep_path.write_text(json.dumps(cfg, indent=2) + "\n")
            print(f"Patched tokenizer_name: {old} -> {vlm}")
        else:
            print(f"tokenizer_name already local: {vlm}")
        break
else:
    raise SystemExit("tokenizer_processor step not found in policy_preprocessor.json")
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
    echo "  FRESH=1 bash train_smolvla.sh"
    echo "  RESUME=1 bash train_smolvla.sh"
    echo "  OUTPUT_DIR=/new/path bash train_smolvla.sh"
    exit 1
fi
echo ""

# --- 训练 ---
ACCEL_ARGS=(
  --num_processes="${NUM_GPUS}"
  --num_machines=1
  --mixed_precision=bf16
  --dynamo_backend=no
)
if [ "${NUM_GPUS}" -gt 1 ]; then
  ACCEL_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT:-29501}")
fi

POLICY_ARGS=(
  --policy.device=cuda
  --policy.vlm_model_name="${VLM_MODEL}"
  --policy.load_vlm_weights=false
  --policy.empty_cameras=1
)
if [ "${RESUME}" != "1" ]; then
  POLICY_ARGS=(--policy.path="${BASE_MODEL}" "${POLICY_ARGS[@]}")
fi

accelerate launch "${ACCEL_ARGS[@]}" \
  "$(which lerobot-train)" \
  "${POLICY_ARGS[@]}" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id="${DATA_REPO}" \
  --rename_map="${RENAME_MAP}" \
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
