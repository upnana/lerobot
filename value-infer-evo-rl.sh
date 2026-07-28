#!/bin/bash
# Evo-RL value inference: write value / advantage / acp_indicator into dataset
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate}"
DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_snacks_tray_stack_bowls_plate}"
RUN_NAME="${RUN_NAME:-snacks_bowls_v1}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/outputs/value_train/${RUN_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/value_infer/${RUN_NAME}}"
TAG="${TAG:-snacks_bowls_v1}"

NUM_GPUS="${NUM_GPUS:-2}"
GPUS="${GPUS:-0,1}"
# Per-GPU batch; no grads so can be larger than train. Start moderate for 4 cams.
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
N_STEP="${N_STEP:-50}"
POSITIVE_RATIO="${POSITIVE_RATIO:-0.3}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "=============================="
echo "Evo-RL value-infer (${RUN_NAME})"
echo "=============================="
echo "GPUs:       ${GPUS} (num=${NUM_GPUS})"
echo "Dataset:    ${DATA_ROOT}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Output:     ${OUTPUT_DIR}"
echo "ACP tag:    ${TAG}  n_step=${N_STEP} pos=${POSITIVE_RATIO}"
echo "Batch/gpu:  ${BATCH_SIZE}"
echo "=============================="

INFER_ARGS=(
  --dataset.repo_id="${DATA_REPO}"
  --dataset.root="${DATA_ROOT}"
  --inference.checkpoint_path="${CHECKPOINT_PATH}"
  --runtime.device=cuda
  --runtime.batch_size="${BATCH_SIZE}"
  --runtime.num_workers="${NUM_WORKERS}"
  --acp.enable=true
  --acp.n_step="${N_STEP}"
  --acp.positive_ratio="${POSITIVE_RATIO}"
  --acp.value_field="complementary_info.value_${TAG}"
  --acp.advantage_field="complementary_info.advantage_${TAG}"
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${RUN_NAME}.infer"
  --viz.enable=false
)

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  accelerate launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    "$(which lerobot-value-infer)" \
    "${INFER_ARGS[@]}"
else
  lerobot-value-infer "${INFER_ARGS[@]}"
fi
