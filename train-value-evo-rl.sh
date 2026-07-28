#!/bin/bash
# Evo-RL value training (pistar06) on local snacks/bowls dataset
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate}"
DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_snacks_tray_stack_bowls_plate}"
RUN_NAME="${RUN_NAME:-snacks_bowls_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/value_train/${RUN_NAME}}"

# Dual-GPU by default (2x 3090). BATCH_SIZE is per-GPU; effective = BATCH_SIZE * NUM_GPUS.
# 4 cams x batch: SigLIP activations dominate VRAM; 32/GPU OOMs, 4–8/GPU is safer.
NUM_GPUS="${NUM_GPUS:-2}"
GPUS="${GPUS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
STEPS="${STEPS:-8000}"
SAVE_FREQ="${SAVE_FREQ:-4000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
USE_GRAD_CHECKPOINT="${USE_GRAD_CHECKPOINT:-true}"
FREEZE_VISION="${FREEZE_VISION:-false}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

# Prefer local backbones (ModelScope / HF cache) to avoid gated HF 403.
LANGUAGE_REPO_ID="${LANGUAGE_REPO_ID:-/home/rxn/models/modelscope/models/google--gemma-3-270m/snapshots/master}"
VISION_REPO_ID="${VISION_REPO_ID:-/home/rxn/.cache/huggingface/hub/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3}"

# Offline once local paths are set; unset these only if you intentionally want hub download.
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Do NOT pre-create OUTPUT_DIR: lerobot-value-train refuses if it already exists
# (unless --resume=true). Only ensure parent exists.
mkdir -p "$(dirname "${OUTPUT_DIR}")"
rm -rf "${OUTPUT_DIR}"

if [[ ! -d "${LANGUAGE_REPO_ID}" ]]; then
  echo "Missing language model at: ${LANGUAGE_REPO_ID}"
  echo "Download with: python -c \"from modelscope import snapshot_download; print(snapshot_download('google/gemma-3-270m', cache_dir='/home/rxn/models/modelscope'))\""
  exit 1
fi
if [[ ! -d "${VISION_REPO_ID}" ]]; then
  echo "Missing vision model at: ${VISION_REPO_ID}"
  exit 1
fi

EFFECTIVE_BATCH=$((BATCH_SIZE * NUM_GPUS))

echo "=============================="
echo "Evo-RL value-train (${RUN_NAME})"
echo "=============================="
echo "Env:      ${CONDA_ENV}"
echo "GPUs:     ${GPUS}  (num=${NUM_GPUS})"
echo "Dataset:  ${DATA_ROOT}"
echo "Language: ${LANGUAGE_REPO_ID}"
echo "Vision:   ${VISION_REPO_ID}"
echo "Output:   ${OUTPUT_DIR}"
echo "Steps:    ${STEPS}  batch/gpu=${BATCH_SIZE}  effective=${EFFECTIVE_BATCH}"
echo "GradCkpt: ${USE_GRAD_CHECKPOINT}  freeze_vision=${FREEZE_VISION}"
echo "=============================="

# Optional: limit cameras, e.g. CAMERA_FEATURES='["observation.images.right_top_right","observation.images.left_wrist","observation.images.right_wrist"]'
CAMERA_FEATURES="${CAMERA_FEATURES:-}"

TRAIN_ARGS=(
  --dataset.repo_id="${DATA_REPO}"
  --dataset.root="${DATA_ROOT}"
  --value.type=pistar06
  --value.dtype=bfloat16
  --value.device=cuda
  --value.language_repo_id="${LANGUAGE_REPO_ID}"
  --value.vision_repo_id="${VISION_REPO_ID}"
  --value.use_gradient_checkpointing="${USE_GRAD_CHECKPOINT}"
  --value.freeze_vision_encoder="${FREEZE_VISION}"
  --value.push_to_hub=false
  --batch_size="${BATCH_SIZE}"
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --num_workers="${NUM_WORKERS}"
  --output_dir="${OUTPUT_DIR}"
  --job_name="${RUN_NAME}"
  --wandb.enable=false
)
if [[ -n "${CAMERA_FEATURES}" ]]; then
  TRAIN_ARGS+=(--value.camera_features="${CAMERA_FEATURES}")
  echo "Cameras:  ${CAMERA_FEATURES}"
fi

export CUDA_VISIBLE_DEVICES="${GPUS}"

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  accelerate launch \
    --multi_gpu \
    --num_processes="${NUM_GPUS}" \
    --mixed_precision=bf16 \
    "$(which lerobot-value-train)" \
    "${TRAIN_ARGS[@]}"
else
  lerobot-value-train "${TRAIN_ARGS[@]}"
fi
