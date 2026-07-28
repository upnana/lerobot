#!/bin/bash
# HIL-SERL Step 4b: train reward classifier
#
# Prerequisites:
#   bash crop_hilserl_demos.sh
#   OVERWRITE=1 bash relabel_hilserl_classifier.sh
#
# Usage:
#   bash train_reward_classifier_so101.sh
#   RESUME=1 bash train_reward_classifier_so101.sh
#
# After training, point configs/hilserl_so101_train.json reward_classifier.pretrained_path
# to the checkpoint below, then run learner + actor.
set -euo pipefail

PROJECT_ROOT="/home/rxn/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot}"
CONFIG_PATH="${CONFIG_PATH:-${PROJECT_ROOT}/configs/hilserl_so101_reward_classifier_train.json}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/hilserl_push_black_block_reward_classifier}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/rxn/lerobot/outputs/train/reward_classifier_so101_push_black_block}"
RESUME="${RESUME:-0}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export CUDA_VISIBLE_DEVICES

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=0
: "${HF_ENDPOINT:=https://hf-mirror.com}"
export HF_ENDPOINT

if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
    echo "ERROR: relabeled dataset not found: ${DATA_ROOT}"
    echo "  Run: OVERWRITE=1 bash relabel_hilserl_classifier.sh"
    exit 1
fi

if [ ! -f "${CONFIG_PATH}" ]; then
    echo "ERROR: config not found: ${CONFIG_PATH}"
    exit 1
fi

EXTRA_ARGS=()
if [ "${RESUME}" = "1" ]; then
    EXTRA_ARGS+=(--resume=true)
elif [ -d "${OUTPUT_DIR}" ] && [ "$(ls -A "${OUTPUT_DIR}" 2>/dev/null || true)" != "" ]; then
    echo "ERROR: output dir exists: ${OUTPUT_DIR}"
    echo "  Use RESUME=1 to continue, or remove the directory to retrain."
    exit 1
fi

echo "=============================="
echo "HIL-SERL Step 4b: Train classifier"
echo "=============================="
echo "  config:  ${CONFIG_PATH}"
echo "  dataset: ${DATA_ROOT}"
echo "  output:  ${OUTPUT_DIR}"
echo "  GPU:     ${CUDA_VISIBLE_DEVICES}"
echo "=============================="

lerobot-train \
  --config_path="${CONFIG_PATH}" \
  --dataset.repo_id="my_rl/push_black_block_reward_classifier" \
  --dataset.root="${DATA_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="reward_classifier_so101_push_black_block" \
  "${EXTRA_ARGS[@]}"

CHECKPOINT="${OUTPUT_DIR}/checkpoints/last/pretrained_model"
echo ""
echo "Done."
echo "Classifier checkpoint: ${CHECKPOINT}"
echo ""
echo "Next: set env.processor.reward_classifier.pretrained_path in train config, then:"
echo "  bash train_hilserl_so101.sh learner"
echo "  bash train_hilserl_so101.sh actor"
