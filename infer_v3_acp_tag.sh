#!/bin/bash
# =============================================================================
# 真机评测：bowls_tray_3cam_v3 + Advantage: positive（带 ACP tag）
#
# Usage:
#   bash /home/rxn/lerobot/infer_v3_acp_tag.sh              # 默认 100000
#   bash /home/rxn/lerobot/infer_v3_acp_tag.sh 120000
#   NUM_EPISODES=2 EPISODE_TIME_S=300 bash /home/rxn/lerobot/infer_v3_acp_tag.sh
#   OVERLAY_VALUE=0 bash /home/rxn/lerobot/infer_v3_acp_tag.sh
# =============================================================================
set -euo pipefail

export TRAIN_ROOT="${TRAIN_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/train/smolvla_acp_bowls_tray_3cam_v3}"
export EVAL_ROOT="${EVAL_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/eval/smolvla_acp_bowls_tray_3cam_v3}"
export DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_v3}"
export DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_bowls_stack_lipstick_tissue_v3}"
export VALUE_RUN="${VALUE_RUN:-bowls_tray_3cam_v3}"
export TAG="${TAG:-bowls_tray_3cam_v3}"
export PRESET="${PRESET:-n10}"   # Advantage: positive
export OVERLAY_VALUE="${OVERLAY_VALUE:-1}"
export NUM_EPISODES="${NUM_EPISODES:-1}"
export EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
export MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-50}"

STEP="${1:-100000}"

echo "=============================="
echo "infer v3 + ACP tag  ckpt ${STEP}"
echo "TRAIN_ROOT=${TRAIN_ROOT}"
echo "PRESET=${PRESET}  (Advantage: positive)"
echo "OVERLAY_VALUE=${OVERLAY_VALUE}"
echo "=============================="

exec bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh "${STEP}"
