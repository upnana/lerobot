#!/bin/bash
# =============================================================================
# 真机评测：bowls_tray_3cam_v2 @ 120000（对照 v3 是否变慢）
#
# Usage:
#   bash /home/rxn/lerobot/infer_v2_ckpt_120k.sh
#   NUM_EPISODES=2 EPISODE_TIME_S=300 bash /home/rxn/lerobot/infer_v2_ckpt_120k.sh
#   OVERLAY_VALUE=0 bash /home/rxn/lerobot/infer_v2_ckpt_120k.sh   # 不叠 value
#   PRESET=n10 bash /home/rxn/lerobot/infer_v2_ckpt_120k.sh         # Advantage: positive
# =============================================================================
set -euo pipefail

export TRAIN_ROOT="${TRAIN_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/train/smolvla_acp_bowls_tray_3cam_v2}"
export EVAL_ROOT="${EVAL_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/eval/smolvla_acp_bowls_tray_3cam_v2}"
export DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_v2}"
export DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_bowls_stack_lipstick_tissue_v2}"
# overlay 仍用 v2 value（和当时训练标签一致）
export VALUE_RUN="${VALUE_RUN:-bowls_tray_3cam_v2}"
export TAG="${TAG:-bowls_tray_3cam_v2}"
export PRESET="${PRESET:-n10_acp_none}"
export OVERLAY_VALUE="${OVERLAY_VALUE:-1}"
export NUM_EPISODES="${NUM_EPISODES:-1}"
export EPISODE_TIME_S="${EPISODE_TIME_S:-300}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
export MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-50}"

STEP="${1:-120000}"

echo "=============================="
echo "infer v2 ckpt ${STEP}"
echo "TRAIN_ROOT=${TRAIN_ROOT}"
echo "PRESET=${PRESET}  OVERLAY_VALUE=${OVERLAY_VALUE}"
echo "=============================="

exec bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh "${STEP}"
