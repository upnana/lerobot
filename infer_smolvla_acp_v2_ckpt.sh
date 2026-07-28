#!/bin/bash
# =============================================================================
# 快速真机评测：smolvla_acp_snacks_bowls_v2 各 checkpoint（按 step / epoch）
#
# Usage:
#   bash /home/rxn/lerobot/infer_smolvla_acp_v2_ckpt.sh              # 默认 100k
#   bash /home/rxn/lerobot/infer_smolvla_acp_v2_ckpt.sh 50000
#   bash /home/rxn/lerobot/infer_smolvla_acp_v2_ckpt.sh 100000
#   bash /home/rxn/lerobot/infer_smolvla_acp_v2_ckpt.sh last
#   STEP=150000 bash /home/rxn/lerobot/infer_smolvla_acp_v2_ckpt.sh
#
# 常用覆盖:
#   EPISODE_TIME_S=300 NUM_EPISODES=2 bash ... 100000
#   PRESET=n10 bash ... 50000          # 带 Advantage: positive
#   PRESET=n10_acp_none bash ... 100000  # 默认：无 ACP（50k/100k 更敢动）
# =============================================================================
set -euo pipefail

STEP="${1:-${STEP:-100000}}"
TRAIN_ROOT="${TRAIN_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/train/smolvla_acp_snacks_bowls_v2}"

if [ "${STEP}" = "last" ]; then
  CKPT_DIR="${TRAIN_ROOT}/checkpoints/last"
else
  # accept 50k / 50000 / 050000
  if [[ "${STEP}" =~ ^[0-9]+k$ ]]; then
    STEP=$((${STEP%k} * 1000))
  fi
  STEP_PAD="$(printf '%06d' "${STEP}")"
  CKPT_DIR="${TRAIN_ROOT}/checkpoints/${STEP_PAD}"
fi

CHECKPOINT="${CHECKPOINT:-${CKPT_DIR}/pretrained_model}"
if [ ! -f "${CHECKPOINT}/config.json" ]; then
  echo "ERROR: checkpoint missing: ${CHECKPOINT}"
  echo "Available:"
  ls "${TRAIN_ROOT}/checkpoints" 2>/dev/null || true
  exit 1
fi

# Longer default episode for real task attempts (snacks + bowls)
export EPISODE_TIME_S="${EPISODE_TIME_S:-240}"
export RESET_TIME_S="${RESET_TIME_S:-20}"
export NUM_EPISODES="${NUM_EPISODES:-1}"
export PRESET="${PRESET:-n10_acp_none}"
export N_ACTION_STEPS="${N_ACTION_STEPS:-10}"
export MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-50}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export SKIP_SLEEP="${SKIP_SLEEP:-1}"

export OUTPUT_DIR="${OUTPUT_DIR:-${TRAIN_ROOT}}"
export CHECKPOINT
export EVAL_ROOT="${EVAL_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/eval/smolvla_acp_snacks_bowls_v2}"

# ~197k frames @ eff batch logged ≈ step/12500 epochs (log epch); rough guide:
# 50k≈4ep  100k≈8ep  200k≈16ep
echo ">>> v2 ckpt test: ${CHECKPOINT}"
echo ">>> PRESET=${PRESET}  episode=${EPISODE_TIME_S}s x ${NUM_EPISODES}  n_action=${N_ACTION_STEPS}"

exec bash /home/rxn/lerobot/infer_smolvla_evo_rl_acp.sh robot
