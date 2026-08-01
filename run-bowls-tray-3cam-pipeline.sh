#!/bin/bash
# bowls+lipstick+tissue: value-train -> value-infer -> ACP SmolVLA policy
# 3 cameras (drop observation.images.left_top_left)
set -euo pipefail

LOG_DIR="${LOG_DIR:-/home/rxn/lerobot/logs}"
mkdir -p "${LOG_DIR}"

DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue}"
DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_bowls_stack_lipstick_tissue}"

VALUE_RUN="${VALUE_RUN:-bowls_tray_3cam}"
TAG="${TAG:-bowls_tray_3cam}"
POLICY_RUN="${POLICY_RUN:-smolvla_acp_bowls_tray_3cam}"

VALUE_OUT="${VALUE_OUT:-/home/rxn/lerobot/Evo-RL/outputs/value_train/${VALUE_RUN}}"
VALUE_LOG="${VALUE_LOG:-${LOG_DIR}/value_train_${VALUE_RUN}.log}"
VALUE_PID_FILE="${VALUE_PID_FILE:-${LOG_DIR}/value_train_${VALUE_RUN}.pid}"
PIPE_LOG="${PIPE_LOG:-${LOG_DIR}/pipeline_${VALUE_RUN}.log}"

# 3 cams: top-right + both wrists (no left_top_left)
if [[ -z "${CAMERA_FEATURES:-}" ]]; then
  CAMERA_FEATURES='["observation.images.right_top_right","observation.images.left_wrist","observation.images.right_wrist"]'
fi
if [[ -z "${RENAME_MAP:-}" ]]; then
  RENAME_MAP='{"observation.images.right_top_right":"observation.images.camera1","observation.images.left_wrist":"observation.images.camera2","observation.images.right_wrist":"observation.images.camera3"}'
fi

VALUE_STEPS="${VALUE_STEPS:-25000}"
VALUE_SAVE_FREQ="${VALUE_SAVE_FREQ:-5000}"
# ~158k frames @ eff bs=8 → ~20k steps/epoch; 200k ≈ ~10 epochs
POLICY_STEPS="${POLICY_STEPS:-200000}"

NUM_GPUS="${NUM_GPUS:-2}"
GPUS="${GPUS:-0,1}"

exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "========================================"
echo "[$(date '+%F %T')] pipeline start (${VALUE_RUN})"
echo "Dataset:  ${DATA_ROOT}"
echo "Cameras:  ${CAMERA_FEATURES}"
echo "Rename:   ${RENAME_MAP}"
echo "Value:    ${VALUE_STEPS} steps -> ${VALUE_OUT}"
echo "Policy:   ${POLICY_STEPS} steps -> smolvla_acp / tag=${TAG}"
echo "========================================"

run_value_train() {
  echo "========================================"
  echo "[$(date '+%F %T')] starting value-train"
  echo "========================================"
  DATA_ROOT="${DATA_ROOT}" DATA_REPO="${DATA_REPO}" \
  RUN_NAME="${VALUE_RUN}" \
  STEPS="${VALUE_STEPS}" SAVE_FREQ="${VALUE_SAVE_FREQ}" \
  CAMERA_FEATURES="${CAMERA_FEATURES}" \
  NUM_GPUS="${NUM_GPUS}" GPUS="${GPUS}" BATCH_SIZE=4 \
  bash /home/rxn/lerobot/train-value-evo-rl.sh
  echo "[$(date '+%F %T')] value-train done"
}

run_value_infer() {
  echo "========================================"
  echo "[$(date '+%F %T')] starting value-infer (tag=${TAG})"
  echo "========================================"
  DATA_ROOT="${DATA_ROOT}" DATA_REPO="${DATA_REPO}" \
  RUN_NAME="${VALUE_RUN}" TAG="${TAG}" \
  NUM_GPUS="${NUM_GPUS}" GPUS="${GPUS}" BATCH_SIZE=8 \
  bash /home/rxn/lerobot/value-infer-evo-rl.sh
  echo "[$(date '+%F %T')] value-infer done"
}

run_policy_train() {
  echo "========================================"
  echo "[$(date '+%F %T')] starting policy train (${POLICY_RUN}, 3 cams)"
  echo "========================================"
  for i in $(seq 1 60); do
    free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
    free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    echo "GPU free: ${free0} / ${free1} MiB"
    if [[ "${free0}" -ge 8000 && "${free1}" -ge 8000 ]]; then
      break
    fi
    sleep 30
  done

  FRESH=1 FORCE_4_CAMS=0 \
  DATA_ROOT="${DATA_ROOT}" DATA_REPO="${DATA_REPO}" \
  RUN_NAME="${POLICY_RUN}" TAG="${TAG}" \
  RENAME_MAP="${RENAME_MAP}" \
  STEPS="${POLICY_STEPS}" \
  NUM_GPUS="${NUM_GPUS}" GPUS="${GPUS}" BATCH_SIZE=4 \
  bash /home/rxn/lerobot/train-policy-evo-rl-acp.sh
  echo "[$(date '+%F %T')] policy train done"
}

# If value already finished, skip retrain
if [[ -L "${VALUE_OUT}/checkpoints/last" ]] && grep -q "End of value training" "${VALUE_LOG}" 2>/dev/null; then
  echo "[$(date '+%F %T')] value-train already complete; skipping"
else
  # also skip if launched separately and still running
  if [[ -f "${VALUE_PID_FILE}" ]] && kill -0 "$(cat "${VALUE_PID_FILE}")" 2>/dev/null; then
    echo "[$(date '+%F %T')] value-train already running pid=$(cat "${VALUE_PID_FILE}"); waiting..."
    while kill -0 "$(cat "${VALUE_PID_FILE}")" 2>/dev/null; do sleep 60; done
    if ! grep -q "End of value training" "${VALUE_LOG}" 2>/dev/null; then
      echo "ERROR: value-train did not finish cleanly"; exit 1
    fi
  else
    run_value_train
  fi
fi

run_value_infer
run_policy_train

echo "========================================"
echo "[$(date '+%F %T')] pipeline ALL DONE"
echo "value ckpt:  ${VALUE_OUT}/checkpoints/last"
echo "policy out:  /home/rxn/lerobot/Evo-RL/outputs/train/${POLICY_RUN}"
echo "pipe log:    ${PIPE_LOG}"
echo "========================================"
