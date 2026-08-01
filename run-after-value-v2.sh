#!/bin/bash
# Wait for snacks_bowls_v2 value-train to finish, then value-infer -> ACP policy train.
# Designed to run unattended overnight (no prompts).
set -euo pipefail

LOG_DIR="${LOG_DIR:-/home/rxn/lerobot/logs}"
mkdir -p "${LOG_DIR}"
PIPE_LOG="${PIPE_LOG:-${LOG_DIR}/pipeline_after_value_v2.log}"
VALUE_PID_FILE="${VALUE_PID_FILE:-${LOG_DIR}/value_train_snacks_bowls_v2.pid}"
VALUE_LOG="${VALUE_LOG:-${LOG_DIR}/value_train_snacks_bowls_v2.log}"
VALUE_RUN="${VALUE_RUN:-snacks_bowls_v2}"
VALUE_OUT="${VALUE_OUT:-/home/rxn/lerobot/Evo-RL/outputs/value_train/${VALUE_RUN}}"
TAG="${TAG:-snacks_bowls_v2}"
POLICY_RUN="${POLICY_RUN:-smolvla_acp_snacks_bowls_v2}"
# ~197k frames @ eff bs=8 → ~25k steps/epoch; 250k ≈ ~10 epochs
POLICY_STEPS="${POLICY_STEPS:-250000}"

exec > >(tee -a "${PIPE_LOG}") 2>&1

echo "========================================"
echo "[$(date '+%F %T')] pipeline start (after value ${VALUE_RUN})"
echo "========================================"

wait_value_train() {
  local pid=""
  if [[ -f "${VALUE_PID_FILE}" ]]; then
    pid="$(cat "${VALUE_PID_FILE}")"
  fi

  echo "[$(date '+%F %T')] waiting for value-train pid=${pid:-unknown}"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    while kill -0 "${pid}" 2>/dev/null; do
      # progress crumb every ~10 min
      tail -n 1 "${VALUE_LOG}" 2>/dev/null || true
      sleep 60
    done
    echo "[$(date '+%F %T')] value-train pid ${pid} exited"
  else
    echo "[$(date '+%F %T')] value-train pid not running; checking completion markers"
  fi

  # Wait until training log says finished (handles race / orphan children)
  for i in $(seq 1 180); do
    if grep -q "End of value training" "${VALUE_LOG}" 2>/dev/null; then
      echo "[$(date '+%F %T')] found 'End of value training'"
      break
    fi
    # Also accept final checkpoint present
    if [[ -L "${VALUE_OUT}/checkpoints/last" ]]; then
      target="$(readlink "${VALUE_OUT}/checkpoints/last")"
      if [[ "${target}" == "025000" ]] || [[ -d "${VALUE_OUT}/checkpoints/025000" ]]; then
        echo "[$(date '+%F %T')] found checkpoint 025000"
        break
      fi
    fi
    # If no process and log stalled long after start, fail
    if ! pgrep -f "lerobot-value-train|outputs/value_train/${VALUE_RUN}" >/dev/null 2>&1; then
      if grep -q "End of value training\|Error\|Traceback" "${VALUE_LOG}" 2>/dev/null; then
        break
      fi
    fi
    sleep 60
  done

  if ! grep -q "End of value training" "${VALUE_LOG}" 2>/dev/null; then
    echo "[$(date '+%F %T')] ERROR: value-train did not finish cleanly"
    tail -n 80 "${VALUE_LOG}" || true
    exit 1
  fi

  # Ensure GPUs free a bit (children may linger briefly)
  sleep 30
  for i in $(seq 1 30); do
    if ! pgrep -af "accelerate|lerobot-value-train" | grep -q "${VALUE_RUN}"; then
      break
    fi
    echo "[$(date '+%F %T')] waiting for value-train GPU processes to exit..."
    sleep 20
  done
}

run_value_infer() {
  echo "========================================"
  echo "[$(date '+%F %T')] starting value-infer (${VALUE_RUN} / tag=${TAG})"
  echo "========================================"
  RUN_NAME="${VALUE_RUN}" \
  TAG="${TAG}" \
  NUM_GPUS=2 GPUS=0,1 BATCH_SIZE=8 \
  bash /home/rxn/lerobot/value-infer-evo-rl.sh
  echo "[$(date '+%F %T')] value-infer done"
}

run_policy_train() {
  echo "========================================"
  echo "[$(date '+%F %T')] starting policy train (${POLICY_RUN} / tag=${TAG} / 4 cams)"
  echo "========================================"
  # Wait for free VRAM
  for i in $(seq 1 60); do
    free0=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 | tr -d ' ')
    free1=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1 | tr -d ' ')
    echo "GPU free: ${free0} / ${free1} MiB"
    if [[ "${free0}" -ge 8000 && "${free1}" -ge 8000 ]]; then
      break
    fi
    sleep 30
  done

  FRESH=1 \
  FORCE_4_CAMS=1 \
  RUN_NAME="${POLICY_RUN}" \
  TAG="${TAG}" \
  STEPS="${POLICY_STEPS}" \
  NUM_GPUS=2 GPUS=0,1 BATCH_SIZE=4 \
  bash /home/rxn/lerobot/train-policy-evo-rl-acp.sh
  echo "[$(date '+%F %T')] policy train done"
}

wait_value_train
run_value_infer
run_policy_train

echo "========================================"
echo "[$(date '+%F %T')] pipeline ALL DONE"
echo "value ckpt:  ${VALUE_OUT}/checkpoints/last"
echo "policy out:  /home/rxn/lerobot/Evo-RL/outputs/train/${POLICY_RUN}"
echo "pipe log:    ${PIPE_LOG}"
echo "========================================"
