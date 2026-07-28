#!/bin/bash
# Evo-RL ACP policy training (SmolVLA) on snacks/bowls dataset with value-infer tags
#
# Usage:
#   bash /home/rxn/lerobot/train-policy-evo-rl-acp.sh
#   NUM_GPUS=2 BATCH_SIZE=4 bash ...
#   FRESH=1 bash ...
#   RESUME=1 bash ...
#   BACKGROUND=1 bash ...
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_snacks_tray_stack_bowls_plate}"
DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_snacks_tray_stack_bowls_plate}"
BASE_MODEL="${BASE_MODEL:-/home/rxn/.cache/modelscope/models/lerobot--smolvla_base/snapshots/master}"
VLM_MODEL="${VLM_MODEL:-/home/rxn/.cache/modelscope/models/HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/master}"
RUN_NAME="${RUN_NAME:-smolvla_acp_snacks_bowls_v1}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/${RUN_NAME}}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
TAG="${TAG:-snacks_bowls_v1}"
INDICATOR_FIELD="${INDICATOR_FIELD:-complementary_info.acp_indicator_${TAG}}"

# Camera names from Evo-RL collection -> SmolVLA camera1..N (override via RENAME_MAP=...)
if [[ -z "${RENAME_MAP:-}" ]]; then
  RENAME_MAP='{"observation.images.left_top_left":"observation.images.camera1","observation.images.right_top_right":"observation.images.camera2","observation.images.left_wrist":"observation.images.camera3","observation.images.right_wrist":"observation.images.camera4"}'
fi
# Force 4 cams into policy (smolvla_base only ships camera1-3; without this right_wrist is dropped).
# Set FORCE_4_CAMS=0 for 3-cam runs that match smolvla_base.
FORCE_4_CAMS="${FORCE_4_CAMS:-1}"
INPUT_FEATURES_4CAM='{"observation.state":{"type":"STATE","shape":[12]},"observation.images.camera1":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera2":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera3":{"type":"VISUAL","shape":[3,256,256]},"observation.images.camera4":{"type":"VISUAL","shape":[3,256,256]}}'

NUM_GPUS="${NUM_GPUS:-2}"
GPUS="${GPUS:-0,1}"
BATCH_SIZE="${BATCH_SIZE:-4}"   # per-GPU; 4 cams — OOM then try 2
NUM_WORKERS="${NUM_WORKERS:-4}"
# ~75693 frames @ eff bs=8 → ~9462 steps/epoch; 170000 ≈ 18 ep
STEPS="${STEPS:-170000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-200}"
TOLERANCE_S="${TOLERANCE_S:-0.05}"
DROPOUT="${DROPOUT:-0.3}"
MIN_FREE_MIB="${MIN_FREE_MIB:-8000}"
LOG_FILE="${LOG_FILE:-/home/rxn/lerobot/logs/${JOB_NAME}.log}"
PID_FILE="${PID_FILE:-/home/rxn/lerobot/logs/${JOB_NAME}.pid}"

if [ "${BACKGROUND:-0}" = "1" ] && [ -z "${TRAIN_ACP_BG:-}" ]; then
  mkdir -p /home/rxn/lerobot/logs
  if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
    echo "Already running PID $(cat "${PID_FILE}")"
    echo "Log: ${LOG_FILE}"
    exit 1
  fi
  echo "Starting background ACP policy train..."
  nohup env TRAIN_ACP_BG=1 \
    FRESH="${FRESH:-0}" RESUME="${RESUME:-0}" \
    STEPS="${STEPS}" NUM_GPUS="${NUM_GPUS}" BATCH_SIZE="${BATCH_SIZE}" \
    SAVE_FREQ="${SAVE_FREQ}" LOG_FREQ="${LOG_FREQ}" NUM_WORKERS="${NUM_WORKERS}" \
    GPUS="${GPUS}" OUTPUT_DIR="${OUTPUT_DIR}" JOB_NAME="${JOB_NAME}" \
    DATA_ROOT="${DATA_ROOT}" DATA_REPO="${DATA_REPO}" TAG="${TAG}" \
    FORCE_4_CAMS="${FORCE_4_CAMS:-1}" RUN_NAME="${RUN_NAME}" \
    RENAME_MAP="${RENAME_MAP}" \
    bash "$0" > "${LOG_FILE}" 2>&1 &
  echo $! > "${PID_FILE}"
  echo "  pid: $(cat "${PID_FILE}")"
  echo "  tail -f ${LOG_FILE}"
  exit 0
fi

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ACCELERATE_DISABLE_RICH=1

for path in "${DATA_ROOT}" "${BASE_MODEL}" "${VLM_MODEL}"; do
  [ -d "${path}" ] || { echo "ERROR: not found: ${path}"; exit 1; }
done
[ -f "${DATA_ROOT}/meta/info.json" ] || { echo "ERROR: missing info.json"; exit 1; }

python - <<PY
import json
from pathlib import Path
info = json.loads(Path("${DATA_ROOT}/meta/info.json").read_text())
feats = info["features"]
need = "${INDICATOR_FIELD}"
if need not in feats:
    raise SystemExit(f"ERROR: missing ACP field {need}. Run value-infer first.")
print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}")
print(f"ACP field OK: {need} dtype={feats[need].get('dtype')}")
bs = ${BATCH_SIZE} * ${NUM_GPUS}
print(f"Approx epochs @ ${STEPS}: {${STEPS} * bs / info['total_frames']:.1f}")
PY

# Point SmolVLA preprocessor tokenizer to local VLM
python - <<PY
import json
from pathlib import Path
vlm = "${VLM_MODEL}"
prep_path = Path("${BASE_MODEL}") / "policy_preprocessor.json"
cfg = json.loads(prep_path.read_text())
for step in cfg.get("steps", []):
    if step.get("registry_name") == "tokenizer_processor":
        if step["config"].get("tokenizer_name") != vlm:
            step["config"]["tokenizer_name"] = vlm
            prep_path.write_text(json.dumps(cfg, indent=2) + "\n")
            print(f"Patched tokenizer_name -> {vlm}")
        break
PY

RESUME="${RESUME:-0}"
CONFIG_PATH=""
if [ "${FRESH:-0}" = "1" ] && [ -d "${OUTPUT_DIR}" ]; then
  echo "FRESH=1: removing ${OUTPUT_DIR}"
  rm -rf "${OUTPUT_DIR}"
elif [ "${RESUME}" = "1" ]; then
  CONFIG_PATH="${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
  [ -f "${CONFIG_PATH}" ] || { echo "ERROR: missing ${CONFIG_PATH}"; exit 1; }
  echo "Resume from: ${CONFIG_PATH}"
elif [ -d "${OUTPUT_DIR}" ]; then
  echo "ERROR: ${OUTPUT_DIR} exists. Use FRESH=1 or RESUME=1"
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  IFS=',' read -ra GPU_IDS <<< "${GPUS}"
  for ((i=0; i<NUM_GPUS; i++)); do
    GPU_IDX="${GPU_IDS[$i]}"
    FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
    echo "  GPU ${GPU_IDX}: free ${FREE_MIB} MiB"
    if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
      echo "ERROR: GPU ${GPU_IDX} free memory too low (< ${MIN_FREE_MIB} MiB)"
      exit 1
    fi
  done
fi

echo "=============================="
echo "Evo-RL ACP policy train (${RUN_NAME})"
echo "=============================="
echo "Env:       ${CONDA_ENV}"
echo "GPUs:      ${GPUS} (num=${NUM_GPUS})"
echo "Dataset:   ${DATA_ROOT}"
echo "Base:      ${BASE_MODEL}"
echo "Output:    ${OUTPUT_DIR}"
echo "Indicator: ${INDICATOR_FIELD}  dropout=${DROPOUT}"
echo "Steps:     ${STEPS}  batch/gpu=${BATCH_SIZE}  effective=$((BATCH_SIZE * NUM_GPUS))"
echo "=============================="

ACCEL_ARGS=(--num_processes="${NUM_GPUS}" --num_machines=1 --mixed_precision=bf16 --dynamo_backend=no)
[ "${NUM_GPUS}" -gt 1 ] && ACCEL_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT:-29507}")

POLICY_ARGS=(
  --policy.device=cuda
  --policy.vlm_model_name="${VLM_MODEL}"
  --policy.load_vlm_weights=false
  --policy.empty_cameras=0
  --policy.push_to_hub=false
  --policy.scheduler_decay_steps="${STEPS}"
)
if [ "${FORCE_4_CAMS}" = "1" ] && [ "${RESUME}" != "1" ]; then
  POLICY_ARGS+=(--policy.input_features="${INPUT_FEATURES_4CAM}")
  echo "Forcing 4-cam input_features (camera1-4)"
fi
if [ "${RESUME}" != "1" ]; then
  POLICY_ARGS=(--policy.path="${BASE_MODEL}" "${POLICY_ARGS[@]}")
fi

TRAIN_EXTRA=()
if [ "${RESUME}" = "1" ]; then
  TRAIN_EXTRA+=(--resume=true --config_path="${CONFIG_PATH}")
fi

accelerate launch "${ACCEL_ARGS[@]}" \
  "$(which lerobot-train)" \
  "${POLICY_ARGS[@]}" \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id="${DATA_REPO}" \
  --tolerance_s="${TOLERANCE_S}" \
  --rename_map="${RENAME_MAP}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq="${SAVE_FREQ}" \
  --log_freq="${LOG_FREQ}" \
  --eval_freq=-1 \
  --seed=1000 \
  --acp.enable=true \
  --acp.indicator_field="${INDICATOR_FIELD}" \
  --acp.indicator_dropout_prob="${DROPOUT}" \
  "${TRAIN_EXTRA[@]}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --wandb.enable=false

echo "=============================="
echo "Training finished: ${OUTPUT_DIR}"
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/last/pretrained_model"
echo "=============================="
