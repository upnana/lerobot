#!/bin/bash
# =============================================================================
# PI0.5 微调训练（SO101 + front/wrist）
# Usage:
#   conda activate lerobot          # pip install -e ".[pi,feetech]"
#   bash train_pi05.sh              # 默认 2 卡 DDP
#   NUM_GPUS=1 bash train_pi05.sh   # 单卡（显存紧张时用）
#   FRESH=1 bash train_pi05.sh      # 删除旧 output 重新训练
#   RESUME=1 bash train_pi05.sh     # 从 checkpoint 续训
#   BACKGROUND=1 bash train_pi05.sh # nohup 后台
#   tail -f logs/pi05_blue_block_yellow_tray.log
# =============================================================================
set -euo pipefail

PROJECT_ROOT="/home/rxn/lerobot"
MODELS_ROOT="${MODELS_ROOT:-/home/rxn/models}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/blue_block_yellow_tray}"
DATA_REPO="${DATA_REPO:-my_pick_place/blue_block_yellow_tray}"
BASE_MODEL="${BASE_MODEL:-${MODELS_ROOT}/pi05_base}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODELS_ROOT}/paligemma-3b-pt-224}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/train/pi05_blue_block_yellow_tray}"
CONDA_ENV="${CONDA_ENV:-lerobot}"
JOB_NAME="${JOB_NAME:-pi05_blue_block_yellow_tray}"

# ~90 ep ≈ 30k frames @ effective bs=4 → ~7500 steps/epoch；~8 epoch ≈ 60000 steps
# RTX 3090 24GB 双卡：bs=4/GPU 仍 OOM（GPU0 有桌面占用），默认 bs=2/GPU（effective=4）
NUM_GPUS="${NUM_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-2}"   # per-GPU；双卡 effective=4
NUM_WORKERS="${NUM_WORKERS:-2}"
STEPS="${STEPS:-60000}"
FREEZE_VISION="${FREEZE_VISION:-false}"  # OOM 时: FREEZE_VISION=1 bash train_pi05.sh
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-200}"
LOG_FILE="${LOG_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.log}"
PID_FILE="${PID_FILE:-${PROJECT_ROOT}/logs/${JOB_NAME}.pid}"
MIN_FREE_MIB="${MIN_FREE_MIB:-18000}"

if [ "${BACKGROUND:-0}" = "1" ] && [ -z "${TRAIN_PI05_BG:-}" ]; then
    mkdir -p "${PROJECT_ROOT}/logs"
    if [ -f "${PID_FILE}" ] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null; then
        echo "Already running PID $(cat "${PID_FILE}")"
        echo "Log: ${LOG_FILE}"
        exit 1
    fi
    echo "Starting nohup background training..."
    echo "  log: ${LOG_FILE}"
    nohup env TRAIN_PI05_BG=1 \
        FRESH="${FRESH:-0}" RESUME="${RESUME:-0}" \
        STEPS="${STEPS}" NUM_GPUS="${NUM_GPUS}" BATCH_SIZE="${BATCH_SIZE}" \
        SAVE_FREQ="${SAVE_FREQ}" LOG_FREQ="${LOG_FREQ}" NUM_WORKERS="${NUM_WORKERS}" \
        CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" \
        OUTPUT_DIR="${OUTPUT_DIR}" JOB_NAME="${JOB_NAME}" DATA_ROOT="${DATA_ROOT}" \
        bash "$0" > "${LOG_FILE}" 2>&1 &
    echo $! > "${PID_FILE}"
    echo "  pid: $(cat "${PID_FILE}")"
    echo "  tail -f ${LOG_FILE}"
    exit 0
fi

echo "=============================="
echo "PI0.5 Training"
echo "=============================="

source /home/rxn/miniconda3/etc/profile.d/conda.sh
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
    echo "ERROR: conda env '${CONDA_ENV}' not found."
    exit 1
fi
conda activate "${CONDA_ENV}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export ACCELERATE_DISABLE_RICH=1
export ACCELERATE_USE_META=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available — check driver / nvidia-smi"
print(f"CUDA OK: {torch.cuda.device_count()} GPU(s), using CUDA_VISIBLE_DEVICES={__import__('os').environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
PY

python -c "from lerobot.policies.pi05.modeling_pi05 import PI05Policy" 2>/dev/null || {
    echo "ERROR: PI0.5 not installed. Run: pip install -e \".[pi,feetech]\""
    exit 1
}

if [ ! -d "${DATA_ROOT}" ]; then
    echo "ERROR: dataset not found: ${DATA_ROOT}"
    exit 1
fi
if [ ! -f "${TOKENIZER_PATH}/config.json" ]; then
    echo "ERROR: tokenizer not found: ${TOKENIZER_PATH}"
    exit 1
fi

echo "Dataset:  ${DATA_ROOT} (${DATA_REPO})"
echo "Base:     ${BASE_MODEL}"
echo "Tokenizer:${TOKENIZER_PATH}"
echo "Output:   ${OUTPUT_DIR}"
echo "GPUs:     ${NUM_GPUS}"
echo "Batch:    ${BATCH_SIZE} per GPU (effective: $((BATCH_SIZE * NUM_GPUS)))"
echo "Steps:    ${STEPS}"
echo ""

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_COUNT=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
    else
        GPU_IDS=()
        for ((i=0; i<GPU_COUNT; i++)); do GPU_IDS+=("$i"); done
    fi
    if [ "${NUM_GPUS}" -gt "${#GPU_IDS[@]}" ]; then
        echo "ERROR: NUM_GPUS=${NUM_GPUS} but only ${#GPU_IDS[@]} GPU(s) available"
        exit 1
    fi
    echo "GPU status:"
    for ((i=0; i<NUM_GPUS; i++)); do
        GPU_IDX="${GPU_IDS[$i]}"
        FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
        TOTAL_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU_IDX}" | tr -d ' ')
        echo "  GPU ${GPU_IDX}: free/total ${FREE_MIB} / ${TOTAL_MIB} MiB"
        if [ "${FREE_MIB}" -lt "${MIN_FREE_MIB}" ]; then
            echo "ERROR: GPU ${GPU_IDX} needs >= ${MIN_FREE_MIB} MiB free (PI0.5 training)."
            echo "Try: NUM_GPUS=1 BATCH_SIZE=4 CUDA_VISIBLE_DEVICES=1 bash train_pi05.sh"
            exit 1
        fi
    done
    echo ""
fi

python - <<PY
import json
from pathlib import Path

root = Path("${DATA_ROOT}")
info = json.loads((root / "meta/info.json").read_text())
stats = json.loads((root / "meta/stats.json").read_text())
print(f"Episodes: {info['total_episodes']}, Frames: {info['total_frames']}, FPS: {info['fps']}")
assert info["total_episodes"] > 0 and info["total_frames"] > 0
assert "q01" in stats.get("action", {}), "Missing action quantile stats (q01). Re-convert dataset or run augment_dataset_quantile_stats.py"
bs = ${BATCH_SIZE} * ${NUM_GPUS}
epochs = ${STEPS} * bs / info["total_frames"]
print(f"Approx epochs at effective batch {bs}: {epochs:.1f}")
PY

# 把 pi05_base 里 preprocessor 的 tokenizer 指到本机路径（否则会拉 google/paligemma-3b-pt-224）
python - <<PY
import json
from pathlib import Path

base = Path("${BASE_MODEL}")
prep_path = base / "policy_preprocessor.json"
if not prep_path.is_file():
    raise SystemExit(f"Missing {prep_path}")

tokenizer = "${TOKENIZER_PATH}"
cfg = json.loads(prep_path.read_text())
for step in cfg.get("steps", []):
    if step.get("registry_name") == "tokenizer_processor":
        old = step["config"].get("tokenizer_name")
        if old != tokenizer:
            step["config"]["tokenizer_name"] = tokenizer
            prep_path.write_text(json.dumps(cfg, indent=2) + "\n")
            print(f"Patched tokenizer_name: {old} -> {tokenizer}")
        else:
            print(f"tokenizer_name already local: {tokenizer}")
        break
else:
    raise SystemExit("tokenizer_processor step not found in policy_preprocessor.json")
PY

if [ ! -f "${BASE_MODEL}/model.safetensors" ]; then
    echo "ERROR: base model not found: ${BASE_MODEL}/model.safetensors"
    echo "Download: modelscope download --model lerobot/pi05_base --local_dir ${MODELS_ROOT}/pi05_base"
    exit 1
fi
echo "Base model:  ${BASE_MODEL}"
echo "Tokenizer:   ${TOKENIZER_PATH}"

RESUME="${RESUME:-0}"
CONFIG_PATH=""
if [ "${FRESH:-0}" = "1" ] && [ -d "${OUTPUT_DIR}" ]; then
    echo "FRESH=1: removing ${OUTPUT_DIR}"
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
    echo "  FRESH=1 bash train_pi05.sh"
    echo "  RESUME=1 bash train_pi05.sh"
    exit 1
fi
echo ""

ACCEL_ARGS=(
  --num_processes="${NUM_GPUS}"
  --num_machines=1
  --mixed_precision=bf16
  --dynamo_backend=no
)
if [ "${NUM_GPUS}" -gt 1 ]; then
  ACCEL_ARGS+=(--multi_gpu --main_process_port="${MAIN_PROCESS_PORT:-29502}")
fi

accelerate launch "${ACCEL_ARGS[@]}" \
  "$(which lerobot-train)" \
  --policy.type=pi05 \
  --policy.pretrained_path="${BASE_MODEL}" \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.tokenizer_path="${TOKENIZER_PATH}" \
  --policy.scheduler_decay_steps="${STEPS}" \
  $([ "${FREEZE_VISION}" = "1" ] || [ "${FREEZE_VISION}" = "true" ] && echo "--policy.freeze_vision_encoder=true") \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id="${DATA_REPO}" \
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
echo "Infer: CHECKPOINT=${OUTPUT_DIR}/checkpoints/last/pretrained_model bash infer_pi05.sh offline"
echo "=============================="
