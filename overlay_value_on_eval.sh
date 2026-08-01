#!/bin/bash
# =============================================================================
# 对推理/HITL 录制的 eval 数据集：跑 value-infer + 把 value 曲线叠到视频上
#
# 依赖: bowls_tray_3cam_v2 value checkpoint（已训好）
#
# Usage:
#   # 指向一次 infer 保存目录（含 meta/info.json）
#   bash overlay_value_on_eval.sh /path/to/eval_run_dir
#
#   # 只叠某几条 episode
#   VIZ_EPISODES=0,1 bash overlay_value_on_eval.sh /path/to/eval_run_dir
#
#   # 多相机拼图
#   VIZ_VIDEO_KEYS="observation.images.right_top_right,observation.images.left_wrist,observation.images.right_wrist" \
#     bash overlay_value_on_eval.sh /path/to/eval_run_dir
#
# 输出:
#   <eval_dir>_value_viz/viz/*.mp4   （画面上同步 value 曲线 + 当前帧数值）
# =============================================================================
set -euo pipefail

EVAL_ROOT="${1:-}"
if [ -z "${EVAL_ROOT}" ] || [ ! -f "${EVAL_ROOT}/meta/info.json" ]; then
  echo "Usage: bash overlay_value_on_eval.sh /path/to/eval_dataset_root"
  echo "ERROR: missing dataset at '${EVAL_ROOT}'"
  exit 1
fi
EVAL_ROOT="$(readlink -f "${EVAL_ROOT}")"

PROJECT_ROOT="${PROJECT_ROOT:-/home/rxn/lerobot/Evo-RL}"
CONDA_ENV="${CONDA_ENV:-evo-rl}"
VALUE_RUN="${VALUE_RUN:-bowls_tray_3cam_v2}"
TAG="${TAG:-bowls_tray_3cam_v2}"
# training output dir (resolves checkpoints/last/pretrained_model internally)
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${PROJECT_ROOT}/outputs/value_train/${VALUE_RUN}}"

OUT_DIR="${OUT_DIR:-${EVAL_ROOT}_value_viz}"
DATA_REPO="${DATA_REPO:-eval/$(basename "${EVAL_ROOT}")}"
NUM_GPUS="${NUM_GPUS:-1}"
GPUS="${GPUS:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
VIZ_EPISODES="${VIZ_EPISODES:-all}"
VIZ_VIDEO_KEY="${VIZ_VIDEO_KEY:-observation.images.right_top_right}"
VIZ_VIDEO_KEYS="${VIZ_VIDEO_KEYS:-}"
VCODEC="${VCODEC:-libsvtav1}"
N_STEP="${N_STEP:-50}"
POSITIVE_RATIO="${POSITIVE_RATIO:-0.3}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
cd "${PROJECT_ROOT}"

export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export CUDA_VISIBLE_DEVICES="${GPUS}"

if [ ! -d "${CHECKPOINT_PATH}/checkpoints/last" ] && [ ! -f "${CHECKPOINT_PATH}/config.json" ]; then
  echo "ERROR: value checkpoint not found under ${CHECKPOINT_PATH}"
  exit 1
fi

mkdir -p "${OUT_DIR}"

echo "=============================="
echo "Value overlay on eval videos"
echo "=============================="
echo "Eval:       ${EVAL_ROOT}"
echo "Value ckpt: ${CHECKPOINT_PATH}"
echo "Tag:        ${TAG}"
echo "Viz out:    ${OUT_DIR}/viz"
echo "Episodes:   ${VIZ_EPISODES}"
echo "Video:      ${VIZ_VIDEO_KEYS:-${VIZ_VIDEO_KEY}}"
echo "=============================="

INFER_ARGS=(
  --dataset.repo_id="${DATA_REPO}"
  --dataset.root="${EVAL_ROOT}"
  --inference.checkpoint_path="${CHECKPOINT_PATH}"
  --runtime.device=cuda
  --runtime.batch_size="${BATCH_SIZE}"
  --runtime.num_workers=4
  --acp.enable=true
  --acp.n_step="${N_STEP}"
  --acp.positive_ratio="${POSITIVE_RATIO}"
  --acp.value_field="complementary_info.value_${TAG}"
  --acp.advantage_field="complementary_info.advantage_${TAG}"
  --acp.indicator_field="complementary_info.acp_indicator_${TAG}"
  --output_dir="${OUT_DIR}"
  --job_name="overlay_${TAG}"
  --viz.enable=true
  --viz.episodes="${VIZ_EPISODES}"
  --viz.overwrite=true
  --viz.vcodec="${VCODEC}"
  --viz.smooth_window=5
)

if [ -n "${VIZ_VIDEO_KEYS}" ]; then
  INFER_ARGS+=(--viz.video_keys="${VIZ_VIDEO_KEYS}")
else
  INFER_ARGS+=(--viz.video_key="${VIZ_VIDEO_KEY}")
fi

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

echo "=============================="
echo "Done. Overlay videos under:"
find "${OUT_DIR}" -name '*.mp4' -printf '%p (%k KB)\n' | head -30
echo "=============================="
