#!/bin/bash
set -e

echo "=============================="
echo "🚀 GR00T + LeRobot START"
echo "=============================="

# =========================
# 0. 路径配置
# =========================
PROJECT_ROOT="/home/rxn/lerobot"
DATA_ROOT="/home/rxn/datasets/20260701_145251"
DATA_REPO="my_pick_place/so101_test10"
BASE_MODEL="${PROJECT_ROOT}/src/lerobot/GR00T-basemodel"
EAGLE_ASSETS="${PROJECT_ROOT}/src/lerobot/eagle"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/train/groot_final"
CONDA_ENV="groot"

# =========================
# 1. 激活 conda 环境
# =========================
echo "[1/7] Activating conda env: ${CONDA_ENV}..."
source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

# =========================
# 2. 网络 / HF 配置
# =========================
echo "[2/7] Fix HF offline mode..."
unset HF_HUB_OFFLINE
export HF_HUB_OFFLINE=0
export HF_HUB_DISABLE_TELEMETRY=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export ACCELERATE_DISABLE_RICH=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# =========================
# 3. 安装关键依赖
# =========================
echo "[3/7] Installing dependencies..."
pip install -U --no-cache-dir \
    huggingface_hub==0.34.4 \
    fsspec==2025.9.0 \
    packaging==25.0 \
    "decord>=0.6.0,<1.0.0"

# flash-attn 可选：未安装时使用 fallback attention
python -c "import flash_attn" 2>/dev/null || \
    echo "[WARN] flash-attn not installed, using fallback attention"

# =========================
# 4. 环境检查
# =========================
echo "[4/7] Environment check..."
python - <<'PY'
import os
import torch
import transformers
import huggingface_hub

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
print("Transformers:", transformers.__version__)
print("HF Hub:", huggingface_hub.__version__)
print("HF_OFFLINE:", os.getenv("HF_HUB_OFFLINE"))
print("✅ ENV OK")
PY

# =========================
# 5. 资源检查
# =========================
echo "[5/7] Resource check..."

if [ ! -d "${DATA_ROOT}" ]; then
    echo "❌ dataset not found: ${DATA_ROOT}"
    exit 1
fi

if [ ! -d "${BASE_MODEL}" ]; then
    echo "❌ base model not found: ${BASE_MODEL}"
    exit 1
fi

if [ ! -d "${EAGLE_ASSETS}" ]; then
    echo "❌ eagle assets not found: ${EAGLE_ASSETS}"
    exit 1
fi

# 刷新 eagle cache（仅复制 vendor 文件，不删除整个 HF cache）
EAGLE_CACHE="${HOME}/.cache/huggingface/lerobot/lerobot/eagle2hg-processor-groot-n1p5"
mkdir -p "${EAGLE_CACHE}"
cp -r "${PROJECT_ROOT}/src/lerobot/policies/groot/eagle2_hg_model/." "${EAGLE_CACHE}/"

echo "✅ dataset OK"
echo "✅ base model OK"
echo "✅ eagle assets OK"

# =========================
# 6. 启动训练
# =========================
echo "[6/7] Starting GR00T training..."

lerobot-train \
  --dataset.root="${DATA_ROOT}" \
  --dataset.repo_id="${DATA_REPO}" \
  --policy.type=groot \
  --policy.device=cuda \
  --policy.base_model_path="${BASE_MODEL}" \
  --policy.tokenizer_assets_repo="lerobot/eagle2hg-processor-groot-n1p5" \
  --policy.tune_llm=false \
  --policy.tune_visual=false \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=false \
  --policy.use_bf16=true \
  --policy.video_backend=decord \
  --batch_size=1 \
  --num_workers=0 \
  --steps=100000 \
  --save_freq=20000 \
  --log_freq=200 \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=groot_final \
  --wandb.enable=false \
  --policy.push_to_hub=false

echo "=============================="
echo "🎉 DONE"
echo "=============================="
