#!/bin/bash
# 合并 yellow_white + yellow_white_base2 → yellow_white_merged
# Usage: bash merge_yellow_white.sh
set -euo pipefail

PROJECT_ROOT="/home/rxn/lerobot"
SRC1="${SRC1:-/home/rxn/datasets/yellow_white}"
SRC2="${SRC2:-/home/rxn/datasets/yellow_white_base2}"
OUT="${OUT:-/home/rxn/datasets/yellow_white_merged}"
REPO1="${REPO1:-my_pick_place/yellow_white}"
REPO2="${REPO2:-my_pick_place/yellow_white_base2}"
OUT_REPO="${OUT_REPO:-my_pick_place/yellow_white_merged}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-lerobot}"

unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE
export HF_HUB_OFFLINE=0

for d in "${SRC1}" "${SRC2}"; do
    if [ ! -f "${d}/meta/info.json" ]; then
        echo "ERROR: missing dataset: ${d}"
        exit 1
    fi
done

echo "Merging:"
echo "  ${SRC1}"
echo "  ${SRC2}"
echo "→ ${OUT}"

python - <<PY
from pathlib import Path
import shutil
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

out = Path("${OUT}")
if out.exists():
    shutil.rmtree(out)

sources = [
    ("${REPO1}", "${SRC1}"),
    ("${REPO2}", "${SRC2}"),
]
datasets = [LeRobotDataset(r, root=p) for r, p in sources]
for ds in datasets:
    print(f"  {ds.root}: {ds.meta.total_episodes} ep, {ds.meta.total_frames} frames")

merged = merge_datasets(datasets, output_repo_id="${OUT_REPO}", output_dir=out)
print(f"Done: {out} — {merged.meta.total_episodes} ep, {merged.meta.total_frames} frames")
PY

echo ""
echo "Train PI0.5:"
echo "  DATA_ROOT=${OUT} DATA_REPO=${OUT_REPO} OUTPUT_DIR=${PROJECT_ROOT}/outputs/train/pi05_yellow_white_merged \\"
echo "  FRESH=1 CUDA_VISIBLE_DEVICES=1 BACKGROUND=1 bash train_pi05.sh"
