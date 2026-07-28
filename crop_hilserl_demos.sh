#!/bin/bash
# HIL-SERL Step 3: crop + resize demo dataset
#
# ROI format in JSON: [top, left, height, width] on 128x128 images
# Preview frames: configs/hilserl_crop_previews/*.png
#
# Usage:
#   bash crop_hilserl_demos.sh
#   # edit configs/hilserl_so101_crop_params.json if needed, then rerun
#   CROP_PARAMS=configs/my_crop.json bash crop_hilserl_demos.sh
set -euo pipefail

PROJECT_ROOT="/home/rxn/lerobot"
CONDA_ENV="${CONDA_ENV:-lerobot}"
DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/hilserl_push_black_block_demos}"
DATA_REPO="${DATA_REPO:-my_rl/push_black_block_demos}"
CROP_PARAMS="${CROP_PARAMS:-${PROJECT_ROOT}/configs/hilserl_so101_crop_params.json}"
SINGLE_TASK="${SINGLE_TASK:-push the black block}"

source /home/rxn/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

if [ ! -f "${DATA_ROOT}/meta/info.json" ]; then
    echo "ERROR: demo dataset not found: ${DATA_ROOT}"
    echo "  Record first: bash record_hilserl_demos.sh"
    exit 1
fi

if [ ! -f "${CROP_PARAMS}" ]; then
    echo "ERROR: crop params not found: ${CROP_PARAMS}"
    exit 1
fi

echo "=============================="
echo "HIL-SERL Step 3: Crop demos"
echo "=============================="
echo "Input:  ${DATA_ROOT}"
echo "Task:   ${SINGLE_TASK}"
echo "ROI:    ${CROP_PARAMS}"
echo "=============================="

echo "Exporting preview frames..."
python3 - <<PY
from pathlib import Path
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

out = Path("${PROJECT_ROOT}/configs/hilserl_crop_previews")
out.mkdir(parents=True, exist_ok=True)
ds = LeRobotDataset("${DATA_REPO}", root="${DATA_ROOT}")
row = ds[0]
for k, v in row.items():
    if "image" in k:
        img = (v.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        p = out / f"{k.replace('.', '_')}.png"
        Image.fromarray(img).save(p)
        print(f"  {p}  ({img.shape[1]}x{img.shape[0]})")
PY

echo ""
echo "Crop params: ${CROP_PARAMS}"
cat "${CROP_PARAMS}"
echo ""
echo "Edit JSON if ROI is wrong: [top, left, height, width] per camera (on 128x128)."
echo "Preview images: ${PROJECT_ROOT}/configs/hilserl_crop_previews/"
echo ""

python -m lerobot.rl.crop_dataset_roi \
  --repo-id="${DATA_REPO}" \
  --root="${DATA_ROOT}" \
  --crop-params-path="${CROP_PARAMS}" \
  --task="${SINGLE_TASK}"

echo ""
echo "Done. Cropped dataset: ${DATA_ROOT}_cropped_resized"
echo "Next: bash relabel_hilserl_classifier.sh"
