#!/bin/bash
# HIL-SERL pipeline for: push the black block
#
# Usage:
#   bash hilserl_push_black_block_steps.sh           # print checklist
#   bash hilserl_push_black_block_steps.sh status    # check what exists
#   bash hilserl_push_black_block_steps.sh record-fail  # resume + more eps (failures with q)
#   bash hilserl_push_black_block_steps.sh crop
#   bash hilserl_push_black_block_steps.sh relabel
#   bash hilserl_push_black_block_steps.sh train-classifier
#
# Full order:
#   1) record success demos (done if 15/15)
#   2) record failure demos (q) via resume
#   3) crop
#   4) relabel
#   5) train classifier
#   6) RL learner + actor
set -euo pipefail

PROJECT_ROOT="/home/rxn/lerobot"
MODE="${1:-help}"

DEMO_ROOT="${DEMO_ROOT:-/home/rxn/datasets/hilserl_push_black_block_demos}"
CROPPED_ROOT="${CROPPED_ROOT:-${DEMO_ROOT}_cropped_resized}"
CLASSIFIER_DATA="${CLASSIFIER_DATA:-/home/rxn/datasets/hilserl_push_black_block_reward_classifier}"
CLASSIFIER_CKPT="${CLASSIFIER_CKPT:-${PROJECT_ROOT}/outputs/train/reward_classifier_so101_push_black_block/checkpoints/last/pretrained_model}"

_ep_count() {
  local root="$1"
  if [ -f "${root}/meta/info.json" ]; then
    python3 -c "import json; print(json.load(open('${root}/meta/info.json'))['total_episodes'])"
  else
    echo "missing"
  fi
}

status() {
  echo "=============================="
  echo "Push-black-block pipeline status"
  echo "=============================="
  echo "1 demos:      ${DEMO_ROOT}  → $(_ep_count "${DEMO_ROOT}") episodes"
  echo "2 cropped:    ${CROPPED_ROOT}  → $(_ep_count "${CROPPED_ROOT}") episodes"
  echo "3 classifier: ${CLASSIFIER_DATA}  → $(_ep_count "${CLASSIFIER_DATA}") episodes"
  if [ -d "${CLASSIFIER_CKPT}" ]; then
    echo "4 ckpt:       ${CLASSIFIER_CKPT}  → OK"
  else
    echo "4 ckpt:       ${CLASSIFIER_CKPT}  → missing"
  fi
  echo "=============================="
  echo ""
  echo "Suggested next action:"
  if [ ! -f "${DEMO_ROOT}/meta/info.json" ]; then
    echo "  bash record_hilserl_demos.sh"
  elif [ "$(_ep_count "${DEMO_ROOT}")" -lt 20 ]; then
    echo "  # add failure eps (press q), keep success ones:"
    echo "  NUM_EPISODES=25 bash record_hilserl_demos.sh"
  elif [ ! -f "${CROPPED_ROOT}/meta/info.json" ]; then
    echo "  bash crop_hilserl_demos.sh"
    echo "  # then edit configs/hilserl_so101_crop_params.json if ROI wrong and rerun"
  elif [ ! -f "${CLASSIFIER_DATA}/meta/info.json" ]; then
    echo "  OVERWRITE=1 bash relabel_hilserl_classifier.sh"
  elif [ ! -d "${CLASSIFIER_CKPT}" ]; then
    echo "  bash train_reward_classifier_so101.sh"
  else
    echo "  Classifier + push train config ready. Next:"
    echo "  # stop old pick-place learner first if still running"
    echo "  FRESH=1 bash train_hilserl_so101.sh learner"
    echo "  bash train_hilserl_so101.sh actor"
  fi
}

help() {
  cat <<'EOF'
HIL-SERL — push the black block

  Step 2  Record demos
            bash record_hilserl_demos.sh
            # success → press s | failure → press q
            # add more failures without wiping data:
            NUM_EPISODES=25 bash record_hilserl_demos.sh

  Step 3  Crop (ROI on 128x128)
            bash crop_hilserl_demos.sh
            # check configs/hilserl_crop_previews/*.png
            # edit configs/hilserl_so101_crop_params.json = [top,left,h,w]

  Step 4a Relabel for classifier (offline demos, uses s/q rewards)
            OVERWRITE=1 bash relabel_hilserl_classifier.sh

  Step 4b Train reward classifier
            bash train_reward_classifier_so101.sh

  Step 5  HIL-SERL RL  (config: configs/hilserl_so101_push_black_block_train.json)
            FRESH=1 bash train_hilserl_so101.sh learner   # terminal 1
            bash train_hilserl_so101.sh actor             # terminal 2

Commands:
  bash hilserl_push_black_block_steps.sh status
  bash hilserl_push_black_block_steps.sh record-fail
  bash hilserl_push_black_block_steps.sh crop
  bash hilserl_push_black_block_steps.sh relabel
  bash hilserl_push_black_block_steps.sh train-classifier
EOF
}

case "${MODE}" in
  help|-h|--help|"")
    help
    ;;
  status)
    status
    ;;
  record-fail)
    # Resume and allow more episodes for failures (default +10).
    NUM_EPISODES="${NUM_EPISODES:-25}" bash "${PROJECT_ROOT}/record_hilserl_demos.sh"
    ;;
  crop)
    bash "${PROJECT_ROOT}/crop_hilserl_demos.sh"
    ;;
  relabel)
    OVERWRITE="${OVERWRITE:-1}" bash "${PROJECT_ROOT}/relabel_hilserl_classifier.sh"
    ;;
  train-classifier)
    bash "${PROJECT_ROOT}/train_reward_classifier_so101.sh"
    ;;
  *)
    echo "Unknown mode: ${MODE}"
    help
    exit 1
    ;;
esac
