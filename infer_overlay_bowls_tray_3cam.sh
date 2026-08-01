#!/bin/bash
# =============================================================================
# 真机推理 + value 曲线叠视频（一条命令跑完）
#
# 流程:
#   1) policy 控制机械臂并录制 eval 数据集
#   2) 用 bowls_tray_3cam_v2 value 模型打分
#   3) 导出带同步 value 曲线的 mp4
#
# Usage:
#   bash /home/rxn/lerobot/infer_overlay_bowls_tray_3cam.sh            # 默认 80000
#   bash /home/rxn/lerobot/infer_overlay_bowls_tray_3cam.sh 50000
#   bash /home/rxn/lerobot/infer_overlay_bowls_tray_3cam.sh 100000
#   bash /home/rxn/lerobot/infer_overlay_bowls_tray_3cam.sh last
#
#   NUM_EPISODES=3 EPISODE_TIME_S=300 bash ... 80000
#   PRESET=n10 bash ... 80000                    # Advantage: positive
#   OVERLAY_VALUE=0 bash ... 80000               # 只要推理、不叠曲线
#
# 叠好后视频在:
#   Evo-RL/outputs/eval/smolvla_acp_bowls_tray_3cam_v2/<run>_value_viz/**/viz/*.mp4
# =============================================================================
set -euo pipefail

export OVERLAY_VALUE="${OVERLAY_VALUE:-1}"
export TRAIN_ROOT="${TRAIN_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/train/smolvla_acp_bowls_tray_3cam_v2}"
export DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue_v2}"
export DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_bowls_stack_lipstick_tissue_v2}"
export EVAL_ROOT="${EVAL_ROOT:-/home/rxn/lerobot/Evo-RL/outputs/eval/smolvla_acp_bowls_tray_3cam_v2}"
export PRESET="${PRESET:-n10_acp_none}"

echo "=============================="
echo "infer + value overlay (bowls_tray_3cam_v2)"
echo "OVERLAY_VALUE=${OVERLAY_VALUE}  PRESET=${PRESET}"
echo "=============================="

exec bash /home/rxn/lerobot/infer_smolvla_acp_bowls_tray_3cam.sh "$@"
