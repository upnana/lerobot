#!/bin/bash
# =============================================================================
# 采集：叠三碗 → 口红+蓝纸巾进黄 tray（默认 4 相机）
#
# 固定顺序（每条 demo 都按这个做）:
#   1) 先把黄→棕→白碗叠到白盘上（黄底棕中白顶，叠稳）
#   2) 再把口红 + 蓝色纸巾包放进黄 tray
#
# 建议手臂分工（可按习惯改，但整次采集保持一致）:
#   右臂：黄碗、白碗
#   左臂：棕碗、口红、纸巾
#   少双手同时抓；一只动时另一只可收拢
#
# Success(s): 碗叠对且稳 + 口红和纸巾都在黄 tray
# Failure(f): 掉落 / 顺序错 / 缺物 / 倒了 / 只完成一半
#
# Cameras (TOP_CAMS=2 默认):
#   left:  wrist + top_left
#   right: wrist + top_right
# 若只要 3 相机: TOP_CAMS=1 bash get-data-bowls-tray.sh ...
#
# Usage:
#   bash get-data-bowls-tray.sh check
#   bash get-data-bowls-tray.sh teleop
#   bash get-data-bowls-tray.sh              # 新采 50 条
#   bash get-data-bowls-tray.sh record
#   RESUME=1 NUM_EPISODES=50 bash get-data-bowls-tray.sh
#   bash get-data-bowls-tray.sh report
#
# 常用:
#   NUM_EPISODES=30 EPISODE_TIME_S=150 bash get-data-bowls-tray.sh
#   DISPLAY_DATA=false bash get-data-bowls-tray.sh
# =============================================================================
set -euo pipefail

MODE="${1:-record}"

export TOP_CAMS="${TOP_CAMS:-2}"
export DATA_ROOT="${DATA_ROOT:-/home/rxn/datasets/evo_rl_bowls_stack_lipstick_tissue}"
export DATA_REPO="${DATA_REPO:-my_bimanual/evo_rl_bowls_stack_lipstick_tissue}"
export NUM_EPISODES="${NUM_EPISODES:-50}"
export EPISODE_TIME_S="${EPISODE_TIME_S:-180}"
export RESET_TIME_S="${RESET_TIME_S:-10}"
export REQUIRE_EPISODE_SUCCESS_LABEL="${REQUIRE_EPISODE_SUCCESS_LABEL:-true}"

export SINGLE_TASK="${SINGLE_TASK:-First stack the yellow, brown, and white bowls from bottom to top onto the white plate, then put the lipstick and the blue tissue pack into the yellow tray.}"

echo "=============================="
echo "Bowls→plate then lipstick+tissue→tray (4 cams default)"
echo "=============================="
echo "Order:   1) stack bowls  2) fill yellow tray"
echo "Task:    ${SINGLE_TASK}"
echo "Dataset: ${DATA_ROOT}"
echo "Cams:    TOP_CAMS=${TOP_CAMS} (2=4cam; 1=3cam)"
echo "Tips:    小幅随机摆放；固定左右分工；做完立刻 s/f；失败也录并打 f"
echo "=============================="

exec bash /home/rxn/lerobot/get-data-evo-rl-bimanual.sh "${MODE}"
