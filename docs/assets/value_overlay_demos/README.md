# Value-overlay demo videos (bowls_tray)

Each mp4 is `right_top_right` with a synchronized **value** \(V(s)\) curve overlay
(from the matching `bowls_tray_3cam_v*` value model), not advantage/reward spikes.
Encoded as **H.264** for GitHub / Douyin compatibility.

## HITL demos (policy 80k)

Source: `hitl_step-080000` (preset `n10_acp_none`).

| File | Episode |
|------|---------|
| `hitl80k_ep0_value_overlay.mp4` | ep0 |
| `hitl80k_ep1_value_overlay.mp4` | ep1 |
| `hitl80k_ep2_value_overlay.mp4` | ep2 |

## Pure policy eval — no ACP tag (`n10_acp_none`)

| File | Checkpoint | Notes |
|------|------------|--------|
| `v2_step120k_ep0_value_overlay.mp4` | v2 120k | sweet-spot A/B baseline |
| `v2_step140k_ep0_value_overlay.mp4` | v2 140k | mid–late compare |

## Pure policy eval — with ACP tag (`Advantage: positive`, preset `n10`)

| File | Checkpoint | Notes |
|------|------------|--------|
| `v2_step120k_acp_positive_ep0_value_overlay.mp4` | v2 120k | tag A/B vs no-tag 120k |
| `v3_step100k_acp_positive_ep0_value_overlay.mp4` | v3 100k | HITL-merged prior + tag |

Regenerate: `bash overlay_value_on_eval.sh <eval_or_hitl_dataset_root>`
