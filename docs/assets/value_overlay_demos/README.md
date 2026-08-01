# Value-overlay demo videos (bowls_tray)

Each mp4 is `right_top_right` with a synchronized **value** \(V(s)\) curve overlay
(from `bowls_tray_3cam_v2` value model), not advantage/reward spikes.
Encoded as **H.264** for GitHub / Douyin compatibility.

## HITL demos (policy 80k)

Source: `hitl_step-080000` (preset `n10_acp_none`).

| File | Episode |
|------|---------|
| `hitl80k_ep0_value_overlay.mp4` | ep0 |
| `hitl80k_ep1_value_overlay.mp4` | ep1 |
| `hitl80k_ep2_value_overlay.mp4` | ep2 |

## Pure policy eval (v2)

| File | Checkpoint | Notes |
|------|------------|--------|
| `v2_step120k_ep0_value_overlay.mp4` | 120k | sweet-spot A/B baseline |
| `v2_step140k_ep0_value_overlay.mp4` | 140k | mid–late compare |

Regenerate: `bash overlay_value_on_eval.sh <eval_or_hitl_dataset_root>`
