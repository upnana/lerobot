# Value-overlay demo videos (bowls_tray HITL)

Source rollout: `hitl_step-080000` (policy 80k, preset `n10_acp_none`).

Each mp4 is `right_top_right` with a synchronized **value** curve overlay
(from `bowls_tray_3cam_v2` value model), not advantage/reward spikes.

| File | Episode |
|------|---------|
| `hitl80k_ep0_value_overlay.mp4` | ep0 |
| `hitl80k_ep1_value_overlay.mp4` | ep1 |
| `hitl80k_ep2_value_overlay.mp4` | ep2 |

Regenerate: `bash overlay_value_on_eval.sh <hitl_dataset_root>`
