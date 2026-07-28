# Bimanual Handover with SmolVLA — Project Summary

**Platform:** LeRobot + SO101 dual-arm (`bi_so101_follower`)  
**Task:** Grasp double-sided tape roll → handover to other arm → place in yellow tray  
**Period:** July 2026  
**Role focus:** Data collection, training, deployment debugging, evaluation methodology

---

## 1. Project Overview (Elevator Pitch)

Built an end-to-end **vision-language-action (VLA)** pipeline for **bimanual manipulation** on a dual SO101 robot. Collected 80 teleoperated demonstrations (65k frames, 12-DoF state/action, 3 cameras), fine-tuned **SmolVLA** from a single-arm base model, and deployed on real hardware. Through systematic offline metrics and on-robot debugging, identified **overfitting**, **control-loop parameters**, and **language–task mismatch** as primary deployment bottlenecks—not sensor calibration or scene changes.

**One-liner for interviews:**  
*"Fine-tuned SmolVLA for dual-arm handover; diagnosed deployment failure via MAE-based offline eval and control-parameter analysis, separating model limits from infrastructure issues."*

---

## 2. System Architecture

```
Teleop demos (80 ep, 3 cams, 12-dim joints)
        ↓
LeRobot dataset (bimanual_handover)
        ↓
SmolVLA fine-tune (smolvla_base → 12-dim action, 3 cameras)
        ↓
Checkpoint + preprocessor/postprocessor (normalize/unnormalize)
        ↓
Real robot: bi_so101_follower + lerobot-record inference loop
```

| Component | Configuration |
|-----------|---------------|
| Robot | Left + right SO101 followers, 6 joints + gripper each |
| Cameras | Top (UGREEN), left wrist (YHTek), right (Sonix) → renamed to camera1/2/3 |
| Action / state | 12-D (left 6 + right 6), padded to 32 inside model |
| Training | 2× GPU, batch 16, 150k steps (~37 epochs), loss → 0.001 |
| Inference | `n_action_steps=5`, `max_relative_target=50` (after tuning) |

---

## 3. Key Technical Findings

### 3.1 Checkpoint metadata vs. actual training dimensions

When fine-tuning from `smolvla_base` (single-arm, state=6):

- **`output_features.action`** correctly updated to **12-D** from dataset.
- **`input_features.observation.state`** stayed **6-D in `config.json`** because `factory.py` only overwrites `input_features` when empty.
- **Training still used 12-D state** from the dataset; normalizer stats were 12-D via training overrides.
- **Model `state_proj`** always accepts **32-D** padded input (`max_state_dim=32`).

**Interview point:** Distinguish **config metadata bugs** from **runtime tensor shapes**. Always verify safetensors stats and batch tensors, not only `config.json`.

### 3.2 Action storage: bimanual, not single-arm

Checkpoint stores **12-D action** (left + right arm joint targets). Postprocessor unnormalizer also uses shape `[12]`. This part of the pipeline was correct.

### 3.3 Task language vs. real task

| Layer | Text |
|-------|------|
| Recorded / trained | `"Grab and handover the object to the other arm"` |
| Actual manipulation | Grasp tape roll → handover → place in **yellow tray** |

SmolVLA conditions on language tokens, but **imitation is primarily visual**. Generic task text omits the placement phase; the model must learn step 3 from pixels alone. **Do not change inference task to a longer description** unless retraining with that text.

### 3.4 Offline MAE methodology

Per-frame metric on training dataset (5 sample frames):

```python
mae = mean(|pred_action - gt_action|)   # 12 joints, normalized motor units
delta = mean(|pred_action - current_state|)  # motion intent
```

| Frame | MAE | Interpretation |
|-------|-----|----------------|
| 0 | 0.72 | Good initial pose matching |
| 1000–5000 | ~1.0 | Acceptable |
| 10000–20000 | 3.4–3.6 | Weak at handover phase |

**Units:** Not degrees—SO101 uses normalized range (~−100..100 body, 0..100 gripper).

**Limitations:** Single-frame, `policy.reset()` each time, on training distribution only. Good for **sanity check**, not task success rate.

### 3.5 On-robot behavior

Observed: right arm rises slowly to mid height, then **hovers and stops**. Scene, lighting, and cameras unchanged.

**Ruled out:** Normalizer 6/12 bug (offline MAE and right-arm errors reasonable), scene mismatch.

**Most likely causes (ranked):**

1. **Overfitting** at 150k steps (loss ≈ 0.001) → policy outputs ≈ current state ("hold").
2. **Closed-loop divergence** with absolute joint targets—small errors compound mid-trajectory.
3. **Initial joint pose** slightly off vs. demo frame 0.
4. **Control parameters:** `n_action_steps` (replan interval) and `max_relative_target` (per-step speed cap).

### 3.6 Control parameters (deployment tuning)

| Parameter | Layer | Effect |
|-----------|-------|--------|
| `N_ACTION_STEPS` | Model replanning | Smaller → more frequent vision updates (5 ≈ 0.17s @ 30fps) |
| `MAX_RELATIVE_TARGET` | Robot safety clip | Larger → faster motion per step (50 vs 25) |

These are **orthogonal**: one controls how often the VLA re-thinks; the other caps how far each joint moves per command.

---

## 4. What We Built / Modified

| Artifact | Purpose |
|----------|---------|
| `get-data-bimanual.sh` | Dual-arm teleop recording, 3 cameras |
| `train_smolvla_bimanual.sh` | Fine-tune with rename_map, `tolerance_s=0.05`, `empty_cameras=0` |
| `infer_smolvla_bimanual.sh` | Robot / offline / viz modes; tuned defaults |
| `src/lerobot/robots/utils.py` | `max_relative_target` int→float fix |
| `bi_so101_follower` config | Float type for max_relative_target |

---

## 5. Results Summary

| Metric | Value |
|--------|-------|
| Dataset | 80 episodes, 65,301 frames, 12-D action/state |
| Training | 150k steps, final loss ~0.001 |
| Offline MAE (150k) | 0.7–3.6 (avg ~2) on sampled frames |
| Real robot | Partial success: arm lift, then stall mid-trajectory |
| Deployment | Improved from "barely moves" to "reaches mid pose" after param tuning |

---

## 6. Lessons Learned (Interview Talking Points)

### Debugging methodology

1. **Separate offline vs. on-robot failure modes** — Low offline MAE does not guarantee task success; check `delta` (pred vs. state) for "frozen policy."
2. **Checkpoint sweep** — Later checkpoints are not always better; high train-fit (loss → 0.001) often hurts closed-loop control.
3. **Read the full stack** — Config JSON, preprocessor stats, `factory.py` feature wiring, robot action clipping, and VLA action queue.
4. **Fix one variable at a time** — User confirmed scene unchanged; redirected analysis to overfitting and control params.

### Robotics / ML

1. **Absolute joint policies are vision-sensitive** even with identical scenes—closed-loop error accumulation matters.
2. **VLA task strings should match the full task** if language conditioning is relied upon; otherwise vision must carry all sub-goals.
3. **Bimanual handover** is a hard benchmark for smaller VLAs; GR00T is a justified alternative for same dataset.

### Engineering

1. **`tolerance_s`** required for dataset loading (timestamp sync across 3 video streams).
2. **Bimanual calibration files** are separate from single-arm; motor EEPROM gets last written calibration.
3. **Pretrained processor overrides** during training (`normalizer_processor.stats`) differ from inference path—worth aligning explicitly.

---

## 7. Recommended Next Steps

| Priority | Action |
|----------|--------|
| P0 | On-robot eval with **50k / 100k** checkpoints (not 150k) |
| P1 | Offline MAE comparison across checkpoints |
| P2 | Collect 30+ more demos with **full task description** |
| P3 | Retrain 250k steps; deploy mid-training checkpoints |
| P4 | Parallel **GR00T** training on same dataset |
| P5 | Fix `factory.py` to sync `input_features` on fine-tune |

---

## 8. Sample Interview Q&A

**Q: What was the hardest debugging moment?**  
A: The robot barely moved despite successful model loading. I traced it through checkpoint metadata (state=6 vs 12), normalizer stats, offline MAE, and finally control parameters. Offline MAE ~1–3 showed the model wasn't broken—it was overfitting and outputting near-static actions in closed loop.

**Q: How did you evaluate without a sim?**  
A: Built offline inference on the training dataset (MAE per joint + gripper), plus on-robot eval recording with `lerobot-record`. Used checkpoint sweeps and motion-intent metric (|pred − state|).

**Q: What would you do differently?**  
A: (1) Use descriptive task language from day one. (2) Save and evaluate multiple checkpoints during training. (3) Start GR00T earlier for bimanual. (4) Fix feature metadata in the training factory for cleaner handoff.

**Q: What did you learn about VLA deployment?**  
A: Training loss and offline accuracy are insufficient. Deployment needs replanning frequency, action rate limits, initial pose alignment, and early-stop checkpoint selection—especially for absolute joint control on real arms.

---

## 9. Commands Reference

```bash
# Offline eval
bash infer_smolvla_bimanual.sh offline

# Robot with earlier checkpoint
CHECKPOINT=outputs/train/smolvla_bimanual_handover/checkpoints/050000/pretrained_model \
  CUDA_VISIBLE_DEVICES=1 bash infer_smolvla_bimanual.sh robot

# Visualize training demos
bash infer_smolvla_bimanual.sh viz

# Retrain (longer)
STEPS=250000 bash train_smolvla_bimanual.sh
```

---

*Document generated from SmolVLA bimanual handover experiment, July 2026.*
