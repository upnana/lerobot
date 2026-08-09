# Isaac Sim 6.0 + SO101 Setup Notes

Local workstation notes for getting **NVIDIA Isaac Sim 6.0.1** running on dual **RTX 3090**, importing the **SO101** URDF, building a simple stack scene, and controlling joints with UI sliders.

Goal (job / portfolio): show a working sim path toward sim-to-real with the same robot used in real LeRobot experiments.

## Hardware / software

| Item | Value |
|------|--------|
| GPUs | 2× NVIDIA GeForce RTX 3090 (24GB) |
| OS | Ubuntu 22.04 |
| Isaac Sim | `6.0.1.0` via pip |
| Conda env | `env_isaacsim` (Python 3.12) |
| Robot URDF | `/home/rxn/SO-ARM100/Simulation/SO101/so101_new_calib.urdf` |

## Install Isaac Sim (pip, recommended)

```bash
conda create -y -n env_isaacsim python=3.12
conda activate env_isaacsim
pip install --upgrade pip
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url https://pypi.nvidia.com
```

Launch:

```bash
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=0   # single GPU is more stable for GUI
isaacsim
```

Notes:

- First launch can take several minutes (shader / RtPso compile). If the OS says “not responding”, prefer **Wait**.
- The Extensions registry UI may show “No versions available”; pip already bundles URDF importer — use scripts instead of the store.

## Scripts in this repo

All scripts are meant to be pasted into Isaac Sim **Window → Script Editor** (or opened from disk) and **Run**.

| Script | Purpose |
|--------|---------|
| `scripts/isaac_import_so101.py` | Import SO101 URDF → USD and open stage |
| `scripts/isaac_move_so101.py` | Set large joint targets (degrees) + raise `maxForce` |
| `scripts/isaac_prepare_so101_mouse_drag.py` | One-time prep so Joint Inspector drag works |
| `scripts/isaac_so101_sliders.py` | **Best UX**: popup long sliders for fast joint control |
| `scripts/isaac_so101_stack_scene.py` | Add large table + white/blue/black blocks |
| `scripts/isaac_move_franka_usd.py` | Earlier Franka smoke test via USD drives |

Imported USD output (local, not committed):

`isaac_assets/so101/so101_new_calib/so101_new_calib.usda`

## Recommended workflow

### 1) Import SO101

1. `File → New`
2. Run `scripts/isaac_import_so101.py`
3. Confirm log lines like `Imported USD: ...` and `Opened: True`
4. Stage should contain `/so101_new_calib`

### 2) Build stack scene

1. Run `scripts/isaac_so101_stack_scene.py`
2. Expect:
   - larger table (~1.2m × 0.8m)
   - white / blue / black cubes
   - robot placed on table top (`z ≈ 0.30`)

### 3) Control with sliders

1. Click **Play**
2. Run `scripts/isaac_so101_sliders.py`
3. Drag long sliders in the **SO101 Joint Sliders** window (values are **degrees**)

Why not only Joint Inspector?

- Inspector float fields scrub very slowly.
- URDF import sets `maxForce≈10` (too weak) and angular units are **degrees**.
- Sliders script bumps stiffness / damping / maxForce and uses full joint ranges.

## Important gotchas learned

1. **Angular drive targets are in degrees** in the imported USD (`physics:lowerLimit` etc. are degree-valued). Setting `0.5` rad-style values barely moves the arm.
2. **`maxForce=10`** from URDF effort is too small; raise to `1e5` for visible motion.
3. Joints live under `/so101_new_calib/Physics/<joint_name>` (not under `Geometry/`).
4. `UsdPhysics.JointStateAPI` is unavailable in this build; use `DriveAPI` only (or `PhysxSchema.JointStateAPI` if needed).
5. `UsdGeom.DistantLight` may be missing; use `UsdLux.DistantLight` for lighting.
6. Timeline panel may be hidden; for physics joint control, left toolbar **Play/Stop** is enough.
7. Experimental `Articulation("/Franka").set_dof_position_targets(...)` can hit `ApplyMultipleApplyAPI` errors; USD `DriveAPI` is more reliable for quick GUI scripting.

## Joint names (SO101)

- `shoulder_pan`
- `shoulder_lift`
- `elbow_flex`
- `wrist_flex`
- `wrist_roll`
- `gripper`

## Demo recording

Example converted screen capture:

- Source: `~/视频/录屏/录屏 2026年08月09日 23时35分24秒.webm`
- MP4: `~/视频/录屏/录屏_20260809_233524.mp4`

```bash
ffmpeg -y -i "input.webm" -c:v libx264 -preset fast -crf 23 -c:a aac -movflags +faststart "output.mp4"
```

## Next steps (sim-to-real / data)

1. Add a virtual camera and scripted trajectories.
2. Auto-collect episodes (joint states ± images) into **LeRobot dataset** format.
3. Optional later: Isaac Lab for RL / large-scale training; keep this machine for GUI + asset bring-up.

## Resume one-liner

```bash
conda activate env_isaacsim
export OMNI_KIT_ACCEPT_EULA=YES
export CUDA_VISIBLE_DEVICES=0
isaacsim
# then Script Editor: import SO101 → stack scene → sliders
```
