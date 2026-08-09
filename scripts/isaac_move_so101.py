# Isaac Sim Script Editor — move SO101 (no JointStateAPI)
# Fix: angular targets in DEGREES + bump maxForce (URDF effort=10 is too weak)

import omni.usd
import omni.timeline
from pxr import UsdPhysics, Sdf, Gf

stage = omni.usd.get_context().get_stage()
tl = omni.timeline.get_timeline_interface()

# Ensure PhysicsScene
if not any(p.GetTypeName() == "PhysicsScene" for p in stage.Traverse()):
    sc = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)
    print("Created /PhysicsScene")

if not tl.is_playing():
    tl.play()
    print("Pressed Play")

root = stage.GetPrimAtPath("/so101_new_calib")
print("root ok:", bool(root and root.IsValid()))

# DEGREES — big visible motion
POSE_DEG = {
    "shoulder_pan": 35.0,
    "shoulder_lift": 40.0,
    "elbow_flex": -50.0,
    "wrist_flex": 30.0,
    "wrist_roll": 25.0,
    "gripper": 20.0,
}

for name, deg in POSE_DEG.items():
    prim = stage.GetPrimAtPath(f"/so101_new_calib/Physics/{name}")
    if not prim or not prim.IsValid():
        print("MISSING", name)
        continue

    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")

    drive.GetTypeAttr().Set("force")
    drive.GetStiffnessAttr().Set(200.0)
    drive.GetDampingAttr().Set(20.0)
    drive.GetMaxForceAttr().Set(1.0e5)  # was 10
    drive.GetTargetPositionAttr().Set(float(deg))
    print("set", name, "->", deg, "deg")

print("Done. Check Joint Inspector Target/State, and the viewport.")
