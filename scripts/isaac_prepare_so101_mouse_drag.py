# Isaac Sim Script Editor — one-time prep so mouse-drag in Joint Inspector works
# Run once while Play is ON, then drag Target Position with the mouse.

import omni.usd
import omni.timeline
from pxr import UsdPhysics, Sdf, Gf

stage = omni.usd.get_context().get_stage()
tl = omni.timeline.get_timeline_interface()

if not any(p.GetTypeName() == "PhysicsScene" for p in stage.Traverse()):
    sc = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)
    print("Created /PhysicsScene")

if not tl.is_playing():
    tl.play()
    print("Pressed Play")

joints = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

for name in joints:
    prim = stage.GetPrimAtPath(f"/so101_new_calib/Physics/{name}")
    if not prim or not prim.IsValid():
        print("MISSING", name)
        continue
    drive = UsdPhysics.DriveAPI.Get(prim, "angular") or UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.GetTypeAttr().Set("force")
    drive.GetStiffnessAttr().Set(1000.0)
    drive.GetDampingAttr().Set(50.0)
    drive.GetMaxForceAttr().Set(1.0e5)  # critical: URDF default 10 is too weak
    print("ready for mouse drag:", name)

print("Now drag Target Position numbers left/right in Joint Inspector (units: degrees).")
