# Isaac Sim Script Editor
# 用法：
# 1) 场景里已有 Franka
# 2) 左侧点 Play
# 3) 整段粘贴到 Script Editor，点 Run
#
# 这版不依赖 experimental Articulation（避免 ApplyMultipleApplyAPI 报错）

import omni.usd
import omni.timeline
from pxr import Usd, UsdPhysics

STAGE = omni.usd.get_context().get_stage()
ROOT_CANDIDATES = ["/Franka", "/World/Franka", "/World/Franka_1", "/Franka_01"]


def find_robot_root():
    for path in ROOT_CANDIDATES:
        prim = STAGE.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return prim
    # fallback: search any prim named Franka
    for prim in STAGE.Traverse():
        if "franka" in prim.GetName().lower():
            return prim
    return None


def find_joints(root_prim):
    joints = []
    for prim in Usd.PrimRange(root_prim):
        t = prim.GetTypeName()
        if t in ("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"):
            joints.append(prim)
            continue
        # some assets store joints as generic prims with DriveAPI
        if "joint" in prim.GetName().lower() and prim.HasAPI(UsdPhysics.DriveAPI):
            joints.append(prim)
    return joints


def set_drive_target(joint_prim, value, drive_type="angular"):
    # prismatic finger joints use "linear"
    name = joint_prim.GetName().lower()
    if "finger" in name:
        drive_type = "linear"

    drive = UsdPhysics.DriveAPI.Get(joint_prim, drive_type)
    if not drive:
        # try the other type
        other = "linear" if drive_type == "angular" else "angular"
        drive = UsdPhysics.DriveAPI.Get(joint_prim, other)
        drive_type = other if drive else drive_type
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(joint_prim, drive_type)

    # ensure it can actually move
    stiff = drive.GetStiffnessAttr()
    damp = drive.GetDampingAttr()
    if stiff and (stiff.Get() is None or stiff.Get() < 1.0):
        stiff.Set(1000.0)
    if damp and (damp.Get() is None or damp.Get() < 0.1):
        damp.Set(50.0)

    tgt = drive.GetTargetPositionAttr()
    if tgt:
        tgt.Set(float(value))
        return True, drive_type
    return False, drive_type


# ---- main ----
timeline = omni.timeline.get_timeline_interface()
if not timeline.is_playing():
    timeline.play()
    print("Timeline was stopped -> pressed Play")

root = find_robot_root()
if root is None:
    raise RuntimeError("Cannot find Franka root. Check Stage path.")

print("Robot root:", root.GetPath())
joints = find_joints(root)
print("Found joints:", len(joints))
for j in joints:
    print(" -", j.GetPath(), j.GetTypeName())

if not joints:
    raise RuntimeError("No physics joints found under robot root.")

# A clear bent pose for panda_joint1..7 if present; fingers slightly open
POSE = {
    "panda_joint1": -0.5,
    "panda_joint2": 0.3,
    "panda_joint3": 0.0,
    "panda_joint4": -1.5,
    "panda_joint5": 0.0,
    "panda_joint6": 1.2,
    "panda_joint7": 0.5,
    "panda_finger_joint1": 0.04,
    "panda_finger_joint2": 0.04,
}

ok = 0
for j in joints:
    name = j.GetName()
    if name in POSE:
        success, dtype = set_drive_target(j, POSE[name])
        print(f"set {name} -> {POSE[name]} ({dtype}) success={success}")
        ok += int(success)

print(f"Done. Updated {ok} joints. Watch the arm for ~1-2 seconds.")
