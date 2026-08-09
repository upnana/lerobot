# Isaac Sim Script Editor — add table + white/blue/black blocks for SO101 stack scene
# Prerequisite: /so101_new_calib already on stage (imported USD)
# Usage: Stop or Play both OK → Run once → then Play + sliders

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf

stage = omni.usd.get_context().get_stage()

# ---- config (meters) ----
# Bigger desktop so arm + blocks have room (was 0.60 x 0.40 — too small)
TABLE_SIZE = (1.20, 0.80, 0.05)          # L, W, thickness
TABLE_TOP_Z = 0.30                       # table top height
# Center table under robot with more space in front (+X) for blocks
TABLE_CENTER = (0.25, 0.0, TABLE_TOP_Z - TABLE_SIZE[2] / 2.0)

BLOCK = 0.045                            # ~4.5cm cubes
# place blocks farther in front of SO101
BLOCK_XY = {
    "block_white": (0.28, -0.10),
    "block_blue": (0.28, 0.00),
    "block_black": (0.28, 0.10),
}
BLOCK_COLOR = {
    "block_white": (0.92, 0.92, 0.92),
    "block_blue": (0.15, 0.35, 0.95),
    "block_black": (0.08, 0.08, 0.08),
}

ROOT = "/World/StackScene"


def ensure_physics_scene():
    for p in stage.Traverse():
        if p.GetTypeName() == "PhysicsScene":
            return
    sc = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)
    print("Created /PhysicsScene")


def bind_color(prim: Usd.Prim, rgb, name: str):
    mat_path = f"/World/Looks/{name}"
    if not stage.GetPrimAtPath(mat_path).IsValid():
        mat = UsdShade.Material.Define(stage, mat_path)
        shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    else:
        mat = UsdShade.Material.Get(stage, mat_path)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)


def make_box(path, scale_xyz, translate_xyz, rgb, dynamic: bool, mass=0.05):
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(cube.GetPrim())
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*translate_xyz))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale_xyz))

    prim = cube.GetPrim()
    UsdPhysics.CollisionAPI.Apply(prim)
    if dynamic:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mapi = UsdPhysics.MassAPI.Apply(prim)
        mapi.CreateMassAttr(mass)
    bind_color(prim, rgb, path.split("/")[-1] + "_mat")
    return prim


def place_robot_on_table():
    robot = stage.GetPrimAtPath("/so101_new_calib")
    if not robot or not robot.IsValid():
        print("WARN: /so101_new_calib not found — import SO101 first")
        return
    xf = UsdGeom.Xformable(robot)
    # Put robot base on table top, slightly back from blocks
    # Clear and set a single translate for predictability
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, TABLE_TOP_Z))
    print("Moved /so101_new_calib onto table top z=", TABLE_TOP_Z)


ensure_physics_scene()

# container
if not stage.GetPrimAtPath("/World").IsValid():
    UsdGeom.Xform.Define(stage, "/World")
if stage.GetPrimAtPath(ROOT).IsValid():
    stage.RemovePrim(ROOT)
UsdGeom.Xform.Define(stage, ROOT)
UsdGeom.Xform.Define(stage, "/World/Looks")

# table (static)
make_box(
    f"{ROOT}/Table",
    scale_xyz=TABLE_SIZE,
    translate_xyz=TABLE_CENTER,
    rgb=(0.55, 0.42, 0.28),
    dynamic=False,
)

# three blocks (dynamic) sitting on table
z_block = TABLE_TOP_Z + BLOCK / 2.0 + 0.001
for name, (x, y) in BLOCK_XY.items():
    make_box(
        f"{ROOT}/{name}",
        scale_xyz=(BLOCK, BLOCK, BLOCK),
        translate_xyz=(x, y, z_block),
        rgb=BLOCK_COLOR[name],
        dynamic=True,
        mass=0.03,
    )
    print("Added", name, "at", (x, y, z_block))

place_robot_on_table()

# simple dome/distant light if available in this USD build
if not stage.GetPrimAtPath("/World/DistantLight").IsValid():
    try:
        from pxr import UsdLux

        light = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
        light.CreateIntensityAttr(3000)
        print("Added DistantLight")
    except Exception as e:
        print("Skip light (optional):", e)

print("Stack scene ready: table + white/blue/black blocks.")
print("Next: Play → run isaac_so101_sliders.py → try moving arm toward blocks.")
