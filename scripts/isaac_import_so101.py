# Isaac Sim Script Editor: Import SO101 URDF -> USD and open it on stage
# 1) Stop simulation if playing
# 2) Window > Script Editor
# 3) Paste & Run

import os

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig
import isaacsim.core.experimental.utils.stage as stage_utils

URDF_PATH = "/home/rxn/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"
USD_OUT_DIR = "/home/rxn/lerobot/isaac_assets/so101"

os.makedirs(USD_OUT_DIR, exist_ok=True)

assert os.path.isfile(URDF_PATH), f"URDF not found: {URDF_PATH}"
assert os.path.isdir(os.path.join(os.path.dirname(URDF_PATH), "assets")), "assets/ folder missing next to URDF"

importer = URDFImporter(
    URDFImporterConfig(
        urdf_path=URDF_PATH,
        usd_path=USD_OUT_DIR,
        merge_mesh=True,
        fix_base=True,  # mount robot base (don't fall through floor)
        allow_self_collision=False,
        joint_drive_type="force",
        joint_target_type="position",
        override_joint_stiffness=1000.0,
        override_joint_damping=50.0,
        collision_type="Convex Hull",
    )
)

output_path = importer.import_urdf()
print("Imported USD:", output_path)

ok, _ = stage_utils.open_stage(output_path)
print("Opened stage:", ok, "path:", output_path)
print("Next: click Play, then run scripts/isaac_move_so101.py")
