# Isaac Sim Script Editor — big SO101 joint sliders (mouse drag friendly)
# 1) Play ON
# 2) Run this once
# 3) Drag the long sliders in the popup window (much faster than Joint Inspector)

import omni.ui as ui
import omni.usd
import omni.timeline
from pxr import UsdPhysics, Sdf, Gf

stage = omni.usd.get_context().get_stage()
tl = omni.timeline.get_timeline_interface()

if not any(p.GetTypeName() == "PhysicsScene" for p in stage.Traverse()):
    sc = UsdPhysics.Scene.Define(stage, Sdf.Path("/PhysicsScene"))
    sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    sc.CreateGravityMagnitudeAttr().Set(9.81)

if not tl.is_playing():
    tl.play()

# (name, min_deg, max_deg)
JOINTS = [
    ("shoulder_pan", -110.0, 110.0),
    ("shoulder_lift", -100.0, 100.0),
    ("elbow_flex", -97.0, 97.0),
    ("wrist_flex", -95.0, 95.0),
    ("wrist_roll", -157.0, 163.0),
    ("gripper", -10.0, 100.0),
]


def _drive(name: str):
    prim = stage.GetPrimAtPath(f"/so101_new_calib/Physics/{name}")
    if not prim or not prim.IsValid():
        return None
    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    if not drive:
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
    drive.GetTypeAttr().Set("force")
    drive.GetStiffnessAttr().Set(1000.0)
    drive.GetDampingAttr().Set(50.0)
    drive.GetMaxForceAttr().Set(1.0e5)
    return drive


# prepare drives
for name, _, _ in JOINTS:
    d = _drive(name)
    print("ready" if d else "MISSING", name)


class SO101SliderWindow:
    def __init__(self):
        self._models = {}
        self._window = ui.Window("SO101 Joint Sliders (degrees)", width=420, height=320)
        with self._window.frame:
            with ui.VStack(spacing=6, height=0):
                ui.Label("Drag sliders — values are degrees, large travel")
                for name, lo, hi in JOINTS:
                    with ui.HStack(height=0):
                        ui.Label(name, width=120)
                        model = ui.SimpleFloatModel(0.0)
                        self._models[name] = model

                        def _on_change(m, joint_name=name):
                            drive = _drive(joint_name)
                            if drive:
                                drive.GetTargetPositionAttr().Set(float(m.get_value_as_float()))

                        model.add_value_changed_fn(_on_change)
                        # min/max = full joint range → one drag sweeps a lot
                        ui.FloatSlider(model, min=lo, max=hi, step=1.0)

                def _zero():
                    for n, m in self._models.items():
                        m.set_value(0.0)

                ui.Button("Reset all to 0", clicked_fn=_zero)


# keep a global ref so the window is not GC'd
_so101_slider_win = SO101SliderWindow()
print("Slider window opened. Drag there for fast motion.")
