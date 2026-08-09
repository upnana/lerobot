# Run inside Isaac Sim: Window > Script Editor
# 1) Make sure your stage has /Franka (Create > Robots > Franka...)
# 2) Click Play (left toolbar)
# 3) Paste this script and click Run
#
# The arm should move to a bent pose.

from isaacsim.core.experimental.prims import Articulation

# Matches Create > Robots Franka default path in your screenshots
robot = Articulation("/Franka")

# 7 arm joints + 2 fingers (radians)
target = [-1.5, 0.0, 0.0, -1.5, 0.0, 1.5, 0.5, 0.04, 0.04]
robot.set_dof_position_targets(target)

print("Sent Franka target pose:", target)
print("Current DOF positions:", robot.get_dof_positions())
