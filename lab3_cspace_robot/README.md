# Robot-only joint-space Lab 3

Copy or pull **this folder** onto the robot PC. It has no simulation code,
MuJoCo, viewer, SciPy, mesh assets, or imports from the other project folders.
It requires NumPy and the robot's existing FrankaPy/ROS environment.

Files:

- `run.py`: robot entry point and read-only preflight/state capture.
- `joint_executor.py`: executes only `goto_joints(..., use_impedance=False)`
  for arm motion and `goto_gripper()` for the jaws.
- `panda_model.py`: forward kinematics, joint limits and timing checks; no IK.
- `config.json`: geometry, tool frame, gripper settings and calibration status.
- `plans/`: offline collision-checked joint waypoints for all patterns, an
  optional final reset, and the B0 rotate/place diagnostic.
- `manifest.json`: plan checksums and the geometry fingerprint.
- `requirements.txt`: NumPy only; use the existing FrankaPy/ROS installation.

All coordinates/lengths are **metres**, joint angles are **radians**, force is
**newtons**. Keep a partner at the emergency stop as required by the lab.

## Before moving

The included calibration is a **draft**, not proof of the physical setup:

- Waiting/grid XY and grasp Z came from the existing measured config.
- `table_z = 0.036679661` was inferred from taught grasp Z minus half a cube.
  Independently measure it; a taught tool height need not be the cube centre.
- Verify `flange_to_tcp`, all block/grid locations, the table and shelf geometry.
- Cube size is 0.043 m, grip force is 10 N, and travel Z is 0.22 m.
- `hardware_calibrated` starts false. Set it true **after** verifying the setup.
  Changing only this acknowledgement does not require regenerating joint angles.

Changing geometry, gripper parameters, timing or tool calibration requires new
offline plans. The runner rejects a changed config or edited plan rather than
silently executing stale joint targets. Do not edit the manifest to bypass it.

## Commands on the robot

Run in this folder using the lab's configured FrankaPy/ROS Python environment.
The default command only validates the files and prints a summary; it does not
connect to or move the robot:

```bash
python3 run.py
```

Capture current joints and tool pose without moving:

```bash
python3 run.py --state robot_state.json
```

Check measured start joints and model/tool alignment without moving or opening
the jaws (allowed even while calibration is still marked unverified):

```bash
python3 run.py --test rotate-place --check
```

The included plans begin at the standard home joint configuration in the
config. The runner does not automatically move to home. If the robot differs
from the planned starting joints, capture `--state` and regenerate from those
joints, or use the lab's established procedure to restore the expected starting
configuration. Do not reuse a plan after moving blocks out of its assumed setup.

After calibration and preflight, start with an empty gripper, an empty grid,
and all blocks at their waiting positions. First run the diagnostic:

```bash
python3 run.py --test rotate-place --execute
```

It deliberately moves B0 to the grid with the required 90-degree axis change,
then picks it again and returns it to waiting. For the actual assignment:

```bash
python3 run.py --execute
```

This builds pickaxe, skips infeasible axe, transitions to box and bow, then
leaves the final pattern in place. An optional complete reset is:

```bash
python3 run.py --final-reset --execute
```

The runner checks each motion's status, measured joints and TCP endpoint. It
also checks jaw width and stops before releasing if motion fails. It does not
retry, release the held object, or reset automatically on a fault. A pattern
checkpoint is nominal bookkeeping, not a camera measurement of the cubes.
Speeds come from the exported joint plan; there is no runtime speed multiplier.

## Regenerating plans on the planning machine

The collision planner stays in the **separate** `lab3_cspace` folder on the
planning machine. Only the files in this robot folder need to be on the robot.
After editing this folder's config, generate plans with the appropriate
MuJoCo/NumPy environment on the planning machine, from the repository root:

```bash
python3 lab3_cspace/export_robot.py --config lab3_cspace_robot/config.json
```

To plan from a captured robot state rather than the default starting joints:

```bash
python3 lab3_cspace/export_robot.py --config lab3_cspace_robot/config.json --start-state robot_state.json
```

Copy the updated robot folder back to the robot PC. No simulation package is
installed or imported by the robot runner. Export checks collision geometry
offline; the robot runtime checks plan structure, integrity and feedback, not
a new collision search. The environment must match the calibrated plan.

The joint-space simulation has passed, and this executor is tested with a mock
Franka arm. Physical operation of this new implementation is still unverified.
