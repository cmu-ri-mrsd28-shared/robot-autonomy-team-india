# Lab 3 physical Franka

This folder is a standalone robot project. Copy this whole folder to the robot
PC, or extract `lab3_franka.zip`. It contains its own task planner and uses
FrankaPy for execution. It has no MuJoCo imports, models, controllers, or paths
to the simulation project.

Use your existing Lab 2 FrankaPy/ROS environment and running robot services.
Dry runs require NumPy; physical operations also require `frankapy` and
`autolab_core`.

## Calibration

Keep your measured `calibration.json` when updating code. The supplied values
are examples unless you have replaced them with measurements. All positions
are meters and describe `franka_tool` in `world` at the grasp pose. Measure the
grid center, staging positions, grid axes/spacing, tool rotations, transit
height, workspace bounds, and block/gripper dimensions. Set `calibrated` to
`true` after checking the measured setup.

The physical cube width is configured as `block_width_m: 0.043` (4.3 cm).
`gripper_grasp_width_m: 0.0` requests the fully closed target, matching the
closing target used by Lab 2's physical `close_gripper()` call. Execution
uses force-limited grasping with the configured 15 N cap. A successful grasp
is expected to stop around 0.043 m due to block contact; it is not expected
to reach zero while holding a cube. Width feedback remains checked.

The opening target remains 0.055 m, and the feedback tolerance remains
0.004 m. The block size does not automatically adjust taught grasp poses or
prove finger clearance. Lab 2's simulation value `GRIP_CLOSED = 0.0125` is a
finger/tendon coordinate, not the full FrankaPy jaw-opening width. The
simulation project has its own independent dimensions and settings.

Print the current pose without commanding motion:

```bash
python3 run.py --read-pose
```

## Run

Run these commands from this folder. The default previews commands without
connecting to the robot:

```bash
python3 run.py --final-reset
```

After calibration and physical clearance checks, use the lab's partner and
emergency-stop procedure. Begin with an empty grid, empty gripper, and all
blocks in their assigned staging slots.

```bash
# One block to the grid and back.
python3 run.py --execute --test-transfer

# Entire sequence, including the final reset.
python3 run.py --execute --final-reset
```

The full program attempts pickaxe, axe, box, and bow. Axe is skipped under the
NS/EW grasp model. There are 38 transfers with the final reset, or 32 without.
`--test-transfer` selects a two-transfer round trip instead of all patterns.
Hardware motion speeds are configured in `calibration.json`.

## Optional 2x arm speed

Preview the full sequence with arm-motion durations halved:

```bash
python3 run.py --final-reset --speed-scale 2
```

After resolving the existing reflex fault and validating the calibrated setup,
the corresponding physical run with blocks is:

```bash
python3 run.py --execute --final-reset --speed-scale 2
```

The default is `--speed-scale 1`. Scale 2 doubles the configured translation,
vertical, and rotation speed settings and halves the minimum motion duration.
The measured `calibration.json` file is unchanged. Gripper speed/force, jaw
widths, grasp checks, pose tolerances, workspace checks, and reflex handling
are unchanged. Total elapsed time may not halve because gripper operations
and pauses retain their timing. The faster profile has not been tested on
the physical robot.

Normal physical execution expects blocks. With no blocks, the first empty
grasp still fails its width check and stops. No empty-gripper rehearsal mode
has been added. A preview without `--execute` generates all actions without
robot motion; it cannot establish physical clearance or successful grasps.

## Failures and reports

`results/franka_hardware_run/progress.json` is saved during execution. It shows
the current transfer, phase, target, planned duration, and original error.
The final `report.json` includes `error` and `failed_phase`. Dry-run output is
saved separately under `results/franka_dry_run/`.
Both reports and progress record `motion_speed_scale`; the final report also
records the base motion settings and effective configuration.

The wrist rotates at the higher source clearance before lowering, and the
executor rejects substantial rotations below transit height. This software
check does not prove physical clearance or resolve an existing reflex stop.
Resolve the complete controller error before retrying hardware motion.
After a failed or partial run, reconcile block positions before restarting.

A completed hardware report means commands passed tool-pose and gripper-width
checks; it does not measure object poses. There is no perception or full-arm
collision planner. No physical robot was tested during this restructuring.
