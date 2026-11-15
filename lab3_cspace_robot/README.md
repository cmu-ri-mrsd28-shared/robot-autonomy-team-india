# Robot-only joint-space Lab 3

Copy or pull this folder onto the robot PC. Both planning and execution require
only NumPy plus the robot's existing FrankaPy/ROS environment. No MuJoCo,
viewer, SciPy, XML models, STL/OBJ meshes, or parent project folder is required.

## Commands

From this folder, generate plans after editing `config.json`:

```bash
python3 generate_plan.py
```

This computes IK, checks geometric collisions, and searches joint-space paths
without connecting to the robot. All three plans must succeed before files are
replaced. Then check the robot's starting joints and tool frame without moving:

```bash
python3 run.py --check
```

After calibration is verified, with an empty gripper, an empty grid, and all
blocks at their configured waiting positions, execute the assignment:

```bash
python3 run.py --execute
```

For fresh planning followed by execution in one command:

```bash
python3 run.py --replan --execute
```

Planning must finish successfully before any robot command is sent. Without
`--execute` or `--check`, `run.py` only validates files and prints a summary.
The assignment builds pickaxe, skips infeasible axe, transitions to box and bow,
and leaves the final pattern in place. Optional complete reset:

```bash
python3 run.py --final-reset --execute
```

The separate rotate/place diagnostic deliberately moves B0 to the grid with an
X turn, picks it again, and returns it to waiting:

```bash
python3 run.py --test rotate-place --execute
```

## Calibration and starting state

Coordinates/lengths are metres, joint angles radians, force newtons. The
included config remains a draft:

- Waiting/grid XY and taught grasp Z came from the existing measured config.
- `table_z = 0.036679661` was inferred from taught grasp Z minus half a cube;
  independently measure it. A taught tool height need not be the cube centre.
- Verify `flange_to_tcp`, table/shelf geometry, and all block/grid locations.
- Cube size is 0.043 m, grip force 10 N, travel Z 0.22 m.
- Set `hardware_calibrated` to true only after verifying the setup. Changing
  this acknowledgement alone does not change the generated joint angles.

Geometry, tool, gripper, or timing edits require new plans. The runner rejects
stale config/plan combinations; regenerate rather than editing the manifest.
Keep a partner at the emergency stop as required by the lab.

Plans start at `home_q` by default; the runner does not automatically home.
To generate from the robot's current joints instead:

```bash
python3 run.py --state robot_state.json
python3 generate_plan.py --start-state robot_state.json
python3 run.py --check
python3 run.py --execute
```

State capture sends no motion/gripper command. It does not observe block
locations: the empty-grid/waiting-block initial arrangement is still required.
A later `--replan` uses `home_q` again; to preserve a captured start, use the
explicit `generate_plan.py --start-state` command instead.

## Files and behavior

- `run.py`: preflight, state capture, optional replanning, and execution.
- `generate_plan.py`: NumPy-only joint-plan generation.
- `planner/`: numerical IK, task search, RRT-Connect, geometric collision checks,
  and precomputed Panda convex vertices; no simulation/controller code.
- `joint_executor.py`: arm motion via `goto_joints(..., use_impedance=False)`;
  jaw motion via `goto_gripper()`.
- `panda_model.py`: FK, joint limits and timing checks for execution.
- `config.json`, `plans/`, `manifest.json`: calibration, joint plans and hashes.
- `requirements.txt`: NumPy; use the existing FrankaPy/ROS installation.

Like Lab2, planning uses geometric collision checks rather than a physics
engine. Oriented boxes reject clearly separated pairs; a NumPy convex GJK
check resolves potential contacts using the supplied Panda collision vertices.
This avoids rejecting valid approaches due to coarse bounding-box corners.
Checks cover self-collision, table, shelves, other cubes and the held cube.
Only fingers may contact the selected cube; shallow held-cube/table contact
is allowed at placement. Edges are sampled, so collision checking is not a
mathematical guarantee of continuous clearance or calibration accuracy.

The executor checks motion status, measured joints/TCP and jaw width. A motion
fault stops execution before release, without automatic retry/reset/homing.
Checkpoints describe nominal occupancy, not camera measurements. Speeds are
baked into generated plans through the config's joint timing settings.

Simulation remains separate in `lab3_cspace/simulate.py`. The canonical NumPy
planning sources live in `lab3_cspace`; `export_robot.py` packages copies into
`planner/`. After changing planner source, regenerate the robot package from
the repository root with `python3 lab3_cspace/export_robot.py`.
Physical operation of this new implementation remains unverified.
