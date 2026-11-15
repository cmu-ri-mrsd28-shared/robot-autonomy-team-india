# Lab 3 in configuration space

Block locations are Cartesian XYZ values in metres. Robot configurations are
seven joint angles in radians. This implementation follows the Lab2 approach:
convert grasp poses to joint configurations with numerical IK, collision-check
configuration-space edges, use RRT-Connect when a direct edge is blocked, and
execute joint waypoints. The existing `minifranka` implementation is independent.

## Simulation

Use a Python environment with the packages in `requirements.txt`. There is no
SciPy dependency. The existing working environment on the development machine is
`/home/ashutosh/miniconda3/envs/hw2_autonomy/bin/python`.

From this folder:

```bash
python3 plan.py --test all --output results/patterns_plan.json
python3 simulate.py --plan results/patterns_plan.json --gui --speed 5
```

This is the assignment sequence: build pickaxe, skip infeasible axe, transition
to box, then to bow. Blocks are left in the final arrangement. The planner may
move individual blocks back to waiting when a transition requires it, but does
not reset the full arrangement between patterns.

Without `--gui`, simulation runs headlessly as fast as the CPU permits. With
the viewer, `--speed 2` displays at twice real time, and `--speed 0` removes
display pacing. Neither changes the joint plan or physics timestep.

GUI shutdown waits for MuJoCo's native renderer to finish cleanup before the
process exits. Calling the passive viewer's `close()` alone only requests
shutdown and can race Python/GLFW teardown. MuJoCo currently has no public
viewer-thread join; the adapter checks the handle's weak native-owner
reference with a bounded wait. An incompatible viewer handle or cleanup
timeout raises an error. The physics report is saved before GUI teardown.

For the assignment sequence and a final reset:

```bash
python3 plan.py --test all --final-reset --output results/reset_plan.json
python3 simulate.py --plan results/reset_plan.json --report results/reset_report.json
```

The following is a separate diagnostic, not the assignment run. It deliberately
picks B0, places it at grid cell 0 after an X turn, picks the same cube again,
and returns it to its original waiting position:

```bash
python3 plan.py --test rotate-place --output results/rotate_plan.json
python3 simulate.py --plan results/rotate_plan.json --gui --speed 5
```

`--test pickaxe`, `box`, `bow`, and `axe` plan an individual pattern from the
initial arrangement. An infeasible pattern is reported and skipped. The task
search is A*, with interchangeable same-colour blocks. The pattern order is
pickaxe, axe, box, bow. Final reset returns blocks to waiting slots of their
colour; same-colour identities may be interchanged.

The simulation scene uses the existing Panda meshes from `../simulation/assets/`, with
their existing licence, and explicit Lab2 shelf boxes stored in the JSON
configuration. It does not import any previous simulation/planning controller.
Keep that asset folder with the repo when running simulation. Planning uses
the precomputed collision vertices and needs no mesh/XML files.

The simulator executes the saved joint targets and minimum-jerk interpolation.
It never performs runtime IK, welds a cube to the gripper, snaps an object into
position, or changes a free cube's pose to make a transfer succeed. It checks
grasp width, whether the held block follows the tool, and actual block positions
at pattern checkpoints. A failed physical grasp or arrangement raises an error.
MuJoCo's NoSlip postprocessor is enabled with three iterations to reduce the
soft-contact solver's slow static-friction creep during long joint segments;
this is a numerical contact model, not proof of real-world grasp reliability.

## Planner and units

- `kinematics.py`: the same Panda rigid chain as Lab2, an explicit flange-to-TCP
  transform, analytic Jacobian, damped numerical IK, and minimum-jerk timing.
- `task.py` / `grasp.py`: task search and exact downward X/Y grasp choices.
  There are no hidden 3- or 5-degree yaw offsets.
- `collision.py` / `panda_boxes.json`: NumPy geometric collision checks for
  planning, including self-collision, table, shelves, other cubes and the held
  cube. Oriented-box broad phase plus convex GJK; no MuJoCo dependency.
- `scene.py`: separate MuJoCo physics model used only by the simulator.
- `motion.py`: sampled vertical TCP approaches, IK seeded from the previous
  configuration, sampled edge collision checks, and configuration-space RRT.
- `plan.py`: prepares the full sequence before execution and writes a JSON plan.
- `simulate.py`: physical simulation executor.
- `export_robot.py`: exports offline joint plans to the separate robot folder.

Transfers first try a collision-free straight edge in **joint space**. RRT is
needed only when that edge is blocked. Pickup and placement instead sample a
vertical Cartesian line at 10 mm intervals and convert it into a continuous IK
branch. Joint edges are checked at at most 0.035 rad per joint; intermediate
TCP deviation is checked during vertical approaches. These are sampled checks,
not a mathematical guarantee of continuous collision freedom.

Each waypoint is a separate rest-to-rest joint movement, like Lab2, rather than
a high-frequency stream of new skills. Duration accounts for the exact quintic
peak velocity, acceleration and jerk. `motion.joint_speed` is rad/s,
`joint_acceleration` is rad/s², `joint_jerk` is rad/s³, and durations are seconds.
Conservative timing makes the first hardware implementation intentionally slow.

`simulation.json` uses physical cube centres at `table_z + cube_size/2` and a
TCP grasp height of 0.0216 m. These are **not** the old taught tool-Z measurements.
The flange-to-TCP transform is fixed and explicit; simulation does not derive
it from a desired grasp height. FK is checked against MuJoCo when the model is
created.

## Separate robot implementation

The standalone robot runtime is in [`../lab3_cspace_robot`](../lab3_cspace_robot/README.md).
Copy only that folder to the robot PC. It includes a standalone NumPy planner,
so joint plans can be generated on the robot PC before execution. It contains
no simulation imports or MuJoCo dependency. Arm motion uses FrankaPy
`goto_joints()`; numerical IK runs during planning, not during movement.

On the planning machine, export all robot plans using the calibrated robot
configuration:

```bash
python3 export_robot.py --config ../lab3_cspace_robot/config.json
```

Optional `--start-state robot_state.json` uses joints captured with the robot
runner's `--state` command. See the robot README for calibration, read-only
checks and execution commands. Its `config.json` is the operational hardware
configuration; `hardware.json` here is only the original draft profile.
Planning needs only NumPy. The `requirements.txt` here also includes MuJoCo
for simulation; it is not needed for `plan.py` or `export_robot.py`. Physical
execution remains unverified. The former prototype hardware entry
point in this simulation folder has been removed.

## Development checks

```bash
python3 -m pytest tests -q
```

Tests exercise exact-angle IK/Jacobians, RRT around a blocking configuration-space
obstacle, invalid waypoint timing/continuity, joint-only hardware execution, and
stopping before release after a fault. Simulation reports contain measured block
position errors, not only task-planner bookkeeping.
