# Robot Autonomy Lab 3: Task Planning

Group submission for 16-662 Lab 3. A Franka Panda rearranges four red and four
blue blocks on a 3x3 grid into the patterns from the handout (pickaxe, axe,
box, bow), choosing a top-grasp orientation that avoids neighbouring blocks.

> **The submitted implementation is [`minifranka/`](minifranka/).**
> It is the code that ran on the robot for the video.
> Full documentation is in [`minifranka/README.md`](minifranka/README.md).

The lab handout is `Robot_Autonomy_Lab_3.pdf`.

## Running the submitted code

From `minifranka/`:

```bash
python3 main.py --hw --joints   # physical robot, as used for the video
python3 main.py                 # MuJoCo simulation on Linux (mjpython on macOS)
python3 main.py --plan-only     # print the plan for each pattern, no motion
python3 -m pytest -q tests      # offline tests, no robot needed
```

Start a hardware run with all eight blocks in their waiting positions, the
grid empty and the gripper empty. The run pauses for Enter between patterns.

## What it does

- **Planning** (`planner.py`): A* search over block positions finds the
  shortest sequence of pick-and-place moves from the current arrangement to
  the next pattern, so blocks already in a useful cell are reused instead of
  resetting between patterns.
- **Grasp orientation** (`grasp.py`): each pick and place uses a downward
  grasp closing along X or Y, whichever keeps the fingers clear of blocks in
  neighbouring cells.
- **Infeasible patterns are skipped**: the axe has no valid top-grasp order
  with this finger clearance, so the robot reports that and moves on to the
  box, as the handout allows.
- **Execution** (`robot.py`, `joint_motion.py`, `kinematics.py`): simulation
  and hardware share the same pick and place routines. On hardware, `--joints`
  solves inverse kinematics and sends joint-space moves through FrankaPy.
- **Measured setup** (`config.json`): block and grid positions, grasp and
  travel heights, gripper settings and speed.

## Other folders

`simulation/`, `franka_hardware/` and `dist/` are earlier, separate
implementations kept for reference. They are not part of the submitted run;
each has its own README.
