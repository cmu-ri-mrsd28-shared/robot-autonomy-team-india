# Robot Autonomy Lab 3

The implementations are two independent projects:

- `simulation/`: MuJoCo simulation, its own planner, controller, assets, config,
  dependencies, and launch script.
- `franka_hardware/`: physical FrankaPy execution, its own planner, calibration,
  dependencies, and `run.py`. It imports no simulation modules.

The lab handout is `Robot_Autonomy_Lab_3.pdf`.

## Simulation

```bash
cd simulation
MUJOCO_GL=glfw ./simulate.sh --gui --no-record --speed 5
```

See `simulation/README.md` for headless runs and recording.

## Physical robot

Copy only `franka_hardware/` or `dist/lab3_franka.zip` to the robot PC.
Keep your measured `calibration.json` when updating an existing installation.
From the robot project folder:

```bash
# Preview without connecting to the robot.
python3 run.py --final-reset

# Preview the full sequence with 2x arm-motion speed.
python3 run.py --final-reset --speed-scale 2

# Print the current pose without movement.
python3 run.py --read-pose
```

Physical execution and calibration instructions are in
`franka_hardware/README.md`. An existing reflex stop still needs diagnosis
before retrying robot motion; project separation does not resolve that fault.
