# Lab 3 simulation

This folder is a standalone MuJoCo project. Its planner, physics model,
configuration, controller, and rendering code are all inside this folder.
Dependencies are in `requirements.txt`. It does not import the robot project.

```bash
# Run these commands from this folder.
python3 run.py --mode plan
MUJOCO_GL=glfw ./simulate.sh --gui --no-record --speed 5 --output results/viewer
./simulate.sh --final-reset
```

Use `--speed 0` for unlimited GUI speed. For a fast headless check:

```bash
./simulate.sh --no-record --final-reset
```

Edit `config/scene.json` for simulation settings. Assets and their license are
in `assets/`. Generated outputs go to this folder's `results/`.
The optional `LAB3_PYTHON` variable selects a Python environment containing
MuJoCo, NumPy, and Pillow.
