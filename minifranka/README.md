# Lab 3: patterns, top grasps, pick/place

```bash
python3 main.py       # simulation on Linux
mjpython main.py      # simulation on macOS
python3 main.py --hw  # hardware, with an input pause between patterns
```

## Parts

- `planner.py`: `Pattern().get()` returns all four patterns;
  `Pattern().get('box')` returns one 3x3 array. No global pattern data.
- `grasp.py`: `Grasp` considers only downward grasps closing along X or Y.
- `planner.py`: `Planner.plan(pattern, locations)` plans a transition from the
  current arrangement. Blocks already in suitable cells can stay there.
- `robot.py`: `Robot` runs simulation. `HardwareRobot(Robot)` runs FrankaPy.
- `simulation/`: MuJoCo pose/move/grip adapter, viewer and contact physics.

`Robot` has four movement operations:

```python
robot.move_above("B0", "Y")  # travel height, above a waiting location
robot.pick("B0", "Y")  # pick this block from its current location
robot.place((0, 0), "X")  # place the held block at grid row 0, column 0
robot.pick_place(move)  # execute one instruction returned by Planner
```

`pick` and `place` include the approach and lift. `move_above` can be exercised
on its own. On hardware these methods command actual movement.

Example setup for calling them individually:

```python
import json
from planner import Planner
from robot import Robot, HardwareRobot

config = json.load(open("config.json"))
planner = Planner(config)
robot = Robot(config, planner)  # use HardwareRobot(config, planner) on the lab PC
```

## Patterns and transitions

```python
from planner import Pattern

robot.run_pattern("pickaxe", Pattern().get("pickaxe"))
robot.run_pattern("box", Pattern().get("box"))
```

`robot.locations` remembers each block, for example:

```python
{'R0': 'grid:4', 'R1': 'waiting:R1', ...}
```

During a pick it records `'held'`; after placement it records the destination.
Locations are nominal software bookkeeping, not camera measurements. They
persist across patterns in the current run. Starting a new Robot object assumes
all blocks are back in their configured waiting positions.

The planner combines transfers between grid cells, additions from waiting and
removals to waiting. It does not reset the entire arrangement between patterns.
Waiting slots of the same color may hold different individual blocks; the
location dictionary records which block actually occupies each slot.

The planner checks feasibility before moving. Under the intended crowded-grid
model, axe's filled 2x2 group has no possible last insertion and is skipped.
Pickaxe, box and bow are feasible. In the current configuration the planner
uses 5 moves to construct pickaxe, 5 to transition to box, and 5 to transition
to bow. Move counts can change with configuration or starting arrangement.

Hardware pauses before the next feasible pattern, after showing its move count.
Press Enter to continue or `q` to finish. Infeasible patterns are skipped without
a hardware prompt. Simulation continues automatically. Both backends inherit the same move_above, pick and place routines.

## Hardware motion and configuration

The working hardware path is retained:

```
raise to travel Z -> above source XY -> orient -> descend -> grasp -> lift
-> above destination XY -> orient -> descend -> release -> lift
```

All hardware descent uses `grasp_height`; all XY transfer uses `travel_height`
(currently 0.30 m). Speed is controlled by `speed` in m/s. Opening and force come
from `gripper.opening` and `gripper.force`. Coordinate triples stay on one line
in `config.json`. Existing measurements are preserved.

X/Y means the direction the jaws close in the table plane. Both grasps point
straight down. Hardware chooses the closer equivalent wrist heading, avoiding
unnecessary 180-degree turns. Orientation changes happen at travel height.
There are no side grasps, application pose-error rejection gates, custom
workspace/travel rejection gates, or custom jaw-width rejection gates.
FrankaPy/controller error reporting remains enabled.

Waiting and grid X/Y coordinates come from the config. Hardware uses them as
taught tool targets and ignores individual Z values in favor of `grasp_height`.
Simulation derives
cube-center Z as `table_height + box_width/2`, so taught tool heights do not
make simulated boxes float above the table. Simulation requires the existing
lab2 folder/model assets, discovered from a sibling lab2 or an ancestor's labs/lab2.

Grasp clearance uses the original approximate finger footprint, including the
previously requested 10 mm overlap allowance. It is a task-level model;
hardware does not run full-arm collision planning or measure block placement.
MuJoCo handles physical contacts. Its adapter uses numerical joint control for Cartesian commands; there is no separate simulation task planner, RRT or pick/place routine.

## Speed and editing

Arrays in `config.json` stay on one line, including each R0–B3 XYZ triple.
The unused `tool_to_grasp`, `approach_distance` and `workspace` keys are removed.

`speed` remains a translational speed in m/s: 0.05 is 5 cm/s. It controls all
hardware arm moves and calibration moves. Doubling it halves translation,
rotation and minimum arm-motion durations. Simulation receives exactly the same arm-motion durations as hardware; its physics
timestep is unchanged. Jaw-operation speed, force, grasp settling time and
user-input pauses do not scale.

With the tool at the desired gripping height, `python3 calibration.py --grasp`
reads and prints `grasp_height`. It does not move the robot or write the config.

## Shared execution

`main.py` constructs one Planner. Robot.move_above(), pick(), place() and
pick_place() are shared by simulation and HardwareRobot through inheritance.
The hardware subclass only supplies tool-pose feedback, Cartesian movement,
gripper commands and the requested inter-pattern prompt.

The simulator implements the corresponding pose/move/grip interface. It does
not instantiate another Planner or generate its own pick/place trajectories.
It interpolates the same Cartesian commands and converts them to joint targets
for the MuJoCo actuators, analogous to FrankaPy's low-level Cartesian interface.
Physics advances normally; blocks are neither attached nor snapped to targets.
The simulated reported tool frame is aligned with grasp_height so both modes
use the same two tool heights. Cube centers rest at table_height + box_width/2.
