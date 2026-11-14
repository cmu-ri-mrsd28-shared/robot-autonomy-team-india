# Lab 3

Run the simulation (Linux):

```bash
python3 main.py
```

On macOS, use `mjpython main.py` for MuJoCo's viewer.

Run the physical robot in the lab's existing FrankaPy environment:

```bash
python3 main.py --hw
```

There are no other run modes. The four assignment patterns are attempted in
order. The planner includes reset moves between patterns; the final pattern
is left in place. The initial grid must be empty and all boxes at their
configured starting locations. Hardware starts with an empty, open gripper.

## Files

- `config.json`: box/grid positions and a few motion/gripper settings.
- `planner.py`: your previous lab3 placement/removal search wrapped in `Planner`,
  with top and side approach-clearance checks.
- `robot.py`: `Robot.pick_place()` selects simulation or FrankaPy.
- `main.py`: plans the sequence and runs it.
- `simulation/`: your previous lab3 MuJoCo scene, IK/RRT and physics executor.
  These reuse the sibling `lab2` model, meshes and kinematics as before.

The previous teammate-style replacement is not used or imported.

## Configuration

The example grid and box coordinates come from your original lab3 layout:
40 mm cubes, 58 mm grid spacing, staging at y=+/-0.23 and x=0.35/0.44/0.53/0.62.
They are not measurements of your lab setup. All positions are meters in the
robot base frame. `boxes` and the grid center refer to **cube centers**.

`tool_to_grasp` is deliberately `null`: no end-effector calibration is invented.
Simulation uses the original lab2 TCP transform. For hardware, measure the
vector from your configured FrankaPy tool frame to the pinch center, expressed
in tool coordinates, and replace null with that XYZ vector. The tool target is
`box_center - rotation @ tool_to_grasp`. If the tool frame already is the pinch
center, the measured offset can be `[0, 0, 0]`.

`travel_height` is absolute hardware tool Z, not a downward distance. Hardware
requires it to be at least 10 cm above pick and place tool targets. `workspace`
is the lower/upper XYZ limit on tool commands. `gripper.opening` is full jaw
width in meters; grasping requests zero width with `gripper.force` N and checks
that the jaws stop at the box width. `speed` is translational speed in m/s.

Top grasps close along x or y. Side grasps approach from +/-x or +/-y and check
an approach lane, finger space, palm space and table clearance. The gripper
clearance dimensions retain the original lab3 finger model, extended with an
approximate Panda palm. Low tabletop side grasps can be rejected because the
hand would hit the table; they are not a universal escape from blocked top
grasps. The planner searches monotone constructions, not arbitrary temporary
rearrangements. It skips targets with no plan under that model.

Simulation checks full mesh collisions and runs contact-based grasp physics.
Hardware uses Cartesian FrankaPy moves with workspace, pose and jaw feedback;
it does not run the MuJoCo arm planner or measure actual block positions. A
successful simulation therefore does not establish physical calibration or
clearance. On hardware failure, the skill is stopped without releasing a held
box. Reconcile box positions before restarting.
