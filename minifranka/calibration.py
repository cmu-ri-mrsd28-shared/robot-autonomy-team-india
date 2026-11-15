"""Calibration and individual hardware checks, using config.json motion settings."""

import argparse
import json
from pathlib import Path

import numpy as np
from console import log
from robot import Robot, HardwareRobot
from planner import Pattern, Planner


def move_to_position(config, name):
    """Move to box XY at the common hardware grasp height, keeping orientation."""
    xyz = np.asarray(config['boxes'][name], dtype=float).copy()
    if 'grasp_height' in config:
        xyz[2] = config['grasp_height']
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f'boxes.{name} must contain three finite XYZ coordinates')
    speed = float(config.get('speed', 0.025))
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('speed must be positive')

    from frankapy import FrankaArm

    arm = FrankaArm()
    pose = arm.get_pose()
    duration = Robot.motion_duration(float(np.linalg.norm(xyz - pose.translation)), 0.0, speed)
    pose.translation = xyz
    log(f'Moving tool directly to {name}: {xyz.tolist()}, duration {duration:.1f}s; keeping current orientation', flush=True)
    arm.goto_pose(pose, duration=duration, use_impedance=False, ignore_errors=False)
    log('Reported tool XYZ:', arm.get_pose().translation.tolist(), flush=True)


def calibrate_grasp():
    """Read the tool height while positioned at a correct grasp; no motion."""
    from frankapy import FrankaArm

    pose = FrankaArm().get_pose()
    log("Current tool XYZ:", pose.translation.tolist())
    log("Copy this height into config.json:")
    log(json.dumps({"grasp_height": float(pose.translation[2])}, indent=2))


def check1_travel(robot):
    """Move above B0 and try both downward grasp orientations; no pickup."""
    robot.move_above("B0", "Y")
    robot.move_above("B0", "X")


def check2_pickone(robot):
    """Pick B0 and return it to the same waiting position."""
    robot.pick("B0", "Y")
    robot.place("B0", "Y")


def check3_grid(robot):
    """Place B0 in the top-left grid cell, then return it to waiting."""
    robot.pick("B0", "Y")
    robot.place((0, 0), "X")
    robot.pick("B0", "X")
    robot.place("B0", "Y")


def check4_pattern(robot):
    """Run pickaxe and leave the completed pattern in the grid."""
    robot.run_pattern("pickaxe", Pattern().get("pickaxe"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--grasp', action='store_true', help='Print the current tool Z as grasp_height; no motion')
    mode.add_argument('--move', choices=[f'{color}{i}' for color in 'RB' for i in range(4)], metavar='BOX', help='Move to a box XY at grasp_height')
    mode.add_argument('--check1-travel', dest='check', action='store_const', const=check1_travel, help='Above B0: try Y and X orientations without descending')
    mode.add_argument('--check2-pickone', dest='check', action='store_const', const=check2_pickone, help='Pick B0 and return it to its waiting position')
    mode.add_argument('--check3-grid', dest='check', action='store_const', const=check3_grid, help='B0 to top-left grid cell and back to waiting')
    mode.add_argument('--check4-pattern', dest='check', action='store_const', const=check4_pattern, help='Build pickaxe; leave the pattern in the grid')
    args = parser.parse_args()
    if not args.grasp and args.move is None and args.check is None:
        parser.print_help()
        return
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.move is not None:
        move_to_position(config, args.move)
    elif args.grasp:
        calibrate_grasp()
    else:
        log(f"Hardware check: speed={config['speed']} m/s, "
              f"travel Z={config['travel_height']}, grasp Z={config['grasp_height']}", flush=True)
        robot = HardwareRobot(config, Planner(config))
        try:
            args.check(robot)
        finally:
            robot.close()


if __name__ == '__main__':
    main()
