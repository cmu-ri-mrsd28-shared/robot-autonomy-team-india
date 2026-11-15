"""Calibration: --grasp reads the common tool height; --move commands motion."""

import argparse
import json
from pathlib import Path

import numpy as np
from robot import Robot


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
    print(f'Moving tool directly to {name}: {xyz.tolist()}, duration {duration:.1f}s; keeping current orientation', flush=True)
    arm.goto_pose(pose, duration=duration, use_impedance=False, ignore_errors=False)
    print('Reported tool XYZ:', arm.get_pose().translation.tolist(), flush=True)


def calibrate_grasp():
    """Read the tool height while positioned at a correct grasp; no motion."""
    from frankapy import FrankaArm

    pose = FrankaArm().get_pose()
    print("Current tool XYZ:", pose.translation.tolist())
    print("Copy this height into config.json:")
    print(json.dumps({"grasp_height": float(pose.translation[2])}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--grasp', action='store_true', help='Print the current tool Z as grasp_height; no motion')
    mode.add_argument('--move', choices=[f'{color}{i}' for color in 'RB' for i in range(4)], metavar='BOX', help='Move the tool directly to a configured box XYZ, e.g. --move R0')
    args = parser.parse_args()
    if not args.grasp and args.move is None:
        parser.print_help()
        return
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.move is not None:
        move_to_position(config, args.move)
    else:
        calibrate_grasp()


if __name__ == '__main__':
    main()
