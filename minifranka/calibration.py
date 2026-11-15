"""Calibration helpers: --grasp reads an offset; --move commands tool motion."""

import argparse
import json
from pathlib import Path

import numpy as np


def move_to_position(config, name):
    """Move the tool origin directly to boxes[name], keeping its orientation."""
    xyz = np.asarray(config['boxes'][name], dtype=float)
    if xyz.shape != (3,) or not np.isfinite(xyz).all():
        raise ValueError(f'boxes.{name} must contain three finite XYZ coordinates')
    speed = float(config.get('speed', 0.025))
    if not np.isfinite(speed) or speed <= 0:
        raise ValueError('speed must be positive')

    from frankapy import FrankaArm

    arm = FrankaArm()
    pose = arm.get_pose()
    duration = max(10.0, 1.875 * float(np.linalg.norm(xyz - pose.translation)) / speed)
    pose.translation = xyz
    print(f'Moving tool directly to {name}: {xyz.tolist()}, duration {duration:.1f}s; keeping current orientation', flush=True)
    arm.goto_pose(pose, duration=duration, use_impedance=False, ignore_errors=False)
    print('Reported tool XYZ:', arm.get_pose().translation.tolist(), flush=True)


def calibrate_grasp(config):
    center = np.asarray(config['boxes']['R0'], dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError('boxes.R0 must contain three measured, finite coordinates')

    from frankapy import FrankaArm

    pose = FrankaArm().get_pose()
    translation = np.asarray(pose.translation, dtype=float)
    rotation = np.asarray(pose.rotation, dtype=float)
    if (translation.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(translation).all() or not np.isfinite(rotation).all()):
        raise ValueError('Invalid tool-pose feedback')
    offset = rotation.T @ (center - translation)
    print('Assumes the gripper is aligned to grasp R0 and boxes.R0 is measured.')
    print('Copy into config.json; no robot motion or config changes were requested:')
    print(json.dumps({'tool_to_grasp': offset.round(6).tolist()}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--grasp', action='store_true', help='Print tool_to_grasp from current pose and configured R0 center')
    mode.add_argument('--move', choices=[f'{color}{i}' for color in 'RB' for i in range(4)], metavar='BOX', help='Move the tool directly to a configured box XYZ, e.g. --move R0')
    args = parser.parse_args()
    if not args.grasp and args.move is None:
        parser.print_help()
        return
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.move is not None:
        move_to_position(config, args.move)
    else:
        calibrate_grasp(config)


if __name__ == '__main__':
    main()
