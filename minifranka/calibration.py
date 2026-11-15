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


def report_state():
    """Read SDK state and frame information without arm or gripper commands."""
    import frankapy

    arm = frankapy.FrankaArm()
    report = {
        'frankapy_file': getattr(frankapy, '__file__', 'unknown'),
        'robot_state': arm.get_robot_state(),
        'reported_tool_pose': arm.get_pose(),
        'tool_delta_pose': arm.get_tool_base_pose(),
        'units': 'positions m; joint angles rad; joint torques Nm; wrench [N,N,N,Nm,Nm,Nm]',
        'note': 'pose_desired is libfranka state, not necessarily the impedance trajectory goal; reads are sequential',
    }
    constants = getattr(frankapy, 'FrankaConstants', None)
    for key in ('DEFAULT_TRANSLATIONAL_STIFFNESSES', 'DEFAULT_ROTATIONAL_STIFFNESSES'):
        report[key] = getattr(constants, key, 'not exported by this SDK')

    def encode(value):
        if hasattr(value, 'rotation') and hasattr(value, 'translation'):
            rotation = np.asarray(value.rotation, dtype=float)
            return {
                'translation': np.asarray(value.translation).tolist(),
                'rotation': rotation.tolist(),
                'rotation_determinant': float(np.linalg.det(rotation)),
                'rotation_orthogonality_error': float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)

    log('Read-only robot state; no motion or gripper command sent:')
    log(json.dumps(report, default=encode, indent=2))


def check1_travel(robot):
    """Move above B0 and try both downward grasp orientations; no pickup."""
    robot.move_above("B0", "Y")
    robot.move_above("B0", "X")


def check2_pickone(robot):
    """Pick B0 and return it to the same waiting position."""
    robot.pick("B0", "Y")
    robot.place("B0", "Y")


def check3_grid(robot, grid_orientation="Y", x_offset_deg=0.0):
    """Round trip via top-left; optionally test a fixed X heading with yaw offset."""
    robot.planner.grasps.x_yaw_offset_deg = x_offset_deg
    if grid_orientation != "Y":
        heading = (90 if grid_orientation == "X90" else 270) + x_offset_deg
        log(f"Testing {grid_orientation} with offset {x_offset_deg:+g} degrees: "
            f"target heading {heading:g} degrees about base +Z from canonical downward Y; "
            "waiting uses normal Y")
    robot.pick("B0", "Y")
    robot.place((0, 0), grid_orientation)
    robot.pick("B0", grid_orientation)
    robot.place("B0", "Y")


def check3_grid_rotate(robot):
    """Repeat the original Y pickup -> X placement -> return experiment."""
    robot.pick("B0", "Y")
    robot.place((0, 0), "X")
    robot.pick("B0", "X")
    robot.place("B0", "Y")


def check4_pattern(robot):
    """Run pickaxe and leave the completed pattern in the grid."""
    robot.run_pattern("pickaxe", Pattern().get("pickaxe"))


def check5_x_to_y(robot):
    """Reproduce grid X -> travel above B2 -> Y, without descending or gripping."""
    grasps = robot.planner.grasps
    grasps.x_yaw_offset_deg = grasps.yaw_offsets["X"]
    log(f"X-to-Y check: grid heading {90 + grasps.yaw_offsets['X']:g} degrees; "
        f"normal Y offset {grasps.yaw_offsets['Y']:+g} degrees; no descent or gripper commands")
    robot.move_above((0, 0), "X90")
    robot.move_above("B2", "Y")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--grasp', action='store_true', help='Print the current tool Z as grasp_height; no motion')
    mode.add_argument('--state', action='store_true', help='Read robot state, forces, frame data and SDK location; no motion')
    mode.add_argument('--move', choices=[f'{color}{i}' for color in 'RB' for i in range(4)], metavar='BOX', help='Move to a box XY at grasp_height')
    mode.add_argument('--check1-travel', dest='check', action='store_const', const=check1_travel, help='Above B0: try Y and X orientations without descending')
    mode.add_argument('--check2-pickone', dest='check', action='store_const', const=check2_pickone, help='Pick B0 and return it to its waiting position')
    mode.add_argument('--check3-grid', dest='check', action='store_const', const=check3_grid, help='B0 to top-left grid cell and back to waiting')
    mode.add_argument('--check3-grid-rotate', dest='check', action='store_const', const=check3_grid_rotate, help='B0 Y pickup, X placement at top-left cell, then return; stop on first fault')
    mode.add_argument('--check4-pattern', dest='check', action='store_const', const=check4_pattern, help='Build pickaxe; leave the pattern in the grid')
    mode.add_argument('--check5-x-to-y', dest='check', action='store_const', const=check5_x_to_y,
                      help='At travel height: grid X heading, then travel to B2 and rotate to Y; no grasp')
    parser.add_argument('--speed', type=float, help='Override arm speed in m/s for this run; config.json is not edited')
    parser.add_argument('--relative-z', action='store_true',
                        help='Use translation-only goto_pose_delta for vertical travel, descents and lifts')
    parser.add_argument('--cartesian-impedance', action='store_true',
                        help='Use FrankaPy Cartesian impedance control with SDK default gains for all arm motions in this check')
    heading = parser.add_mutually_exclusive_group()
    heading.add_argument('--x90', dest='grid_orientation', action='store_const', const='X90',
                         help='With --check3-grid: fixed downward X heading, +90 degrees about base +Z from canonical Y')
    heading.add_argument('--x270', dest='grid_orientation', action='store_const', const='X270',
                         help='With --check3-grid: fixed downward X heading, +270 degrees (-90) about base +Z from canonical Y')
    parser.add_argument('--x-offset-deg', type=float,
                        help='With --check3-grid and --x90/--x270: add this yaw in degrees; e.g. 2 gives 92/272, -2 gives 88/268')
    args = parser.parse_args()
    if args.relative_z and args.check is None:
        parser.error('--relative-z requires a hardware check')
    if args.cartesian_impedance and args.check is None:
        parser.error('--cartesian-impedance requires a hardware check')
    if args.x_offset_deg is not None:
        if args.grid_orientation is None:
            parser.error('--x-offset-deg requires --x90 or --x270')
        if not np.isfinite(args.x_offset_deg):
            parser.error('--x-offset-deg must be finite')
    if args.grid_orientation is not None and args.check is not check3_grid:
        parser.error('--x90 and --x270 require --check3-grid')
    if args.speed is not None and (not np.isfinite(args.speed) or args.speed <= 0):
        parser.error('--speed must be a positive finite value in m/s')
    if not args.grasp and not args.state and args.move is None and args.check is None:
        parser.print_help()
        return
    if args.state:
        report_state()
        return
    config = json.loads(Path(__file__).with_name('config.json').read_text())
    if args.speed is not None:
        config['speed'] = args.speed
    if args.relative_z:
        config['relative_vertical_motion'] = True
    if args.cartesian_impedance:
        config['cartesian_impedance'] = True
    if args.move is not None:
        move_to_position(config, args.move)
    elif args.grasp:
        calibrate_grasp()
    else:
        log(f"Hardware check: speed={config['speed']} m/s, "
              f"travel Z={config['travel_height']}, grasp Z={config['grasp_height']}, "
              f"relative Z={config.get('relative_vertical_motion', False)}, "
              f"cartesian impedance={config.get('cartesian_impedance', False)}", flush=True)
        if config.get('cartesian_impedance', False):
            log(f"Impedance final-target tracking buffer: {HardwareRobot.IMPEDANCE_BUFFER_TIME}s; "
                "position error limit remains 0.005 m", flush=True)
        robot = HardwareRobot(config, Planner(config))
        try:
            if args.check is check3_grid:
                check3_grid(robot, args.grid_orientation or "Y", args.x_offset_deg or 0.0)
            else:
                args.check(robot)
        finally:
            robot.close()


if __name__ == '__main__':
    main()
