#!/usr/bin/env python3
"""Lab 3 FrankaPy runner. Default: dry run, with no robot connection."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
import time

from .patterns import PATTERNS
from .task import compile_program, validate_config
from .backend import DryRunBackend, FrankaPyBackend
from .executor import Executor


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'calibration.json')
    parser.add_argument('--patterns',nargs='+',choices=PATTERNS,default=list(PATTERNS))
    parser.add_argument('--final-reset',action='store_true')
    parser.add_argument('--test-transfer',action='store_true',help='One staged block to the first target cell and back')
    parser.add_argument('--speed-scale',type=float,choices=(1.,2.),default=1.,
                        help='Arm-motion speed relative to calibration.json: 1=unchanged, 2=double (explicit opt-in)')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute',action='store_true',help='Connect to FrankaPy and command the physical robot')
    mode.add_argument('--read-pose',action='store_true',help='Connect and print current tool pose; send no motion commands')
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'results/franka_dry_run')
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    if args.read_pose:
        from frankapy import FrankaArm
        arm = FrankaArm(robot_num=config['robot_num'])
        pose = arm.get_pose()
        print(json.dumps({'translation_m':pose.translation.tolist(),'rotation_matrix':pose.rotation.tolist(),
                          'quaternion_wxyz':pose.quaternion.tolist(),
                          'gripper_width_m':float(arm.get_gripper_width()),
                          'ee_force_torque':arm.get_ee_force_torque().tolist()},indent=2))
        return 0
    # Keep the measured configuration file intact. Scaling all three speed
    # terms and the minimum duration divides every requested arm duration by
    # the same factor, including short motions limited by minimum duration.
    config = deepcopy(config)
    base_motion_settings = {key:config[key] for key in (
        'max_translation_speed_m_s','max_vertical_speed_m_s',
        'max_rotation_speed_rad_s','minimum_motion_duration_s')}
    for key in ('max_translation_speed_m_s','max_vertical_speed_m_s','max_rotation_speed_rad_s'):
        config[key] *= args.speed_scale
    config['minimum_motion_duration_s'] /= args.speed_scale
    # Validate before constructing FrankaArm, importing ROS, or sending commands.
    validate_config(config,execute=args.execute)
    program = compile_program(config,args.patterns,args.final_reset,args.test_transfer)
    if not any(step['kind'] == 'transfer' for step in program):
        print('No block transfers: all requested targets are model-infeasible. No robot connection made.')
        return 0
    if args.execute and args.output.resolve() == (Path(__file__).resolve().parents[1]/'results/franka_dry_run').resolve():
        args.output = args.output.with_name('franka_hardware_run')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'program.json').write_text(json.dumps(program,indent=2)+'\n')
    transfer_count = sum(step['kind']=='transfer' for step in program)
    scope = 'ONE-BLOCK ROUND TRIP (--test-transfer)' if args.test_transfer else 'PATTERN SEQUENCE'
    print(f'Program: {scope}; {transfer_count} transfers; final reset={args.final_reset}',flush=True)
    print(f'Arm speed scale: {args.speed_scale:g}x; gripper speed, force, and protection settings unchanged',flush=True)
    print(f'Diagnostics: {args.output.resolve()}',flush=True)
    report = {'status':'RUNNING','mode':'hardware' if args.execute else 'dry_run',
              'motion_speed_scale':args.speed_scale,'base_motion_settings':base_motion_settings,
              'config':config,'program':program,'object_poses_verified':False,
              'grasp_model':'NS/EW top grasps; physical geometry requires calibration',
              'description':'Robot feedback checks do not measure block positions or prove completed physical patterns'}
    backend = None
    executor = None
    started = time.monotonic()

    def checkpoint(state):
        payload = dict(status='RUNNING',mode=report['mode'],motion_speed_scale=args.speed_scale,
                       elapsed_s=time.monotonic()-started)
        payload.update(state)
        temporary = args.output/'progress.json.tmp'
        temporary.write_text(json.dumps(payload,indent=2)+'\n')
        temporary.replace(args.output/'progress.json')

    try:
        backend = FrankaPyBackend(config) if args.execute else DryRunBackend(config)
        executor = Executor(config,backend,progress=checkpoint)
        executor.run(program)
        report['status'] = 'COMMAND_SEQUENCE_COMPLETED' if args.execute else 'DRY_RUN_COMPLETED'
        checkpoint(dict(executor.snapshot(),status=report['status']))
        print(f"{report['status']}: {len(executor.transfers)} block transfers",flush=True)
    except BaseException as error:
        report['status'] = 'INTERRUPTED' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'FAILED'
        report['error'] = f'{type(error).__name__}: {error}'
        print(f"STOPPED: {report['error']}",file=sys.stderr,flush=True)
        print(f"Read {args.output/'progress.json'} and {args.output/'report.json'} for the failed phase.",file=sys.stderr,flush=True)
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        if executor:
            report.update(transfers=executor.transfers,events=executor.events,
                          nominal_locations=executor.locations,held_block=executor.held_block,
                          failed_phase=executor.phase if executor.state_uncertain else None,
                          state_uncertain=executor.state_uncertain)
        if backend:
            report['trace'] = backend.trace
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
