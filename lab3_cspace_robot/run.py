"""Standalone joint-space Franka runner. No imports from the simulation project."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
if __package__:
    from .joint_executor import Executor,validate
    from .panda_model import Panda,rotvec,validate_config
else:
    from joint_executor import Executor,validate
    from panda_model import Panda,rotvec,validate_config

ROOT=Path(__file__).resolve().parent


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def geometry_hash(config):
    # A calibration acknowledgement or description change does not change q.
    return fingerprint({key:value for key,value in config.items()
                        if key not in ('description','hardware_calibrated')})


def read_plan(config,name):
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if geometry_hash(config)!=manifest['geometry_hash']:
        raise ValueError('Config geometry/timing changed: regenerate the joint plans offline before running')
    entry=manifest['plans'][name]
    path=ROOT/entry['file']
    plan=json.loads(path.read_text())
    if fingerprint(plan)!=entry['sha256']:
        raise ValueError('Joint plan differs from the exported plan; regenerate it offline')
    if geometry_hash(plan['config'])!=geometry_hash(config):
        raise ValueError('Joint plan does not match the local calibration config')
    validate(plan)
    return plan


def measured_state(arm,config):
    q=np.asarray(arm.get_joints(),dtype=float)
    if q.shape!=(7,) or not np.isfinite(q).all(): raise RuntimeError('Invalid measured joint feedback')
    pose=arm.get_pose(); predicted=Panda(config).fk(q)
    if (np.asarray(pose.translation).shape!=(3,) or np.asarray(pose.rotation).shape!=(3,3)
            or not np.isfinite(pose.translation).all() or not np.isfinite(pose.rotation).all()):
        raise RuntimeError('Invalid measured tool pose')
    return {'q':q.tolist(),'tool_translation':pose.translation.tolist(),
            'tool_rotation':pose.rotation.tolist(),'predicted_tool_translation':predicted[:3,3].tolist(),
            'model_position_error_m':float(np.linalg.norm(pose.translation-predicted[:3,3])),
            'model_rotation_error_rad':float(np.linalg.norm(rotvec(pose.rotation@predicted[:3,:3].T)))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test',choices=['patterns','rotate-place'],default='patterns')
    parser.add_argument('--final-reset',action='store_true',help='Return blocks to waiting after all patterns')
    parser.add_argument('--replan',action='store_true',help='Generate fresh joint plans from config before checking/executing (NumPy only)')
    modes=parser.add_mutually_exclusive_group()
    modes.add_argument('--state',metavar='JSON',help='Save current joints/tool pose, no motion')
    modes.add_argument('--check',action='store_true',help='Read current robot and verify the plan start/tool model, no motion')
    modes.add_argument('--execute',action='store_true',help='Execute the exported joint plan')
    args=parser.parse_args()
    if args.final_reset and args.test!='patterns': parser.error('--final-reset is only for the pattern sequence')
    config=validate_config(json.loads((ROOT/'config.json').read_text()))
    if args.state:
        if args.replan: parser.error('--replan cannot be combined with --state')
        from frankapy import FrankaArm
        state=measured_state(FrankaArm(),config)
        path=Path(args.state); path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state,indent=2)); print('Saved state; no motion or gripper command sent')
        return
    if args.replan:
        if __package__:
            from .planner.robot_plans import generate
        else:
            from planner.robot_plans import generate
        generate(ROOT/'config.json',ROOT)
        config=validate_config(json.loads((ROOT/'config.json').read_text()))
    name='patterns-reset' if args.final_reset else args.test
    plan=read_plan(config,name)
    print(f"Plan: {name}; {len(plan['actions'])} actions; nominal duration {plan['predicted_duration']:.1f}s")
    print('Calibration verified:',config.get('hardware_calibrated',False))
    for item in plan['reports']: print(item)
    if args.execute and not config.get('hardware_calibrated',False):
        raise RuntimeError('Verify the physical table/tool/locations, then set hardware_calibrated=true in config.json. No robot command sent.')
    if not (args.check or args.execute):
        print('Offline validation passed. Use --check for read-only robot checks or --execute to move.')
        return
    from frankapy import FrankaArm
    arm=FrankaArm()
    state=measured_state(arm,config)
    print(json.dumps(state,indent=2))
    if np.max(abs(np.asarray(state['q'])-plan['start_q']))>.02:
        raise RuntimeError('Robot does not match exported start joints. Capture --state and regenerate plans. No motion commanded.')
    if state['model_position_error_m']>.005 or state['model_rotation_error_rad']>.035:
        raise RuntimeError('Tool/model frame mismatch. Correct calibration before execution. No motion commanded.')
    if args.check:
        print('Start joints and tool frame match; no motion or gripper command sent')
        return
    Executor(arm,plan).execute()
    print('Joint sequence completed. Checkpoints describe nominal block locations, not camera measurements.')


if __name__=='__main__': main()
