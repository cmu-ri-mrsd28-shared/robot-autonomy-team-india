"""Small tests for geometry, planning and hardware failure behavior."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from lab3_cspace.kinematics import Panda,load_config,rotvec,rz,duration,QMIN,QMAX
from lab3_cspace.motion import JointPlanner
from lab3_cspace_robot.joint_executor import validate,Executor
from lab3_cspace.task import Planner,Pattern
from lab3_cspace.kinematics import task_config


@pytest.fixture
def cfg(): return load_config(Path(__file__).resolve().parents[1]/'simulation.json')


def test_ik_recovers_pose_including_exact_half_turn(cfg):
    robot=Panda(cfg); target=robot.fk(robot.home)
    target[:3,3]+=[.03,-.08,-.05]
    target[:3,:3]=rz(np.pi/2)@target[:3,:3]
    q=robot.ik(target,robot.home)
    assert q is not None
    actual=robot.fk(q)
    assert np.linalg.norm(actual[:3,3]-target[:3,3])<.0002
    assert np.linalg.norm(rotvec(actual[:3,:3]@target[:3,:3].T))<.001
    assert np.isclose(np.linalg.norm(rotvec(rz(np.pi))),np.pi)


def test_analytic_jacobian_matches_finite_difference(cfg):
    robot=Panda(cfg); q=robot.home.copy(); t,j=robot.fk(q,True)
    for i in range(7):
        step=q.copy(); step[i]+=1e-6; moved=robot.fk(step)
        numerical=np.r_[(moved[:3,3]-t[:3,3])/1e-6,rotvec(moved[:3,:3]@t[:3,:3].T)/1e-6]
        np.testing.assert_allclose(j[:,i],numerical,atol=1e-5)


def test_rrt_routes_around_configuration_space_obstacle(cfg):
    planner=JointPlanner(None,SimpleNamespace(last_collision=None),cfg,seed=13)
    a=np.array(cfg['home_q']); b=a.copy(); a[0]=-.7; b[0]=.7
    def valid(q):
        return bool(np.all(q>=QMIN) and np.all(q<=QMAX) and not (abs(q[0])<.2 and abs(q[1]+np.pi/4)<.2))
    assert not planner.edge(a,b,valid)
    path=planner.transfer(a,b,valid)
    assert planner.rrt_searches==1 and len(path)>2
    np.testing.assert_allclose(path[0],a); np.testing.assert_allclose(path[-1],b)
    assert all(planner.edge(x,y,valid) for x,y in zip(path[:-1],path[1:]))


@pytest.fixture
def plan(cfg):
    a=np.array(cfg['home_q']); b=a.copy(); b[0]+=.03
    return {'schema':1,'config':cfg,'config_hash':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
            'start_q':a.tolist(),'actions':[
                {'kind':'move','label':'test','start_q':a.tolist(),'goal_q':b.tolist(),'duration':duration(a,b,cfg['motion'])},
                {'kind':'grip','label':'release','width':.08}]}


def test_plan_rejects_zero_duration_or_joint_jump(plan):
    invalid=copy.deepcopy(plan); invalid['actions'][0]['duration']=0
    with pytest.raises(ValueError,match='timing'): validate(invalid)
    invalid=copy.deepcopy(plan); invalid['actions'][0]['start_q'][0]+=.2
    with pytest.raises(ValueError,match='discontinuous'): validate(invalid)


def test_hardware_sends_joint_targets_and_never_cartesian_pose(plan):
    arm=Mock(); robot=Panda(plan['config']); state=[np.array(plan['start_q'])]
    arm.get_joints.side_effect=lambda:state[0]
    def pose():
        t=robot.fk(state[0]); return SimpleNamespace(translation=t[:3,3],rotation=t[:3,:3])
    arm.get_pose.side_effect=pose
    arm.goto_joints.side_effect=lambda q,**kwargs:state.__setitem__(0,np.array(q))
    arm.is_skill_done.return_value=True; arm._client.get_state.return_value=3
    arm._get_current_franka_interface_status.return_value=SimpleNamespace(franka_interface_status=SimpleNamespace(error_description=''))
    arm.get_gripper_width.return_value=.08
    Executor(arm,plan).execute()
    arm.goto_joints.assert_called_once(); arm.goto_gripper.assert_called_once()
    arm.goto_pose.assert_not_called(); arm.goto_pose_delta.assert_not_called()


def test_hardware_fault_stops_before_release(plan):
    arm=Mock(); robot=Panda(plan['config']); t=robot.fk(plan['start_q'])
    arm.get_joints.return_value=np.array(plan['start_q'])
    arm.get_pose.return_value=SimpleNamespace(translation=t[:3,3],rotation=t[:3,:3])
    arm.goto_joints.side_effect=RuntimeError('joint reflex')
    arm._client.get_state.return_value=3
    with pytest.raises(RuntimeError,match='joint reflex'): Executor(arm,plan).execute()
    arm.goto_gripper.assert_not_called(); arm.reset_joints.assert_not_called()
    arm._client.cancel_goal.assert_not_called()  # Terminal goal: avoid duplicate cancellation.


def test_task_sequence_builds_requested_patterns_and_skips_axe(cfg):
    planner=Planner(task_config(cfg)); locations=planner.initial_locations.copy()
    for name,pattern in Pattern().get().items():
        moves=planner.plan(pattern,locations)
        if name=='axe':
            assert moves is None
            continue
        assert moves is not None
        for move in moves:
            assert move.target not in locations.values()
            locations[move.block]=move.target
        board={place:block[0] for block,place in locations.items() if place.startswith('grid:')}
        expected={f'grid:{3*r+c}':color for r,row in enumerate(pattern) for c,color in enumerate(row) if color!='.'}
        assert board==expected


def test_invalid_joint_feedback_never_sends_a_command(plan):
    arm=Mock(); arm.get_joints.return_value=np.full(7,np.nan)

    with pytest.raises(RuntimeError,match='Invalid measured joint feedback'):
        Executor(arm,plan).execute()

    arm.goto_joints.assert_not_called(); arm.goto_gripper.assert_not_called()
