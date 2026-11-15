"""Check the exported robot folder independently of the simulator."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock
import numpy as np
import pytest


@pytest.fixture
def bundle(tmp_path,monkeypatch):
    source=Path(__file__).resolve().parents[2]/'lab3_cspace_robot'
    dest=tmp_path/'standalone_robot'
    shutil.copytree(source,dest)
    monkeypatch.syspath_prepend(str(dest))
    spec=importlib.util.spec_from_file_location('standalone_robot_runner',dest/'run.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module,dest


def fake_arm(module,config,fail_after_grasp=False):
    robot=module.Panda(config); q=[np.array(config['home_q'])]; closed=[False]
    arm=Mock(); arm.get_joints.side_effect=lambda:q[0]
    def pose():
        t=robot.fk(q[0]); return SimpleNamespace(translation=t[:3,3],rotation=t[:3,:3])
    def joints(goal,**options):
        assert options['use_impedance'] is False and options['block'] is False
        if closed[0] and fail_after_grasp: raise RuntimeError('joint reflex')
        q[0]=np.array(goal)
    def grip(width,**options): closed[0]=width==0
    arm.get_pose.side_effect=pose; arm.goto_joints.side_effect=joints
    arm.goto_gripper.side_effect=grip
    arm.get_gripper_width.side_effect=lambda:.0405 if closed[0] else .0743
    arm.is_skill_done.return_value=True; arm._client.get_state.return_value=3
    arm._get_current_franka_interface_status.return_value=SimpleNamespace(franka_interface_status=SimpleNamespace(error_description=''))
    return arm


def test_folder_runs_after_copying_without_parent_project(bundle,tmp_path):
    _,dest=bundle
    result=subprocess.run([sys.executable,str(dest/'run.py')],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert 'Offline validation passed' in result.stdout
    assert not (dest.parent/'lab3_cspace').exists()


def test_geometry_change_requires_new_joint_plans(bundle):
    module,dest=bundle
    config=json.loads((dest/'config.json').read_text())
    config['travel_z']+=.01
    with pytest.raises(ValueError,match='regenerate'):
        module.read_plan(config,'patterns')


def test_read_only_check_never_moves_or_opens_gripper(bundle,monkeypatch):
    module,dest=bundle; config=json.loads((dest/'config.json').read_text())
    arm=fake_arm(module,config)
    monkeypatch.setitem(sys.modules,'frankapy',SimpleNamespace(FrankaArm=lambda:arm))
    monkeypatch.setattr(sys,'argv',['run.py','--check'])
    with contextlib.redirect_stdout(io.StringIO()): module.main()
    arm.goto_joints.assert_not_called(); arm.goto_gripper.assert_not_called()


def test_unverified_config_is_rejected_before_connecting(bundle,monkeypatch):
    module,_=bundle
    factory=Mock()
    monkeypatch.setitem(sys.modules,'frankapy',SimpleNamespace(FrankaArm=factory))
    monkeypatch.setattr(sys,'argv',['run.py','--execute'])
    with contextlib.redirect_stdout(io.StringIO()),pytest.raises(RuntimeError,match='hardware_calibrated=true'):
        module.main()
    factory.assert_not_called()


def test_full_assignment_executes_joint_only_with_mock_robot(bundle,monkeypatch):
    module,dest=bundle; config=json.loads((dest/'config.json').read_text())
    config['hardware_calibrated']=True
    (dest/'config.json').write_text(json.dumps(config))
    arm=fake_arm(module,config)
    monkeypatch.setitem(sys.modules,'frankapy',SimpleNamespace(FrankaArm=lambda:arm))
    monkeypatch.setattr(sys,'argv',['run.py','--execute'])
    with contextlib.redirect_stdout(io.StringIO()): module.main()
    assert arm.goto_joints.call_count>100
    assert sum(call.args[0]==0 for call in arm.goto_gripper.call_args_list)==15
    arm.goto_pose.assert_not_called(); arm.goto_pose_delta.assert_not_called()


def test_fault_after_grasp_does_not_release_or_reset(bundle,monkeypatch):
    module,dest=bundle; config=json.loads((dest/'config.json').read_text())
    config['hardware_calibrated']=True
    (dest/'config.json').write_text(json.dumps(config))
    arm=fake_arm(module,config,fail_after_grasp=True)
    monkeypatch.setitem(sys.modules,'frankapy',SimpleNamespace(FrankaArm=lambda:arm))
    monkeypatch.setattr(sys,'argv',['run.py','--test','rotate-place','--execute'])
    with contextlib.redirect_stdout(io.StringIO()),pytest.raises(RuntimeError,match='joint reflex'):
        module.main()
    assert [call.args[0] for call in arm.goto_gripper.call_args_list]==[.08,0.]
    arm.reset_joints.assert_not_called()
