"""Offline IK and joint-backend checks; no ROS or physical robot required."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibration import check3_grid
from grasp import Grasp
from joint_motion import JointHardwareRobot, JointMotion
from kinematics import PandaIK
from planner import Planner
from robot import Robot


class FakeArm:
    def __init__(self, ik, joints):
        self.ik, self.joints = ik, joints.copy()
        self.joint_calls, self.gripper_calls = [], []
        self.width = 0.08
        self.fail_next = False

    def get_joints(self):
        return self.joints.copy()

    def get_joint_velocities(self):
        return np.zeros(7)

    def get_pose(self):
        pose = self.ik.forward(self.joints)
        return SimpleNamespace(translation=pose[:3, 3], rotation=pose[:3, :3])

    def joint_move(self, joints, seconds):
        if self.fail_next:
            raise RuntimeError("joint controller fault")
        self.joint_calls.append((joints, seconds))
        self.joints = np.asarray(joints)

    def goto_gripper(self, width, **kwargs):
        self.gripper_calls.append(width)
        self.width = 0.04 if width == 0 else width

    def get_gripper_width(self):
        return self.width


@pytest.fixture
def ik():
    model = PandaIK()
    home = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
    model.align_tool(home, [0.3069, 0, 0.4867], np.diag([1, -1, -1]))
    return model


@pytest.fixture
def robot(ik):
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    config.update(speed=0.05, travel_height=0.2, grasp_height=0.058)
    config["gripper"]["force"] = 15.0
    result = JointHardwareRobot.__new__(JointHardwareRobot)
    Robot._setup(result, config, Planner(config))
    home = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
    result.arm = FakeArm(ik, home)
    result.ik, result.motion = ik, JointMotion(ik, config["speed"])
    result._start_joint_move = result.arm.joint_move
    result._wait_for_joint_motion = Mock()
    result._wait_until_settled = Mock()
    result._cancel_motion = Mock()
    return result


def test_fk_matches_known_panda_home_flange():
    q = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4]
    pose = PandaIK().forward(q)
    np.testing.assert_allclose(pose[:3, 3], [0.30689057, 0, 0.59028205], atol=1e-7)


def test_ik_reaches_pose_from_nearby_seed(ik):
    target = np.array([-0.35, 0.3, -0.1, -2.3, 0.2, 2.5, 0.5])
    pose = ik.forward(target)
    q = ik.solve(pose[:3, 3], pose[:3, :3], target + 0.04)
    np.testing.assert_allclose(ik.forward(q)[:3, 3], pose[:3, 3], atol=0.0002)
    assert np.all(q >= ik.lower) and np.all(q <= ik.upper)


def test_ik_rejects_unreachable_target(ik):
    with pytest.raises(RuntimeError, match="did not converge"):
        ik.solve([3, 3, 3], np.diag([1, -1, -1]), [0, 0, 0, -2, 0, 2, 0])


def test_joint_descent_approximates_vertical_line(ik):
    start = ik.solve([0.426, -0.175, 0.2], np.diag([1, -1, -1]),
                     [0, -0.5, 0, -2, 0, 1.5, 0.8])
    waypoints = JointMotion(ik, 0.05).plan([0.426, -0.175, 0.058],
                                         np.diag([1, -1, -1]), start, 6)
    previous = start
    for target, seconds in waypoints:
        assert seconds > 0
        for fraction in (0.25, 0.5, 0.75, 1):
            point = ik.forward(previous + fraction * (target - previous))[:3, 3]
            assert np.linalg.norm(point[:2] - [0.426, -0.175]) <= JointMotion.TOLERANCE
            assert point[2] >= 0.057
        previous = target


def test_first_hardware_move_is_few_waypoints(ik):
    # Home down to travel height was 15 sub-second waypoints, and the first faulted.
    home = np.array([0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4])
    waypoints = JointMotion(ik, 0.1).plan([0.3069, 0, 0.2], np.diag([1, -1, -1]), home, 5.4)
    assert len(waypoints) <= 5
    assert all(seconds >= 1.0 for _, seconds in waypoints)


def test_travel_is_one_move_and_descent_stays_straight(robot):
    down = np.diag([1, -1, -1])
    robot._send_move([0.35, -0.23, 0.2], down, 5)   # home to above B0
    assert len(robot.arm.joint_calls) == 1
    robot._send_move([0.35, -0.23, 0.058], down, 5)  # descent
    assert len(robot.arm.joint_calls) > 2


def test_x_to_y_returns_to_home_side_heading(ik):
    # Logged failure: at grid:4 in X with zero yaw offsets, both Y headings were
    # a quarter turn away and the chosen one needed joint 7 beyond its limit.
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    config["grasp_yaw_offsets_deg"] = {"X": 0.0, "Y": 0.0}
    joints = np.array([0.1529, 0.08188, -0.194915, -2.330366, 0.02412, 2.411188, 2.29752])
    pose = ik.forward(joints)
    heading = Grasp(config).rotation("Y", pose[:3, :3])
    np.testing.assert_allclose(heading, np.diag([1, -1, -1]), atol=1e-9)
    assert len(JointMotion(ik, 0.1).plan(pose[:3, 3], heading, joints, 5.9, False)) == 1


def test_check3_round_trip_uses_joints_and_existing_gripper(robot):
    check3_grid(robot)
    assert robot.locations["B0"] == "waiting:B0"
    assert robot.held_block is None
    assert len(robot.arm.joint_calls) > 0
    assert robot.arm.gripper_calls == [0.08, 0.0, 0.08, 0.08, 0.0, 0.08]


def test_joint_fault_does_not_release_held_block(robot):
    robot.pick("B0", "Y")
    before = list(robot.arm.gripper_calls)
    robot.arm.fail_next = True
    with pytest.raises(RuntimeError, match="joint controller fault"):
        robot.place((0, 0), "Y")
    assert robot.arm.gripper_calls == before
    assert robot.held_block == "B0"


def test_failed_ik_plan_sends_no_joint_command(robot):
    robot.motion.ik.solve = Mock(side_effect=RuntimeError("IK failure"))
    with pytest.raises(RuntimeError, match="IK failure"):
        robot._send_move([0.4, 0, 0.2], np.diag([1, -1, -1]), 5)
    assert robot.arm.joint_calls == []


def test_joint_wait_propagates_controller_fault(robot):
    robot.arm = Mock()
    robot.arm.is_skill_done.side_effect = RuntimeError("joint reflex")
    with pytest.raises(RuntimeError, match="joint reflex"):
        JointHardwareRobot._wait_for_joint_motion(robot, 3)
    robot.arm.is_skill_done.assert_called_once_with(ignore_errors=False)


def test_joint_wait_checks_terminal_status(robot):
    robot.arm = Mock()
    robot.arm.is_skill_done.return_value = True
    robot.arm._get_current_franka_interface_status.return_value.franka_interface_status.error_description = ""
    robot.arm._client.get_state.return_value = 4
    robot.arm._client.get_goal_status_text.return_value = "aborted"
    with pytest.raises(RuntimeError, match="aborted"):
        JointHardwareRobot._wait_for_joint_motion(robot, 3)
