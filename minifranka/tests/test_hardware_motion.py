"""Simple success/failure checks; no FrankaPy, ROS or physical robot."""

from itertools import count
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import robot as robot_module
from robot import HardwareRobot


@pytest.fixture
def robot(monkeypatch):
    # Advance test time without waiting for the real settling timeout.
    ticks = count(0, 0.1)
    monkeypatch.setattr(robot_module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(robot_module.time, "sleep", lambda seconds: None)

    robot = HardwareRobot.__new__(HardwareRobot)
    robot.config = {"speed": 0.05, "travel_height": 0.3, "grasp_height": 0.058}
    robot.arm = Mock()
    robot.arm.get_joints.return_value = np.zeros(7)
    robot.arm.get_joint_velocities.return_value = np.zeros(7)
    robot.arm.get_pose.return_value = SimpleNamespace(
        translation=np.array([0.4, 0, 0.3]), rotation=np.eye(3))
    robot.planner = Mock()
    robot.planner.xyz.return_value = np.array([0.4, 0, 0.058])
    robot.move_above = Mock()
    robot._send_move = Mock()
    robot._cancel_motion = Mock()
    robot._grip = Mock()
    return robot


def test_settling_accepts_stationary_arm(robot):
    robot._wait_until_settled("test")

    assert robot.arm.get_joint_velocities.call_count >= 6


def test_settling_rejects_moving_arm_before_descent(robot):
    robot.arm.get_joint_velocities.return_value = np.full(7, 0.02)

    with pytest.raises(TimeoutError, match="did not settle"):
        robot._descend([0.4, 0, 0.058], np.eye(3), "Descend to place")

    robot._send_move.assert_not_called()


def test_place_releases_after_reaching_target(robot):
    def reach_target(xyz, rotation, duration):
        robot.arm.get_pose.return_value = SimpleNamespace(
            translation=np.array(xyz), rotation=rotation)

    robot._send_move.side_effect = reach_target
    robot._place("grid:0", "X")

    robot._grip.assert_called_once_with(False)
    assert robot._send_move.call_count == 2  # Descend, then lift.


def test_place_stops_on_reflex_without_release_or_retry(robot):
    robot._send_move.side_effect = RuntimeError("joint acceleration discontinuity")

    with pytest.raises(RuntimeError, match="joint acceleration discontinuity"):
        robot._place("grid:0", "X")

    robot._send_move.assert_called_once()
    robot._cancel_motion.assert_called_once()
    robot._grip.assert_not_called()


def test_place_rejects_motion_that_returns_without_descending(robot):
    # The mock returns successfully but its measured pose stays at travel height.
    with pytest.raises(RuntimeError, match="target not reached"):
        robot._place("grid:0", "X")

    robot._send_move.assert_called_once()
    robot._grip.assert_not_called()


@pytest.mark.parametrize('impedance', [False, True])
def test_relative_vertical_move_sends_only_world_z(robot, impedance):
    robot.config['relative_vertical_motion'] = True
    robot.config['cartesian_impedance'] = impedance
    robot.transform = lambda **values: SimpleNamespace(**values)
    robot._check_relative_target = Mock()
    robot._wait_for_motion = Mock()

    robot._send_vertical_move([0.4, 0, 0.058], np.eye(3), 5.0)

    delta = robot.arm.goto_pose_delta.call_args.args[0]
    np.testing.assert_allclose(delta.translation, [0, 0, -0.242])
    np.testing.assert_allclose(delta.rotation, np.eye(3))
    assert delta.from_frame == delta.to_frame == 'world'
    assert robot.arm.goto_pose_delta.call_args.kwargs['use_impedance'] is impedance
    robot._send_move.assert_not_called()
    robot._check_relative_target.assert_called_once()
    robot._wait_for_motion.assert_called_once_with(5.0)


def test_relative_vertical_move_rejects_lateral_motion(robot):
    robot.config['relative_vertical_motion'] = True

    with pytest.raises(ValueError, match='XY within 1 mm'):
        robot._send_vertical_move([0.45, 0, 0.058], np.eye(3), 5.0)

    robot.arm.goto_pose_delta.assert_not_called()


def test_relative_vertical_move_cannot_replace_rotation(robot):
    robot.config['relative_vertical_motion'] = True
    turn = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    with pytest.raises(ValueError, match='unchanged orientation'):
        robot._send_vertical_move([0.4, 0, 0.058], turn, 5.0)

    robot.arm.goto_pose_delta.assert_not_called()


def test_relative_descent_fault_stops_before_release(robot):
    robot.config['relative_vertical_motion'] = True
    robot.config['cartesian_impedance'] = True
    robot.transform = lambda **values: SimpleNamespace(**values)
    robot._check_relative_target = Mock()
    robot.arm.goto_pose_delta.side_effect = RuntimeError('relative motion fault')

    with pytest.raises(RuntimeError, match='relative motion fault'):
        robot._place('grid:0', 'X')

    robot.arm.goto_pose_delta.assert_called_once()
    robot._send_move.assert_not_called()
    robot._grip.assert_not_called()


@pytest.mark.parametrize('impedance', [False, True])
def test_absolute_move_uses_selected_control_mode(robot, impedance):
    robot.config['cartesian_impedance'] = impedance
    robot.transform = lambda **values: SimpleNamespace(**values)
    robot._wait_for_motion = Mock()

    # Call the real sender; the fixture replaces it for task-level tests.
    HardwareRobot._send_move(robot, [0.4, 0, 0.3], np.eye(3), 5.0)

    options = robot.arm.goto_pose.call_args.kwargs
    assert options['use_impedance'] is impedance
    assert options['ignore_errors'] is False
    assert options['block'] is False
    robot._wait_for_motion.assert_called_once_with(5.0)
