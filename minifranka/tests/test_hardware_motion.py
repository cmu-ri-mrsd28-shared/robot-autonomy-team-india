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
