"""Opt-in joint execution for the existing Robot pick/place routines."""

import time
import numpy as np
from scipy.spatial.transform import Rotation

from console import log
from kinematics import PandaIK
from robot import HardwareRobot


class JointMotion:
    def __init__(self, ik, speed):
        self.ik = ik
        self.speed = float(speed)
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("speed must be a positive number in m/s")

    def plan(self, xyz, rotation, seed, duration):
        """Precompute nearby joint targets along a Cartesian segment before moving."""
        start = self.ik.forward(seed)
        goal = self.ik.pose(xyz, rotation)
        delta = goal[:3, 3] - start[:3, 3]
        turn = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()
        count = max(1, int(np.ceil(np.linalg.norm(delta) / 0.02)),
                    int(np.ceil(np.linalg.norm(turn) / 0.15)))
        previous = np.asarray(seed, dtype=float).copy()
        waypoints = []
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            point = start[:3, 3] + fraction * delta
            heading = Rotation.from_rotvec(fraction * turn).as_matrix() @ start[:3, :3]
            joints = self.ik.solve(point, heading, previous)
            seconds = self.duration(previous, joints, duration / count)
            waypoints.append((joints, seconds))
            previous = joints
        return waypoints

    def duration(self, start, end, minimum):
        """Scale conservative joint velocity/acceleration/jerk with global speed."""
        distance = float(np.max(np.abs(np.asarray(end) - start)))
        scale = self.speed / 0.05
        # Peak derivatives of a rest-to-rest minimum-jerk trajectory are
        # 1.875*d/T, 5.774*d/T^2 and 60*d/T^3. These set timing, not rejection gates.
        return max(float(minimum), 0.5 / scale,
                   1.875 * distance / (0.5 * scale),
                   np.sqrt(5.774 * distance) / scale,
                   np.cbrt(60 * distance / 5.0) / scale)


class JointHardwareRobot(HardwareRobot):
    """Only replace Cartesian command execution; inherit picking, placing and grasping."""

    def __init__(self, config, planner):
        super().__init__(config, planner)
        self.ik = PandaIK()
        joints = self.arm.get_joints()
        pose = self._pose()
        self.ik.align_tool(joints, pose.translation, pose.rotation)
        self.motion = JointMotion(self.ik, config["speed"])
        log("Joint backend: Panda model aligned to the current reported tool frame")
        log(f"Flange-to-tool transform: {self.ik.tool.round(6).tolist()}")

    def _send_move(self, xyz, rotation, duration):
        # Plan the entire segment first: an IK failure must not leave a partial move.
        waypoints = self.motion.plan(xyz, rotation, self.arm.get_joints(), duration)
        log(f"Joint motion: {len(waypoints)} waypoints, "
            f"planned duration={sum(seconds for _, seconds in waypoints):.1f}s")
        for i, (joints, seconds) in enumerate(waypoints, 1):
            log(f"goto_joints {i}/{len(waypoints)}: {joints.round(5).tolist()}, {seconds:.2f}s")
            self.arm.goto_joints(joints.tolist(), duration=float(seconds),
                                 use_impedance=False, block=False, ignore_errors=False)
            self._wait_for_joint_motion(seconds)

    def _wait_for_joint_motion(self, duration):
        """Wait with error propagation, matching the existing hardware backend."""
        deadline = time.monotonic() + duration + 10.0
        while not self.arm.is_skill_done(ignore_errors=False):
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for joint motion")
            time.sleep(0.01)
        status = self.arm._get_current_franka_interface_status().franka_interface_status
        if status.error_description:
            raise RuntimeError(status.error_description)
        if self.arm._client.get_state() != 3:
            raise RuntimeError("Joint motion failed: " + self.arm._client.get_goal_status_text())
