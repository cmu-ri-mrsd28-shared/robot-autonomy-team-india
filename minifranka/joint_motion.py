"""Opt-in joint execution for the existing Robot pick/place routines."""

import time
import numpy as np
from scipy.spatial.transform import Rotation

from console import log
from kinematics import PandaIK
from robot import HardwareRobot


class JointMotion:
    TOLERANCE = 0.003  # m, allowed tool deviation from the straight segment
    MAX_WAYPOINTS = 20

    def __init__(self, ik, speed):
        self.ik = ik
        self.speed = float(speed)
        if not np.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("speed must be a positive number in m/s")

    def plan(self, xyz, rotation, seed, duration, straight=True):
        """Fewest joint targets that keep the tool near the Cartesian segment.

        With straight=False the segment is one joint move and the tool path may bow.
        """
        seed = np.asarray(seed, dtype=float)
        start = self.ik.forward(seed)
        goal = self.ik.pose(xyz, rotation)
        delta = goal[:3, 3] - start[:3, 3]
        turn = Rotation.from_matrix(goal[:3, :3] @ start[:3, :3].T).as_rotvec()
        for count in range(1, (self.MAX_WAYPOINTS if straight else 1) + 1):
            joints = self._targets(start, delta, turn, seed, count)
            if self._deviation(start[:3, 3], delta, joints) <= self.TOLERANCE:
                break
        return [(end, self.duration(begin, end, duration / count))
                for begin, end in zip(joints, joints[1:])]

    def _targets(self, start, delta, turn, seed, count):
        """The seed followed by count evenly spaced IK solutions, each seeded by the last."""
        joints = [seed]
        for fraction in np.linspace(0, 1, count + 1)[1:]:
            point = start[:3, 3] + fraction * delta
            heading = Rotation.from_rotvec(fraction * turn).as_matrix() @ start[:3, :3]
            joints.append(self.ik.solve(point, heading, joints[-1]))
        return joints

    def _deviation(self, origin, delta, joints):
        """Largest tool distance from the segment while interpolating between joint targets."""
        length = np.linalg.norm(delta)
        axis = delta / length if length > 1e-9 else np.zeros(3)
        worst = 0.0
        for begin, end in zip(joints, joints[1:]):
            for fraction in (0.25, 0.5, 0.75):
                offset = self.ik.forward(begin + fraction * (end - begin))[:3, 3] - origin
                worst = max(worst, float(np.linalg.norm(offset - axis * (offset @ axis))))
        return worst

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
        # Only moves that start or end below travel height pass close to blocks;
        # everything else is a single joint move with no intermediate stop.
        low = min(xyz[2], self._pose().translation[2]) < self.config["travel_height"] - 0.005
        # Plan the entire segment first: an IK failure must not leave a partial move.
        waypoints = self.motion.plan(xyz, rotation, self.arm.get_joints(), duration, low)
        log(f"Joint motion: {len(waypoints)} waypoints, "
            f"planned duration={sum(seconds for _, seconds in waypoints):.1f}s")
        for i, (joints, seconds) in enumerate(waypoints, 1):
            log(f"Joint move {i}/{len(waypoints)}: {joints.round(5).tolist()}, {seconds:.2f}s")
            self._start_joint_move(joints.tolist(), float(seconds))
            self._wait_for_joint_motion(seconds)

    def _start_joint_move(self, joints, seconds):
        """FrankaPy's goto_joints() skill, but ending on time instead of goal proximity.

        goto_joints() finishes once the generated position is within 1e-4 rad of
        the goal, which is before a short minimum-jerk move has stopped
        decelerating; the controller then faults. This runs the full duration.
        """
        from frankapy.franka_constants import FrankaConstants as FC
        from frankapy.franka_interface_common_definitions import (
            FeedbackControllerType, SkillType, TerminationHandlerType, TrajectoryGeneratorType)
        from frankapy.skill_list import Skill

        if not self.arm.is_joints_reachable(joints):
            raise ValueError("Joint target is outside the joint limits")
        if self.arm.is_joints_in_collision_with_boxes(joints):
            raise ValueError("Joint target collides with the SDK virtual walls")
        skill = Skill(SkillType.JointPositionSkill,
                      TrajectoryGeneratorType.MinJerkJointTrajectoryGenerator,
                      feedback_controller_type=FeedbackControllerType.SetInternalImpedanceFeedbackController,
                      termination_handler_type=TerminationHandlerType.TimeTerminationHandler,
                      skill_desc="GoToJointsTimed")
        skill.add_initial_sensor_values(FC.EMPTY_SENSOR_VALUES)
        skill.set_joint_impedances(False, None, None, None, None)
        skill.add_time_termination_params(FC.DEFAULT_TERM_BUFFER_TIME)
        skill.add_goal_joints(seconds, joints)
        self.arm._send_goal(skill.create_goal(), cb=skill.feedback_callback,
                            block=False, ignore_errors=False)

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
