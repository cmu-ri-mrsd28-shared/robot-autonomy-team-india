"""Shared task and movement routines; only the backend differs on hardware."""

import time
import numpy as np

from console import log


class Robot:
    def __init__(self, config, planner, show_viewer=True):
        self._setup(config, planner)
        from simulation import Simulation

        self.sim = Simulation(planner, show_viewer)

    @staticmethod
    def motion_duration(distance, angle, speed):
        """One speed controls translation, rotation and the minimum duration."""
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("speed must be a positive number in m/s")
        scale = speed / 0.05
        return max(
            3.0 / scale, 1.875 * distance / speed, 1.875 * angle / (0.25 * scale)
        )

    def _setup(self, config, planner):
        self.config, self.planner = config, planner
        self.locations = planner.initial_locations.copy()
        self.held_block = None
        self.completed_patterns = 0
        self.sim = self.arm = None

    def pick(self, block, orientation="Y"):
        if self.held_block is not None:
            raise ValueError("Place the held block before picking another")
        source = self.locations[block]
        self._pick(block, source, orientation)
        self.locations[block] = "held"
        self.held_block = block

    def place(self, location, orientation="Y"):
        if self.held_block is None:
            raise ValueError("Pick a block before placing")
        target = self.planner.location(location)
        if target in self.locations.values():
            raise ValueError("Destination is occupied: " + target)
        self._place(target, orientation)
        self.locations[self.held_block] = target
        self.held_block = None

    def pick_place(self, move):
        if self.locations[move.block] != move.source:
            raise ValueError("Block is not at the planned source")
        log(
            f"{move.block}: {move.source} -> {move.target} [{move.pick_axis}/{move.place_axis}]",
            flush=True,
        )
        try:
            self.pick(move.block, move.pick_axis)
            self.place(move.target, move.place_axis)
        except BaseException:
            if self.arm is not None:
                try:
                    self._cancel_motion()
                except Exception:
                    pass
            raise

    def before_pattern(self, name, move_count):
        return True

    def run_pattern(self, name, pattern):
        moves = self.planner.plan(pattern, self.locations)
        if moves is None:
            log(f"Skip {name}: no feasible top-grasp plan", flush=True)
            return None
        log(f"{name}: {len(moves)} moves from current locations", flush=True)
        if not self.before_pattern(name, len(moves)):
            return False
        for move in moves:
            self.pick_place(move)
        self.completed_patterns += 1
        log("Block locations:", self.locations, flush=True)
        return True

    def close(self):
        if self.sim is not None:
            self.sim.close()

    def move_above(self, location, orientation="Y"):
        point = self.planner.xyz(location)
        current = self._pose()
        z = self.config["travel_height"]
        self._move([*current.translation[:2], z], current.rotation, "Travel height", vertical=True)
        self._move([*point[:2], z], current.rotation, "Move above " + str(location))
        # The two equivalent 180-degree headings grip the same axis. Pick the
        # one closest to the actual wrist orientation to avoid needless turns.
        rotation = self.planner.grasps.rotation(orientation, self._pose().rotation)
        self._move([*point[:2], z], rotation, "Top grasp " + orientation)

    def _pick(self, block, source, orientation):
        self._grip(False)
        self.move_above(source, orientation)
        rotation = self._pose().rotation.copy()
        point = self.planner.xyz(source)
        point[2] = self.config["grasp_height"]
        self._descend(point, rotation, "Descend to pick")
        self._grip(True)
        self._move([*point[:2], self.config["travel_height"]], rotation, "Lift block", vertical=True)

    def _place(self, target, orientation):
        self.move_above(target, orientation)
        rotation = self._pose().rotation.copy()
        point = self.planner.xyz(target)
        point[2] = self.config["grasp_height"]
        self._descend(point, rotation, "Descend to place")
        self._grip(False)
        self._move(
            [*point[:2], self.config["travel_height"]], rotation, "Lift after release", vertical=True
        )

    def _pose(self):
        return self.sim.pose()

    def _descend(self, xyz, rotation, label):
        self._move(xyz, rotation, label, vertical=True)

    def _send_move(self, xyz, rotation, duration):
        self.sim.move(xyz, rotation, duration)

    def _send_vertical_move(self, xyz, rotation, duration):
        self._send_move(xyz, rotation, duration)

    def _send_grip(self, width, force):
        self.sim.grip(width, force)

    def _jaw_width(self):
        return self.sim.jaw_width()

    def _move(self, xyz, rotation, label, duration_scale=1.0, vertical=False):
        xyz = np.asarray(xyz, dtype=float)
        current = self._pose()
        distance = np.linalg.norm(xyz - current.translation)
        angle = np.arccos(
            np.clip((np.trace(current.rotation.T @ rotation) - 1) / 2, -1, 1)
        )
        if distance < 1e-5 and angle < 1e-4:
            log(f"SKIP {label}: reported tool is already at {current.translation.round(5).tolist()}", flush=True)
            return
        duration = duration_scale * self.motion_duration(distance, angle, self.config["speed"])
        log(f"{label}: {xyz.round(5).tolist()}, {duration:.1f}s", flush=True)
        started = time.monotonic()
        if vertical:
            self._send_vertical_move(xyz, rotation, duration)
        else:
            self._send_move(xyz, rotation, duration)
        actual = self._pose()
        rotation_error = np.arccos(np.clip((np.trace(actual.rotation.T@rotation)-1)/2,-1,1))
        log(f"RETURNED {label}: actual XYZ={actual.translation.round(5).tolist()}, "
              f"position error={np.linalg.norm(actual.translation-xyz):.5f} m, "
              f"rotation error={rotation_error:.4f} rad, "
              f"elapsed={time.monotonic()-started:.2f}s", flush=True)

    def _grip(self, close):
        width = 0.0 if close else self.config["gripper"]["opening"]
        force = self.config["gripper"]["force"] if close else 0.0
        log("Grasp" if close else "Release", flush=True)
        self._send_grip(width, force)
        log(f"Jaw width: {self._jaw_width():.4f} m", flush=True)


class HardwareRobot(Robot):
    # Diagnostic settling criteria, in rad/s and seconds. These are application
    # checks, not replacements for the controller's motion limits.
    SETTLE_VELOCITY = 0.01
    SETTLE_WINDOW = 0.5
    SETTLE_TIMEOUT = 5.0

    def __init__(self, config, planner):
        self._setup(config, planner)
        from frankapy import FrankaArm
        from autolab_core import RigidTransform

        self.arm = FrankaArm()
        self.transform = RigidTransform

    def before_pattern(self, name, move_count):
        if self.completed_patterns:
            answer = input(
                f"Next pattern: {name}, {move_count} moves. Enter to continue, q to stop: "
            )
            return answer.strip().lower() != "q"
        return True

    def _pose(self):
        return self.arm.get_pose()

    def _descend(self, xyz, rotation, label):
        self._wait_until_settled("Before " + label)
        # One attempt only: preserve the first fault and leave the gripper alone.
        self._move(xyz, rotation, label, vertical=True)

    def _joint_sample(self):
        q = np.asarray(self.arm.get_joints(), dtype=float)
        dq = np.asarray(self.arm.get_joint_velocities(), dtype=float)
        if q.shape != (7,) or dq.shape != (7,) or not (
            np.isfinite(q).all() and np.isfinite(dq).all()
        ):
            raise RuntimeError("Invalid joint feedback; no further motion commanded")
        return q, dq

    def _log_joint_state(self, label):
        q, dq = self._joint_sample()
        pose = self._pose()
        log(f"STATE {label}: q(rad)={q.round(6).tolist()}, "
            f"dq(rad/s)={dq.round(6).tolist()}, "
            f"XYZ={pose.translation.round(6).tolist()}, "
            f"R={pose.rotation.round(6).tolist()}")

    def _wait_until_settled(self, label):
        started = time.monotonic()
        quiet_since = None
        log(f"SETTLING {label}: all |dq| <= {self.SETTLE_VELOCITY} rad/s "
            f"for {self.SETTLE_WINDOW}s (timeout {self.SETTLE_TIMEOUT}s)")
        while True:
            _, dq = self._joint_sample()
            now = time.monotonic()
            peak = float(np.max(np.abs(dq)))
            if now - started >= self.SETTLE_TIMEOUT:
                raise TimeoutError(f"{label}: arm did not settle; max |dq|={peak:.6f} rad/s; "
                                   "descent cancelled")
            if peak <= self.SETTLE_VELOCITY:
                if quiet_since is None:
                    quiet_since = now
                if now - quiet_since >= self.SETTLE_WINDOW:
                    log(f"SETTLED {label}: max |dq|={peak:.6f} rad/s, "
                        f"elapsed={now-started:.2f}s")
                    self._log_joint_state(label)
                    return
            else:
                quiet_since = None
            time.sleep(0.05)

    def _move(self, xyz, rotation, label, duration_scale=1.0, vertical=False):
        try:
            self._log_joint_state("BEFORE " + label)
            log(f"TARGET {label}: R={np.asarray(rotation).round(6).tolist()}")
            super()._move(xyz, rotation, label, duration_scale, vertical=vertical)
            actual = self._pose()
            position_error = float(np.linalg.norm(actual.translation - np.asarray(xyz)))
            angle_error = float(np.arccos(np.clip(
                (np.trace(actual.rotation.T @ rotation) - 1) / 2, -1, 1)))
            if (not np.isfinite([position_error, angle_error]).all()
                    or position_error > 0.005 or angle_error > 0.03):
                raise RuntimeError(f"{label}: target not reached "
                                   f"(position error={position_error:.5f} m, "
                                   f"rotation error={angle_error:.5f} rad)")
            self._log_joint_state("AFTER " + label)
            if label.startswith("Top grasp "):
                self._wait_until_settled("After " + label)
        except BaseException:
            # Diagnostic reads/cancellation must never replace the original fault.
            try:
                self._log_joint_state("FAILED " + label)
            except Exception as error:
                log(f"Could not read failure state: {error}")
            try:
                self._cancel_motion()
            except Exception as error:
                log(f"Motion cancellation failed: {error}")
            log(f"STOPPED {label}: no automatic retry, detour, or gripper command")
            raise

    def _cancel_motion(self):
        """Bounded cancellation only; never recover and launch another motion."""
        import rospy

        self.arm._client.cancel_goal()
        if not self.arm._client.wait_for_result(rospy.Duration.from_sec(10.0)):
            raise RuntimeError("Motion cancellation did not finish within 10 seconds")

    def _send_move(self, xyz, rotation, duration):
        pose = self.transform(
            translation=xyz,
            rotation=rotation,
            from_frame="franka_tool",
            to_frame="world",
        )
        self.arm.goto_pose(
            pose, duration=float(duration), use_impedance=False, ignore_errors=False,
            block=False,
        )
        self._wait_for_motion(duration)

    def _send_vertical_move(self, xyz, rotation, duration):
        if not self.config.get("relative_vertical_motion", False):
            return self._send_move(xyz, rotation, duration)
        current = self._pose()
        xyz = np.asarray(xyz, dtype=float)
        angle = np.arccos(np.clip((np.trace(current.rotation.T @ rotation) - 1) / 2, -1, 1))
        if (xyz.shape != (3,) or not np.isfinite(xyz).all()
                or np.linalg.norm(xyz[:2] - current.translation[:2]) > 0.001
                or not np.isfinite(angle) or angle > 0.03):
            raise ValueError("Relative Z requires XY within 1 mm and unchanged orientation (0.03 rad tolerance)")
        # Keep the starting commanded orientation rather than sending measured
        # orientation as a fresh absolute goal after the wrist turn.
        target_xyz = current.translation.copy()
        target_xyz[2] = xyz[2]
        target = self.transform(
            translation=target_xyz, rotation=current.rotation,
            from_frame="franka_tool", to_frame="world")
        self._check_relative_target(target)
        dz = float(xyz[2] - current.translation[2])
        delta = self.transform(
            translation=np.array([0.0, 0.0, dz]), rotation=np.eye(3),
            from_frame="world", to_frame="world")
        log(f"RELATIVE Z: dz={dz:.6f} m, identity rotation, {duration:.1f}s")
        self.arm.goto_pose_delta(
            delta, duration=float(duration), use_impedance=False,
            ignore_errors=False, block=False)
        self._wait_for_motion(duration)

    def _check_relative_target(self, target):
        # The SDK world-frame delta branch skips the absolute-pose wall check;
        # apply the same check to the predicted tool-base endpoint here.
        from frankapy import FrankaConstants

        tool_base_target = target * self.arm.get_tool_base_pose().inverse()
        walls = FrankaConstants.WORKSPACE_WALLS[:, :3]
        xyz = tool_base_target.translation
        if (not np.isfinite(xyz).all() or np.any(xyz <= walls.min(axis=0))
                or np.any(xyz >= walls.max(axis=0))):
            raise ValueError("Relative Z target is outside workspace virtual walls")

    def _wait_for_motion(self, duration):
        # FrankaPy's blocking wait uses is_skill_done(ignore_errors=True).
        # Poll explicitly so controller errors reach the calling movement routine.
        deadline = time.monotonic() + float(duration) + 10.0
        while not self.arm.is_skill_done(ignore_errors=False):
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for Cartesian motion")
            time.sleep(0.01)
        status = self.arm._get_current_franka_interface_status().franka_interface_status
        if status.error_description:
            raise RuntimeError(status.error_description)
        # A completed action can also mean aborted/rejected, not success (3).
        if self.arm._client.get_state() != 3:
            raise RuntimeError(
                "Cartesian motion failed: " + self.arm._client.get_goal_status_text()
            )

    def _send_grip(self, width, force):
        self.arm.goto_gripper(
            width,
            speed=0.04,
            grasp=width == 0.0,
            force=force,
            epsilon_inner=0.004,
            epsilon_outer=self.config["box_width"] + 0.004 if width == 0.0 else 0.004,
            block=True,
            ignore_errors=False,
        )

    def _jaw_width(self):
        return self.arm.get_gripper_width()
