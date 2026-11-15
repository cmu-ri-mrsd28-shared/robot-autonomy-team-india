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
                    self.arm.stop_skill()
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
        self._move([*current.translation[:2], z], current.rotation, "Travel height")
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
        self._move([*point[:2], self.config["travel_height"]], rotation, "Lift block")

    def _place(self, target, orientation):
        self.move_above(target, orientation)
        rotation = self._pose().rotation.copy()
        point = self.planner.xyz(target)
        point[2] = self.config["grasp_height"]
        self._descend(point, rotation, "Descend to place")
        self._grip(False)
        self._move(
            [*point[:2], self.config["travel_height"]], rotation, "Lift after release"
        )

    def _pose(self):
        return self.sim.pose()

    def _descend(self, xyz, rotation, label):
        self._move(xyz, rotation, label)

    def _send_move(self, xyz, rotation, duration):
        self.sim.move(xyz, rotation, duration)

    def _send_grip(self, width, force):
        self.sim.grip(width, force)

    def _jaw_width(self):
        return self.sim.jaw_width()

    def _move(self, xyz, rotation, label, duration_scale=1.0):
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
        # Initial descent, one slower retry, then up to three center detours.
        for attempt in range(5):
            try:
                if attempt >= 2:
                    self._reposition_for_descent(xyz, rotation)
                self._move(xyz, rotation, label, duration_scale=1.0 if attempt == 0 else 2.0)
                actual_z = self._pose().translation[2]
                if abs(actual_z - xyz[2]) > 0.01:
                    raise RuntimeError(
                        f"{label} did not reach grasp height: Z={actual_z:.5f}, "
                        f"target={xyz[2]:.5f}; gripper command cancelled"
                    )
                return
            except Exception as error:
                if attempt == 4 or not self._retryable_motion_error(error):
                    log(f"Z failure during {label}: stopping without a gripper command; {error}")
                    raise
                action = "retry descent at half speed" if attempt == 0 else (
                    f"center detour {attempt}/3, then retry descent at half speed"
                )
                log(f"Retrying because of Z failure during {label}: {action}; "
                    f"cancelling the current motion and waiting for recovery. Reason: {error}")
                self._recover_motion()

    @staticmethod
    def _retryable_motion_error(error):
        message = str(error).lower()
        jerk_reflex = '["cartesian_motion_generator_joint_acceleration_discontinuity"]'
        return (isinstance(error, TimeoutError) or jerk_reflex in message
                or ("motion finished commanded" in message and "still moving" in message))

    def _recover_motion(self):
        """Cancel this client's motion, wait for it to end, then await recovery."""
        import rospy

        # Avoid stop_skill(): its internal wait can suppress errors and wait indefinitely.
        self.arm._client.cancel_goal()
        if not self.arm._client.wait_for_result(rospy.Duration.from_sec(10.0)):
            raise RuntimeError("Motion cancellation did not finish within 10 seconds; no retry sent")
        # franka-interface owns libfranka and performs automaticErrorRecovery.
        self.arm.wait_for_franka_interface(timeout=10.0)
        # The action is now terminal and the interface ready; clear SDK skill bookkeeping.
        if not self.arm.is_skill_done(ignore_errors=False):
            raise RuntimeError("Previous motion is still active; no retry sent")

    def _reposition_for_descent(self, xyz, rotation):
        """Return through grid center at travel height, keeping the held block."""
        z = self.config["travel_height"]
        current = self._pose()
        center = self.config["grid"]["center"]
        if abs(current.translation[2] - z) > 0.001:
            self._move([*current.translation[:2], z], rotation,
                       "Z recovery: rise to travel height", duration_scale=2.0)
        self._move([*center[:2], z], rotation,
                   "Z recovery: move to grid center", duration_scale=2.0)
        self._move([*xyz[:2], z], rotation,
                   "Z recovery: return above descent target", duration_scale=2.0)

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
