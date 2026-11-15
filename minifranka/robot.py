"""Shared task and movement routines; only the backend differs on hardware."""

import numpy as np


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
        print(
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
            print(f"Skip {name}: no feasible top-grasp plan", flush=True)
            return None
        print(f"{name}: {len(moves)} moves from current locations", flush=True)
        if not self.before_pattern(name, len(moves)):
            return False
        for move in moves:
            self.pick_place(move)
        self.completed_patterns += 1
        print("Block locations:", self.locations, flush=True)
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
        self._move(point, rotation, "Descend to pick")
        self._grip(True)
        self._move([*point[:2], self.config["travel_height"]], rotation, "Lift block")

    def _place(self, target, orientation):
        self.move_above(target, orientation)
        rotation = self._pose().rotation.copy()
        point = self.planner.xyz(target)
        point[2] = self.config["grasp_height"]
        self._move(point, rotation, "Descend to place")
        self._grip(False)
        self._move(
            [*point[:2], self.config["travel_height"]], rotation, "Lift after release"
        )

    def _pose(self):
        return self.sim.pose()

    def _send_move(self, xyz, rotation, duration):
        self.sim.move(xyz, rotation, duration)

    def _send_grip(self, width, force):
        self.sim.grip(width, force)

    def _jaw_width(self):
        return self.sim.jaw_width()

    def _move(self, xyz, rotation, label):
        xyz = np.asarray(xyz, dtype=float)
        current = self._pose()
        distance = np.linalg.norm(xyz - current.translation)
        angle = np.arccos(
            np.clip((np.trace(current.rotation.T @ rotation) - 1) / 2, -1, 1)
        )
        if distance < 1e-5 and angle < 1e-4:
            return
        duration = self.motion_duration(distance, angle, self.config["speed"])
        print(f"{label}: {xyz.round(5).tolist()}, {duration:.1f}s", flush=True)
        self._send_move(xyz, rotation, duration)

    def _grip(self, close):
        width = 0.0 if close else self.config["gripper"]["opening"]
        force = self.config["gripper"]["force"] if close else 0.0
        print("Grasp" if close else "Release", flush=True)
        self._send_grip(width, force)
        print(f"Jaw width: {self._jaw_width():.4f} m", flush=True)


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

    def _send_move(self, xyz, rotation, duration):
        pose = self.transform(
            translation=xyz,
            rotation=rotation,
            from_frame="franka_tool",
            to_frame="world",
        )
        self.arm.goto_pose(
            pose, duration=float(duration), use_impedance=False, ignore_errors=False
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
