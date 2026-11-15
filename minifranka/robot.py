"""Robot picks/places; simulation is the default, hw=True selects FrankaPy."""

import numpy as np
from planner import INITIAL, NAMES


class Robot:
    def __init__(self, config, planner, hw=False, show_viewer=True):
        self.config, self.planner, self.hw = config, planner, hw
        self.locations = INITIAL
        self.sim = self.arm = None
        if not hw:
            from simulation import Simulation

            self.sim = Simulation(planner, show_viewer)
        else:
            if config["tool_to_grasp"] is None:
                raise ValueError(
                    "Set tool_to_grasp to the measured tool-to-pinch offset before --hw"
                )
            from frankapy import FrankaArm
            from autolab_core import RigidTransform

            self.arm = FrankaArm()
            self.transform = RigidTransform

    def pick_place(self, move):
        print(
            f"{NAMES[move.block]}: {move.source} -> {move.target}; "
            f"{self.planner.labels[move.pick_axis]} / {self.planner.labels[move.place_axis]}",
            flush=True,
        )
        self.planner.apply(
            self.locations, move
        )  # Validate source, destination and grasps.
        if not self.hw:
            self.locations = self.sim.pick_place(self.locations, move)
            return
        try:
            self._hardware_move(move)
            self.locations = self.planner.apply(self.locations, move)
        except BaseException:
            # Preserve a held object. Do not reset joints or open after failure.
            try:
                self.arm.stop_skill()
            except Exception:
                pass
            raise

    def _pose(self, center, rotation):
        return np.array(center) - rotation @ np.array(self.config["tool_to_grasp"])

    def _move(self, xyz, rotation, label="move"):
        xyz = np.array(xyz)
        current = self.arm.get_pose()
        angle = np.arccos(
            np.clip((np.trace(current.rotation.T @ rotation) - 1) / 2, -1, 1)
        )
        print(f"MOVE {label}: current XYZ={current.translation.round(5).tolist()}, "
              f"target XYZ={xyz.round(5).tolist()}, rotation change={angle:.4f} rad", flush=True)
        duration = max(
            3.0,
            1.875 * np.linalg.norm(xyz - current.translation) / self.config["speed"],
            1.875 * angle / 0.25,
        )
        target = self.transform(
            translation=xyz,
            rotation=rotation,
            from_frame="franka_tool",
            to_frame="world",
        )
        self.arm.goto_pose(
            target, duration=float(duration), use_impedance=False, ignore_errors=False
        )
        actual = self.arm.get_pose()
        rotation_error = np.arccos(
            np.clip((np.trace(actual.rotation.T @ rotation) - 1) / 2, -1, 1)
        )
        print(f"RESULT {label}: actual XYZ={actual.translation.round(5).tolist()}, "
              f"position error={np.linalg.norm(actual.translation-xyz):.5f} m, "
              f"rotation error={rotation_error:.5f} rad", flush=True)

    def _grip(self, close):
        width = 0.0 if close else self.config["gripper"]["opening"]
        self.arm.goto_gripper(
            width,
            grasp=close,
            force=self.config["gripper"]["force"] if close else 0.0,
            epsilon_inner=0.004,
            epsilon_outer=self.config["box_width"] + 0.004 if close else 0.004,
            block=True,
            ignore_errors=False,
        )

    def _hardware_move(self, move):
        pick, place = (
            self.planner.rotations[i] for i in (move.pick_axis, move.place_axis)
        )
        source = self._pose(self.planner.xyz(move.source), pick)
        target = self._pose(self.planner.xyz(move.target), place)
        pre_source = source - pick[:, 2] * self.config["approach_distance"]
        pre_target = target - place[:, 2] * self.config["approach_distance"]
        z = self.config["travel_height"]
        current = self.arm.get_pose()
        z = max(z, current.translation[2])
        self._grip(False)
        self._move([*current.translation[:2], z], current.rotation, label="raise at current XY")
        self._move([*pre_source[:2], z], current.rotation, label="above waiting block")
        self._move([*pre_source[:2], z], pick, label="rotate above waiting block")
        self._move(pre_source, pick, label="descend to pre-grasp")
        self._move(source, pick, label="descend to grasp")
        self._grip(True)
        self._move(pre_source, pick, label="retract after grasp")
        self._move([*pre_source[:2], z], pick, label="lift waiting block")
        self._move([*pre_target[:2], z], pick, label="transfer above destination")
        self._move([*pre_target[:2], z], place, label="rotate above destination")
        self._move(pre_target, place, label="descend to pre-place")
        self._move(target, place, label="descend to place")
        self._grip(False)
        self._move(pre_target, place, label="retract after release")
        self._move([*pre_target[:2], z], place, label="final lift")

    def close(self):
        if self.sim:
            self.sim.close()
