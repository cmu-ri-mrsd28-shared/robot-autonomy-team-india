"""Two downward grasps. X/Y denotes the direction in which the jaws close."""

import numpy as np


class Grasp:
    def __init__(self, config):
        offsets = config.get("grasp_yaw_offsets_deg", {})
        self.yaw_offsets = {"X": float(offsets.get("X", 3.0)), "Y": float(offsets.get("Y", 0.0))}
        if not np.isfinite(list(self.yaw_offsets.values())).all():
            raise ValueError("grasp_yaw_offsets_deg must contain finite degrees")
        self.x_yaw_offset_deg = 0.0  # Calibration-only offset for explicit X90/X270.
        self.half_box = config["box_width"] / 2
        # The original lab3 finger footprint, with the requested 10 mm allowance.
        self.reach = config["gripper"]["opening"] / 2 + 0.016
        self.half_finger = 0.014

    def available(self, point, occupied):
        choices = []
        for axis in ("X", "Y"):
            extent = np.array(
                [self.reach, self.half_finger]
                if axis == "X"
                else [self.half_finger, self.reach]
            )
            if not any(
                np.all(
                    abs(np.array(point[:2]) - np.array(other[:2]))
                    < extent + self.half_box - 0.010
                )
                for other in occupied
            ):
                choices.append(axis)
        return choices

    def rotation(self, axis, reference=None):
        # Explicit headings are used only by calibration, without nearest-pose selection.
        y = np.array([1.0, 0, 0] if axis in ("X", "X90", "X270") else [0.0, 1, 0])
        z = np.array([0.0, 0, -1.0])
        first = np.column_stack([np.cross(y, z), y, z])
        second = first @ np.diag([-1.0, -1.0, 1.0])
        if axis in ("X90", "X270"):
            target = second if axis == "X90" else first
            return self._yaw(target, self.x_yaw_offset_deg)
        if axis == "Y0":
            return first
        if axis == "X":
            first = self._yaw(first, -self.yaw_offsets["X"])
            second = self._yaw(second, self.yaw_offsets["X"])
        elif axis == "Y":
            first = self._yaw(first, self.yaw_offsets["Y"])
            second = self._yaw(second, -self.yaw_offsets["Y"])
        if reference is not None and np.trace(reference.T @ second) > np.trace(
            reference.T @ first
        ):
            return second
        return first

    @staticmethod
    def _yaw(rotation, degrees):
        """Apply yaw about base +Z, keeping a downward tool pointed down."""
        angle = np.deg2rad(degrees)
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ rotation
