"""Two downward grasps. X/Y denotes the direction in which the jaws close."""

import numpy as np


class Grasp:
    def __init__(self, config):
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
            angle = np.deg2rad(self.x_yaw_offset_deg)
            c, s = np.cos(angle), np.sin(angle)
            yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            return yaw @ target  # Base-frame yaw offset; tool Z stays downward.
        if axis == "Y0":
            return first
        if reference is not None and np.trace(reference.T @ second) > np.trace(
            reference.T @ first
        ):
            return second
        return first
