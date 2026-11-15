"""Two downward grasps. X/Y denotes the direction in which the jaws close."""

import numpy as np


class Grasp:
    def __init__(self, config):
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
        if axis == "X90":
            return second  # Rz(+90 degrees) @ canonical downward Y rotation.
        if axis in ("X270", "Y0"):
            return first  # X270 = Rz(-90 degrees) @ canonical downward Y.
        if reference is not None and np.trace(reference.T @ second) > np.trace(
            reference.T @ first
        ):
            return second
        return first
