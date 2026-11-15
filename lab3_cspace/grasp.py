"""Exact downward X/Y grasps; no hidden yaw corrections."""
import numpy as np

class Grasp:
    def __init__(self, config):
        self.half_box = config["box_width"] / 2
        self.reach = config["gripper"]["opening"] / 2 + .016
        self.half_finger = .014

    def available(self, point, occupied):
        choices = []
        for axis in ("X", "Y"):
            extent = np.array([self.reach, self.half_finger] if axis == "X" else [self.half_finger, self.reach])
            if not any(np.all(abs(np.asarray(point[:2])-np.asarray(other[:2])) < extent+self.half_box) for other in occupied):
                choices.append(axis)
        return choices

    def rotation(self, axis, reference=None):
        y = np.array([1.,0,0] if axis == "X" else [0.,1,0])
        z = np.array([0.,0,-1])
        first = np.column_stack([np.cross(y,z),y,z])
        second = first @ np.diag([-1.,-1,1])
        return second if reference is not None and np.trace(reference.T@second)>np.trace(reference.T@first) else first
