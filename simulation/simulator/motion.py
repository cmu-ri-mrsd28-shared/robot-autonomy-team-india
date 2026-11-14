"""IK against MuJoCo's own model, avoiding a second tool-frame convention."""

import numpy as np
import mujoco as mj


def rotation(axis):
    if axis == "EW":
        return np.diag([1., -1., -1.])
    if axis == "NS":
        return np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    raise ValueError(axis)


def rotation_error(target, current):
    quat = np.empty(4)
    mj.mju_mat2Quat(quat, (target @ current.T).ravel())
    if quat[0] < 0:
        quat *= -1
    norm = np.linalg.norm(quat[1:])
    return quat[1:] * (2*np.arctan2(norm, quat[0]) / max(norm, 1e-12))


class IK:
    def __init__(self, model, site, arm_qpos, arm_dofs):
        self.model = model
        self.data = mj.MjData(model)
        self.site = site
        self.qpos = arm_qpos
        self.dofs = arm_dofs
        self.limits = np.array([model.joint(f"joint{i}").range for i in range(1, 8)])
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))

    def solve(self, position, desired_rotation, seed):
        q = np.asarray(seed).copy()
        for _ in range(250):
            self.data.qpos[self.qpos] = q
            mj.mj_kinematics(self.model, self.data)
            mj.mj_comPos(self.model, self.data)
            pos_error = np.asarray(position) - self.data.site_xpos[self.site]
            rot_error = rotation_error(desired_rotation, self.data.site_xmat[self.site].reshape(3,3))
            if np.linalg.norm(pos_error) < 0.00005 and np.linalg.norm(rot_error) < 0.0005:
                return q
            mj.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site)
            jac = np.vstack((self.jacp[:, self.dofs], self.jacr[:, self.dofs]))
            error = np.r_[pos_error, rot_error]
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6)*0.00005, error)
            delta *= min(1., 0.15 / max(np.max(np.abs(delta)), 1e-12))
            q = np.clip(q + delta, self.limits[:,0]+0.002, self.limits[:,1]-0.002)
        raise RuntimeError(f"IK failed for {np.round(position,4)}: position={np.linalg.norm(pos_error):.5f}, rotation={np.linalg.norm(rot_error):.5f}")


def min_jerk(t):
    return 10*t**3 - 15*t**4 + 6*t**5
