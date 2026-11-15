"""MuJoCo adapter for pose(), move() and grip(); no task or motion planner."""

from contextlib import nullcontext
from types import SimpleNamespace
import time
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
from .scene import Scene


class Simulation:
    def __init__(self, planner, show_viewer=True):
        self.scene = Scene(planner)
        self.model, self.data = self.scene.model, self.scene.data
        self.ik_data = mj.MjData(self.model)
        self.q_target = self.data.qpos[:7].copy()
        self.opening = planner.config["gripper"]["opening"] / 2
        self.viewer = None
        if show_viewer:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 1.8
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -40
            self.viewer.cam.lookat[:] = [0.45, 0, 0.15]

    def pose(self):
        return SimpleNamespace(
            translation=self.data.site_xpos[self.scene.site].copy(),
            rotation=self.data.site_xmat[self.scene.site].reshape(3, 3).copy(),
        )

    def jaw_width(self):
        return float(self.data.qpos[7] + self.data.qpos[8])

    def _joint_target(self, xyz, rotation):
        """Numerical IK only, equivalent to the Cartesian interface in FrankaPy."""
        d, m = self.ik_data, self.model
        d.qpos[:] = self.data.qpos
        q = self.q_target.copy()
        position_jac = np.zeros((3, m.nv))
        rotation_jac = np.zeros((3, m.nv))
        for _ in range(35):
            d.qpos[:7] = q
            mj.mj_forward(m, d)
            actual = d.site_xmat[self.scene.site].reshape(3, 3)
            error = np.r_[
                xyz - d.site_xpos[self.scene.site],
                Rotation.from_matrix(rotation @ actual.T).as_rotvec(),
            ]
            if np.linalg.norm(error[:3]) < 0.0002 and np.linalg.norm(error[3:]) < 0.003:
                break
            mj.mj_jacSite(m, d, position_jac, rotation_jac, self.scene.site)
            jac = np.vstack((position_jac[:, :7], rotation_jac[:, :7]))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + 0.0004 * np.eye(6), error)
            delta *= min(1.0, 0.10 / max(np.max(abs(delta)), 1e-12))
            q = np.clip(
                q + delta, m.jnt_range[:7, 0] + 0.001, m.jnt_range[:7, 1] - 0.001
            )
        return q

    def move(self, xyz, rotation, duration):
        self._advance(duration, np.asarray(xyz), np.asarray(rotation))

    def grip(self, width, force):
        self.opening = width / 2
        limit = force if force > 0 else self.scene.layout.config["gripper"]["force"]
        self.model.actuator_forcelimited[self.scene.grip_act] = True
        self.model.actuator_forcerange[self.scene.grip_act] = [-limit, limit]
        self._advance(1.0)

    def _advance(self, duration, xyz=None, rotation=None):
        start = self.pose()
        if xyz is not None:
            turn = Rotation.from_matrix(start.rotation.T @ rotation).as_rotvec()
        dt = self.model.opt.timestep
        steps = max(1, int(np.ceil(duration / dt)))
        wall_start = time.monotonic()
        for k in range(steps + 1):
            if self.viewer is not None and not self.viewer.is_running():
                raise RuntimeError("Viewer closed")
            t = min(k * dt / duration, 1.0)
            if xyz is not None and (k % 10 == 0 or k == steps):
                blend = 10 * t**3 - 15 * t**4 + 6 * t**5
                position = start.translation + blend * (xyz - start.translation)
                orientation = (
                    start.rotation @ Rotation.from_rotvec(blend * turn).as_matrix()
                )
                self.q_target = self._joint_target(position, orientation)
            with self.viewer.lock() if self.viewer is not None else nullcontext():
                self.data.ctrl[self.scene.arm_act] = self.q_target
                self.data.ctrl[self.scene.grip_act] = self.opening
                self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
                mj.mj_step(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError("MuJoCo produced a nonfinite numerical state")
            if self.viewer is not None and k % 8 == 0:
                self.viewer.sync()
                delay = k * dt - (time.monotonic() - wall_start)
                if delay > 0:
                    time.sleep(delay)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
