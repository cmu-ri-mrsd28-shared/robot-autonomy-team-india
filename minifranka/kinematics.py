"""Small Panda FK/IK model using the rigid-link geometry from lab2/Franka.py."""

import numpy as np
from scipy.spatial.transform import Rotation


class PandaIK:
    def __init__(self):
        # (roll, x, y, z) of each fixed joint origin, followed by a Z revolute joint.
        origins = [
            (0, 0, 0, 0.333), (-np.pi / 2, 0, 0, 0),
            (np.pi / 2, 0, -0.316, 0), (np.pi / 2, 0.0825, 0, 0),
            (-np.pi / 2, -0.0825, 0.384, 0), (np.pi / 2, 0, 0, 0),
            (np.pi / 2, 0.088, 0, 0),
        ]
        self.links = []
        for roll, x, y, z in origins:
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_rotvec([roll, 0, 0]).as_matrix()
            transform[:3, 3] = [x, y, z]
            self.links.append(transform)
        self.lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.flange = np.eye(4)
        self.flange[2, 3] = 0.107
        self.tool = np.eye(4)

    @staticmethod
    def pose(xyz, rotation):
        result = np.eye(4)
        result[:3, 3] = np.asarray(xyz, dtype=float)
        # SDK measured rotations can have small numerical orthogonality errors.
        result[:3, :3] = Rotation.from_matrix(rotation).as_matrix()
        if not np.isfinite(result).all():
            raise ValueError("Tool pose must be finite")
        return result

    def align_tool(self, joints, xyz, rotation):
        """Fit the fixed flange-to-reported-tool transform at the current pose."""
        self.tool = np.eye(4)
        self.tool = np.linalg.inv(self.forward(joints)) @ self.pose(xyz, rotation)

    def forward(self, joints):
        return self._evaluate(joints)[0]

    def _evaluate(self, joints):
        q = np.asarray(joints, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite Panda joint angles in radians")
        transform = np.eye(4)
        origins, axes = [], []
        for link, angle in zip(self.links, q):
            transform = transform @ link
            origins.append(transform[:3, 3].copy())
            axes.append(transform[:3, 2].copy())
            turn = np.eye(4)
            c, s = np.cos(angle), np.sin(angle)
            turn[:2, :2] = [[c, -s], [s, c]]
            transform = transform @ turn
        transform = transform @ self.flange @ self.tool
        jacobian = np.empty((6, 7))
        for i, (origin, axis) in enumerate(zip(origins, axes)):
            jacobian[:3, i] = np.cross(axis, transform[:3, 3] - origin)
            jacobian[3:, i] = axis
        return transform, jacobian

    def solve(self, xyz, rotation, seed):
        """Damped Jacobian IK, seeded by the previous/current joint configuration."""
        goal = self.pose(xyz, rotation)
        q = np.asarray(seed, dtype=float).copy()
        self.forward(q)  # Validate the seed before clipping or iteration.
        q = np.clip(q, self.lower + 1e-5, self.upper - 1e-5)
        for _ in range(200):
            actual, jacobian = self._evaluate(q)
            position_error = goal[:3, 3] - actual[:3, 3]
            rotation_error = Rotation.from_matrix(goal[:3, :3] @ actual[:3, :3].T).as_rotvec()
            if np.linalg.norm(position_error) < 0.0002 and np.linalg.norm(rotation_error) < 0.002:
                return q
            error = np.r_[position_error, rotation_error]
            step = jacobian.T @ np.linalg.solve(jacobian @ jacobian.T + 1e-4 * np.eye(6), error)
            step *= min(1.0, 0.12 / max(float(np.max(np.abs(step))), 1e-12))
            q = np.clip(q + step, self.lower + 1e-5, self.upper - 1e-5)
        raise RuntimeError(
            f"Panda IK did not converge: position error={np.linalg.norm(position_error):.5f} m, "
            f"rotation error={np.linalg.norm(rotation_error):.4f} rad; no joint command sent"
        )
