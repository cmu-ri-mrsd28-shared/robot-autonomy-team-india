"""Simulation backend retained from your original lab3 implementation."""

from contextlib import nullcontext
import time
import numpy as np
from .scene import Scene
from .motion_planner import MotionPlanner
from .lab2_bridge import lab2


class Simulation:
    def __init__(self, planner, show_viewer=True):
        self.query = Scene(planner)
        self.scene = Scene(planner)
        self.motion = MotionPlanner(self.query)
        self.q = lab2.Q_HOME.copy()
        self.viewer = None
        if show_viewer:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.scene.model, self.scene.data)
            self.viewer.cam.distance = 1.8
            self.viewer.cam.azimuth = 135
            self.viewer.cam.elevation = -40
            self.viewer.cam.lookat[:] = [0.45, 0, 0.15]

    def pick_place(self, locations, move):
        segments, result, q = self.motion.plan_move(locations, move, self.q)
        run_physics(self.scene, segments, self.viewer)
        self.q = q
        return result

    def close(self):
        if self.viewer is not None:
            self.viewer.close()


def run_physics(scene, segments, viewer=None, speed=1):
    import mujoco as mj

    from .lab2_bridge import lab2, rt

    d, m = scene.data, scene.model
    dt = m.opt.timestep
    arm = scene.arm_act
    # Only initialization sets block poses; executing a plan never snaps a block
    # to its intended destination or attaches it to the gripper.
    observations = []
    released = None
    for segment in segments:
        path = segment.path
        pairs = (
            list(zip(path[:-1], path[1:])) if len(path) > 1 else [(path[0], path[0])]
        )
        for a, b in pairs:
            duration = (
                max(0.06, float(np.max(np.abs(b - a))) / lab2.JOINT_SPEED)
                if len(path) > 1
                else 1.0
            )
            steps = max(1, int(np.ceil(duration / dt)))
            start = time.monotonic()
            for k in range(steps + 1):
                if viewer is not None and not viewer.is_running():
                    raise RuntimeError("Viewer closed; execution stopped")
                with viewer.lock() if viewer is not None else nullcontext():
                    q, _ = rt.interp_min_jerk(a, b, min(k * dt, duration), duration)
                    d.ctrl[arm] = q
                    d.ctrl[scene.grip_act] = segment.grip
                    d.qfrc_applied[:7] = d.qfrc_bias[:7]
                    mj.mj_step(m, d)
                    if not np.all(np.isfinite(d.qpos)):
                        raise RuntimeError("Nonfinite simulation state")
                if viewer is not None and k % 8 == 0:
                    viewer.sync()
                    delay = (k * dt) / speed - (time.monotonic() - start)
                    if delay > 0:
                        time.sleep(delay)
        if np.max(np.abs(d.qpos[:7] - path[-1])) > 0.10:
            raise RuntimeError(
                f"Arm tracking error during {segment.label or segment.event}"
            )
        if segment.event == "lift":
            p = d.xpos[scene.block_bodies[segment.block]]
            if p[2] < scene.layout.table_z + 0.08:
                raise RuntimeError(
                    f"Grasp failed: block {segment.block} was not lifted"
                )
        if segment.label == "retreat":
            # The preceding release identifies the block and its target.
            if released is not None:
                block, target = released
                expected = np.array(scene.layout.xyz(target))
                actual = d.xpos[scene.block_bodies[block]].copy()
                if (
                    np.linalg.norm(actual[:2] - expected[:2]) > 0.012
                    or abs(actual[2] - expected[2]) > 0.012
                ):
                    raise RuntimeError(
                        f"Placement failed for block {block}: expected {expected.round(3)}, got {actual.round(3)}"
                    )
                R = d.xmat[scene.block_bodies[block]].reshape(3, 3)
                if np.max(np.abs(R[2, :])) < 0.98:
                    raise RuntimeError(
                        f"Block {block} is tilted after placement: {R[2, :]}"
                    )
                observations.append(
                    dict(
                        block=block,
                        target=target,
                        expected=expected.tolist(),
                        actual=actual.tolist(),
                    )
                )
                print(f"  verified block {block} at slot {target}", flush=True)
                released = None
        if segment.event == "place":
            released = (segment.block, segment.target)
        elif segment.event == "grasp":
            released = None

    return observations
