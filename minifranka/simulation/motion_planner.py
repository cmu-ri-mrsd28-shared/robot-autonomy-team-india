"""Ground task moves using lab2 FK/Jacobians, RRT and timing helpers."""

import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .lab2_bridge import lab2
from planner import apply


class MotionFailure(RuntimeError):
    def __init__(self, message, move=None, state=None):
        super().__init__(message)
        self.move, self.state = move, state


@dataclass
class Segment:
    path: list
    grip: float
    event: str = ""
    block: int = -1
    target: int = -1
    label: str = ""


class MotionPlanner:
    def __init__(self, scene, seed=13, timeout=30):
        self.scene = scene
        self.bot = lab2.mybot
        self.layout = scene.layout
        self.rng = np.random.default_rng(seed)
        self.timeout = timeout
        self.deadline = float("inf")
        self.ik_cache = {}

    def check_time(self):
        if time.monotonic() > self.deadline:
            raise MotionFailure("Motion planning time budget exhausted")

    def fk(self, q):
        frames, J = self.bot.ForwardKin(list(q))
        return np.array(frames[-1]), np.array(J)

    def tcp_pose(self, xyz, axis):
        pose = np.eye(4)
        pose[:3, :3] = self.layout.rotations[axis]
        pose[:3, 3] = xyz
        return pose

    def ik(self, T, q_seed):
        goal = T @ lab2.T_TCP_FLANGE
        q = np.asarray(q_seed, dtype=float).copy()
        low, high = np.array(self.bot.qmin), np.array(self.bot.qmax)
        for _ in range(200):
            self.check_time()
            pose, J = self.fk(q)
            err = np.r_[
                goal[:3, 3] - pose[:3, 3],
                Rotation.from_matrix(goal[:3, :3] @ pose[:3, :3].T).as_rotvec(),
            ]
            if np.linalg.norm(err[:3]) < 0.0006 and np.linalg.norm(err[3:]) < 0.005:
                return q
            delta = J.T @ np.linalg.solve(J @ J.T + 0.0004 * np.eye(6), err)
            delta *= min(1.0, 0.15 / max(np.max(np.abs(delta)), 1e-12))
            q = np.clip(q + delta, low + 0.001, high - 0.001)
        return None

    def edge(self, a, b, state, grip, touch=None, carried=None):
        n = max(1, int(np.ceil(np.max(np.abs(np.array(b) - a)) / 0.035)))
        return all(
            self.scene.valid(
                np.array(a) + s * (np.array(b) - a), state, grip, touch, carried
            )
            for s in np.linspace(0, 1, n + 1)
        )

    def transit(self, a, b, state, grip, carried=None):
        def edge(a, b):
            self.check_time()
            return self.edge(a, b, state, grip, carried=carried)

        if edge(a, b):
            return [np.array(a), np.array(b)]
        path = lab2.RRTQuery(
            a, b, [], [], max_vertices=600, edge_validator=edge, max_iterations=1500
        )
        if path is None:
            raise MotionFailure("RRT could not connect elevated poses")
        return path

    def cartesian(self, start, T_end, state, grip, touch=None, carried=None):
        flange, _ = self.fk(start)
        T_start = flange @ lab2.T_FLANGE_TCP
        distance = np.linalg.norm(T_end[:3, 3] - T_start[:3, 3])
        steps = max(1, int(np.ceil(distance / 0.008)))
        path = [np.array(start)]
        for t in np.linspace(0, 1, steps + 1)[1:]:
            T = T_end.copy()
            T[:3, 3] = T_start[:3, 3] + t * (T_end[:3, 3] - T_start[:3, 3])
            q = self.ik(T, path[-1])
            if (
                q is None
                or np.max(np.abs(q - path[-1])) > 0.3
                or not self.edge(path[-1], q, state, grip, touch, carried)
            ):
                raise MotionFailure(
                    "Cartesian approach/retreat is unreachable or in collision"
                )
            path.append(q)
        return path

    def approach_candidates(self, T, state, block):
        key = tuple(np.round(T.ravel(), 5))
        seeds = []
        if key in self.ik_cache:
            seeds.append(self.ik_cache[key])
        home = lab2.Q_HOME.copy()
        home[0] = np.arctan2(T[1, 3], T[0, 3])
        seeds.extend([home, lab2.Q_HOME.copy()])
        seeds.extend(self.rng.uniform(self.bot.qmin, self.bot.qmax) for _ in range(8))
        for seed in seeds:
            q = self.ik(T, seed)
            if q is not None and self.scene.valid(q, state, 0.04):
                self.ik_cache[key] = q
                yield q

    def plan_move(self, state, move, q_current):
        self.deadline = time.monotonic() + self.timeout
        block = move.block
        grip = 0.012
        after = apply(state, move, self.layout)
        pick = np.array(self.layout.xyz(move.source))
        place = np.array(self.layout.xyz(move.target))
        place[2] += 0.002
        T_pick = self.tcp_pose(pick, move.pick_axis)
        T_place = self.tcp_pose(place, move.place_axis)
        T_pre = T_pick.copy()
        T_pre[:3, 3] -= self.layout.config["approach_distance"] * T_pick[:3, 2]
        T_post = T_place.copy()
        T_post[:3, 3] -= self.layout.config["approach_distance"] * T_place[:3, 2]
        last_error = "No collision-free IK for pick pre-pose"
        for q_pre in self.approach_candidates(T_pre, state, block):
            try:
                to_pick = self.transit(q_current, q_pre, state, 0.04)
                down = self.cartesian(q_pre, T_pick, state, 0.04, touch=block)
                self.scene.set_state(state, down[-1], grip)
                held = (block, self.scene.relative_block(block))
                up = self.cartesian(
                    down[-1], T_pre, state, grip, touch=block, carried=held
                )
                # The object may rotate 90 degrees when changing grasp orientation;
                # squares are symmetric, and its full attached box is collision-checked.
                for q_post in self.approach_candidates(T_post, state, block):
                    try:
                        transfer = self.transit(up[-1], q_post, state, grip, held)
                        lower = self.cartesian(
                            q_post, T_place, state, grip, touch=block, carried=held
                        )
                        retreat = self.cartesian(
                            lower[-1], T_post, after, 0.04, touch=block
                        )
                        return (
                            [
                                Segment(to_pick, 0.04, label="approach pick"),
                                Segment(down, 0.04, label="descend pick"),
                                Segment([down[-1]], grip, "grasp", block),
                                Segment(up, grip, "lift", block, label="lift"),
                                Segment(transfer, grip, label="transfer"),
                                Segment(lower, grip, label="descend place"),
                                Segment([lower[-1]], 0.04, "place", block, move.target),
                                Segment(retreat, 0.04, label="retreat"),
                            ],
                            after,
                            retreat[-1],
                        )
                    except MotionFailure as exc:
                        last_error = str(exc)
            except MotionFailure as exc:
                last_error = str(exc)
            self.check_time()
        raise MotionFailure(last_error, move, state)

    def compile(self, state, moves, progress=print):
        q = lab2.Q_HOME.copy()
        segments = []
        for i, move in enumerate(moves):
            progress(
                f"  motion {i + 1}/{len(moves)}: block {move.block}, {move.source} -> {move.target}, axes {move.pick_axis}/{move.place_axis}"
            )
            try:
                piece, state, q = self.plan_move(state, move, q)
            except MotionFailure as exc:
                raise MotionFailure(str(exc), move, state) from exc
            segments.extend(piece)
        self.deadline = time.monotonic() + self.timeout
        segments.append(
            Segment(
                self.transit(q, lab2.Q_HOME, state, 0.04), 0.04, label="return home"
            )
        )
        return segments, state
