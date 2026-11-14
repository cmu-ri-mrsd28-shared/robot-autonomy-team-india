"""Your original lab3 placement/removal search, exposed as Planner.

The search below is retained from the previous task_planner.py. Only its
geometry/configuration adapter is new. No teammate modules are imported.
"""

from dataclasses import dataclass
import numpy as np

PATTERNS = {
    "pickaxe": ("BBB", ".R.", ".R."),
    "axe": ("BB.", "BR.", ".R."),
    "box": ("BRB", "R.R", "BRB"),
    "bow": ("BR.", "B.R", "BR."),
}
NAMES = tuple(f"{color}{i}" for color in "RB" for i in range(4))
COLORS = tuple(name[0] for name in NAMES)
INITIAL = tuple(range(9, 17))


@dataclass(frozen=True)
class Move:
    block: int
    source: int
    target: int
    pick_axis: int
    place_axis: int

    def key(self, state):
        return (state, self.block, self.target, self.pick_axis, self.place_axis)


class Planner:
    def __init__(self, config):
        self.config = config
        self.half = config["box_width"] / 2
        self.table_z = config["table_height"]
        self._axes_cache = {}
        self.rotations, self.labels = [], []

        def grasp(label, approach, closing):
            y, z = np.array(closing), np.array(approach)
            self.labels.append(label)
            self.rotations.append(np.column_stack((np.cross(y, z), y, z)))

        grasp("top, closing X", [0, 0, -1], [1, 0, 0])
        grasp("top, closing Y", [0, 0, -1], [0, 1, 0])
        for label, direction, closing in (
            ("+X", [1, 0, 0], [0, 1, 0]),
            ("-X", [-1, 0, 0], [0, 1, 0]),
            ("+Y", [0, 1, 0], [1, 0, 0]),
            ("-Y", [0, -1, 0], [1, 0, 0]),
        ):
            grasp("side " + label, direction, closing)
            grasp("side " + label + ", vertical closing", direction, [0, 0, 1])
        if set(config["boxes"]) != set(NAMES):
            raise ValueError("boxes must contain R0..R3 and B0..B3")
        if not 0 < config["box_width"] < config["gripper"]["opening"] <= 0.08:
            raise ValueError("Require box width < gripper opening <= 0.08 m")
        if not 0 < config["gripper"]["force"] <= 70 or config["speed"] <= 0:
            raise ValueError("Invalid force or speed")
        if config["grid"]["spacing"] <= config["box_width"]:
            raise ValueError("Grid cells overlap")
        limits = np.array(config["workspace"])
        if (
            limits.shape != (2, 3)
            or not np.isfinite(limits).all()
            or np.any(limits[0] >= limits[1])
        ):
            raise ValueError("workspace must contain lower and upper XYZ bounds")
        offset = config["tool_to_grasp"]
        if offset is not None and (
            np.shape(offset) != (3,) or not np.isfinite(offset).all()
        ):
            raise ValueError("tool_to_grasp must be a measured XYZ vector or null")
        for slot in range(17):
            point = self.xyz(slot)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Locations must contain three finite coordinates")
            if point[2] - self.half < config["table_height"] - 0.001:
                raise ValueError("A box extends below the table")
        if config["approach_distance"] <= 0:
            raise ValueError("approach_distance must be positive")

    def xyz(self, slot):
        if slot >= 9:
            return np.array(self.config["boxes"][NAMES[slot - 9]], dtype=float)
        row, col = divmod(slot, 3)
        return (
            np.array(self.config["grid"]["center"])
            + np.array([1 - row, 1 - col, 0]) * self.config["grid"]["spacing"]
        )

    def axes(self, slot, state, ignore=None):
        """Extend the original finger-clearance check with side approach lanes.

        These are conservative Panda gripper envelopes, not full-arm geometry.
        Top finger extents retain the original 14/56 mm clearance dimensions.
        """
        key = (slot, tuple(state), ignore)
        if key in self._axes_cache:
            return self._axes_cache[key]
        point = self.xyz(slot)
        valid = []
        for axis, rotation in enumerate(self.rotations):
            if (
                min(
                    point[2] - (abs(rotation) @ np.array([0.014, 0.056, 0.01]))[2],
                    (point - rotation[:, 2] * 0.10)[2]
                    - (abs(rotation) @ np.array([0.04, 0.05, 0.03]))[2],
                )
                < self.table_z - 0.001
            ):
                continue
            pre = point - rotation[:, 2] * self.config["approach_distance"]
            high = pre.copy()
            high[2] = max(high[2], self.config["travel_height"])
            clear = True
            for a, b in ((high, pre), (pre, point)):
                for t in np.linspace(
                    0, 1, max(2, int(np.linalg.norm(b - a) / 0.01) + 1)
                ):
                    center = a + t * (b - a)
                    # Fingers near the pinch center, palm 10 cm behind it.
                    for p, half, is_finger in (
                        (center, abs(rotation) @ np.array([0.014, 0.056, 0.01]), True),
                        (
                            center - rotation[:, 2] * 0.10,
                            abs(rotation) @ np.array([0.04, 0.05, 0.03]),
                            False,
                        ),
                    ):
                        if p[2] - half[2] < self.config["table_height"] - 0.001:
                            clear = False
                            break
                        for block, other in enumerate(state):
                            if other is None:
                                continue
                            if block == ignore and is_finger:
                                continue
                            if np.all(
                                abs(p - self.xyz(other)) < half + self.half + 0.002
                            ):
                                clear = False
                                break
                        if not clear:
                            break
                    if not clear:
                        break
                if not clear:
                    break
            if clear:
                valid.append(axis)
        self._axes_cache[key] = valid
        return valid

    def plan(self, name, locations=INITIAL):
        return plan_pattern(locations, name, self)

    def reset(self, locations):
        return reset_plan(locations, self)

    def apply(self, locations, move):
        return apply(locations, move, self)


def occupancy(state):
    return {slot: COLORS[i] for i, slot in enumerate(state)}


def matches(state, pattern):
    occ = occupancy(state)
    return all(
        occ.get(3 * r + c, ".") == pattern[r][c] for r in range(3) for c in range(3)
    )


def apply(state, move, layout):
    if state[move.block] != move.source or move.target in state:
        raise ValueError("Invalid move source or occupied destination")
    if move.pick_axis not in layout.axes(move.source, state, move.block):
        raise ValueError("Blocked pick")
    if move.place_axis not in layout.axes(move.target, state, move.block):
        raise ValueError("Blocked placement")
    result = list(state)
    result[move.block] = move.target
    return tuple(result)


def reset_plan(state, layout, banned=frozenset()):
    """Backtrack over removals to return all blocks to color-specific parking.

    A temporary obstruction is fine if other blocks can be removed first.
    If no removal order exists, this is a genuine discrete deadlock.
    """
    failed = set()

    def search(current):
        if all(s >= 9 for s in current):
            return []
        if current in failed:
            return None
        for block, source in enumerate(current):
            if source >= 9:
                continue
            targets = range(9, 13) if COLORS[block] == "R" else range(13, 17)
            for target in targets:
                if target in current:
                    continue
                for pick in layout.axes(source, current, block):
                    for place in layout.axes(target, current, block):
                        move = Move(block, source, target, pick, place)
                        if move.key(current) in banned:
                            continue
                        after = apply(current, move, layout)
                        rest = search(after)
                        if rest is not None:
                            return [move] + rest
        failed.add(current)
        return None

    result = search(state)
    if result is None:
        return None
    for move in result:
        state = apply(state, move, layout)
    return state, result


def construct_plan(state, pattern, layout, banned=frozenset()):
    """Complete DFS over monotone constructions from empty grid, including yaw.

    Parking supplies interchangeable blocks. Memoization prunes failed states;
    geometric failures can ban a particular state/action and trigger replanning.
    """
    goals = {
        3 * r + c: color
        for r, row in enumerate(pattern)
        for c, color in enumerate(row)
        if color != "."
    }
    shape = tuple(list(goals) + [None] * (8 - len(goals)))
    while any(s is not None for s in shape):
        for block, slot in enumerate(shape):
            if slot is not None and layout.axes(slot, shape, block):
                shape = tuple(None if i == block else s for i, s in enumerate(shape))
                break
        else:
            return None
    failed = set()

    def search(current):
        if matches(current, pattern):
            return []
        if current in failed:
            return None
        # Fill constrained cells first; backtracking still explores other orders.
        todo = [s for s in goals if s not in current]
        todo.sort(key=lambda s: len(layout.axes(s, current)))
        for target in todo:
            for block, source in enumerate(current):
                if source < 9 or COLORS[block] != goals[target]:
                    continue
                for pick in layout.axes(source, current, block):
                    for place in layout.axes(target, current, block):
                        move = Move(block, source, target, pick, place)
                        if move.key(current) in banned:
                            continue
                        after = apply(current, move, layout)
                        rest = search(after)
                        if rest is not None:
                            return [move] + rest
        failed.add(current)
        return None

    result = search(state)
    if result is None:
        return None
    final = state
    for move in result:
        final = apply(final, move, layout)
    # A completed pattern must admit autonomous removal, not just construction.
    if reset_plan(final, layout) is None:
        return None
    return final, result


def plan_pattern(state, name, layout, banned=frozenset()):
    reset = reset_plan(state, layout, banned)
    if reset is None:
        return None
    parked, reset_moves = reset
    construction = construct_plan(parked, PATTERNS[name], layout, banned)
    if construction is None:
        return None
    final, moves = construction
    return final, reset_moves + moves
