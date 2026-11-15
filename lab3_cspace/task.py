"""Plan direct pattern transitions using grid/waiting occupancy and top grasps."""

from dataclasses import dataclass
from heapq import heappop, heappush
from itertools import count
from operator import itemgetter
import numpy as np
from lab3_cspace.grasp import Grasp


class Pattern:
    def get(self, name=None):
        patterns = {
            "pickaxe": ["BBB", ".R.", ".R."],
            "axe": ["BB.", "BR.", ".R."],
            "box": ["BRB", "R.R", "BRB"],
            "bow": ["BR.", "B.R", "BR."],
        }
        return patterns if name is None else patterns[name]


@dataclass(frozen=True)
class Move:
    block: str
    source: str
    target: str
    pick_axis: str
    place_axis: str


class Planner:
    def __init__(self, config):
        self.config = config
        self.blocks = tuple(f"{c}{i}" for c in "RB" for i in range(4))
        self.initial_locations = {b: "waiting:" + b for b in self.blocks}
        self.initial_slots = tuple(range(9, 17))
        self.half = config["box_width"] / 2
        self.table_z = config["table_height"]
        self.grasps = Grasp(config)
        self.rotations = {a: self.grasps.rotation(a) for a in ("X", "Y")}
        if set(config["boxes"]) != set(self.blocks):
            raise ValueError("boxes must contain R0..R3 and B0..B3")

    def location(self, value):
        if isinstance(value, int):
            return (
                "grid:" + str(value)
                if value < 9
                else "waiting:" + self.blocks[value - 9]
            )
        if isinstance(value, tuple):
            return "grid:" + str(3 * value[0] + value[1])
        return "waiting:" + value if value in self.blocks else value

    def slot(self, location):
        location = self.location(location)
        if location.startswith("grid:"):
            return int(location[5:])
        return 9 + self.blocks.index(location.split(":")[1])

    def xyz(self, location):
        location = self.location(location)
        if location.startswith("waiting:"):
            return np.array(self.config["boxes"][location.split(":")[1]], dtype=float)
        row, col = divmod(int(location[5:]), 3)
        return (
            np.array(self.config["grid"]["center"])
            + np.array([1 - row, 1 - col, 0]) * self.config["grid"]["spacing"]
        )

    def orientations(self, location, locations, ignore=None):
        occupied = [
            self.xyz(pos)
            for block, pos in locations.items()
            if block != ignore and pos != "held"
        ]
        return self.grasps.available(self.xyz(location), occupied)

    def feasible(self, pattern):
        """A shape is constructible only if its blocks can be removed in reverse."""
        occupied = {
            "grid:" + str(3 * r + c)
            for r, row in enumerate(pattern)
            for c, color in enumerate(row)
            if color != "."
        }
        while occupied:
            for cell in sorted(occupied):
                others = [self.xyz(other) for other in occupied if other != cell]
                if self.grasps.available(self.xyz(cell), others):
                    occupied.remove(cell)
                    break
            else:
                return False
        return True

    def _board(self, state):
        return {
            pos: self.blocks[i][0]
            for i, pos in enumerate(state)
            if pos.startswith("grid:")
        }

    def _distance(self, state, goals):
        occupied = self._board(state)
        missing = sum(occupied.get(cell) != color for cell, color in goals.items())
        wrong = sum(goals.get(cell) != color for cell, color in occupied.items())
        return max(missing, wrong)

    def _state_key(self, state):
        # Same-color blocks are interchangeable, but each returned path
        # retains real block IDs for the Robot's location dictionary.
        return tuple(
            sorted((pos, self.blocks[i][0]) for i, pos in enumerate(state))
        )

    def plan(self, pattern, locations):
        if len(pattern) != 3 or any(
            len(row) != 3 or set(row) - set("RB.") for row in pattern
        ):
            raise ValueError("A pattern must be three rows containing R, B or .")
        if not self.feasible(pattern):
            return None
        start = tuple(locations[b] for b in self.blocks)
        if len(set(start)) != len(start) or "held" in start:
            raise ValueError(
                "Finish the held move and use distinct block locations before planning"
            )
        goals = {
            "grid:" + str(3 * r + c): color
            for r, row in enumerate(pattern)
            for c, color in enumerate(row)
            if color != "."
        }

        queue = []
        serial = count()
        best = {self._state_key(start): 0}
        heappush(queue, (self._distance(start, goals), self._distance(start, goals), next(serial), 0, start, []))
        while queue:
            _, _, _, cost, state, path = heappop(queue)
            if cost != best.get(self._state_key(state)):
                continue
            if self._distance(state, goals) == 0:
                return path
            current = dict(zip(self.blocks, state))
            occupied = set(state)
            candidates = []
            for i, block in enumerate(self.blocks):
                source = state[i]
                picks = self.orientations(source, current, block)
                if not picks:
                    continue
                destinations = [
                    cell
                    for cell, color in goals.items()
                    if color == block[0] and cell not in occupied
                ]
                if source.startswith("grid:"):
                    waiting = ["waiting:" + block] + [
                        "waiting:" + b for b in self.blocks
                        if b[0] == block[0] and b != block
                    ]
                    destinations += [pos for pos in waiting if pos not in occupied]
                for target in destinations:
                    places = self.orientations(target, current, block)
                    if not places:
                        continue
                    # Keep one grasp orientation for the transfer whenever possible.
                    # Prefer Y when both ends are open; use X when clearance requires it.
                    shared = [axis for axis in ("Y", "X") if axis in picks and axis in places]
                    pick_axis = shared[0] if shared else ("Y" if "Y" in picks else picks[0])
                    place_axis = pick_axis if pick_axis in places else places[0]
                    move = Move(block, source, target, pick_axis, place_axis)
                    after = list(state)
                    after[i] = target
                    after = tuple(after)
                    new_cost = cost + 1
                    fingerprint = self._state_key(after)
                    if new_cost >= best.get(fingerprint, float("inf")):
                        continue
                    best[fingerprint] = new_cost
                    remaining = self._distance(after, goals)
                    candidates.append((remaining, move, after))
            for remaining, move, after in sorted(candidates, key=itemgetter(0)):
                heappush(
                    queue,
                    (
                        cost + 1 + remaining,
                        remaining,
                        next(serial),
                        cost + 1,
                        after,
                        path + [move],
                    ),
                )
        return None
