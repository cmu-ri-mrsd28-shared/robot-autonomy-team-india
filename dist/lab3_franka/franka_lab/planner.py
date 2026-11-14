"""Exhaustive reverse search under an explicit two-direction grasp model.

Each edge removes one block with both finger-side neighbors clear. Reversing
the edges gives a construction plan and its reset certificate. This geometric
abstraction is checked separately against the MuJoCo finger collision geometry.
"""

from collections import deque
from dataclasses import dataclass

from .patterns import target_mask


@dataclass(frozen=True)
class Placement:
    cell: int
    orientation: str
    color: str


def orientations(mask, cell):
    row, col = divmod(cell, 3)
    choices = []
    for name, neighbors in (
        ("NS", ((row - 1, col), (row + 1, col))),
        ("EW", ((row, col - 1), (row, col + 1))),
    ):
        if all(not (0 <= r < 3 and 0 <= c < 3) or not mask & (1 << (3*r+c))
               for r, c in neighbors):
            choices.append(name)
    return choices


def plan_mask(mask):
    """Return (build order, visited count); None means exhaustive model failure."""
    if mask < 0 or mask >= 512:
        raise ValueError("Occupancy must be a nine-bit mask")
    queue = deque([mask])
    parents = {mask: None}
    while queue:
        state = queue.popleft()
        if state == 0:
            build = []
            while parents[state] is not None:
                previous, cell, axis = parents[state]
                build.append((cell, axis))
                state = previous
            return build, len(parents)
        for cell in range(9):
            if not state & (1 << cell):
                continue
            for axis in orientations(state, cell):
                successor = state & ~(1 << cell)
                if successor not in parents:
                    parents[successor] = (state, cell, axis)
                    queue.append(successor)
    return None, len(parents)


def plan_pattern(rows):
    build, visited = plan_mask(target_mask(rows))
    if build is None:
        return None, visited
    colors = "".join(rows)
    return [Placement(cell, axis, colors[cell]) for cell, axis in build], visited
