"""Run lab3 in simulation; --hw selects the physical Franka."""

import argparse
import json
from pathlib import Path
from planner import INITIAL, PATTERNS, Planner
from robot import Robot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    planner = Planner(config)
    locations = INITIAL
    actions = []
    for pattern in PATTERNS:
        result = planner.plan(pattern, locations)
        if result is None:
            print(
                f"Skip {pattern}: no plan for the configured grasp clearances",
                flush=True,
            )
            continue
        locations, moves = result
        print(
            f"{pattern}: {len(moves)} moves, including any reset from the previous pattern",
            flush=True,
        )
        actions.extend(moves)
    if not actions:
        return
    robot = Robot(config, planner, hw=args.hw)
    try:
        for move in actions:
            robot.pick_place(move)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
