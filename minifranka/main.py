"""Simulation by default; --hw runs FrankaPy and pauses between patterns.

Examples:
  python3 main.py --list                        # show the available patterns
  python3 main.py --pattern box                 # build only box (from waiting)
  python3 main.py --pattern pickaxe box --hw    # pickaxe, then transition to box
  python3 main.py --pattern bow --plan-only     # print the moves; no robot/sim
"""

import argparse
import json
from pathlib import Path
from planner import Pattern, Planner
from robot import Robot, HardwareRobot


def show_patterns(patterns):
    for name, rows in patterns.items():
        print(name)
        for row in rows:
            print("   " + " ".join(row))


def plan_only(planner, selected):
    """Print each pattern's moves, carrying block locations forward like a run."""
    locations = planner.initial_locations.copy()
    for name, pattern in selected.items():
        moves = planner.plan(pattern, locations)
        if moves is None:
            print(f"{name}: skipped (no feasible top-grasp plan)")
            continue
        print(f"{name}: {len(moves)} moves")
        for move in moves:
            print(f"   {move.block}: {move.source} -> {move.target} "
                  f"[pick {move.pick_axis} / place {move.place_axis}]")
            locations[move.block] = move.target


def main():
    patterns = Pattern().get()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hw", action="store_true", help="run on the real robot")
    parser.add_argument(
        "--pattern", nargs="+", choices=list(patterns), metavar="NAME",
        help=f"pattern(s) to run, in the order given; default: all "
             f"({', '.join(patterns)}). Start: blocks in waiting positions")
    parser.add_argument("--list", action="store_true",
                        help="print the available patterns and exit")
    parser.add_argument("--plan-only", action="store_true",
                        help="print the planned moves without moving anything")
    args = parser.parse_args()

    selected = {name: patterns[name] for name in (args.pattern or patterns)}
    if args.list:
        show_patterns(selected)
        return
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    planner = Planner(config)
    if args.plan_only:
        plan_only(planner, selected)
        return

    print("Patterns to run:", ", ".join(selected), flush=True)
    robot = HardwareRobot(config, planner) if args.hw else Robot(config, planner)
    try:
        for name, pattern in selected.items():
            if robot.run_pattern(name, pattern) is False:
                break
    finally:
        robot.close()


if __name__ == "__main__":
    main()
