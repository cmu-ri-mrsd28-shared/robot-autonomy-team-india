"""Simulation by default; --hw runs FrankaPy and pauses between patterns."""

import argparse
import json
from pathlib import Path
from planner import Pattern, Planner
from robot import Robot, HardwareRobot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(__file__).with_name("config.json").read_text())
    planner = Planner(config)
    robot = HardwareRobot(config, planner) if args.hw else Robot(config, planner)
    try:
        for name, pattern in Pattern().get().items():
            if robot.run_pattern(name, pattern) is False:
                break
    finally:
        robot.close()


if __name__ == "__main__":
    main()
