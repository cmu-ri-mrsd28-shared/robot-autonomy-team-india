"""Read-only calibration helpers. Run --grasp while aligned to grasp R0."""

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grasp", action="store_true",
        help="Print tool_to_grasp using the current tool pose and R0's configured center",
    )
    args = parser.parse_args()
    if not args.grasp:
        parser.print_help()
        return

    config = json.loads(Path(__file__).with_name("config.json").read_text())
    center = np.asarray(config["boxes"]["R0"], dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("boxes.R0 must contain three measured, finite coordinates")

    from frankapy import FrankaArm

    pose = FrankaArm().get_pose()
    translation = np.asarray(pose.translation, dtype=float)
    rotation = np.asarray(pose.rotation, dtype=float)
    if (translation.shape != (3,) or rotation.shape != (3, 3)
            or not np.isfinite(translation).all() or not np.isfinite(rotation).all()):
        raise ValueError("Invalid tool-pose feedback")
    offset = rotation.T @ (center - translation)
    print("Assumes the gripper is aligned to grasp R0 and boxes.R0 is measured.")
    print("Copy into config.json; no robot motion or config changes were requested:")
    print(json.dumps({"tool_to_grasp": offset.round(6).tolist()}, indent=2))


if __name__ == "__main__":
    main()
