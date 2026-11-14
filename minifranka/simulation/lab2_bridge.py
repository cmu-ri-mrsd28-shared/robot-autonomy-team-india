"""Import lab2 without copying its kinematics, RRT, scene, or interpolation code."""

import importlib
import sys
from pathlib import Path

LAB2 = Path(__file__).resolve().parents[2] / "lab2"
if not (LAB2 / "RRTPickPlace.py").is_file():
    raise ImportError(f"Expected sibling lab2 implementation at {LAB2}")
sys.path.insert(0, str(LAB2))
lab2 = importlib.import_module("RRTPickPlace")

rt = lab2.rt
