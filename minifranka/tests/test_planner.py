"""Two cases per method: a successful outcome and an expected rejection.

These tests use fixture coordinates, never the hardware calibration/config.
Planner.orientation is named Planner.orientations in the implementation.
"""

import json
from pathlib import Path
import sys

import pytest

# This project uses standalone modules rather than an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grasp import Grasp
from planner import Pattern, Planner


@pytest.fixture
def config():
    return json.loads(Path(__file__).with_name("config.json").read_text())


@pytest.fixture
def planner(config):
    return Planner(config)


def test_plan_builds_pickaxe(planner):
    locations = planner.initial_locations.copy()
    pattern = Pattern().get("pickaxe")

    moves = planner.plan(pattern, locations)

    assert moves is not None
    for move in moves:
        assert locations[move.block] == move.source
        assert move.target not in locations.values()
        locations[move.block] = move.target
    actual = {location: block[0] for block, location in locations.items()
              if location.startswith("grid:")}
    expected = {f"grid:{3 * row + col}": color
                for row, cells in enumerate(pattern)
                for col, color in enumerate(cells) if color != "."}
    assert actual == expected


def test_plan_rejects_infeasible_axe(planner):
    moves = planner.plan(Pattern().get("axe"), planner.initial_locations.copy())

    assert moves is None


def test_feasible_accepts_box(planner):
    assert planner.feasible(Pattern().get("box")) is True


def test_feasible_rejects_axe(planner):
    assert planner.feasible(Pattern().get("axe")) is False


def test_orientations_returns_clear_axis(planner):
    locations = planner.initial_locations.copy()
    locations["R0"] = "grid:1"  # North of the center: obstructs closing along X.

    assert planner.orientations("grid:4", locations) == ["Y"]


def test_orientations_rejects_both_blocked_axes(planner):
    locations = planner.initial_locations.copy()
    locations["R0"] = "grid:1"  # North: blocks X.
    locations["R1"] = "grid:3"  # West: blocks Y.

    assert planner.orientations("grid:4", locations) == []


def test_location_resolves_waiting_block(planner):
    assert planner.location("R0") == "waiting:R0"


def test_location_rejects_out_of_range_slot(planner):
    # Grid slots are 0..8, waiting slots 9..16.
    with pytest.raises(IndexError):
        planner.location(17)


def test_available_accepts_unobstructed_grasps(config):
    grasp = Grasp(config)

    assert grasp.available([0.0, 0.0, 0.02], []) == ["X", "Y"]


def test_available_rejects_obstructed_grasps(config):
    grasp = Grasp(config)
    neighbors = [[0.058, 0.0, 0.02], [0.0, 0.058, 0.02]]

    assert grasp.available([0.0, 0.0, 0.02], neighbors) == []
