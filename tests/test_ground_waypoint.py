"""Rescue waypoint height follows the terrain actually loaded by Gazebo."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from rescue_population.ground_waypoint import GroundWaypoints
from rescue_population.world import create_population_world


def test_waypoint_interpolates_the_ground_under_the_target(tmp_path):
    mesh = tmp_path / "slope.obj"
    mesh.write_text(
        "v 0 0 1\nv 2 0 3\nv 0 2 5\n"
        "v 0 0 8\nv 2 0 8\nv 0 2 8\n"
        "usemtl Terrain\nf 1 2 3\nusemtl Tree\nf 4 5 6\n"
    )
    waypoints = GroundWaypoints.from_mesh(mesh)

    assert waypoints.at((0.5, 1.0, 6.0)) == pytest.approx((0.5, 1.0, 3.5))
    assert waypoints.at((2.0, 2.0, 6.0)) is None
    assert waypoints.at((float("nan"), 1.0, 6.0)) is None


def test_waypoint_uses_the_collision_mesh_for_each_seed(tmp_path):
    heights = []
    for seed in (42, 43):
        output = tmp_path / str(seed) / "terrain_new.sdf"
        result = create_population_world(
            REPO, "terrain_new", output, humans=0, animals=0, terrain_height=8,
            map_size=60, spawn_seed=seed,
        )
        mesh_uri = ET.parse(output).findtext(
            "world/model[@name='generated_ground']/link/collision/geometry/mesh/uri"
        )
        assert result.ground_mesh_path.as_uri() == mesh_uri
        waypoint = GroundWaypoints.from_mesh(result.ground_mesh_path).at((24.0, 19.0, 10.0))
        assert waypoint[:2] == (24.0, 19.0)
        heights.append(waypoint[2])
    assert heights[0] != pytest.approx(heights[1])


def test_waypoint_surface_is_absent_for_other_worlds(tmp_path):
    result = create_population_world(
        REPO, "simple_trees", tmp_path / "simple_trees.sdf", humans=0, animals=0,
    )
    assert result.ground_mesh_path is None
