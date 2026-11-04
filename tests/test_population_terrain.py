import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rescue_population.terrain import TerrainSurface


def write_model(tmp_path, obj, *, pose="0 0 0 0 0 0", scale="1 1 1",
                link_pose="0 0 0 0 0 0", collision_pose="0 0 0 0 0 0"):
    model = tmp_path / "terrain_fixture"
    model.mkdir()
    (model / "terrain.obj").write_text(obj)
    (model / "model.sdf").write_text(
        '<sdf version="1.8"><model name="fixture"><pose>' + pose +
        '</pose><link name="ground"><pose>' + link_pose +
        '</pose><collision name="surface"><pose>' + collision_pose +
        '</pose><geometry><mesh><uri>model://terrain_fixture/terrain.obj</uri><scale>' +
        scale + '</scale></mesh></geometry></collision></link></model></sdf>')
    return model


def square(z=0):
    return np.array([[[0, 0, z], [10, 0, z], [10, 10, z]],
                     [[0, 0, z], [10, 10, z], [0, 10, z]]], dtype=float)


def test_y_up_mesh_uses_scale_then_full_pose(tmp_path):
    # Raw Y is height. After scale and rotation the plane rises 0.75 m per world X.
    model = write_model(
        tmp_path,
        "v 0 0 0\nv 4 2 0\nv 4 2 4\nv 0 0 4\nusemtl Terrain\nf 1 2 3 4\n",
        pose="10 20 3 1.5707963267948966 0 0", scale="2 3 4")
    surface = TerrainSurface.from_model(model)
    assert surface.bounds == pytest.approx((10, 18, 4, 20))
    point = surface.sample(14, 12)
    assert point.z == pytest.approx(6)
    assert point.slope_deg == pytest.approx(math.degrees(math.atan(0.75)))
    assert not surface.is_clear(14, 12, 0.2, max_slope_deg=30)
    assert surface.is_clear(14, 12, 0.2, max_slope_deg=40)


def test_link_and_collision_poses_are_composed(tmp_path):
    model = write_model(
        tmp_path,
        "v 0 0 0\nv 4 0 0\nv 4 4 0\nv 0 4 0\nusemtl Terrain\nf 1 2 3 4\n",
        pose="10 20 1 0 0 1.5707963267948966",
        link_pose="2 0 1 0 0 0", collision_pose="0 3 2 0 0 0")
    surface = TerrainSurface.from_model(model)
    assert surface.sample(5, 24).z == pytest.approx(4)
    assert surface.bounds == pytest.approx((3, 7, 22, 26))


def test_concave_ngon_and_negative_obj_indices_do_not_fill_missing_ground(tmp_path):
    model = write_model(
        tmp_path,
        "v 0 0 0\nv 3 0 0\nv 3 1 0\nv 1 1 0\nv 1 3 0\nv 0 3 0\n"
        "usemtl Terrain\nf -6 -5 -4 -3 -2 -1 # Concave ground outline\n")
    surface = TerrainSurface.from_model(model)
    assert surface.sample(0.5, 2).z == pytest.approx(0)
    assert surface.sample(2, 0.5).z == pytest.approx(0)
    assert surface.sample(2, 2) is None
    assert not surface.is_clear(0.9, 1.5, 0.2)


def test_clearance_rejects_tree_canopy_and_vertical_trunk():
    canopy = [[[6, 6, 4], [8, 6, 4], [7, 8, 4]]]
    # This wall is degenerate in XY, so point-in-triangle alone cannot detect it.
    trunk = [[[5, 4, 0], [5, 6, 0], [5, 5, 3]]]
    surface = TerrainSurface(square(), canopy + trunk)
    assert not surface.is_clear(7, 6.5, 0.2)
    assert not surface.is_clear(5.3, 5, 0.4)
    assert surface.is_clear(3, 3, 0.4)


def test_footprint_stays_inside_terrain_and_rejects_unknown_points():
    surface = TerrainSurface(square())
    assert surface.is_clear(0.5, 5, 0.5)
    assert not surface.is_clear(0.5, 5, 0.6)
    assert surface.sample(-1, 5) is None
    assert surface.sample(float("nan"), 5) is None
    assert not surface.is_clear(5, 5, -1)


def test_highest_ground_layer_is_sampled():
    surface = TerrainSurface(np.concatenate((square(0), square(2))))
    assert surface.sample(5, 5).z == pytest.approx(2)


def test_missing_ground_material_fails_with_clear_error(tmp_path):
    model = write_model(
        tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Rock\nf 1 2 3\n")
    with pytest.raises(ValueError, match="Terrain material"):
        TerrainSurface.from_model(model)


def test_nonpositive_scale_is_rejected(tmp_path):
    model = write_model(
        tmp_path, "v 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl Terrain\nf 1 2 3\n",
        scale="1 0 1")
    with pytest.raises(ValueError, match="three positive"):
        TerrainSurface.from_model(model)
