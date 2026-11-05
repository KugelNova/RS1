"""Placement and physical geometry checks for the supplied tree/rock assets."""
from pathlib import Path
import math
import sys
from urllib.parse import unquote, urlparse

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from rescue_population.props import (  # noqa: E402
    PROP_SPECS, fitted_prop, footprint_radius, ground_radius, tree_model, rock_model,
)


def _vertices(path):
    return np.array([
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ])


@pytest.mark.parametrize("kind", ["tree", "rock"])
@pytest.mark.parametrize("multiplier", [0.6, 1, 1.4])
def test_meshes_are_upright_centered_grounded_and_size_varied(kind, multiplier):
    fitted = fitted_prop(kind, REPO, multiplier)
    vertices = _vertices(fitted.path) * fitted.scale
    transformed = np.column_stack(
        (vertices[:, 0], -vertices[:, 2], vertices[:, 1])
    ) + fitted.translation
    lo, hi = transformed.min(axis=0), transformed.max(axis=0)
    assert lo[2] == pytest.approx(0, abs=1e-9)
    assert hi[2] == pytest.approx(PROP_SPECS[kind].height_m * multiplier)
    assert (lo[:2] + hi[:2]) / 2 == pytest.approx([0, 0], abs=1e-9)
    assert footprint_radius(kind, REPO, multiplier) > np.linalg.norm(hi[:2])


def test_tree_collision_fits_trunk_and_ground_radius_excludes_foliage():
    fitted = fitted_prop("tree", REPO)
    model = tree_model("rescue_tree_001", 1, 2, 3, .4, REPO)
    collision = model.find("link/collision")
    assert collision.get("name") == "trunk_collision"
    assert float(collision.findtext("geometry/cylinder/length")) == pytest.approx(fitted.trunk_height)
    assert fitted.trunk_height < fitted.dimensions[2] / 2
    assert 0 < ground_radius("tree", REPO) < footprint_radius("tree", REPO)
    assert fitted.trunk_center[2] - fitted.trunk_height / 2 == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("builder", [tree_model, rock_model])
def test_props_keep_original_materials_and_ambient_temperature(builder):
    model = builder("prop", -2, 4, 7, math.pi, REPO)
    assert model.findtext("static") == "true"
    assert [float(v) for v in model.findtext("pose").split()] == pytest.approx([-2, 4, 7, 0, 0, math.pi])
    visual = model.find("link/visual")
    assert visual.findtext("cast_shadows") == "true"
    assert visual.find("plugin") is None
    mesh_path = Path(unquote(urlparse(visual.findtext("geometry/mesh/uri")).path))
    assert mesh_path.is_file()
    mtllib = next(line.split(maxsplit=1)[1] for line in mesh_path.read_text().splitlines()
                  if line.startswith("mtllib "))
    assert (mesh_path.parent / mtllib).is_file()


def test_rock_collision_uses_exact_visual_mesh_and_transform():
    model = rock_model("rescue_rock_001", 0, 0, 0, 0, REPO, scale=1.3)
    visual = model.find("link/visual")
    collision = model.find("link/collision")
    assert collision.findtext("pose") == visual.findtext("pose")
    assert collision.findtext("geometry/mesh/uri") == visual.findtext("geometry/mesh/uri")
    assert collision.findtext("geometry/mesh/scale") == visual.findtext("geometry/mesh/scale")


@pytest.mark.parametrize("scale", [0, -1, float("nan"), float("inf"), "bad"])
def test_invalid_scale_is_rejected(scale):
    with pytest.raises(ValueError, match="scale"):
        tree_model("tree", 0, 0, 0, 0, REPO, scale=scale)


def test_invalid_name_coordinates_and_kind_are_rejected():
    with pytest.raises(ValueError, match="nonempty"):
        tree_model("", 0, 0, 0, 0, REPO)
    with pytest.raises(ValueError, match="finite"):
        rock_model("rock", 0, 0, float("inf"), 0, REPO)
    with pytest.raises(ValueError, match="kind"):
        fitted_prop("house", REPO)
