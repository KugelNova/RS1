"""Physical placement checks for the supplied meshes and generated models."""
from pathlib import Path
import math
import sys
from urllib.parse import unquote, urlparse
import xml.etree.ElementTree as ET

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from rescue_population.entities import (  # noqa: E402
    ANIMAL_SPECS, HUMAN_SPEC, SUPPORTED_ANIMALS,
    animal_model, fitted_mesh, footprint_radius, human_model,
)


def _vertices(path):
    return np.array([
        [float(value) for value in line.split()[1:4]]
        for line in path.read_text().splitlines()
        if line.startswith("v ")
    ])


@pytest.mark.parametrize("kind,species", [("human", None)] +
                         [("animal", species) for species in SUPPORTED_ANIMALS])
def test_supplied_meshes_are_upright_centered_and_grounded(kind, species):
    fitted = fitted_mesh(kind, REPO, species)
    vertices = _vertices(fitted.path) * fitted.scale
    rotated = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1]))
    transformed = rotated + np.array(fitted.translation)
    lo, hi = transformed.min(axis=0), transformed.max(axis=0)
    expected_height = HUMAN_SPEC.height_m if kind == "human" else ANIMAL_SPECS[species].height_m
    assert lo[2] == pytest.approx(0, abs=1e-10)
    assert hi[2] == pytest.approx(expected_height)
    assert (lo[:2] + hi[:2]) / 2 == pytest.approx([0, 0], abs=1e-10)
    assert footprint_radius(kind, REPO, species) > np.linalg.norm(hi[:2])


@pytest.mark.parametrize("species", SUPPORTED_ANIMALS)
def test_animals_keep_material_library_and_have_warm_visuals(species):
    model = animal_model("animal_001", 4, -3, 2.5, 0.7, REPO, species)
    assert model.get("name") == "animal_001"
    assert model.findtext("static") == "true"
    assert [float(v) for v in model.findtext("pose").split()] == pytest.approx([4, -3, 2.5, 0, 0, 0.7])
    visual = model.find("link/visual")
    path = Path(unquote(urlparse(visual.findtext("geometry/mesh/uri")).path))
    mtllib = next(line.split(maxsplit=1)[1] for line in path.read_text().splitlines()
                  if line.startswith("mtllib "))
    assert (path.parent / mtllib).is_file()
    temperature = float(visual.findtext("plugin/temperature"))
    assert 308.15 <= temperature <= 315.15
    box_size = [float(v) for v in model.findtext("link/collision/geometry/box/size").split()]
    assert box_size == pytest.approx(fitted_mesh("animal", REPO, species).dimensions)
    collision_z = float(model.findtext("link/collision/pose").split()[2])
    assert collision_z == pytest.approx(box_size[2] / 2)


def test_human_is_a_warm_full_height_person():
    model = human_model("human_001", 0, 0, 4, math.pi, REPO)
    assert model.findtext("link/visual/plugin/temperature") == "310.15"
    assert float(model.findtext("link/collision/geometry/box/size").split()[2]) == 1.8
    assert model.findtext("link/visual/geometry/mesh/uri").endswith("/Human/Person.obj")
    # The generated element can be embedded in an SDFormat world without XML escaping errors.
    assert ET.fromstring(ET.tostring(model)).get("name") == "human_001"


def test_invalid_species_and_positions_are_rejected():
    with pytest.raises(ValueError, match="Unknown animal"):
        animal_model("animal", 0, 0, 0, 0, REPO, "dragon")
    with pytest.raises(ValueError, match="finite"):
        human_model("person", float("nan"), 0, 0, 0, REPO)
    with pytest.raises(ValueError, match="nonempty"):
        human_model("", 0, 0, 0, 0, REPO)
