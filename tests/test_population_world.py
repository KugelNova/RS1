"""Population behaviour on a small, real SDF/OBJ fixture."""
from pathlib import Path
import math
import sys
import xml.etree.ElementTree as ET

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from rescue_population.world import create_population_world


@pytest.fixture
def package(tmp_path):
    root = tmp_path / "package"
    (root / "worlds").mkdir(parents=True)
    (root / "models/terrain_new").mkdir(parents=True)
    (root / "Human").symlink_to(REPO / "Human", target_is_directory=True)
    (root / "models/animals").symlink_to(REPO / "models/animals", target_is_directory=True)
    (root / "worlds/terrain_new.sdf").write_text(
        '<sdf version="1.8"><world name="terrain_new"><include>'
        '<uri>model://terrain_new</uri></include></world></sdf>')
    (root / "models/terrain_new/model.sdf").write_text(
        '<sdf version="1.8"><model name="terrain_new"><link name="ground">'
        '<collision name="surface"><geometry><mesh><uri>'
        'model://terrain_new/ground.obj</uri></mesh></geometry></collision>'
        '</link></model></sdf>')
    (root / "models/terrain_new/ground.obj").write_text(
        'v -20 -20 2\nv 20 -20 2\nv 20 20 2\nv -20 20 2\n'
        'usemtl Terrain\nf 1 2 3 4\n')
    return root


def build(package, tmp_path, **kwargs):
    return create_population_world(package, "terrain_new", tmp_path / "populated.sdf", **kwargs)


def test_exact_requested_counts_and_surface_height(package, tmp_path):
    source = (package / "worlds/terrain_new.sdf").read_bytes()
    result = build(package, tmp_path, humans=4, animals=7)
    models = ET.parse(result.path).getroot().find("world").findall("model")
    assert sum(m.get("name").startswith("rescue_human_") for m in models) == 4
    assert sum(m.get("name").startswith("rescue_animal_") for m in models) == 7
    assert len({m.get("name") for m in models}) == 11
    assert len({p["species"] for p in result.placements if p["kind"] == "animal"}) == 6
    for p in result.placements:
        assert 2 <= p["z"] <= 2.05
        assert -20 < p["x"] < 20 and -20 < p["y"] < 20
        assert math.hypot(p["x"], p["y"]) >= p["radius"] + 3
    for i, a in enumerate(result.placements):
        for b in result.placements[i+1:]:
            assert math.hypot(a["x"]-b["x"], a["y"]-b["y"]) >= a["radius"]+b["radius"]+0.5
    assert (package / "worlds/terrain_new.sdf").read_bytes() == source


def test_same_seed_repeats_positions_and_new_seed_changes_them(package, tmp_path):
    a = build(package, tmp_path, humans=2, animals=3, spawn_seed=19)
    b = build(package, tmp_path, humans=2, animals=3, spawn_seed=19)
    c = build(package, tmp_path, humans=2, animals=3, spawn_seed=20)
    assert a.placements == b.placements
    assert a.placements != c.placements


def test_zero_counts_disable_every_target(package, tmp_path):
    result = build(package, tmp_path, humans=0, animals=0)
    assert not result.placements
    assert not ET.parse(result.path).getroot().find("world").findall("model")


def test_single_species_selection(package, tmp_path):
    result = build(package, tmp_path, humans=0, animals=4, species="deer")
    assert {p["species"] for p in result.placements} == {"deer"}


@pytest.mark.parametrize("field,value", [
    ("humans", "-1"), ("animals", "1.5"), ("humans", True),
    ("spawn_seed", "bad"), ("animals", 501), ("species", "dragon")])
def test_invalid_settings_fail_before_writing(package, tmp_path, field, value):
    with pytest.raises(ValueError):
        build(package, tmp_path, **{field: value})
    assert not (tmp_path / "populated.sdf").exists()


def test_no_suitable_space_fails_instead_of_returning_a_partial_population(package, tmp_path):
    with pytest.raises(ValueError, match="Could not fit"):
        build(package, tmp_path, humans=1, animals=0,
              reserved_points=((0, 0, 1000),))
    assert not (tmp_path / "populated.sdf").exists()


def test_refuses_to_overwrite_source_world(package):
    with pytest.raises(ValueError, match="must not replace"):
        create_population_world(package, "terrain_new", package / "worlds/terrain_new.sdf")


def test_world_include_cannot_override_the_sampled_model_pose(package, tmp_path):
    path = package / "worlds/terrain_new.sdf"
    tree = ET.parse(path)
    ET.SubElement(tree.getroot().find("world/include"), "pose").text = "0 0 0 0 0 0"
    tree.write(path)
    with pytest.raises(ValueError, match="include pose"):
        build(package, tmp_path, humans=1, animals=0)


def test_upright_figures_do_not_intersect_the_uphill_ground(package, tmp_path):
    from rescue_population.entities import footprint_radius, fitted_mesh
    radius = footprint_radius("human", package)
    gradient = 0.08 / radius
    obj = package / "models/terrain_new/ground.obj"
    obj.write_text(
        "\n".join("v %s %s %s" % (x, y, 2 + gradient*x)
                  for x, y in [(-20,-20),(20,-20),(20,20),(-20,20)]) +
        "\nusemtl Terrain\nf 1 2 3 4\n")
    result = build(package, tmp_path, humans=3, animals=0)
    width, length, _ = fitted_mesh("human", package).dimensions
    for person in result.placements:
        for dx in [-width/2, width/2]:
            for dy in [-length/2, length/2]:
                wx = person["x"] + dx*math.cos(person["yaw"]) - dy*math.sin(person["yaw"])
                assert person["z"] >= 2 + gradient*wx


def test_excessive_relief_cannot_leave_an_upright_target_partly_buried(package, tmp_path):
    from rescue_population.entities import footprint_radius
    gradient = 0.13 / footprint_radius("human", package)
    obj = package / "models/terrain_new/ground.obj"
    obj.write_text(
        "\n".join("v %s %s %s" % (x, y, 2 + gradient*x)
                  for x, y in [(-20,-20),(20,-20),(20,20),(-20,20)]) +
        "\nusemtl Terrain\nf 1 2 3 4\n")
    with pytest.raises(ValueError, match="Could not fit"):
        build(package, tmp_path, humans=1, animals=0)
