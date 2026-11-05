import hashlib
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from rescue_population.generated_ground import generate_ground


def source_asset(tmp_path, count=21):
    package = tmp_path / "package"
    asset = package / "models" / "ground"
    asset.mkdir(parents=True)
    lines = ["mtllib terrain.mtl", "o BareGround"]
    for z in np.linspace(-4, 4, count):
        for x in np.linspace(-2, 2, count):
            lines.append("v %g %g %g" % (x, 1 + x * 0.3, z))
    for z in np.linspace(0, 1, count):
        for x in np.linspace(0, 1, count):
            lines.append("vt %.8f %.8f" % (x, z))
    lines.extend(["vn 0 1 0", "usemtl Terrain"])
    for row in range(count - 1):
        for col in range(count - 1):
            a = row * count + col + 1
            corners = [a, a + 1, a + count + 1, a + count]
            lines.append("f " + " ".join("%d/%d/1" % (i, i) for i in corners))
    (asset / "terrain.obj").write_text("\n".join(lines) + "\n")
    (asset / "terrain.mtl").write_text("newmtl Terrain\nKd 0.1 0.3 0.1\nmap_Kd grass.png\n")
    (asset / "grass.png").write_bytes(b"texture fixture")
    return package


def points_and_normals(path):
    vertices, normals, uv = [], [], []
    for line in path.read_text().splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "vn":
            normals.append([float(v) for v in fields[1:4]])
        elif fields[0] == "vt":
            uv.append(line)
    return np.asarray(vertices), np.asarray(normals), uv


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_seed_reproduces_geometry_and_other_seed_changes_hills(tmp_path):
    package = source_asset(tmp_path)
    first = generate_ground(package, tmp_path / "first", seed=42)
    repeat = generate_ground(package, tmp_path / "repeat", seed=42)
    changed = generate_ground(package, tmp_path / "changed", seed=43)
    assert first.mesh_path.read_bytes() == repeat.mesh_path.read_bytes()
    assert first.mesh_path.read_bytes() != changed.mesh_path.read_bytes()
    assert first.metadata == repeat.metadata


def test_map_size_and_height_bounds_are_physical_metres(tmp_path):
    package = source_asset(tmp_path)
    ground = generate_ground(package, tmp_path / "generated", seed=5,
                             map_size=120, terrain_height=12)
    vertices, normals, _ = points_and_normals(ground.mesh_path)
    assert ground.bounds == pytest.approx((-60, 60, -60, 60))
    assert vertices[:, 2].min() >= -6 - 1e-8
    assert vertices[:, 2].max() <= 6 + 1e-8
    assert np.ptp(vertices[:, 2]) > 5
    assert np.all(normals[:, 2] > 0)
    assert np.linalg.norm(normals, axis=1) == pytest.approx(np.ones(len(normals)))


def test_zero_relief_is_flat_with_clear_supported_footprints(tmp_path):
    package = source_asset(tmp_path)
    ground = generate_ground(package, tmp_path / "flat", seed=9, map_size=80,
                             terrain_height=0, terrain_roughness=1)
    vertices, normals, _ = points_and_normals(ground.mesh_path)
    assert np.all(vertices[:, 2] == 0)
    assert normals[:, 2] == pytest.approx(np.ones(len(normals)))
    assert ground.surface.sample(15, -10).z == 0
    assert ground.surface.is_clear(15, -10, radius=1, max_slope_deg=1)
    assert not ground.surface.is_clear(39.5, 0, radius=1)


def test_robot_spawn_pads_have_zero_height_and_flat_support(tmp_path):
    package = source_asset(tmp_path, count=31)
    ground = generate_ground(package, tmp_path / "pads", seed=12,
                             map_size=120, terrain_height=15)
    for x, y in ((0, 0), (-10, 7)):
        point = ground.surface.sample(x, y)
        assert point.z == pytest.approx(0, abs=1e-10)
        assert point.slope_deg == pytest.approx(0, abs=1e-10)
        assert ground.surface.is_clear(x, y, radius=0.5, max_slope_deg=1)


def test_visual_collision_match_and_original_uv_materials_remain_intact(tmp_path):
    package = source_asset(tmp_path)
    asset = package / "models" / "ground"
    originals = {p.name: digest(p) for p in asset.iterdir()}
    ground = generate_ground(package, tmp_path / "generated", seed=10)
    uri_visual = ground.model.findtext("link/visual/geometry/mesh/uri")
    uri_collision = ground.model.findtext("link/collision/geometry/mesh/uri")
    assert uri_visual == uri_collision == ground.mesh_path.as_uri()
    assert ground.model.findtext("static") == "true"
    assert {p.name: digest(p) for p in asset.iterdir()} == originals
    _, _, original_uv = points_and_normals(asset / "terrain.obj")
    _, generated_normals, generated_uv = points_and_normals(ground.mesh_path)
    assert generated_uv == original_uv
    assert len(generated_normals) == ground.metadata["triangle_count"]
    assert (ground.mesh_path.parent / "generated_ground.mtl").read_bytes() == (
        asset / "terrain.mtl").read_bytes()
    assert (ground.mesh_path.parent / "grass.png").read_bytes() == b"texture fixture"


def test_roughness_adds_shorter_wavelength_detail(tmp_path):
    package = source_asset(tmp_path, count=41)
    smooth = generate_ground(package, tmp_path / "smooth", seed=8,
                             terrain_roughness=0)
    rough = generate_ground(package, tmp_path / "rough", seed=8,
                            terrain_roughness=1)
    smooth_z = points_and_normals(smooth.mesh_path)[0][:, 2].reshape(41, 41)
    rough_z = points_and_normals(rough.mesh_path)[0][:, 2].reshape(41, 41)
    # Compare curvature outside protected spawn pads.
    smooth_curvature = np.abs(np.diff(smooth_z, n=2, axis=1)).mean()
    rough_curvature = np.abs(np.diff(rough_z, n=2, axis=1)).mean()
    assert rough_curvature > smooth_curvature


@pytest.mark.parametrize("changes", [
    {"map_size": 0}, {"map_size": float("inf")}, {"terrain_height": -1},
    {"terrain_roughness": 1.1}, {"terrain_roughness": float("nan")},
    {"seed": -1}, {"seed": 1.5}, {"seed": True},
])
def test_invalid_controls_fail_before_writing_geometry(tmp_path, changes):
    package = source_asset(tmp_path)
    options = dict(seed=42)
    options.update(changes)
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        generate_ground(package, output, **options)
    assert not output.exists()


def test_cannot_write_generated_geometry_over_original_assets(tmp_path):
    package = source_asset(tmp_path)
    with pytest.raises(ValueError, match="original ground assets"):
        generate_ground(package, package / "models" / "ground", seed=1)
