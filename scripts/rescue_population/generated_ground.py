"""Reproducible, Z-up ground geometry generated from the supplied bare terrain OBJ."""

from dataclasses import dataclass
import math
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

import numpy as np

from .terrain import TerrainSurface, _triangulate


@dataclass(frozen=True)
class GeneratedGround:
    model: ET.Element
    surface: TerrainSurface
    mesh_path: Path
    metadata: dict

    @property
    def bounds(self):
        return self.surface.bounds


def _number(value, name, *, minimum=0.0, positive=False, maximum=None):
    if isinstance(value, bool):
        raise ValueError("%s must be a finite number." % name)
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a finite number." % name) from None
    if (not math.isfinite(result) or result < minimum or
            (positive and result <= minimum) or
            (maximum is not None and result > maximum)):
        relation = "greater than" if positive else "at least"
        message = "%s must be finite and %s %g" % (name, relation, minimum)
        if maximum is not None:
            message += " and at most %g" % maximum
        raise ValueError(message + ".")
    return result


def _seed(value):
    if isinstance(value, bool):
        raise ValueError("spawn_seed must be a nonnegative whole number.")
    text = str(value).strip()
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError("spawn_seed must be a nonnegative whole number.")
    return int(text)


def _read_source(path):
    """Keep original UV corners and material names while triangulating polygons."""
    vertices, texture_lines, triangles = [], [], []
    material = None
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                point = [float(value) for value in fields[1:4]]
                if len(point) != 3 or not np.isfinite(point).all():
                    raise ValueError("Invalid ground vertex at OBJ line %d." % line_number)
                vertices.append(point)
            elif fields[0] == "vt":
                texture_lines.append(line.rstrip("\r\n"))
            elif fields[0] == "usemtl":
                material = " ".join(fields[1:])
            elif fields[0] == "f":
                if material is None or material.lower() != "terrain":
                    raise ValueError("The bare ground OBJ must use only the Terrain material.")
                corners = []
                for part in fields[1:]:
                    indices = part.split("/")
                    vertex = int(indices[0])
                    vertex = vertex - 1 if vertex > 0 else len(vertices) + vertex
                    if vertex < 0 or vertex >= len(vertices):
                        raise ValueError("Invalid ground vertex index at line %d." % line_number)
                    uv = None
                    if len(indices) > 1 and indices[1]:
                        uv = int(indices[1])
                        uv = uv - 1 if uv > 0 else len(texture_lines) + uv
                        if uv < 0 or uv >= len(texture_lines):
                            raise ValueError("Invalid ground UV index at line %d." % line_number)
                    corners.append((vertex, uv))
                if len(corners) < 3:
                    raise ValueError("Invalid ground face at line %d." % line_number)
                by_vertex = {corner[0]: corner for corner in corners}
                for triangle in _triangulate(vertices, [corner[0] for corner in corners]):
                    triangles.append((tuple(by_vertex[index] for index in triangle), material))
    if not vertices or not triangles:
        raise ValueError("The supplied ground OBJ contains no terrain faces.")
    return np.asarray(vertices, dtype=float), texture_lines, triangles


def _normalise(values):
    low, high = float(values.min()), float(values.max())
    if high - low < 1e-12:
        return np.zeros_like(values)
    return (values - (high + low) * 0.5) / (high - low)


def _height_field(xy, *, seed, map_size, terrain_height, terrain_roughness):
    if terrain_height == 0:
        return np.zeros(len(xy))
    rng = np.random.default_rng(seed)
    coordinates = xy / map_size
    broad, detail = np.zeros(len(xy)), np.zeros(len(xy))
    for target, count, frequency_range in [
            (broad, 8, (0.65, 2.2)), (detail, 12, (4.0, 9.0))]:
        for _ in range(count):
            angle = rng.uniform(0, 2 * math.pi)
            frequency = rng.uniform(*frequency_range)
            direction = np.asarray((math.cos(angle), math.sin(angle)))
            phase = rng.uniform(0, 2 * math.pi)
            amplitude = rng.uniform(0.7, 1.3)
            target += amplitude * np.sin(
                2 * math.pi * frequency * (coordinates[:, 0] * direction[0] + coordinates[:, 1] * direction[1]) + phase)
    # Convexly mix two bounded fields: all elevations remain inside +/- relief/2.
    detail_weight = 0.35 * terrain_roughness
    return terrain_height * ((1 - detail_weight) * _normalise(broad) +
                             detail_weight * _normalise(detail))


def _flatten_spawns(vertices, triangle_indices, map_size, spawn_points):
    """Flatten each containing triangle and blend out smoothly around its pad."""
    xy = vertices[:, :2]
    triangles = xy[triangle_indices]
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    edge, other = b - a, c - a
    determinant = edge[:, 0] * other[:, 1] - edge[:, 1] * other[:, 0]
    valid = np.abs(determinant) > 1e-12
    pads = []
    for x, y in spawn_points:
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("Ground spawn points must have finite XY coordinates.")
        point = np.asarray((x, y))
        delta = point - a
        u, v = np.zeros(len(triangles)), np.zeros(len(triangles))
        u[valid] = (delta[valid, 0] * other[valid, 1] -
                    delta[valid, 1] * other[valid, 0]) / determinant[valid]
        v[valid] = (edge[valid, 0] * delta[valid, 1] -
                    edge[valid, 1] * delta[valid, 0]) / determinant[valid]
        inside = valid & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        if not inside.any():
            continue
        # Cover the actual triangle's vertices, including coarser mesh resolutions.
        required_radius = float(np.linalg.norm(triangles[inside] - point, axis=2).max())
        inner = max(2.5, required_radius * 1.05)
        outer = inner + max(3.0, map_size * 0.035)
        distance = np.linalg.norm(xy - point, axis=1)
        blend = np.clip((distance - inner) / (outer - inner), 0, 1)
        blend = blend * blend * blend * (blend * (6 * blend - 15) + 10)
        vertices[:, 2] *= blend
        pads.append(dict(x=float(x), y=float(y), inner_radius_m=inner,
                         blend_radius_m=outer))
    return pads


def _copy_material(source_dir, destination):
    source = source_dir / "terrain.mtl"
    if not source.is_file():
        raise ValueError("The supplied ground mesh is missing terrain.mtl.")
    shutil.copyfile(source, destination / "generated_ground.mtl")
    # Preserve texture resources if they accompany a later textured ground asset.
    # Copying their relative directory layout leaves the original MTL unchanged.
    for item in source_dir.iterdir():
        if item == source or item.suffix.lower() in (".obj", ".sdf", ".config", ".md"):
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file():
            shutil.copyfile(item, target)


def _write_obj(path, vertices, texture_lines, faces):
    triangles = []
    corners_and_materials = []
    for corners, material in faces:
        corners = list(corners)
        triangle = vertices[[corner[0] for corner in corners]]
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        magnitude = float(np.linalg.norm(normal))
        if magnitude < 1e-12:
            continue
        if normal[2] < 0:
            corners[1], corners[2] = corners[2], corners[1]
            triangle = triangle[[0, 2, 1]]
            normal = -normal
        triangles.append(triangle)
        corners_and_materials.append((corners, material, normal / magnitude))
    if not triangles:
        raise ValueError("The supplied ground mesh has no nondegenerate terrain faces.")
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write("# Seeded ground derived from the original bare terrain asset.\n")
        stream.write("mtllib generated_ground.mtl\no generated_ground\n")
        for vertex in vertices:
            stream.write("v %.9f %.9f %.9f\n" % tuple(vertex))
        for line in texture_lines:
            stream.write(line + "\n")
        for _, _, normal in corners_and_materials:
            stream.write("vn %.9f %.9f %.9f\n" % tuple(normal))
        stream.write("s off\n")
        previous_material = None
        for index, (corners, material, _) in enumerate(corners_and_materials, 1):
            if material != previous_material:
                stream.write("usemtl " + material + "\n")
                previous_material = material
            tokens = [("%d/%d/%d" % (vertex + 1, uv + 1, index) if uv is not None
                       else "%d//%d" % (vertex + 1, index))
                      for vertex, uv in corners]
            stream.write("f " + " ".join(tokens) + "\n")
    return np.asarray(triangles)


def generate_ground(package_dir, output_dir, *, seed, map_size=150.0,
                    terrain_height=8.0, terrain_roughness=0.5,
                    spawn_points=((0.0, 0.0), (-10.0, 7.0))):
    """Generate a ground OBJ, matching static SDF model and placement surface.

    map_size is the map width and length in metres. terrain_height bounds the
    peak-to-valley relief; zero produces flat ground. terrain_roughness in [0, 1]
    blends broad hills with finer detail. The original asset is always read-only.
    """
    seed = _seed(seed)
    map_size = _number(map_size, "map_size", positive=True)
    terrain_height = _number(terrain_height, "terrain_height")
    terrain_roughness = _number(terrain_roughness, "terrain_roughness", maximum=1.0)
    source_dir = (Path(package_dir) / "models" / "ground").resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(source_dir):
        raise ValueError("Generated ground must not replace the original ground assets.")
    raw, uv_lines, faces = _read_source(source_dir / "terrain.obj")
    low, high = raw.min(axis=0), raw.max(axis=0)
    if high[0] - low[0] < 1e-10 or high[2] - low[2] < 1e-10:
        raise ValueError("The bare terrain OBJ must span both horizontal X/Z axes.")
    vertices = np.empty_like(raw)
    vertices[:, 0] = (raw[:, 0] - (low[0] + high[0]) * 0.5) * map_size / (high[0] - low[0])
    vertices[:, 1] = -(raw[:, 2] - (low[2] + high[2]) * 0.5) * map_size / (high[2] - low[2])
    vertices[:, 2] = _height_field(
        vertices[:, :2], seed=seed, map_size=map_size,
        terrain_height=terrain_height, terrain_roughness=terrain_roughness)
    indices = np.asarray([[corner[0] for corner in face] for face, _ in faces])
    pads = _flatten_spawns(vertices, indices, map_size, spawn_points)
    output_dir.mkdir(parents=True, exist_ok=True)
    _copy_material(source_dir, output_dir)
    mesh_path = output_dir / "generated_ground.obj"
    triangles = _write_obj(mesh_path, vertices, uv_lines, faces)
    surface = TerrainSurface(triangles)
    model = ET.Element("model", name="generated_ground")
    ET.SubElement(model, "static").text = "true"
    link = ET.SubElement(model, "link", name="ground")
    for tag in ("visual", "collision"):
        item = ET.SubElement(link, tag, name="ground_" + tag)
        geometry = ET.SubElement(item, "geometry")
        mesh = ET.SubElement(geometry, "mesh")
        ET.SubElement(mesh, "uri").text = mesh_path.as_uri()
        ET.SubElement(mesh, "scale").text = "1 1 1"
    metadata = dict(seed=seed, map_size_m=map_size, terrain_height_m=terrain_height,
                    terrain_roughness=terrain_roughness,
                    min_height_m=float(vertices[:, 2].min()),
                    max_height_m=float(vertices[:, 2].max()),
                    vertex_count=len(vertices), triangle_count=len(triangles),
                    spawn_pads=pads)
    return GeneratedGround(model, surface, mesh_path, metadata)
