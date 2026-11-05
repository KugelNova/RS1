"""Grounded, stationary tree and rock models using the supplied local assets.

The OBJ files are Y-up and have modelling-origin offsets. A size multiplier is
applied around a nominal 7 m tree or 0.75 m rock, then each mesh is centred and
its lowest point is placed at local z=0. Source meshes and materials are never
rewritten by this module.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Optional, Tuple
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class PropSpec:
    asset: str
    height_m: float


PROP_SPECS = {
    "tree": PropSpec("models/tree/tree.obj", 7.0),
    "rock": PropSpec("models/rock/rock.obj", 0.75),
}
SUPPORTED_PROPS = tuple(PROP_SPECS)


@dataclass(frozen=True)
class FittedProp:
    path: Path
    # Uniform scale applied to the original OBJ coordinates.
    scale: float
    translation: Tuple[float, float, float]
    # Z-up dimensions: horizontal width, horizontal length, height.
    dimensions: Tuple[float, float, float]
    trunk_center: Optional[Tuple[float, float, float]] = None
    trunk_radius: Optional[float] = None
    trunk_height: Optional[float] = None


@lru_cache(maxsize=16)
def _mesh_data(asset_path):
    vertices = []
    material_vertices = {}
    material = ""
    with Path(asset_path).open(encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "v":
                if len(fields) < 4:
                    raise ValueError("Invalid OBJ vertex in %s." % asset_path)
                vertex = tuple(float(value) for value in fields[1:4])
                if not all(math.isfinite(value) for value in vertex):
                    raise ValueError("Non-finite OBJ vertex in %s." % asset_path)
                vertices.append(vertex)
            elif fields[0] == "usemtl":
                material = " ".join(fields[1:])
            elif fields[0] == "f":
                group = material_vertices.setdefault(material, set())
                for token in fields[1:]:
                    index = int(token.split("/")[0])
                    index = index - 1 if index > 0 else len(vertices) + index
                    if not 0 <= index < len(vertices):
                        raise ValueError("OBJ face has an invalid vertex index in %s." % asset_path)
                    group.add(index)
    if not vertices:
        raise ValueError("OBJ contains no vertices: %s." % asset_path)
    return tuple(vertices), {
        name: tuple(vertices[index] for index in sorted(indices))
        for name, indices in material_vertices.items() if indices
    }


def _bounds(vertices):
    minimum = tuple(min(vertex[axis] for vertex in vertices) for axis in range(3))
    maximum = tuple(max(vertex[axis] for vertex in vertices) for axis in range(3))
    return minimum, maximum


def fitted_prop(kind, package_dir, scale=1.0):
    """Fit a prop, using scale as a multiplier of its nominal real-world size."""
    try:
        spec = PROP_SPECS[kind]
    except KeyError as exc:
        raise ValueError("Prop kind must be 'tree' or 'rock'.") from exc
    try:
        multiplier = float(scale)
    except (TypeError, ValueError) as exc:
        raise ValueError("Prop scale must be finite and greater than zero.") from exc
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Prop scale must be finite and greater than zero.")
    height = spec.height_m * multiplier
    if not math.isfinite(height):
        raise ValueError("Prop scale is too large.")

    path = (Path(package_dir) / spec.asset).resolve()
    vertices, groups = _mesh_data(str(path))
    lo, hi = _bounds(vertices)
    if hi[1] <= lo[1]:
        raise ValueError("OBJ must have positive Y-up height: %s." % path)
    factor = height / (hi[1] - lo[1])
    translation = (
        -factor * (lo[0] + hi[0]) / 2.0,
        factor * (lo[2] + hi[2]) / 2.0,
        -factor * lo[1],
    )
    dimensions = (factor * (hi[0] - lo[0]), factor * (hi[2] - lo[2]), height)
    if not all(math.isfinite(value) and value > 0 for value in dimensions):
        raise ValueError("OBJ dimensions and prop scale must produce finite positive sizes.")

    center = radius = trunk_height = None
    if kind == "tree":
        if not groups:
            raise ValueError("Tree OBJ needs material groups to identify its trunk.")
        # The provided brown trunk is the material group touching the lowest Y.
        # This also remains valid if its material is renamed during re-export.
        trunk = min(groups.values(), key=lambda group: min(v[1] for v in group))
        tlo, thi = _bounds(trunk)
        raw_center_x = (tlo[0] + thi[0]) / 2.0
        raw_center_z = (tlo[2] + thi[2]) / 2.0
        center = (
            factor * raw_center_x + translation[0],
            -factor * raw_center_z + translation[1],
            factor * (tlo[1] + thi[1]) / 2.0 + translation[2],
        )
        radius = factor * max(
            math.hypot(v[0] - raw_center_x, v[2] - raw_center_z) for v in trunk
        )
        trunk_height = factor * (thi[1] - tlo[1])
        if radius <= 0 or trunk_height <= 0:
            raise ValueError("Tree trunk must have positive radius and height.")
    return FittedProp(path, factor, translation, dimensions, center, radius, trunk_height)


def footprint_radius(kind, package_dir, scale=1.0):
    """Full visual radius, including tree foliage, for placement avoidance."""
    width, length, _ = fitted_prop(kind, package_dir, scale).dimensions
    return math.hypot(width, length) / 2.0 + 0.1


def ground_radius(kind, package_dir, scale=1.0):
    """Ground-contact radius around the model origin, excluding tree foliage."""
    fitted = fitted_prop(kind, package_dir, scale)
    if kind == "tree":
        return math.hypot(*fitted.trunk_center[:2]) + fitted.trunk_radius + 0.1
    return math.hypot(*fitted.dimensions[:2]) / 2.0 + 0.1


def _vector(values):
    return " ".join(format(float(value), ".12g") for value in values)


def _mesh_geometry(parent, fitted):
    geometry = ET.SubElement(parent, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "uri").text = fitted.path.as_uri()
    ET.SubElement(mesh, "scale").text = _vector((fitted.scale,) * 3)


def _model(kind, name, x, y, z, yaw, package_dir, scale):
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every spawned prop needs a nonempty name.")
    try:
        coordinates = tuple(float(value) for value in (x, y, z, yaw))
    except (TypeError, ValueError) as exc:
        raise ValueError("Prop positions and yaw must be finite.") from exc
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError("Prop positions and yaw must be finite.")
    x, y, z, yaw = coordinates
    fitted = fitted_prop(kind, package_dir, scale)
    model = ET.Element("model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = _vector((x, y, z, 0, 0, yaw))
    link = ET.SubElement(model, "link", name="body")
    visual = ET.SubElement(link, "visual", name=kind + "_visual")
    mesh_pose = (*fitted.translation, math.pi / 2.0, 0, 0)
    ET.SubElement(visual, "pose").text = _vector(mesh_pose)
    ET.SubElement(visual, "cast_shadows").text = "true"
    _mesh_geometry(visual, fitted)
    # No temperature plugin: props remain at Gazebo's ambient temperature.
    if kind == "tree":
        collision = ET.SubElement(link, "collision", name="trunk_collision")
        ET.SubElement(collision, "pose").text = _vector(
            (*fitted.trunk_center, 0, 0, 0)
        )
        geometry = ET.SubElement(collision, "geometry")
        cylinder = ET.SubElement(geometry, "cylinder")
        ET.SubElement(cylinder, "radius").text = _vector((fitted.trunk_radius,))
        ET.SubElement(cylinder, "length").text = _vector((fitted.trunk_height,))
    else:
        collision = ET.SubElement(link, "collision", name="rock_collision")
        ET.SubElement(collision, "pose").text = _vector(mesh_pose)
        _mesh_geometry(collision, fitted)
    return model


def tree_model(name, x, y, z, yaw, package_dir, scale=1.0):
    """Create a stationary tree; nominal height is 7 m and scale varies its size."""
    return _model("tree", name, x, y, z, yaw, package_dir, scale)


def rock_model(name, x, y, z, yaw, package_dir, scale=1.0):
    """Create an ambient-temperature rock with its actual mesh as collision."""
    return _model("rock", name, x, y, z, yaw, package_dir, scale)
