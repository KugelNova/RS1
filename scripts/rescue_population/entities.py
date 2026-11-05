"""Local human and animal meshes packaged as warm, stationary Gazebo models.

The supplied OBJ files use Y-up coordinates and several have nonzero modelling
origins. Their visual poses place the feet at the requested ground height while
leaving the original assets and material libraries untouched.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
from typing import Optional, Tuple
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class EntitySpec:
    asset: str
    height_m: float
    temperature_k: float


HUMAN_SPEC = EntitySpec("Human/Person.obj", 1.8, 310.15)
ANIMAL_SPECS = {
    "bear": EntitySpec("models/animals/bear/bear.obj", 1.4, 310.15),
    "deer": EntitySpec("models/animals/deer/deermodel.obj", 1.65, 313.15),
    "bird": EntitySpec("models/animals/bird/birdinunder10mins.obj", 0.25, 313.15),
    "mouse": EntitySpec("models/animals/mouse/mouse.obj", 0.08, 310.15),
    "kangaroo": EntitySpec("models/animals/kangaroo/kangaroo.obj", 1.6, 309.15),
    "horse": EntitySpec("models/animals/horse/horse.obj", 2.0, 310.65),
}
SUPPORTED_ANIMALS = tuple(ANIMAL_SPECS)


@dataclass(frozen=True)
class FittedMesh:
    path: Path
    scale: float
    translation: Tuple[float, float, float]
    # Dimensions after rotating the original Y-up mesh into Z-up coordinates.
    dimensions: Tuple[float, float, float]


def _spec(kind: str, species: Optional[str] = None) -> EntitySpec:
    if kind == "human":
        return HUMAN_SPEC
    if kind != "animal":
        raise ValueError("Entity kind must be 'human' or 'animal'.")
    species = "deer" if species is None else species
    try:
        return ANIMAL_SPECS[species]
    except KeyError as exc:
        raise ValueError(
            "Unknown animal species %r; choose from %s."
            % (species, ", ".join(SUPPORTED_ANIMALS))
        ) from exc


@lru_cache(maxsize=32)
def _fit_mesh(asset_path: str, height_m: float) -> FittedMesh:
    path = Path(asset_path)
    minimum = [math.inf, math.inf, math.inf]
    maximum = [-math.inf, -math.inf, -math.inf]
    count = 0
    with path.open(encoding="utf-8") as source:
        for line in source:
            fields = line.split()
            if not fields or fields[0] != "v":
                continue
            if len(fields) < 4:
                raise ValueError("Invalid OBJ vertex in %s." % path)
            vertex = tuple(float(value) for value in fields[1:4])
            if not all(math.isfinite(value) for value in vertex):
                raise ValueError("Non-finite OBJ vertex in %s." % path)
            for axis, value in enumerate(vertex):
                minimum[axis] = min(minimum[axis], value)
                maximum[axis] = max(maximum[axis], value)
            count += 1
    if not count or maximum[1] <= minimum[1]:
        raise ValueError("OBJ must contain vertices with a positive Y-up height: %s." % path)

    scale = height_m / (maximum[1] - minimum[1])
    center_x = (minimum[0] + maximum[0]) / 2.0
    center_z = (minimum[2] + maximum[2]) / 2.0
    # Rx(+pi/2): (x, y, z) -> (x, -z, y).
    translation = (-scale * center_x, scale * center_z, -scale * minimum[1])
    dimensions = (
        scale * (maximum[0] - minimum[0]),
        scale * (maximum[2] - minimum[2]),
        height_m,
    )
    return FittedMesh(path, scale, translation, dimensions)


def fitted_mesh(kind: str, package_dir, species: Optional[str] = None) -> FittedMesh:
    """Return the fitted mesh; package_dir is the package source or share folder."""
    spec = _spec(kind, species)
    return _fit_mesh(str((Path(package_dir) / spec.asset).resolve()), spec.height_m)


def footprint_radius(kind: str, package_dir, species: Optional[str] = None) -> float:
    """Conservative horizontal radius for placement and inter-entity clearance."""
    width, length, _ = fitted_mesh(kind, package_dir, species).dimensions
    return math.hypot(width, length) / 2.0 + 0.15


def _number(value: float) -> str:
    return format(float(value), ".12g")


def _vector(values) -> str:
    return " ".join(_number(value) for value in values)


def _model(name: str, x: float, y: float, z: float, yaw: float,
           package_dir, kind: str, species: Optional[str] = None) -> ET.Element:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Every spawned model needs a nonempty name.")
    if not all(math.isfinite(float(value)) for value in (x, y, z, yaw)):
        raise ValueError("Spawn positions and yaw must be finite.")
    spec = _spec(kind, species)
    fitted = fitted_mesh(kind, package_dir, species)

    model = ET.Element("model", name=name)
    ET.SubElement(model, "static").text = "true"
    ET.SubElement(model, "pose").text = _vector((x, y, z, 0, 0, yaw))
    link = ET.SubElement(model, "link", name="body")
    visual = ET.SubElement(link, "visual", name="%s_visual" % (species or kind))
    ET.SubElement(visual, "pose").text = _vector(
        (*fitted.translation, math.pi / 2.0, 0, 0)
    )
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    ET.SubElement(mesh, "uri").text = fitted.path.as_uri()
    ET.SubElement(mesh, "scale").text = _vector((fitted.scale,) * 3)
    thermal = ET.SubElement(
        visual, "plugin",
        filename="ignition-gazebo-thermal-system",
        name="ignition::gazebo::systems::Thermal",
    )
    ET.SubElement(thermal, "temperature").text = _number(spec.temperature_k)

    collision = ET.SubElement(link, "collision", name="body_collision")
    ET.SubElement(collision, "pose").text = _vector(
        (0, 0, fitted.dimensions[2] / 2.0, 0, 0, 0)
    )
    geometry = ET.SubElement(collision, "geometry")
    box = ET.SubElement(geometry, "box")
    ET.SubElement(box, "size").text = _vector(fitted.dimensions)
    return model


def human_model(name, x, y, z, yaw, package_dir) -> ET.Element:
    """Create a 1.8 m tall, 37 C person with feet at (x, y, z)."""
    return _model(name, x, y, z, yaw, package_dir, "human")


def animal_model(name, x, y, z, yaw, package_dir, species="deer") -> ET.Element:
    """Create a warm animal with feet at (x, y, z); animals remain stationary."""
    return _model(name, x, y, z, yaw, package_dir, "animal", species)
