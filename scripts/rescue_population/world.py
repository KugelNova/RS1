"""Build a populated copy of the terrain world before Gazebo starts."""
from dataclasses import dataclass
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from .entities import SUPPORTED_ANIMALS, animal_model, footprint_radius, human_model
from .terrain import TerrainSurface

MAX_POPULATION = 500


@dataclass(frozen=True)
class PopulationResult:
    path: Path
    humans: int
    animals: int
    seed: int
    placements: tuple


def nonnegative_integer(value, name):
    """Validate launch inputs without accepting fractional counts."""
    if isinstance(value, bool):
        raise ValueError("%s must be a nonnegative whole number." % name)
    text = str(value).strip()
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError("%s must be a nonnegative whole number." % name)
    return int(text)


def animal_species(value):
    if isinstance(value, str):
        values = [part.strip().lower() for part in value.split(",")]
    else:
        values = list(value)
    if not values or any(not part or part not in SUPPORTED_ANIMALS for part in values):
        raise ValueError("animal_species must list one or more of: %s."
                         % ", ".join(SUPPORTED_ANIMALS))
    return tuple(dict.fromkeys(values))


def create_population_world(package_dir, world_name, output_path, *,
                            humans=3, animals=5, spawn_seed=42,
                            species=SUPPORTED_ANIMALS,
                            reserved_points=((0.0, 0.0, 3.0), (-10.0, 7.0, 3.0))):
    """Place the requested population on clear terrain, or fail before launch.

    The original SDF and all mesh assets are read-only. Coordinates and radii are
    in world metres; the terrain model's current pose and scale are honoured.
    """
    humans = nonnegative_integer(humans, "humans")
    animals = nonnegative_integer(animals, "animals")
    seed = nonnegative_integer(spawn_seed, "spawn_seed")
    if humans + animals > MAX_POPULATION:
        raise ValueError("Choose at most %d humans and animals combined." % MAX_POPULATION)
    selected_species = animal_species(species)
    package_dir = Path(package_dir)
    if world_name not in ("simple_trees", "large_demo", "rescue_forest", "terrain_new"):
        raise ValueError("Unknown simulation world: %s." % world_name)
    if humans + animals and world_name != "terrain_new":
        raise ValueError("Adjustable population requires world:=terrain_new; "
                         "use humans:=0 animals:=0 for the other demo worlds.")
    source_path = package_dir / "worlds" / (world_name + ".sdf")
    output_path = Path(output_path)
    if output_path.resolve() == source_path.resolve():
        raise ValueError("The generated population world must not replace the source world.")
    tree = ET.parse(source_path)
    world = tree.getroot().find("world")
    if world is None or world.get("name") != world_name:
        raise ValueError("The SDF world name must match the selected world.")
    if any((m.get("name") or "").startswith(("rescue_human_", "rescue_animal_"))
           for m in world.findall("model")):
        raise ValueError("The source world already contains generated rescue targets.")

    placements = []
    if humans + animals:
        terrain_include = next(
            (item for item in world.findall("include")
             if (item.findtext("uri") or "").strip() == "model://terrain_new"), None)
        if terrain_include is None:
            raise ValueError("terrain_new world must include model://terrain_new.")
        include_pose = terrain_include.find("pose")
        if include_pose is not None:
            raise ValueError("Set terrain placement in models/terrain_new/model.sdf, "
                             "rather than overriding the world's include pose.")
        surface = TerrainSurface.from_model(package_dir / "models" / "terrain_new")
        rng = random.Random(seed)
        requests = [("human", i + 1, None) for i in range(humans)]
        cycle = []
        for i in range(animals):
            if not cycle:
                cycle = list(selected_species)
                rng.shuffle(cycle)
            requests.append(("animal", i + 1, cycle.pop()))
        # Fit larger bodies first so smaller figures do not consume their clearings.
        requests = sorted(
            requests,
            key=lambda item: footprint_radius(item[0], package_dir, item[2]),
            reverse=True)
        xmin, xmax, ymin, ymax = surface.bounds
        for kind, index, species_name in requests:
            radius = footprint_radius(kind, package_dir, species_name)
            if xmax - xmin <= 2 * radius or ymax - ymin <= 2 * radius:
                raise ValueError("The scaled map is too small to place a %s." % kind)
            for attempt in range(600):
                x = rng.uniform(xmin + radius, xmax - radius)
                y = rng.uniform(ymin + radius, ymax - radius)
                if any(math.hypot(x-rx, y-ry) < radius+clearance
                       for rx, ry, clearance in reserved_points):
                    continue
                if any(math.hypot(x-p["x"], y-p["y"]) <
                       radius+p["radius"]+0.5 for p in placements):
                    continue
                if not surface.is_clear(x, y, radius, max_slope_deg=20.0):
                    continue
                footprint = [surface.sample(x, y).z]
                footprint.extend(
                    surface.sample(x + radius * math.cos(i * math.pi / 8),
                                   y + radius * math.sin(i * math.pi / 8)).z
                    for i in range(16))
                # An upright static mesh has a flat base. Reject uneven patches
                # and lift it to the high edge rather than burying uphill feet.
                if max(footprint) - min(footprint) > 0.20:
                    continue
                z = max(footprint) + 0.02
                yaw = rng.uniform(-math.pi, math.pi)
                name = ("rescue_human_%03d" % index if kind == "human" else
                        "rescue_animal_%03d_%s" % (index, species_name))
                maker = human_model if kind == "human" else animal_model
                kwargs = {} if kind == "human" else {"species": species_name}
                world.append(maker(name, x, y, z, yaw, package_dir, **kwargs))
                placements.append(dict(name=name, kind=kind, species=species_name,
                                       x=x, y=y, z=z, yaw=yaw, radius=radius))
                break
            else:
                raise ValueError(
                    "Could not fit the requested %d humans and %d animals on clear "
                    "terrain. Reduce the counts, enlarge the map, or try another spawn_seed."
                    % (humans, animals))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return PopulationResult(output_path, humans, animals, seed, tuple(placements))
