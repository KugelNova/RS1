"""Build a seeded terrain, scenery and population before Gazebo starts."""
from dataclasses import dataclass
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from .entities import SUPPORTED_ANIMALS, animal_model, footprint_radius, human_model
from .generated_ground import generate_ground
from .props import (footprint_radius as prop_radius, ground_radius,
                    rock_model, tree_model)
from .terrain import TerrainSurface

MAX_POPULATION = 500
MAX_PROPS = 2000


@dataclass(frozen=True)
class PopulationResult:
    path: Path
    humans: int
    animals: int
    seed: int
    placements: tuple
    trees: int = 0
    rocks: int = 0
    ground_metadata: object = None
    spawn_ground: tuple = ()
    ground_mesh_path: Path | None = None


def nonnegative_integer(value, name):
    """Validate launch inputs without accepting fractional counts."""
    if isinstance(value, bool):
        raise ValueError("%s must be a nonnegative whole number." % name)
    text = str(value).strip()
    if not text or not text.isascii() or not text.isdecimal():
        raise ValueError("%s must be a nonnegative whole number." % name)
    return int(text)


def finite_number(value, name, minimum, maximum=None):
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("%s must be a finite number." % name) from exc
    if not math.isfinite(number) or number < minimum:
        raise ValueError("%s must be finite and at least %s." % (name, minimum))
    if maximum is not None and number > maximum:
        raise ValueError("%s must be at most %s." % (name, maximum))
    return number


def animal_species(value):
    if isinstance(value, str):
        values = [part.strip().lower() for part in value.split(",")]
    else:
        values = list(value)
    if not values or any(not part or part not in SUPPORTED_ANIMALS for part in values):
        raise ValueError("animal_species must list one or more of: %s."
                         % ", ".join(SUPPORTED_ANIMALS))
    return tuple(dict.fromkeys(values))


def _clear_position(surface, rng, radius, base_radius, occupied, reserved_points,
                    *, max_slope=20.0, max_relief=0.20):
    xmin, xmax, ymin, ymax = surface.bounds
    if xmax - xmin <= 2*radius or ymax - ymin <= 2*radius:
        return None
    for _ in range(1200):
        x = rng.uniform(xmin+radius, xmax-radius)
        y = rng.uniform(ymin+radius, ymax-radius)
        if any(math.hypot(x-rx, y-ry) < radius+clearance
               for rx, ry, clearance in reserved_points):
            continue
        if any(math.hypot(x-p["x"], y-p["y"]) <
               radius+p["radius"]+0.5 for p in occupied):
            continue
        if not surface.is_clear(x, y, radius, max_slope_deg=max_slope):
            continue
        samples = [surface.sample(x, y)]
        samples.extend(
            surface.sample(x+base_radius*math.cos(i*math.pi/8),
                           y+base_radius*math.sin(i*math.pi/8))
            for i in range(16))
        if any(p is None for p in samples):
            continue
        heights = [p.z for p in samples]
        if max(heights)-min(heights) > max_relief:
            continue
        return x, y, max(heights)+0.02, rng.uniform(-math.pi, math.pi)
    return None


def create_population_world(package_dir, world_name, output_path, *,
                            humans=3, animals=5, spawn_seed=42,
                            species=SUPPORTED_ANIMALS, trees=0, rocks=0,
                            terrain_height=None, terrain_roughness=0.5,
                            map_size=None,
                            reserved_points=((0.0, 0.0, 3.0), (-10.0, 7.0, 3.0))):
    """Create an owned launch copy; original worlds and assets remain read-only.

    Passing terrain_height (including zero), map_size or positive scenery counts
    generates a new ground mesh. Omitting them retains the original imported
    surface for callers of the existing population API.
    """
    humans = nonnegative_integer(humans, "humans")
    animals = nonnegative_integer(animals, "animals")
    trees = nonnegative_integer(trees, "trees")
    rocks = nonnegative_integer(rocks, "rocks")
    seed = nonnegative_integer(spawn_seed, "spawn_seed")
    if humans+animals > MAX_POPULATION:
        raise ValueError("Choose at most %d humans and animals combined." % MAX_POPULATION)
    if trees+rocks > MAX_PROPS:
        raise ValueError("Choose at most %d trees and rocks combined." % MAX_PROPS)
    roughness = finite_number(terrain_roughness, "terrain_roughness", 0, 1)
    procedural = terrain_height is not None or map_size is not None or bool(trees+rocks)
    height = finite_number(8 if terrain_height is None else terrain_height,
                           "terrain_height", 0) if procedural else None
    size = finite_number(150 if map_size is None else map_size,
                         "map_size", 30) if procedural else None
    selected_species = animal_species(species)
    package_dir = Path(package_dir)
    if world_name not in ("simple_trees", "large_demo", "rescue_forest", "terrain_new"):
        raise ValueError("Unknown simulation world: %s." % world_name)
    if (humans+animals+trees+rocks or procedural) and world_name != "terrain_new":
        raise ValueError("Seeded terrain and population require world:=terrain_new; "
                         "use zero counts for the other demo worlds.")
    source_path = package_dir/"worlds"/(world_name+".sdf")
    output_path = Path(output_path)
    if output_path.resolve() == source_path.resolve():
        raise ValueError("The generated population world must not replace the source world.")
    tree = ET.parse(source_path)
    world = tree.getroot().find("world")
    if world is None or world.get("name") != world_name:
        raise ValueError("The SDF world name must match the selected world.")
    prefixes = ("rescue_human_", "rescue_animal_", "rescue_tree_", "rescue_rock_")
    if any((m.get("name") or "").startswith(prefixes) for m in world.findall("model")):
        raise ValueError("The source world already contains generated rescue targets.")

    placements = []
    metadata = None
    ground_mesh_path = None
    spawn_ground = ()
    surface = None
    if humans+animals or procedural:
        terrain_include = next(
            (item for item in world.findall("include")
             if (item.findtext("uri") or "").strip() == "model://terrain_new"), None)
        if terrain_include is None:
            raise ValueError("terrain_new world must include model://terrain_new.")
        if terrain_include.find("pose") is not None:
            raise ValueError("Set terrain placement in models/terrain_new/model.sdf, "
                             "rather than overriding the world's include pose.")
        if procedural:
            ground = generate_ground(
                package_dir, output_path.parent/"generated_ground", seed=seed,
                map_size=size, terrain_height=height, terrain_roughness=roughness)
            surface = ground.surface
            world.remove(terrain_include)
            world.append(ground.model)
            metadata = ground.metadata
            ground_mesh_path = ground.mesh_path
            heights = []
            for name, x, y in (("husky", 0, 0), ("parrot", -10, 7)):
                sample = surface.sample(x, y)
                if sample is None:
                    raise ValueError("map_size must include both robot spawn points.")
                heights.append((name, sample.z))
            spawn_ground = tuple(heights)
        else:
            surface = TerrainSurface.from_model(package_dir/"models"/"terrain_new")

        # Independent streams keep the ground and tree layout unchanged when
        # only the number of humans/animals is adjusted.
        for kind, count, maker, low, high in (
                ("tree", trees, tree_model, 0.75, 1.25),
                ("rock", rocks, rock_model, 0.6, 1.4)):
            rng = random.Random("%d:%s" % (seed, kind))
            for index in range(1, count+1):
                scale = rng.uniform(low, high)
                radius = prop_radius(kind, package_dir, scale)
                base = ground_radius(kind, package_dir, scale)
                position = _clear_position(
                    surface, rng, radius, base, placements, reserved_points,
                    max_slope=30, max_relief=0.35 if kind == "tree" else 0.30)
                if position is None:
                    raise ValueError("Could not fit %d trees and %d rocks on clear ground. "
                                     "Reduce counts or enlarge map_size." % (trees, rocks))
                x, y, z, yaw = position
                name = "rescue_%s_%03d" % (kind, index)
                world.append(maker(name, x, y, z, yaw, package_dir, scale=scale))
                placements.append(dict(name=name, kind=kind, species=None, scale=scale,
                                       x=x, y=y, z=z, yaw=yaw, radius=radius))

        rng = random.Random(seed)
        requests = [("human", i+1, None) for i in range(humans)]
        cycle = []
        for i in range(animals):
            if not cycle:
                cycle = list(selected_species)
                rng.shuffle(cycle)
            requests.append(("animal", i+1, cycle.pop()))
        requests.sort(key=lambda item: footprint_radius(item[0], package_dir, item[2]),
                      reverse=True)
        for kind, index, species_name in requests:
            radius = footprint_radius(kind, package_dir, species_name)
            position = _clear_position(
                surface, rng, radius, radius, placements, reserved_points)
            if position is None:
                raise ValueError("Could not fit the requested %d humans and %d animals "
                                 "on clear terrain. Reduce counts, enlarge the map, "
                                 "or try another spawn_seed." % (humans, animals))
            x, y, z, yaw = position
            name = ("rescue_human_%03d" % index if kind == "human" else
                    "rescue_animal_%03d_%s" % (index, species_name))
            maker = human_model if kind == "human" else animal_model
            kwargs = {} if kind == "human" else {"species": species_name}
            world.append(maker(name, x, y, z, yaw, package_dir, **kwargs))
            placements.append(dict(name=name, kind=kind, species=species_name,
                                   x=x, y=y, z=z, yaw=yaw, radius=radius))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_path, encoding="unicode", xml_declaration=True)
    return PopulationResult(output_path, humans, animals, seed, tuple(placements),
                            trees, rocks, metadata, spawn_ground, ground_mesh_path)
