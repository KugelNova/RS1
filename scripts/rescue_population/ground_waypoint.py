"""Ground waypoints on the terrain collision mesh used by Gazebo."""

from pathlib import Path

from .terrain import TerrainSurface, _read_obj


class GroundWaypoints:
    """Sample the ground below a localized target in Gazebo world coordinates."""

    def __init__(self, surface):
        self.surface = surface

    @classmethod
    def from_mesh(cls, path):
        """Load a generated ground OBJ with its vertices already in world axes."""
        vertices, faces = _read_obj(Path(path))
        triangles = [vertices[indices] for indices, material in faces if material == "terrain"]
        if not triangles:
            raise ValueError("The ground collision mesh has no Terrain faces.")
        return cls(TerrainSurface(triangles))

    def at(self, xyz):
        """Keep the estimated horizontal position and sample its ground height."""
        x, y, _ = xyz
        point = self.surface.sample(x, y)
        return (float(x), float(y), point.z) if point is not None else None
