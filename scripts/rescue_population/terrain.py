"""Ground sampling and conservative vegetation clearance for the imported terrain."""

from collections import defaultdict
import math
from pathlib import Path
from typing import NamedTuple
import xml.etree.ElementTree as ET

import numpy as np


class GroundSample(NamedTuple):
    z: float
    slope_deg: float


def _pose(element):
    matrix = np.eye(4)
    if element is None or not (element.text or "").strip():
        return matrix
    if element.get("relative_to"):
        raise ValueError("Population terrain poses must use their parent frame.")
    values = [float(v) for v in element.text.split()]
    if len(values) != 6 or not np.isfinite(values).all():
        raise ValueError("Population terrain poses must contain six finite XYZ/RPY values.")
    x, y, z, roll, pitch, yaw = values
    if element.get("degrees", "false").lower() in ("true", "1"):
        roll, pitch, yaw = map(math.radians, (roll, pitch, yaw))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    matrix[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    matrix[:3, 3] = (x, y, z)
    return matrix


def _cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _triangulate(vertices, indices):
    """Fan convex polygons; ear-clip concave polygons without filling their notches."""
    if len(indices) == 3:
        return [indices]
    points = np.asarray([vertices[i] for i in indices])
    normal = np.cross(points, np.roll(points, -1, axis=0)).sum(axis=0)
    axis = int(np.argmax(np.abs(normal)))
    xy = np.delete(points, axis, axis=1)
    turns = np.asarray([
        _cross2(xy[(i + 1) % len(xy)] - xy[i],
                xy[(i + 2) % len(xy)] - xy[(i + 1) % len(xy)])
        for i in range(len(xy))
    ])
    epsilon = 1e-10
    if np.all(turns >= -epsilon) or np.all(turns <= epsilon):
        return [(indices[0], indices[i], indices[i + 1])
                for i in range(1, len(indices) - 1)]
    area = sum(_cross2(xy[i], xy[(i + 1) % len(xy)]) for i in range(len(xy)))
    orientation = 1 if area >= 0 else -1
    remaining = list(range(len(indices)))
    triangles = []
    while len(remaining) > 3:
        for offset, middle in enumerate(remaining):
            left = remaining[offset - 1]
            right = remaining[(offset + 1) % len(remaining)]
            a, b, c = xy[[left, middle, right]]
            if orientation * _cross2(b - a, c - b) <= epsilon:
                continue
            blocked = False
            for other in remaining:
                if other in (left, middle, right):
                    continue
                p = xy[other]
                signs = [_cross2(b - a, p - a), _cross2(c - b, p - b),
                         _cross2(a - c, p - c)]
                if all(orientation * value >= -epsilon for value in signs):
                    blocked = True
                    break
            if not blocked:
                triangles.append((indices[left], indices[middle], indices[right]))
                remaining.pop(offset)
                break
        else:
            raise ValueError("Terrain OBJ contains a polygon that cannot be triangulated.")
    triangles.append(tuple(indices[i] for i in remaining))
    return triangles


def _read_obj(path):
    vertices, faces = [], []
    material = ""
    with path.open(encoding="utf-8", errors="replace") as source:
        for line_number, line in enumerate(source, 1):
            fields = line.partition("#")[0].split()
            if not fields:
                continue
            if fields[0] == "v":
                point = tuple(float(v) for v in fields[1:4])
                if len(point) != 3 or not np.isfinite(point).all():
                    raise ValueError("Invalid vertex in terrain OBJ at line %d." % line_number)
                vertices.append(point)
            elif fields[0] == "usemtl":
                material = " ".join(fields[1:]).strip().lower()
            elif fields[0] == "f":
                indices = [int(part.split("/")[0]) for part in fields[1:]
                           if not part.startswith("#")]
                indices = [i - 1 if i > 0 else len(vertices) + i for i in indices]
                if len(indices) < 3 or any(i < 0 or i >= len(vertices) for i in indices):
                    raise ValueError("Invalid face in terrain OBJ at line %d." % line_number)
                for triangle in _triangulate(vertices, indices):
                    faces.append((triangle, material))
    if not vertices or not faces:
        raise ValueError("Terrain OBJ contains no usable mesh.")
    return np.asarray(vertices, dtype=float), faces


def _mesh_path(model_dir, uri):
    if uri.startswith("model://"):
        model_name, separator, relative = uri[len("model://"):].partition("/")
        if not separator or model_name != model_dir.name:
            raise ValueError("Population terrain mesh must be inside its model directory.")
        return model_dir / relative
    if uri.startswith("file://"):
        return Path(uri[len("file://"):])
    return model_dir / uri


class TerrainSurface:
    """A spatially indexed terrain surface in Gazebo world coordinates.

    Non-Terrain materials are conservatively treated as obstacles in XY, including
    foliage above the ground. This keeps spawned bodies out of tree canopies.
    """

    def __init__(self, ground_triangles, obstacle_triangles=None):
        ground = np.asarray(ground_triangles, dtype=float).reshape((-1, 3, 3))
        obstacles = np.asarray(
            [] if obstacle_triangles is None else obstacle_triangles,
            dtype=float).reshape((-1, 3, 3))
        if not len(ground) or not np.isfinite(ground).all() or not np.isfinite(obstacles).all():
            raise ValueError("Population needs a finite, nonempty terrain surface.")
        edges = ground[:, 1:] - ground[:, :1]
        normals = np.cross(edges[:, 0], edges[:, 1])
        usable = np.abs(normals[:, 2]) > 1e-10
        self._ground = ground[usable]
        normals = normals[usable]
        if not len(self._ground):
            raise ValueError("Terrain has no ground faces that can be sampled from above.")
        self._slopes = np.degrees(np.arctan2(
            np.linalg.norm(normals[:, :2], axis=1), np.abs(normals[:, 2])))
        xy = self._ground[:, :, :2].reshape((-1, 2))
        low, high = xy.min(axis=0), xy.max(axis=0)
        self.bounds = (float(low[0]), float(high[0]), float(low[1]), float(high[1]))
        self._cell_size = max(float(np.max(high - low)) / 64.0, 0.05)
        self._ground_cells = self._index(self._ground)
        self._obstacles = obstacles
        self._obstacle_cells = self._index(obstacles)

    @classmethod
    def from_model(cls, model_dir):
        model_dir = Path(model_dir)
        model = ET.parse(model_dir / "model.sdf").getroot().find("model")
        if model is None:
            raise ValueError("Missing model in terrain model.sdf.")
        transform = _pose(model.find("pose"))
        ground, obstacles = [], []
        for link in model.findall("link"):
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                if mesh is None:
                    continue
                uri = (mesh.findtext("uri") or "").strip()
                if not uri:
                    raise ValueError("Terrain collision mesh has no URI.")
                scale = np.asarray([float(v) for v in
                                    (mesh.findtext("scale") or "1 1 1").split()])
                if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
                    raise ValueError("Terrain mesh scale must contain three positive numbers.")
                vertices, faces = _read_obj(_mesh_path(model_dir, uri))
                local = vertices * scale
                placement = transform @ _pose(link.find("pose")) @ _pose(collision.find("pose"))
                vertices = local @ placement[:3, :3].T + placement[:3, 3]
                for indices, material in faces:
                    target = ground if material == "terrain" else obstacles
                    target.append(vertices[list(indices)])
        if not ground:
            raise ValueError("Terrain collision OBJ must identify ground with the Terrain material.")
        return cls(ground, obstacles)

    def _cell(self, x, y):
        return math.floor(x / self._cell_size), math.floor(y / self._cell_size)

    def _index(self, triangles):
        buckets = defaultdict(list)
        for index, triangle in enumerate(triangles):
            low, high = triangle[:, :2].min(axis=0), triangle[:, :2].max(axis=0)
            min_x, min_y = self._cell(*low)
            max_x, max_y = self._cell(*high)
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    buckets[(x, y)].append(index)
        return buckets

    def sample(self, x, y):
        if not math.isfinite(x) or not math.isfinite(y):
            return None
        candidates = self._ground_cells.get(self._cell(x, y))
        if not candidates:
            return None
        indices = np.asarray(candidates)
        triangles = self._ground[indices]
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        e, f = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2]
        delta = np.asarray((x, y)) - a[:, :2]
        determinant = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
        u = (delta[:, 0] * f[:, 1] - delta[:, 1] * f[:, 0]) / determinant
        v = (e[:, 0] * delta[:, 1] - e[:, 1] * delta[:, 0]) / determinant
        inside = (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)
        if not inside.any():
            return None
        z = a[inside, 2] + u[inside] * (b[inside, 2] - a[inside, 2]) + v[inside] * (
            c[inside, 2] - a[inside, 2])
        top = float(z.max())
        slopes = self._slopes[indices[inside]]
        return GroundSample(top, float(slopes[np.abs(z - top) <= 1e-7].max()))

    def _obstacle_near(self, x, y, radius):
        lo_x, lo_y = self._cell(x - radius, y - radius)
        hi_x, hi_y = self._cell(x + radius, y + radius)
        candidates = set()
        for cx in range(lo_x, hi_x + 1):
            for cy in range(lo_y, hi_y + 1):
                candidates.update(self._obstacle_cells.get((cx, cy), ()))
        if not candidates:
            return False
        xy = self._obstacles[list(candidates), :, :2]
        point = np.asarray((x, y))
        a, b, c = xy[:, 0], xy[:, 1], xy[:, 2]
        e, f, delta = b - a, c - a, point - a
        det = e[:, 0] * f[:, 1] - e[:, 1] * f[:, 0]
        valid = np.abs(det) > 1e-12
        u, v = np.zeros_like(det), np.zeros_like(det)
        u[valid] = (delta[valid, 0] * f[valid, 1] - delta[valid, 1] * f[valid, 0]) / det[valid]
        v[valid] = (e[valid, 0] * delta[valid, 1] - e[valid, 1] * delta[valid, 0]) / det[valid]
        if (valid & (u >= -1e-8) & (v >= -1e-8) & (u + v <= 1 + 1e-8)).any():
            return True
        for edge in range(3):
            start, end = xy[:, edge], xy[:, (edge + 1) % 3]
            direction = end - start
            length_squared = np.einsum("ij,ij->i", direction, direction)
            numerator = np.einsum("ij,ij->i", point - start, direction)
            fraction = np.zeros_like(numerator)
            np.divide(numerator, length_squared, out=fraction, where=length_squared > 1e-20)
            nearest = start + np.clip(fraction, 0, 1)[:, None] * direction
            if (np.sum((nearest - point) ** 2, axis=1) <= radius ** 2 + 1e-12).any():
                return True
        return False

    def is_clear(self, x, y, radius, max_slope_deg=30.0):
        if not all(math.isfinite(v) for v in (x, y, radius, max_slope_deg)):
            return False
        if radius < 0 or max_slope_deg < 0:
            return False
        xmin, xmax, ymin, ymax = self.bounds
        if x - radius < xmin or x + radius > xmax or y - radius < ymin or y + radius > ymax:
            return False
        if self._obstacle_near(x, y, radius):
            return False
        offsets = [(0.0, 0.0)]
        if radius:
            offsets.extend((radius * math.cos(i * math.pi / 8),
                            radius * math.sin(i * math.pi / 8)) for i in range(16))
        for dx, dy in offsets:
            point = self.sample(x + dx, y + dy)
            if point is None or point.slope_deg > max_slope_deg:
                return False
        return True
