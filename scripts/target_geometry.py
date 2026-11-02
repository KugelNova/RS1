"""Camera-ray / ground-plane intersection for the RS1 Gazebo thermal camera.

Assumptions:
* Gazebo camera points along +X; +Y is left, +Z is up.
* The image has square pixels and a known *horizontal* field of view.
* The configured ground plane has constant height in the selected TF frame.
These approximations must be validated before using targets as ground truth.
"""

import math


def _rotate_by_quaternion(vec, quat):
    """Rotate (x,y,z) by (qx,qy,qz,qw), normalizing the quaternion."""
    x, y, z, w = quat
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-12:
        raise ValueError('Invalid zero-length camera orientation')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    vx, vy, vz = vec
    # q * (vx,vy,vz,0) * conj(q) expressed via 2 cross products.
    tx = 2.0 * (y*vz - z*vy)
    ty = 2.0 * (z*vx - x*vz)
    tz = 2.0 * (x*vy - y*vx)
    return (
        vx + w*tx + y*tz - z*ty,
        vy + w*ty + z*tx - x*tz,
        vz + w*tz + x*ty - y*tx,
    )


def pixel_to_ground(u, v, width, height, horizontal_fov,
                    camera_position, camera_quaternion,
                    ground_z=0.0, max_distance=50.0):
    """Return (x,y,z) in the fixed TF frame, or None if no valid intersection.

    u increases right; v increases down.  Unlike an optical +Z frame,
    the Gazebo sensor frame used here has +X pointing forward.
    """
    if width < 2 or height < 2 or not (0.0 < horizontal_fov < math.pi):
        return None
    focal_px = (width / 2.0) / math.tan(horizontal_fov / 2.0)
    u0, v0 = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    camera_ray = (1.0, -(u - u0) / focal_px, -(v - v0) / focal_px)
    dx, dy, dz = _rotate_by_quaternion(camera_ray, camera_quaternion)
    if abs(dz) < 1e-4:
        return None
    t = (ground_z - camera_position[2]) / dz
    if t <= 0:
        return None
    distance = t * math.sqrt(dx*dx + dy*dy + dz*dz)
    if distance > max_distance:
        return None
    return (camera_position[0] + t*dx,
            camera_position[1] + t*dy,
            float(ground_z))
