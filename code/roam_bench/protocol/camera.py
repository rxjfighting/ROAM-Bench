"""Camera utilities for ROAM-Bench."""
import math

def pose(eye_x, eye_y, eye_z, yaw_deg, pitch_deg, fov_deg, width, height):
    return (float(eye_x), float(eye_y), float(eye_z), float(yaw_deg), float(pitch_deg), float(fov_deg), int(width), int(height))

def _basis(yaw_deg, pitch_deg):
    ry, rp = (math.radians(yaw_deg), math.radians(pitch_deg))
    sy, cy, sp, cp = (math.sin(ry), math.cos(ry), math.sin(rp), math.cos(rp))
    forward = (sy * cp, -sp, cy * cp)
    right = (cy, 0.0, -sy)
    up = (sp * sy, cp, sp * cy)
    return (forward, right, up)

def _half_extents(fov_deg, width, height):
    ty = math.tan(math.radians(fov_deg) / 2.0)
    return (ty * (float(width) / float(height)), ty)

def unproject(cam, u, v, depth):
    ex, ey, ez, yaw, pitch, fov, w, h = cam
    u, v = (int(u), int(v))
    if not (0 <= u < w and 0 <= v < h):
        return None
    try:
        z = float(depth[v][u])
    except (TypeError, IndexError, ValueError):
        return None
    if not math.isfinite(z) or z <= 0.0:
        return None
    forward, right, up = _basis(yaw, pitch)
    tx, ty = _half_extents(fov, w, h)
    ndc_x = 2.0 * (u + 0.5) / w - 1.0
    ndc_y = 1.0 - 2.0 * (v + 0.5) / h
    dx = ndc_x * tx
    dy = ndc_y * ty
    return (ex + forward[0] * z + right[0] * dx * z + up[0] * dy * z, ey + forward[1] * z + right[1] * dx * z + up[1] * dy * z, ez + forward[2] * z + right[2] * dx * z + up[2] * dy * z)

def project(cam, point):
    ex, ey, ez, yaw, pitch, fov, w, h = cam
    forward, right, up = _basis(yaw, pitch)
    rx = float(point[0]) - ex
    ry = float(point[1]) - ey
    rz = float(point[2]) - ez
    z = rx * forward[0] + ry * forward[1] + rz * forward[2]
    if z <= 1e-06:
        return None
    tx, ty = _half_extents(fov, w, h)
    ndc_x = (rx * right[0] + ry * right[1] + rz * right[2]) / (z * tx)
    ndc_y = (rx * up[0] + ry * up[1] + rz * up[2]) / (z * ty)
    return ((ndc_x + 1.0) * w / 2.0 - 0.5, (1.0 - ndc_y) * h / 2.0 - 0.5)
