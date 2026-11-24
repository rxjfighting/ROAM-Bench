"""Seg geom utilities for ROAM-Bench."""
from __future__ import annotations
import math
__all__ = ['point_seg_dist', 'dist_point_aabb', 'seg_seg_dist', 'seg_aabb_min_dist', 'seg_furniture_blocked']

def point_seg_dist(px, pz, ax, az, bx, bz):
    dx, dz = (bx - ax, bz - az)
    denom = dx * dx + dz * dz
    if denom == 0.0:
        return math.hypot(px - ax, pz - az)
    t = ((px - ax) * dx + (pz - az) * dz) / denom
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    return math.hypot(px - (ax + t * dx), pz - (az + t * dz))

def dist_point_aabb(px, pz, cx, cz, hx, hz):
    dx = max(abs(px - cx) - hx, 0.0)
    dz = max(abs(pz - cz) - hz, 0.0)
    return math.hypot(dx, dz)

def _segments_intersect(a, b, c, d):

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_seg(p, q, r):
        return min(p[0], r[0]) - 1e-12 <= q[0] <= max(p[0], r[0]) + 1e-12 and min(p[1], r[1]) - 1e-12 <= q[1] <= max(p[1], r[1]) + 1e-12
    o1, o2 = (orient(a, b, c), orient(a, b, d))
    o3, o4 = (orient(c, d, a), orient(c, d, b))
    if (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0):
        return True
    if abs(o1) < 1e-12 and on_seg(a, c, b):
        return True
    if abs(o2) < 1e-12 and on_seg(a, d, b):
        return True
    if abs(o3) < 1e-12 and on_seg(c, a, d):
        return True
    if abs(o4) < 1e-12 and on_seg(c, b, d):
        return True
    return False

def seg_seg_dist(a, b, c, d):
    if _segments_intersect(a, b, c, d):
        return 0.0
    return min(point_seg_dist(a[0], a[1], c[0], c[1], d[0], d[1]), point_seg_dist(b[0], b[1], c[0], c[1], d[0], d[1]), point_seg_dist(c[0], c[1], a[0], a[1], b[0], b[1]), point_seg_dist(d[0], d[1], a[0], a[1], b[0], b[1]))

def seg_aabb_min_dist(p, q, cx, cz, hx, hz):
    if dist_point_aabb(p[0], p[1], cx, cz, hx, hz) == 0.0 or dist_point_aabb(q[0], q[1], cx, cz, hx, hz) == 0.0:
        return 0.0
    x0, z0, x1, z1 = (cx - hx, cz - hz, cx + hx, cz + hz)
    corners = [(x0, z0), (x1, z0), (x1, z1), (x0, z1)]
    return min((seg_seg_dist(p, q, corners[i], corners[(i + 1) % 4]) for i in range(4)))

def seg_furniture_blocked(p, q, furniture_boxes, r, stats=None):
    sx0 = p[0] if p[0] <= q[0] else q[0]
    sx1 = p[0] if p[0] >= q[0] else q[0]
    sz0 = p[1] if p[1] <= q[1] else q[1]
    sz1 = p[1] if p[1] >= q[1] else q[1]
    for cx, cz, hx, hz in furniture_boxes:
        if cx - hx > sx1 + r or cx + hx < sx0 - r:
            continue
        if cz - hz > sz1 + r or cz + hz < sz0 - r:
            continue
        if stats is not None:
            stats['boxes_dist_calcd'] += 1
        if seg_aabb_min_dist(p, q, cx, cz, hx, hz) < r:
            return True
    return False
