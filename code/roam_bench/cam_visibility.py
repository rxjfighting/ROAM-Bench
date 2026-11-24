"""Cam visibility utilities for ROAM-Bench."""
import math
import numpy as np
FOCAL_MM = 18.0
HAPERTURE_MM = 20.955
DEFAULT_W, DEFAULT_H = (640, 480)
AXIS_ANGLE_MIN_DEG = 55.0

def intrinsics_px(width=DEFAULT_W, height=DEFAULT_H, focal_mm=FOCAL_MM, haperture_mm=HAPERTURE_MM):
    hfov = 2.0 * math.atan(haperture_mm / (2.0 * focal_mm))
    fx = width / 2.0 / math.tan(hfov / 2.0)
    return (fx, fx)

def unit(v):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    return v / (n if n > 1e-12 else 1.0)

def box_corners(mn, mx):
    return np.array([[mn[0] if i & 1 else mx[0], mn[1] if i & 2 else mx[1], mn[2] if i & 4 else mx[2]] for i in range(8)], dtype=np.float64)

def aabb_of(pts):
    pts = np.asarray(pts, dtype=np.float64)
    return (pts.min(axis=0), pts.max(axis=0))

def rot_about_line(pts, pivot, axis, ang):
    axis = unit(axis)
    p = np.asarray(pts, dtype=np.float64) - np.asarray(pivot, dtype=np.float64)
    c, s = (math.cos(ang), math.sin(ang))
    rot = p * c + np.cross(axis, p) * s + np.outer(np.dot(p, axis), axis) * (1.0 - c)
    return rot + np.asarray(pivot, dtype=np.float64)

def frame_corners(rest_mn, rest_mx, kind, axis_w, pivot, pos):
    corners = box_corners(rest_mn, rest_mx)
    if kind == 'prismatic':
        return corners + unit(axis_w) * float(pos)
    return rot_about_line(corners, pivot, axis_w, math.radians(float(pos)))

def box_strictly_foreground(eye, p, mn, mx, eps=0.001):
    eye = np.asarray(eye, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    mn = np.asarray(mn, dtype=np.float64)
    mx = np.asarray(mx, dtype=np.float64)
    d = p - eye
    if np.all(p >= mn - eps) and np.all(p <= mx + eps):
        return False
    tmin, tmax = (0.0, 1.0)
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if eye[i] < mn[i] or eye[i] > mx[i]:
                return False
        else:
            t1 = (mn[i] - eye[i]) / d[i]
            t2 = (mx[i] - eye[i]) / d[i]
            if t1 > t2:
                t1, t2 = (t2, t1)
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return tmax < 1.0 - eps
_FACES = [(0, +1), (0, -1), (1, +1), (1, -1), (2, +1), (2, -1)]

def _face_samples(mn, mx, ax, sign, n=4):
    other = [i for i in range(3) if i != ax]
    u, v = other
    us = np.linspace(mn[u], mx[u], n)
    vs = np.linspace(mn[v], mx[v], n)
    face_val = mx[ax] if sign > 0 else mn[ax]
    pts = []
    for uu in us:
        for vv in vs:
            p = np.zeros(3)
            p[ax] = face_val
            p[u] = uu
            p[v] = vv
            pts.append(p)
    normal = np.zeros(3)
    normal[ax] = float(sign)
    area = float((mx[u] - mn[u]) * (mx[v] - mn[v]))
    return (np.array(pts), normal, area)

def visible_area_px(eye, fmn, fmx, sibling_boxes, fx, fy, n=4):
    eye = np.asarray(eye, dtype=np.float64)
    total = 0.0
    for ax, sign in _FACES:
        pts, normal, area_m = _face_samples(fmn, fmx, ax, sign, n)
        fc = pts.mean(axis=0)
        if float(np.dot(normal, eye - fc)) <= 0.0:
            continue
        vis = sum((1 for p in pts if not any((box_strictly_foreground(eye, p, mn, mx) for mn, mx in sibling_boxes))))
        if vis == 0:
            continue
        frac = vis / float(len(pts))
        dist = float(np.linalg.norm(eye - fc)) or 1e-06
        cos = abs(float(np.dot(normal, unit(eye - fc))))
        total += frac * area_m * cos * (fx * fy) / (dist * dist)
    return total

def candidate_dirs(n_theta=24, n_phi=12):
    dirs = []
    for it in range(n_theta):
        th = 2.0 * math.pi * it / n_theta
        for ip in range(n_phi):
            z = -0.2 + 1.1 * ip / (n_phi - 1)
            r = math.sqrt(max(0.0, 1.0 - z * z))
            dirs.append(np.array([r * math.cos(th), r * math.sin(th), z]))
    return [unit(d) for d in dirs]

def project_px(eye, look_at, world_pt, fx, fy, width=DEFAULT_W, height=DEFAULT_H, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, dtype=np.float64)
    fwd = unit(np.asarray(look_at, dtype=np.float64) - eye)
    right = unit(np.cross(fwd, np.asarray(up, dtype=np.float64)))
    upv = np.cross(right, fwd)
    rel = np.asarray(world_pt, dtype=np.float64) - eye
    zc = float(np.dot(rel, fwd))
    if zc <= 1e-06:
        return None
    xc = float(np.dot(rel, right))
    yc = float(np.dot(rel, upv))
    u = width / 2.0 + fx * (xc / zc)
    v = height / 2.0 - fy * (yc / zc)
    return np.array([u, v], dtype=np.float64)

def predicted_centroid_shift_px(eye, look_at, closed_center, open_center, fx, fy, width=DEFAULT_W, height=DEFAULT_H):
    a = project_px(eye, look_at, closed_center, fx, fy, width, height)
    b = project_px(eye, look_at, open_center, fx, fy, width, height)
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))

def choose_view(half_c, closed_mn, closed_mx, open_mn, open_mx, axis_w, sibling_boxes, dist, fx=None, fy=None, axis_angle_min_deg=AXIS_ANGLE_MIN_DEG, min_area_px=4.8):
    if fx is None or fy is None:
        fx, fy = intrinsics_px()
    half_c = np.asarray(half_c, dtype=np.float64)
    axis_w = unit(axis_w)
    closed_center = (np.asarray(closed_mn) + np.asarray(closed_mx)) / 2.0
    open_center = (np.asarray(open_mn) + np.asarray(open_mx)) / 2.0
    n_off_axis = 0
    n_gate_pass = 0
    best_motion = None
    best_visible = None
    for d in candidate_dirs():
        ang = math.degrees(math.acos(min(1.0, abs(float(np.dot(d, axis_w))))))
        if ang < axis_angle_min_deg:
            continue
        n_off_axis += 1
        eye = half_c + d * dist
        o_px = visible_area_px(eye, open_mn, open_mx, sibling_boxes, fx, fy)
        c_px = visible_area_px(eye, closed_mn, closed_mx, sibling_boxes, fx, fy)
        shift = predicted_centroid_shift_px(eye, half_c, closed_center, open_center, fx, fy)
        rec = {'dir': d, 'eye': eye, 'closed_px': c_px, 'open_px': o_px, 'min_px': min(c_px, o_px), 'off_axis_deg': ang, 'pred_centroid_shift_px': shift}
        if best_visible is None or rec['min_px'] > best_visible['min_px']:
            best_visible = rec
        if c_px > min_area_px and o_px > min_area_px and (shift is not None):
            n_gate_pass += 1
            if best_motion is None or shift > best_motion['pred_centroid_shift_px']:
                best_motion = rec
    best = best_motion if best_motion is not None else best_visible
    if best is None:
        d = candidate_dirs()[0]
        best = {'dir': d, 'eye': half_c + d * dist, 'closed_px': 0.0, 'open_px': 0.0, 'min_px': 0.0, 'off_axis_deg': 0.0, 'pred_centroid_shift_px': None}
    best['gate_ok'] = bool(best_motion is not None)
    best['n_off_axis_candidates'] = n_off_axis
    best['n_gate_pass'] = n_gate_pass
    return best
