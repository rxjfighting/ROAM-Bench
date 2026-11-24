"""Artic open check utilities for ROAM-Bench."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Sequence
import numpy as np
_TOL = 1e-06
_RNG = np.random.default_rng(0)

@dataclass
class Hull:
    eqs: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    vol: float

def hull_from_verts(verts: np.ndarray) -> Optional[Hull]:
    v = np.asarray(verts, float)
    if len(v) < 4:
        return None
    from scipy.spatial import ConvexHull
    try:
        h = ConvexHull(v)
    except Exception:
        return None
    return Hull(eqs=h.equations.copy(), lo=v.min(0), hi=v.max(0), vol=float(h.volume))

def box_hull(center: Sequence[float], half: Sequence[float], R: Optional[np.ndarray]=None) -> Hull:
    c = np.asarray(center, float)
    h = np.asarray(half, float)
    R = np.eye(3) if R is None else np.asarray(R, float)
    eqs = np.zeros((6, 4))
    corners = []
    for k in range(3):
        n = R[:, k]
        for s in (+1.0, -1.0):
            p = c + s * h[k] * n
            a = s * n
            row = len(corners)
            eqs[row, :3] = a
            eqs[row, 3] = -float(a @ p)
            corners.append(0)
    signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
    pts = c + signs * h @ R.T
    return Hull(eqs=eqs, lo=pts.min(0), hi=pts.max(0), vol=float(8.0 * h[0] * h[1] * h[2]))

def capsule_hulls(base: Sequence[float], top: Sequence[float], radius: float, segments: int=8) -> List[Hull]:
    b = np.asarray(base, float)
    t = np.asarray(top, float)
    ang = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.c_[radius * np.cos(ang), radius * np.sin(ang), np.zeros(segments)]
    verts = np.vstack([b + ring, t + ring])
    h = hull_from_verts(verts)
    return [h] if h else []

@dataclass
class MovingLink:
    name: str
    jtype: str
    anchor: np.ndarray
    axis: np.ndarray
    target_q: float
    hulls: List[Hull]

@dataclass
class Obstacle:
    name: str
    kind: str
    hulls: List[Hull]

@dataclass
class OpenCheckResult:
    success: bool
    error_message: str
    target_openness: float
    final_openness: float
    last_safe_openness: float
    colliding_object: Optional[str]
    colliding_kind: Optional[str]
    n_steps: int
    peak_overlap_frac: float
    reason: str = ''
    checked_obstacles: int = 0
    detection_enabled: bool = True

def _rot_about(pts, anchor, axis, ang):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-30)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    return (pts - anchor) @ R.T + anchor

def _translate(pts, axis, dist):
    a = np.asarray(axis, float)
    a = a / (np.linalg.norm(a) + 1e-30)
    return pts + dist * a

def _sample_inside(hulls: List[Hull], n: int) -> np.ndarray:
    if not hulls:
        return np.empty((0, 3))
    vols = np.array([max(h.vol, 1e-12) for h in hulls])
    quota = np.maximum(1, np.round(n * vols / vols.sum()).astype(int))
    out = []
    for h, q in zip(hulls, quota):
        need = int(q)
        got: List = []
        A, b = (h.eqs[:, :3], h.eqs[:, 3])
        tries = 0
        while len(got) < need and tries < 200:
            tries += 1
            c = _RNG.uniform(h.lo, h.hi, size=(max(need, 8) * 4, 3))
            got.extend(c[np.all(c @ A.T + b <= _TOL, axis=1)].tolist())
        out.extend(got[:need])
    return np.array(out) if out else np.empty((0, 3))

def _frac_inside(points: np.ndarray, hulls: List[Hull]) -> float:
    if len(points) == 0 or not hulls:
        return 0.0
    blo = np.min([h.lo for h in hulls], 0)
    bhi = np.max([h.hi for h in hulls], 0)
    cm = np.all((points >= blo - _TOL) & (points <= bhi + _TOL), 1)
    if not cm.any():
        return 0.0
    inside = np.zeros(len(points), bool)
    for h in hulls:
        sub = ~inside & cm
        if not sub.any():
            break
        idx = np.where(sub)[0]
        cs = points[idx]
        m = np.all((cs >= h.lo - _TOL) & (cs <= h.hi + _TOL), 1)
        if not m.any():
            continue
        A, b = (h.eqs[:, :3], h.eqs[:, 3])
        inside[idx[m][np.all(cs[m] @ A.T + b <= _TOL, 1)]] = True
    return float(inside.sum()) / float(len(points))

def _drive_samples(base_samples, link: MovingLink, openness: float) -> np.ndarray:
    if len(base_samples) == 0:
        return base_samples
    q = openness * link.target_q
    if link.jtype == 'revolute':
        return _rot_about(base_samples, link.anchor, link.axis, q)
    return _translate(base_samples, link.axis, q)

def check_open(link: MovingLink, obstacles: List[Obstacle], *, target_openness: float=1.0, n_steps: int=20, overlap_tol: float=0.005, samples: int=12000, rollback: str='start', detection_enabled: bool=True, start_openness: float=0.0) -> OpenCheckResult:
    if rollback not in ('start', 'last_safe'):
        raise ValueError("rollback must be 'start' or 'last_safe', got %r" % rollback)
    target_openness = float(target_openness)
    start_openness = float(start_openness)
    base = _sample_inside(link.hulls, samples)
    if not detection_enabled:
        return OpenCheckResult(success=True, error_message='', target_openness=target_openness, final_openness=target_openness, last_safe_openness=target_openness, colliding_object=None, colliding_kind=None, n_steps=0, peak_overlap_frac=0.0, reason='detection disabled', checked_obstacles=0, detection_enabled=False)
    steps = np.linspace(start_openness, target_openness, max(2, n_steps + 1))
    last_safe = start_openness
    peak = 0.0
    for o in steps[1:]:
        moved = _drive_samples(base, link, o)
        for obs in obstacles:
            fr = _frac_inside(moved, obs.hulls)
            if fr > peak:
                peak = fr
            if fr > overlap_tol:
                final = start_openness if rollback == 'start' else last_safe
                if obs.kind == 'agent':
                    msg = 'Openable object collided with the agent.'
                else:
                    msg = 'Openable object collided with %s.' % obs.name
                return OpenCheckResult(success=False, error_message=msg, target_openness=target_openness, final_openness=float(final), last_safe_openness=float(last_safe), colliding_object=obs.name, colliding_kind=obs.kind, n_steps=len(steps) - 1, peak_overlap_frac=round(float(fr), 5), reason='collided at openness=%.3f (frac=%.4f) with %s[%s]' % (o, fr, obs.name, obs.kind), checked_obstacles=len(obstacles), detection_enabled=True)
        last_safe = float(o)
    return OpenCheckResult(success=True, error_message='', target_openness=target_openness, final_openness=target_openness, last_safe_openness=target_openness, colliding_object=None, colliding_kind=None, n_steps=len(steps) - 1, peak_overlap_frac=round(float(peak), 5), reason='opened to openness=%.3f, no collision' % target_openness, checked_obstacles=len(obstacles), detection_enabled=True)
