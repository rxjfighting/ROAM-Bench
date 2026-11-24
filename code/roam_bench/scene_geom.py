"""Scene geom utilities for ROAM-Bench."""
from __future__ import annotations
import logging
import math
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from roam_bench.protocol.visibility import pixel_fraction
from roam_bench.scene_facts import SceneFacts, _find_house_state, _room_offsets, _obj_world_matrix, _world_aabb_drake, _STRUCTURAL
import json
_log = logging.getLogger('roam_bench.scene_geom')
GEOM_MODEL_VERSION = 'fakesim-vis-geom-v1'
_EYE_Y = 1.5
_ASPECT_FIT_TOL = 0.001
_FRAME_TOL_M = 1.0
_EPS = 1e-09

@dataclass
class Opening:
    t0: float
    t1: float
    y0: float
    y1: float
    kind: str

@dataclass
class Wall:
    x0: float
    z0: float
    x1: float
    z1: float
    openings: list = field(default_factory=list)

    @property
    def length(self) -> float:
        return math.hypot(self.x1 - self.x0, self.z1 - self.z0)

@dataclass(frozen=True)
class CamIntrinsics:
    fov_deg: float
    width: int
    height: int

    @classmethod
    def from_projection_matrix(cls, proj, width, height):
        m = _as_4x4(proj)
        p00, p11 = (float(m[0][0]), float(m[1][1]))
        if p00 <= 0.0 or p11 <= 0.0:
            raise ValueError(f'projection matrix has non-positive focal terms P00={p00} P11={p11}; this is not a perspective matrix in the expected layout.')
        aspect = float(width) / float(height)
        refit = p00 * aspect
        if abs(refit - p11) > _ASPECT_FIT_TOL * max(abs(p11), 1.0):
            raise ValueError(f'non-square pixels: P00*aspect={refit:.6f} != P11={p11:.6f} (w/h={aspect:.4f}Invalid scene geometry. ')
        fovy = 2.0 * math.degrees(math.atan(1.0 / p11))
        return cls(fov_deg=fovy, width=int(width), height=int(height))

def _as_4x4(proj):
    flat = list(proj)
    if len(flat) == 16 and (not hasattr(flat[0], '__len__')):
        return [flat[4 * r:4 * r + 4] for r in range(4)]
    if len(flat) == 4 and all((hasattr(row, '__len__') and len(row) == 4 for row in flat)):
        return [list(row) for row in flat]
    raise ValueError(f'projection matrix must be flat-16 or 4x4 nested, got shape ~{len(flat)}')

def load_cam(path: str) -> CamIntrinsics:
    d = json.load(open(path))
    flat = d['gpu_camera_params']['projection_flat16']
    return CamIntrinsics.from_projection_matrix(flat, int(d['width']), int(d['height']))
_REAL_PROJECTION_FLAT16 = [1.7179671468185143, 0.0, 0.0, 0.0, 0.0, 3.0541638165662475, 0.0, 0.0, 0.0, 0.0, 5.000250087013882e-05, -1.0, 0.0, 0.0, 0.05000250087013882, 0.0]
REAL_RENDER_CAM = CamIntrinsics.from_projection_matrix(_REAL_PROJECTION_FLAT16, 1280, 720)

@dataclass
class SceneGeom:
    version: str
    aabbs: dict = field(default_factory=dict)
    walls: list = field(default_factory=list)
    wall_height: float = 3.4
    room_wall_bounds: dict = field(default_factory=dict)
    cam: 'CamIntrinsics' = None

def _thor_aabb_from_drake(drake_aabb):
    if drake_aabb is None:
        return None
    lo, hi = drake_aabb
    return ((float(lo[0]), float(lo[2]), float(lo[1])), (float(hi[0]), float(hi[2]), float(hi[1])))

def _extract_walls(layout) -> tuple:
    from .exterior_seals import exterior_door_ids
    sealed = exterior_door_ids(layout)
    walls, room_bounds = ([], {})
    for pr in layout.get('placed_rooms', []):
        rid = pr.get('room_id')
        xs, zs = ([], [])
        for w in pr.get('walls') or []:
            sp, ep = (w.get('start_point'), w.get('end_point'))
            if not sp or not ep:
                continue
            x0, z0 = (float(sp[0]), float(sp[1]))
            x1, z1 = (float(ep[0]), float(ep[1]))
            ops = []
            for op in w.get('openings') or []:
                if str(op.get('opening_id') or '') in sealed:
                    continue
                pos = float(op.get('position_along_wall', 0.0))
                ow = float(op.get('width', 0.0))
                oh = float(op.get('height', 0.0))
                sill = float(op.get('sill_height', 0.0))
                ops.append(Opening(t0=pos, t1=pos + ow, y0=sill, y1=sill + oh, kind=str(op.get('opening_type', 'open'))))
            walls.append(Wall(x0, z0, x1, z1, ops))
            xs += [x0, x1]
            zs += [z0, z1]
        if xs:
            room_bounds[rid] = (min(xs), min(zs), max(xs), max(zs))
    return (walls, room_bounds)

def _assert_frame_shared(facts: SceneFacts, geom: SceneGeom) -> dict:
    geom_aliases = set(geom.aabbs)
    facts_aliases = set(facts.by_alias)
    if geom_aliases != facts_aliases:
        missing = sorted(facts_aliases - geom_aliases)[:5]
        extra = sorted(geom_aliases - facts_aliases)[:5]
        raise ValueError(f"scene_geom alias set != scene_facts alias set (missing {missing}, extra {extra}); the AABB extraction drifted from load_facts' alias scheme, so the gate would key on the wrong objects.")
    total = len(facts.instances)
    checked = 0
    unchecked = []
    worst_alias, worst_over = (None, 0.0)
    for inst in facts.instances:
        rid = inst.room_id
        env = geom.room_wall_bounds.get(rid)
        c = facts.centroids.get(inst.alias)
        if env is None or c is None:
            unchecked.append(inst.alias)
            continue
        checked += 1
        x, z = (c[0], c[2])
        ox = max(env[0] - x, x - env[2], 0.0)
        oz = max(env[1] - z, z - env[3], 0.0)
        over = math.hypot(ox, oz)
        if over > worst_over:
            worst_over, worst_alias = (over, inst.alias)
    if checked != total:
        raise ValueError(f'frame linchpin unverifiable: only {checked} of {total} centroids could be compared to a room wall envelope ({len(unchecked)} unchecked, e.g. {sorted(unchecked)[:5]}Invalid scene geometry. ')
    if worst_over > _FRAME_TOL_M:
        raise ValueError(f"checked {checked} of {total}: centroid of {worst_alias!r} is {worst_over:.2f} m outside its room's wall envelope (> {_FRAME_TOL_M}Invalid scene geometry. ")
    _log.info('frame linchpin: checked %d of %d centroids in wall envelope, worst overshoot %.2f m (<= %.1f m)', checked, total, worst_over, _FRAME_TOL_M)
    return {'checked': checked, 'total': total, 'worst_over': worst_over, 'worst_alias': worst_alias}

def load_geom(scene_root: str, facts: SceneFacts, cam: 'CamIntrinsics') -> SceneGeom:
    if cam is None:
        raise TypeError('Invalid scene geometry. ')
    hs = json.load(open(_find_house_state(scene_root)))
    layout = hs.get('layout', {})
    rooms = hs.get('rooms', {})
    offsets = _room_offsets(scene_root)
    aabbs = {}
    for rid, rm in rooms.items():
        off = offsets.get(rid, np.zeros(3))
        for oname, o in (rm.get('objects') or {}).items():
            if o.get('object_type') in _STRUCTURAL:
                continue
            oid = o.get('object_id') or oname
            alias = f'{rid}_{oid}'
            world_mat = _obj_world_matrix(o.get('transform'), off)
            drake_aabb = _world_aabb_drake(o.get('bbox_min'), o.get('bbox_max'), world_mat)
            thor_aabb = _thor_aabb_from_drake(drake_aabb)
            if thor_aabb is not None:
                aabbs[alias] = thor_aabb
    walls, room_bounds = _extract_walls(layout)
    geom = SceneGeom(version=GEOM_MODEL_VERSION, aabbs=aabbs, walls=walls, wall_height=float(layout.get('wall_height', 3.4)), room_wall_bounds=room_bounds, cam=cam)
    _assert_frame_shared(facts, geom)
    return geom

def _cam_from_pose(pose, cam) -> tuple:
    if cam is None:
        raise TypeError('Invalid scene geometry. ')
    return (float(pose.x), _EYE_Y, float(pose.z), float(pose.yaw_deg), float(pose.horizon_deg), float(cam.fov_deg), int(cam.width), int(cam.height))

def _aabb_corners(aabb):
    (lx, ly, lz), (hx, hy, hz) = aabb
    return [(lx if i & 1 else hx, ly if i & 2 else hy, lz if i & 4 else hz) for i in range(8)]
_NEAR_M = 0.001

def _projected_frac(cam, aabb) -> float:
    from roam_bench.protocol import camera
    ex, ey, ez, yaw, pitch, _fov, w, h = cam
    fwd, _, _ = camera._basis(yaw, pitch)
    corners = _aabb_corners(aabb)
    depths = [(p[0] - ex) * fwd[0] + (p[1] - ey) * fwd[1] + (p[2] - ez) * fwd[2] for p in corners]
    pts = [corners[i] for i in range(8) if depths[i] > _NEAR_M]
    for i in range(8):
        for bit in (1, 2, 4):
            j = i ^ bit
            if j < i:
                continue
            di, dj = (depths[i], depths[j])
            if (di > _NEAR_M) == (dj > _NEAR_M):
                continue
            t = (_NEAR_M - di) / (dj - di)
            ci, cj = (corners[i], corners[j])
            pts.append((ci[0] + t * (cj[0] - ci[0]), ci[1] + t * (cj[1] - ci[1]), ci[2] + t * (cj[2] - ci[2])))
    if not pts:
        return 0.0
    us, vs = ([], [])
    for p in pts:
        px = camera.project(cam, p)
        if px is None:
            continue
        us.append(px[0])
        vs.append(px[1])
    if not us:
        return 0.0
    u0, u1 = (max(0.0, min(us)), min(float(w), max(us)))
    v0, v1 = (max(0.0, min(vs)), min(float(h), max(vs)))
    area = max(0.0, u1 - u0) * max(0.0, v1 - v0)
    return area / float(w * h)

def _seg_seg_intersect_param(px, pz, dx, dz, ax, az, bx, bz):
    ex, ez = (bx - ax, bz - az)
    denom = dx * -ez - dz * -ex
    if abs(denom) < _EPS:
        return None
    rx, rz = (ax - px, az - pz)
    s = (rx * -ez - rz * -ex) / denom
    t = (dx * rz - dz * rx) / denom
    return (s, t)

def _wall_blocks_los(cam_xz, eye_y, tgt_xz, tgt_y, walls, wall_height) -> bool:
    px, pz = cam_xz
    tx, tz = tgt_xz
    dx, dz = (tx - px, tz - pz)
    if abs(dx) < _EPS and abs(dz) < _EPS:
        return False
    for w in walls:
        hit = _seg_seg_intersect_param(px, pz, dx, dz, w.x0, w.z0, w.x1, w.z1)
        if hit is None:
            continue
        s, t = hit
        if not _EPS < s < 1.0 - _EPS:
            continue
        if not -_EPS <= t <= 1.0 + _EPS:
            continue
        y_here = eye_y + s * (tgt_y - eye_y)
        if y_here > wall_height:
            continue
        arc = t * w.length
        through = False
        for op in w.openings:
            if op.t0 - _EPS <= arc <= op.t1 + _EPS and op.y0 - _EPS <= y_here <= op.y1 + _EPS:
                through = True
                break
        if not through:
            return True
    return False

def wall_blocks_los(cam_xz, eye_y, tgt_xz, tgt_y, walls, wall_height) -> bool:
    return _wall_blocks_los(cam_xz, eye_y, tgt_xz, tgt_y, walls, wall_height)

def _seg_aabb_hit(p0, p1, aabb) -> bool:
    (lx, ly, lz), (hx, hy, hz) = aabb
    lo = (lx, ly, lz)
    hi = (hx, hy, hz)
    tmin, tmax = (0.0, 1.0)
    for a in range(3):
        d = p1[a] - p0[a]
        if abs(d) < _EPS:
            if p0[a] < lo[a] - _EPS or p0[a] > hi[a] + _EPS:
                return False
            continue
        t1 = (lo[a] - p0[a]) / d
        t2 = (hi[a] - p0[a]) / d
        if t1 > t2:
            t1, t2 = (t2, t1)
        tmin = max(tmin, t1)
        tmax = min(tmax, t2)
        if tmin > tmax:
            return False
    return tmin < 1.0 - _EPS and tmax > _EPS

def _aabb_occluded(eye, tgt, tgt_aabb, occluders) -> bool:
    for aabb in occluders:
        if aabb is tgt_aabb:
            continue
        (lx, ly, lz), (hx, hy, hz) = aabb
        if lx - _EPS <= eye[0] <= hx + _EPS and ly - _EPS <= eye[1] <= hy + _EPS and (lz - _EPS <= eye[2] <= hz + _EPS):
            continue
        if lx - _EPS <= tgt[0] <= hx + _EPS and ly - _EPS <= tgt[1] <= hy + _EPS and (lz - _EPS <= tgt[2] <= hz + _EPS):
            continue
        if _seg_aabb_hit(eye, tgt, aabb):
            return True
    return False

def _centre(aabb):
    (lx, ly, lz), (hx, hy, hz) = aabb
    return ((lx + hx) / 2.0, (ly + hy) / 2.0, (lz + hz) / 2.0)

def visible_from_pose(geom: SceneGeom, pose, aliases, min_pixel_frac=None, *, occlude=True) -> set:
    mpf = pixel_fraction(geom.cam.width, geom.cam.height) if min_pixel_frac is None else float(min_pixel_frac)
    cam = _cam_from_pose(pose, geom.cam)
    eye = (float(pose.x), _EYE_Y, float(pose.z))
    cam_xz = (float(pose.x), float(pose.z))
    passed = []
    for a in aliases:
        aabb = geom.aabbs.get(a)
        if aabb is None:
            continue
        if _projected_frac(cam, aabb) < mpf:
            continue
        cx, cy, cz = _centre(aabb)
        if _wall_blocks_los(cam_xz, _EYE_Y, (cx, cz), cy, geom.walls, geom.wall_height):
            continue
        passed.append((a, aabb, (cx, cy, cz)))
    if not occlude:
        return {a for a, _, _ in passed}
    all_boxes = list(geom.aabbs.values())
    out = set()
    for a, aabb, ctr in passed:
        if not _aabb_occluded(eye, ctr, aabb, all_boxes):
            out.add(a)
    return out

class _Abstain:
    __slots__ = ('reason',)

    def __init__(self, reason: str):
        self.reason = reason

    def __bool__(self):
        raise TypeError(f'visibility is ABSTAIN ({self.reason}Invalid scene geometry. ')

    def __repr__(self):
        return f'ABSTAIN({self.reason})'
ABSTAIN_CROSS_ROOM = _Abstain('Invalid scene geometry. ')
ABSTAIN_CAM_OUTSIDE = _Abstain('camera pose outside every room box (room_localization_point None)')

def visible_one(geom: SceneGeom, facts, pose, alias, min_pixel_frac=None, *, occlude=True):
    inst = facts.by_alias.get(alias)
    if inst is None:
        raise ValueError(f'visible_one: alias {alias!r}Invalid scene geometry. ')
    cam_room = facts.room_localization_point(float(pose.x), float(pose.z))
    if cam_room is None:
        return ABSTAIN_CAM_OUTSIDE
    if getattr(inst, 'room_id', None) != cam_room:
        return ABSTAIN_CROSS_ROOM
    return alias in visible_from_pose(geom, pose, [alias], min_pixel_frac, occlude=occlude)
