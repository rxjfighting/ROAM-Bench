"""Pixel refer utilities for ROAM-Bench."""
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import math
import numpy as np
from . import seg_alias
R_OOB = 'out_of_bounds'
R_VOID = 'no_instance'
R_STRUCTURAL = 'structural'
R_NOT_GROUND = 'not_ground'
R_ABOVE_EYE = 'above_eye'
R_NO_WORLD = 'no_world_point'
R_FOREIGN = 'foreign_label'
R_NO_SEG = 'no_seg_frame'
R_NO_STANDABLE = 'no_standable_within_tol'
R_NO_DEPTH = 'depth_missing'
R_WALL = 'wall_not_floor'
R_CEILING = 'ceiling_not_floor'
_REASON_TEXT = {R_OOB: 'That point is outside your view.', R_VOID: 'Nothing is there -- that point is empty space (sky/void).', R_STRUCTURAL: 'That is floor, wall or ceiling, not an object; use the ground skill to walk there.', R_NOT_GROUND: 'That is an object, not ground; use the object skill to walk to a thing.', R_ABOVE_EYE: 'That is above you, not ground you could stand on.', R_NO_WORLD: 'You cannot work out where that point is.', R_FOREIGN: 'There is an object there, but it is not in our object table (unnamed instance).', R_NO_SEG: 'No segmentation frame yet (render first).', R_NO_STANDABLE: 'There is nowhere to stand near that point.', R_NO_DEPTH: 'Depth channel unavailable for that point.', R_WALL: 'That is a wall, not floor you can stand on.', R_CEILING: 'That is the ceiling, not floor you can stand on.'}
_STRUCTURAL_MARKER = 'room_geometry'

def _scene_query_hit_dict(hit):
    if isinstance(hit, dict):
        if not hit.get('hit', True):
            return None
        return dict(hit, hit=True)
    try:
        return {'hit': True, 'position': hit.position, 'normal': hit.normal, 'distance': float(hit.distance), 'collision': str(getattr(hit, 'collision', '')), 'rigidBody': str(getattr(hit, 'rigid_body', getattr(hit, 'rigidBody', '')))}
    except (AttributeError, TypeError, ValueError):
        return None

def _hit_belongs_to_roots(hit, roots):
    for key in ('collision', 'rigidBody'):
        path = str(hit.get(key, ''))
        for root in roots:
            root = str(root).rstrip('/')
            if root and (path == root or path.startswith(root + '/')):
                return True
    return False

@dataclass(frozen=True)
class PixelHit:
    kind: str
    alias: Optional[str] = None
    xyz: Optional[Tuple[float, float, float]] = None
    surface: Optional[str] = None
    reason: Optional[str] = None
    reason_code: Optional[str] = None

    def __bool__(self) -> bool:
        return True

    @property
    def ok(self) -> bool:
        return self.kind != 'unresolved'

def _unresolved(reason_code: str, **kw) -> PixelHit:
    return PixelHit('unresolved', reason=_REASON_TEXT[reason_code], reason_code=reason_code, **kw)

def classify_label(id_to_labels, safe2alias, seg_id) -> Tuple[str, Optional[str], Optional[str]]:
    label = seg_alias._lookup_label(id_to_labels, seg_id)
    if label is None or str(label) in seg_alias._BACKGROUND_LABELS:
        return ('void', None, None if label is None else str(label))
    s = str(label)
    root = seg_alias.instance_root_of(s)
    name_seg = root.rsplit('/', 1)[-1] if root else s
    if _STRUCTURAL_MARKER in name_seg or _STRUCTURAL_MARKER in s:
        return ('structural', None, s)
    alias = seg_alias.resolve_label(s, safe2alias)
    if alias is not None:
        return ('object', alias, s)
    return ('foreign', None, s)

def _bounds_fail(seg, u, v):
    h, w = np.asarray(seg).shape[:2]
    if not (0 <= v < h and 0 <= u < w):
        return _unresolved(R_OOB)
    return None

def _ray_dir_world(u, v, W, H, tx, ty, R):
    ndc_x = 2.0 * (u + 0.5) / float(W) - 1.0
    ndc_y = 1.0 - 2.0 * (v + 0.5) / float(H)
    d_cam = np.array([ndc_x * tx, ndc_y * ty, -1.0])
    d_world = np.asarray(R) @ d_cam
    n = float(np.linalg.norm(d_world)) or 1.0
    return d_world / n

def classify_structural_surface(xyz, normal, *, floor_z=None, up=(0.0, 0.0, 1.0), up_dot_min=0.85, floor_z_tol=0.15) -> str:
    z = float(xyz[2])
    near_floor_h = floor_z is None or abs(z - float(floor_z)) <= floor_z_tol
    n = None
    if normal is not None:
        v = np.asarray(normal, dtype=float)
        nn = float(np.linalg.norm(v))
        if nn > 0.0:
            n = v / nn
    if n is not None:
        upv = np.asarray(up, dtype=float)
        upv = upv / (float(np.linalg.norm(upv)) or 1.0)
        d = float(np.dot(n, upv))
        if d >= up_dot_min:
            return 'floor' if near_floor_h else 'wall'
        if d <= -up_dot_min:
            return 'ceiling'
        return 'wall'
    if floor_z is not None:
        return 'floor' if near_floor_h else 'wall'
    return 'unknown'

def resolve_pixel_to_object(seg, id_to_labels, safe2alias, u, v) -> PixelHit:
    fail = _bounds_fail(seg, u, v)
    if fail is not None:
        return fail
    seg = np.asarray(seg)
    cat, alias, _ = classify_label(id_to_labels, safe2alias, int(seg[v, u]))
    if cat == 'structural':
        return _unresolved(R_STRUCTURAL)
    if cat == 'void':
        return _unresolved(R_VOID)
    if cat == 'foreign':
        return _unresolved(R_FOREIGN)
    return PixelHit('object', alias=alias)

def resolve_pixel_to_navpoint(seg, id_to_labels, safe2alias, u, v, *, world_from_pixel: Callable[[int, int], Optional[Tuple[Tuple[float, float, float], Optional[Tuple[float, float, float]]]]], floor_z: Optional[float]=None) -> PixelHit:
    fail = _bounds_fail(seg, u, v)
    if fail is not None:
        return fail
    seg = np.asarray(seg)
    cat, alias, _ = classify_label(id_to_labels, safe2alias, int(seg[v, u]))
    if cat == 'object':
        return _unresolved(R_NOT_GROUND)
    if cat == 'void':
        return _unresolved(R_VOID)
    if cat == 'foreign':
        return _unresolved(R_FOREIGN)
    hit = world_from_pixel(u, v)
    if hit is None:
        return _unresolved(R_NO_WORLD)
    xyz, normal = hit
    x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
    surf = classify_structural_surface((x, y, z), normal, floor_z=floor_z)
    if surf == 'floor':
        return PixelHit('floor_point', xyz=(x, y, z), surface='floor')
    if surf == 'ceiling':
        return _unresolved(R_CEILING)
    return _unresolved(R_WALL)

def nearest_standable_free(x, z, *, standable: Callable[[float, float], bool], snap_tolerance: float, step: float=0.05) -> PixelHit:
    if standable(x, z):
        return PixelHit('floor_point', xyz=(float(x), 0.0, float(z)), surface='standable')
    r = step
    while r <= snap_tolerance + 1e-09:
        n = max(8, int(2 * math.pi * r / step))
        best = None
        for k in range(n):
            ang = 2.0 * math.pi * k / n
            cx = x + r * math.cos(ang)
            cz = z + r * math.sin(ang)
            if standable(cx, cz):
                d = (cx - x) ** 2 + (cz - z) ** 2
                if best is None or d < best[0]:
                    best = (d, cx, cz)
        if best is not None:
            return PixelHit('floor_point', xyz=(float(best[1]), 0.0, float(best[2])), surface='standable')
        r += step
    return _unresolved(R_NO_STANDABLE)

class PixelReferer:

    def __init__(self, backend, floor_z: Optional[float]=0.0):
        self._b = backend
        self._floor_z = floor_z

    def object_at(self, u: int, v: int) -> PixelHit:
        sa = self._b._seg_and_aliases()
        if sa is None:
            return PixelHit('unresolved', reason='No segmentation frame yet (render first).', reason_code=R_NO_SEG)
        return resolve_pixel_to_object(*sa, int(u), int(v))

    def navpoint_at(self, u: int, v: int) -> PixelHit:
        sa = self._b._seg_and_aliases()
        if sa is None:
            return PixelHit('unresolved', reason='No segmentation frame yet (render first).', reason_code=R_NO_SEG)
        return resolve_pixel_to_navpoint(*sa, int(u), int(v), world_from_pixel=self._raycast_hit, floor_z=self._floor_z)

    def _live_projection(self):
        if getattr(self, '_proj_cache', '_miss') != '_miss':
            return self._proj_cache
        self._proj_cache, self._proj_err = (None, None)
        try:
            rep = self._b._rep
            annot = rep.AnnotatorRegistry.get_annotator('camera_params')
            annot.attach([self._b._render_product])
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=4)
            data = annot.get_data()
            proj = np.asarray(data['cameraProjection'], dtype=float).reshape(4, 4)
            W, H = (int(self._b._width), int(self._b._height))
            self._proj_cache = (float(proj[0, 0]), float(proj[1, 1]), W, H)
        except Exception as e:
            self._proj_err = repr(e)
        return self._proj_cache

    def _pixel_ray_dir(self, u, v, eye, R):
        import math
        live = self._live_projection()
        if live is not None:
            p00, p11, W, H = live
            tx, ty = (1.0 / p00, 1.0 / p11)
        else:
            from .scene_geom import REAL_RENDER_CAM as cam
            W, H = (int(self._b._width), int(self._b._height))
            ty0 = math.tan(math.radians(cam.fov_deg) / 2.0)
            tx = ty0 * (cam.width / cam.height)
            ty = tx / (float(W) / float(H))
        return _ray_dir_world(u, v, W, H, tx, ty, R)

    def _raycast_world(self, u, v):
        hit = self._raycast_hit(u, v)
        return None if hit is None else hit[0]

    def _raycast_hit(self, u, v):
        try:
            import omni.physx, carb
        except Exception:
            return None
        eye, R = self._live_cam_eye_and_basis()
        if eye is None:
            return None
        d = self._pixel_ray_dir(u, v, eye, R)
        sq = omni.physx.get_physx_scene_query_interface()
        o = carb.Float3(float(eye[0]), float(eye[1]), float(eye[2]))
        dd = carb.Float3(float(d[0]), float(d[1]), float(d[2]))
        ignore_roots = tuple(getattr(self._b, '_pixel_raycast_ignore_roots', ()))
        ignored_hits = []
        if ignore_roots:
            hits = []

            def report(raw):
                hit_row = _scene_query_hit_dict(raw)
                if hit_row is not None:
                    if _hit_belongs_to_roots(hit_row, ignore_roots):
                        ignored_hits.append(hit_row)
                    else:
                        hits.append(hit_row)
                return True
            try:
                sq.raycast_all(o, dd, 10000.0, report, False)
            except TypeError:
                sq.raycast_all(o, dd, 10000.0, report)
            hit = min(hits, key=lambda row: row['distance']) if hits else {'hit': False}
        else:
            hit = sq.raycast_closest(o, dd, 10000.0, False)
        got = hit.get('hit') if isinstance(hit, dict) else getattr(hit, 'hit', False)
        recorder = getattr(self._b, '_record_pixel_raycast', None)
        if callable(recorder):
            try:
                recorder(int(u), int(v), eye, d, hit if got else None, ignored_hits)
            except TypeError:
                recorder(int(u), int(v), eye, d, hit if got else None)
        if not got:
            return None
        pos = hit.get('position') if isinstance(hit, dict) else getattr(hit, 'position')
        nrm = hit.get('normal') if isinstance(hit, dict) else getattr(hit, 'normal', None)
        pos3 = (float(pos[0]), float(pos[1]), float(pos[2]))
        normal = None
        if nrm is not None:
            try:
                normal = (float(nrm[0]), float(nrm[1]), float(nrm[2]))
            except Exception:
                normal = None
        return (pos3, normal)

    def _live_cam_eye_and_basis(self):
        b = self._b
        try:
            pose = b._pose
            if pose is None:
                return (None, None)
            camera_frame = getattr(b, '_camera_frame', None)
            if callable(camera_frame):
                eye, target, up = camera_frame()
            else:
                eye, target = b._pose_to_camera(pose)
                up = (0.0, 0.0, 1.0)
            eye = np.array([float(eye[0]), float(eye[1]), float(eye[2])])
            target = np.array([float(target[0]), float(target[1]), float(target[2])])
            up = np.array([float(up[0]), float(up[1]), float(up[2])])
            fwd = target - eye
            fwd = fwd / (np.linalg.norm(fwd) or 1.0)
            right = np.cross(fwd, up)
            right = right / (np.linalg.norm(right) or 1.0)
            up2 = np.cross(right, fwd)
            R = np.column_stack([right, up2, -fwd])
            return (eye, R)
        except Exception:
            return (None, None)

    def _eye_y(self):
        eye, _ = self._live_cam_eye_and_basis()
        return float(eye[2]) if eye is not None else 1.5
