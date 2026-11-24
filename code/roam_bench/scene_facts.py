"""Scene facts utilities for ROAM-Bench."""
from __future__ import annotations
import json
import math
import os
from dataclasses import dataclass, field, fields, replace
from typing import Optional
import numpy as np
from roam_bench.protocol.types import InstanceFacts
from roam_bench.scene_loader import _find_combined, _DrakeLoader, _translate
_IF_FIELDS = {f.name for f in fields(InstanceFacts)}

def _instance_facts(**kw) -> InstanceFacts:
    return InstanceFacts(**{k: v for k, v in kw.items() if k in _IF_FIELDS})
_STRUCTURAL = {'wall'}

def _quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = (w / n, x / n, y / n, z / n)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])

def _obj_world_matrix(transform, room_offset) -> np.ndarray:
    t = transform or {}
    local = np.eye(4)
    local[:3, :3] = _quat_wxyz_to_R(t.get('rotation_wxyz', [1, 0, 0, 0]))
    local[:3, 3] = np.asarray(t.get('translation', [0, 0, 0]), float)
    return _translate(room_offset) @ local

def _world_aabb_drake(bbox_min, bbox_max, world_mat):
    if not bbox_min or not bbox_max:
        return None
    lo = np.asarray(bbox_min, float)
    hi = np.asarray(bbox_max, float)
    corners = np.array([[lo[0] if i & 1 else hi[0], lo[1] if i & 2 else hi[1], lo[2] if i & 4 else hi[2], 1.0] for i in range(8)])
    world = corners @ world_mat.T
    w = world[:, :3]
    return (w.min(0), w.max(0))

def _footprint_thor(aabb):
    if aabb is None:
        return None
    lo, hi = aabb
    cx = (lo[0] + hi[0]) / 2.0
    cz = (lo[1] + hi[1]) / 2.0
    sx = float(hi[0] - lo[0])
    sz = float(hi[1] - lo[1])
    return (float(cx), float(cz), sx, sz)

def _humanise(room_id: str) -> str:
    return room_id.replace('_', ' ').strip()

@dataclass
class SceneFacts:
    root: str
    instances: list = field(default_factory=list)
    footprints: dict = field(default_factory=dict)
    centroids: dict = field(default_factory=dict)
    aabbs: dict = field(default_factory=dict)
    room_graph: dict = field(default_factory=dict)
    room_types: dict = field(default_factory=dict)
    room_names: dict = field(default_factory=dict)
    room_area: dict = field(default_factory=dict)
    room_bounds: dict = field(default_factory=dict)
    by_alias: dict = field(default_factory=dict)
    placement_relations: dict = field(default_factory=dict)

    def room_labels(self) -> tuple:
        seen, out = (set(), [])
        for rid in self.room_types:
            lbl = self.room_types[rid]
            if lbl not in seen:
                seen.add(lbl)
                out.append(lbl)
        return tuple(out)

    def room_localization_point(self, x: float, z: float, snap_tol: float=0.25):
        best_rid, best_margin = (None, -float('inf'))
        for rid, (x0, z0, x1, z1) in self.room_bounds.items():
            inset_x = min(x - x0, x1 - x)
            inset_z = min(z - z0, z1 - z)
            if inset_x >= 0.0 and inset_z >= 0.0:
                margin = min(inset_x, inset_z)
            else:
                ox = max(x0 - x, x - x1, 0.0)
                oz = max(z0 - z, z - z1, 0.0)
                margin = -(ox * ox + oz * oz) ** 0.5
            if margin > best_margin or (margin == best_margin and best_rid is not None and (rid < best_rid)):
                best_margin, best_rid = (margin, rid)
        if best_margin < -snap_tol:
            return None
        return best_rid

def _find_house_state(scene_root: str) -> str:
    try:
        combined, _dmd = _find_combined(scene_root)
        if combined:
            cand = os.path.join(combined, 'house_state.json')
            if os.path.isfile(cand):
                return cand
            bak = cand + '.bak'
            if os.path.isfile(bak):
                return bak
    except Exception:
        pass
    sidecar_root = os.environ.get('ROAM_BENCH_HOUSE_STATE_SIDECAR_ROOT')
    if sidecar_root:
        cand = os.path.join(sidecar_root, os.path.basename(os.path.abspath(scene_root)), 'house_state.json')
        if os.path.isfile(cand):
            return cand
    for dp, _dn, fn in os.walk(scene_root):
        for f in fn:
            if f in ('house_state.json', 'house_state.json.bak'):
                return os.path.join(dp, f)
    raise FileNotFoundError(f'no house_state.json under {scene_root}')

def _room_offsets(scene_root: str) -> dict:
    import yaml
    try:
        _combined, dmd = _find_combined(scene_root)
        with open(dmd) as f:
            doc = yaml.load(f, Loader=_DrakeLoader)
    except Exception:
        return {}
    offs = {}
    for d in doc.get('directives', []):
        fr = d.get('add_frame')
        if not fr:
            continue
        name = fr.get('name', '')
        if name.startswith('room_') and name.endswith('_frame'):
            rid = name[len('room_'):-len('_frame')]
            t = (fr.get('X_PF', {}) or {}).get('translation', [0, 0, 0])
            offs[rid] = np.asarray(t, float)
    return offs

def _rests_on(alias, aabb, others, floor_tol=0.06):
    if aabb is None:
        return ()
    lo, hi = aabb
    cx, cy = ((lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0)
    bottom = lo[2]
    out = []
    for b_alias, b_aabb in others:
        if b_alias == alias or b_aabb is None:
            continue
        blo, bhi = b_aabb
        top = bhi[2]
        if abs(bottom - top) > floor_tol:
            continue
        if blo[0] - 1e-06 <= cx <= bhi[0] + 1e-06 and blo[1] - 1e-06 <= cy <= bhi[1] + 1e-06:
            out.append((b_alias, top))
    out.sort(key=lambda p: -p[1])
    return tuple((a for a, _ in out[:1]))

def _contained_in(alias, aabb, containers):
    if aabb is None:
        return ()
    lo, hi = aabb
    centre = (lo + hi) / 2.0
    out = []
    for host, host_aabb in containers:
        if host == alias or host_aabb is None:
            continue
        hlo, hhi = host_aabb
        if np.all(centre > hlo + 0.0001) and np.all(centre < hhi - 0.0001):
            out.append(host)
    return tuple(sorted(out))

def _explicit_placement_relation(alias: str, room_id: str, obj: dict, surface_hosts: dict) -> dict:
    placement = obj.get('placement_info') or {}
    parent_surface_id = str(placement.get('parent_surface_id', ''))
    method = str(placement.get('placement_method', ''))
    matches = surface_hosts.get((room_id, parent_surface_id), ())
    host = matches[0] if len(matches) == 1 else {}
    explicit_inside = method in {'inside_placement', 'container_placement'}
    return {'child': alias, 'room_id': room_id, 'parent_surface_id': parent_surface_id or None, 'placement_method': method or None, 'host': host.get('host'), 'link_name': host.get('link_name'), 'relation': 'inside' if explicit_inside and host.get('host') else 'on' if method in {'surface_placement', 'stack_placement', 'pile_placement', 'create_arrangement'} and host.get('host') else 'unknown', 'ambiguous_parent_surface': len(matches) > 1}

def load_facts(scene_root: str) -> SceneFacts:
    path = _find_house_state(scene_root)
    hs = json.load(open(path))
    layout = hs.get('layout', {})
    rooms = hs.get('rooms', {})
    facts = SceneFacts(root=scene_root)
    offsets = _room_offsets(scene_root)
    placed = {pr['room_id']: pr for pr in layout.get('placed_rooms', []) if pr.get('room_id')}
    for r in layout.get('rooms', []):
        rid = r['id']
        facts.room_types[rid] = r.get('type', rid)
        facts.room_names[rid] = _humanise(rid)
        facts.room_graph[rid] = dict(r.get('connections', {}))
        pr = placed.get(rid)
        if pr and pr.get('width') and pr.get('depth'):
            ex, ez = (float(pr['width']), float(pr['depth']))
        else:
            w, ln = (r.get('width'), r.get('length'))
            ex, ez = (float(w), float(ln)) if w and ln else (None, None)
        if ex and ez:
            facts.room_area[rid] = ex * ez
            off = offsets.get(rid, np.zeros(3))
            facts.room_bounds[rid] = (off[0] - ex / 2.0, off[1] - ez / 2.0, off[0] + ex / 2.0, off[1] + ez / 2.0)
    aabbs = {}
    room_localization = {}
    raw = []
    surface_hosts: dict[tuple[str, str], list[dict]] = {}
    for rid, rm in rooms.items():
        off = offsets.get(rid, np.zeros(3))
        for _oname, o in (rm.get('objects') or {}).items():
            otype = o.get('object_type')
            if otype in _STRUCTURAL:
                continue
            oid = o.get('object_id') or _oname
            alias = f'{rid}_{oid}'
            world_mat = _obj_world_matrix(o.get('transform'), off)
            aabb = _world_aabb_drake(o.get('bbox_min'), o.get('bbox_max'), world_mat)
            aabbs[alias] = aabb
            facts.aabbs[alias] = aabb
            room_localization[alias] = rid
            raw.append((alias, rid, o, aabb))
            for surface in o.get('support_surfaces') or ():
                surface_id = str(surface.get('surface_id', ''))
                if surface_id:
                    surface_hosts.setdefault((rid, surface_id), []).append({'host': alias, 'link_name': surface.get('link_name')})
    by_room = {}
    for a, ab in aabbs.items():
        by_room.setdefault(room_localization[a], []).append((a, ab))
    for alias, rid, o, aabb in raw:
        fp = _footprint_thor(aabb)
        facts.footprints[alias] = fp
        if aabb is not None:
            lo, hi = aabb
            facts.centroids[alias] = (float((lo[0] + hi[0]) / 2.0), float((lo[2] + hi[2]) / 2.0), float((lo[1] + hi[1]) / 2.0))
        inst = _instance_facts(alias=alias, kind=o.get('name') or o.get('object_type') or 'object', room=facts.room_types.get(rid, rid), room_id=rid, on=_rests_on(alias, aabb, by_room.get(rid, ())), pickupable=o.get('object_type') == 'manipuland', openable=False, toggleable=False, sliceable=False, breakable=False, receptacle=bool(o.get('support_surfaces')), is_open=None, is_toggled=None, footprint=fp[2] * fp[3] if fp else None, look=o.get('geometry_path') or None)
        facts.instances.append(inst)
        facts.by_alias[alias] = inst
        from .relation_rules import reviewed_scene_relation
        facts.placement_relations[alias] = reviewed_scene_relation(_explicit_placement_relation(alias, rid, o, surface_hosts), scene_root, path)
    try:
        from .sdf_joints import parse_scene_joints
        jointed = {alias for alias, _sdf, specs in parse_scene_joints(scene_root) if specs}
    except Exception:
        jointed = set()
    containers_by_room = {}
    for host in jointed:
        if host in aabbs:
            containers_by_room.setdefault(room_localization.get(host), []).append((host, aabbs[host]))
    for alias, inst in list(facts.by_alias.items()):
        enclosed = _contained_in(alias, aabbs.get(alias), containers_by_room.get(room_localization.get(alias), ())) if inst.pickupable else ()
        updated = replace(inst, openable=alias in jointed, is_open=False if alias in jointed else inst.is_open, inside=enclosed)
        facts.by_alias[alias] = updated
        facts.instances[facts.instances.index(inst)] = updated
    return facts
