"""Artic open assemble utilities for ROAM-Bench."""
from __future__ import annotations
import math
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
import numpy as np
from roam_bench.artic_open_check import Hull, MovingLink, Obstacle, hull_from_verts
from roam_bench.joint_usd_map import map_scene_joints
from roam_bench.sdf_link_meshes import parse_link_collisions
from roam_bench.scene_loader import _AngleAxis, _DrakeLoader, _find_combined, _parse_pose, _resolve_package, _rot_axis_angle, _translate

def _pose_matrix(translation, rotation) -> np.ndarray:
    m = _translate(translation or [0, 0, 0])
    if isinstance(rotation, _AngleAxis):
        m = m @ _rot_axis_angle(rotation.axis, math.radians(rotation.angle_deg))
    return m

def scene_welds(scene_root: str) -> Dict[str, dict]:
    import yaml
    _, dmd = _find_combined(scene_root)
    doc = yaml.load(open(dmd), Loader=_DrakeLoader)
    directives = doc['directives']
    raw_frames = {}
    for d in directives:
        if 'add_frame' in d:
            fr = d['add_frame']
            xpf = fr['X_PF']
            raw_frames[fr['name']] = (xpf['base_frame'], xpf.get('translation', [0, 0, 0]))

    def frame_world(name):
        if name == 'world':
            return np.eye(4)
        base, t = raw_frames[name]
        return frame_world(base) @ _translate(t)
    models = {}
    for d in directives:
        if 'add_model' in d:
            am = d['add_model']
            models[am['name']] = _resolve_package(am['file'], scene_root)
    welds: Dict[str, dict] = {}
    for d in directives:
        if 'add_weld' not in d:
            continue
        w = d['add_weld']
        parent = w['parent']
        model = w['child'].split('::')[0]
        xpc = w.get('X_PC', {}) or {}
        link_world = frame_world(parent) @ _pose_matrix(xpc.get('translation'), xpc.get('rotation'))
        room = parent[:-6] if parent.endswith('_frame') else parent
        room = room[5:] if room.startswith('room_') else room
        kind = 'geometry' if model.startswith('room_geometry') else 'furniture'
        welds[model] = dict(link_world=link_world, room=room, sdf=models.get(model), kind=kind)
    return welds
_VERTS_CACHE: Dict[str, Optional[np.ndarray]] = {}

def _obj_verts(obj_path: str) -> Optional[np.ndarray]:
    if obj_path in _VERTS_CACHE:
        return _VERTS_CACHE[obj_path]
    import trimesh
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        v = np.asarray(mesh.vertices, float)
    except Exception:
        v = None
    v = v if v is not None and len(v) >= 4 else None
    _VERTS_CACHE[obj_path] = v
    return v

def world_hulls_for_link(sdf: str, link_name: str, link_world: np.ndarray) -> List[Hull]:
    hulls: List[Hull] = []
    for pc in parse_link_collisions(sdf).get(link_name, []):
        v = _obj_verts(pc.obj_path)
        if v is None:
            continue
        vh = np.c_[v, np.ones(len(v))]
        world = (vh @ (link_world @ pc.transform).T)[:, :3]
        h = hull_from_verts(world)
        if h is not None:
            hulls.append(h)
    return hulls

def world_hulls_for_model(sdf: str, link_world: np.ndarray) -> List[Hull]:
    out: List[Hull] = []
    for link_name in parse_link_collisions(sdf).keys():
        out.extend(world_hulls_for_link(sdf, link_name, link_world))
    return out

def wall_hulls(sdf: str, link_world: np.ndarray) -> List[Hull]:
    try:
        root = ET.parse(sdf).getroot()
    except Exception:
        return []
    hulls: List[Hull] = []
    for link in root.iter('link'):
        link_pose = _parse_pose(link.find('pose'))
        for col in link.findall('collision'):
            if 'wall' not in (col.get('name') or '').lower():
                continue
            box = col.find('./geometry/box/size')
            if box is None or not box.text:
                continue
            sx, sy, sz = [float(v) for v in box.text.split()]
            cpose = _parse_pose(col.find('pose'))
            M = link_world @ link_pose @ cpose
            half = np.array([sx, sy, sz]) / 2.0
            signs = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1, 3)
            corners = np.c_[signs * half, np.ones(8)]
            world = (corners @ M.T)[:, :3]
            h = hull_from_verts(world)
            if h is not None:
                hulls.append(h)
    return hulls

def _quat_rot(q, v) -> np.ndarray:
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, float)

def joint_world_axis_anchor(p, link_world: np.ndarray):
    unit = {'X': (1.0, 0, 0), 'Y': (0, 1.0, 0), 'Z': (0, 0, 1.0)}[p.axis]
    axis_world = link_world[:3, :3] @ _quat_rot(p.local_rot_wxyz, unit)
    anchor_world = (link_world @ np.r_[np.asarray(p.local_pos, float), 1.0])[:3]
    return (axis_world, anchor_world)

def authored_target_q(p) -> float:
    return math.radians(p.open_pos) if p.drive == 'angular' else float(p.open_pos)

def build_moving_links(scene_root: str, model: str, *, target_q: Optional[Dict[str, float]]=None) -> Dict[str, MovingLink]:
    target_q = target_q or {}
    welds = scene_welds(scene_root)
    w = welds.get(model)
    if w is None:
        raise KeyError(f'model {model!r} is not welded in {scene_root} (not in scene_welds)')
    lw = w['link_world']
    out: Dict[str, MovingLink] = {}
    for m, spec, p in map_scene_joints(scene_root):
        if m != model:
            continue
        hulls = world_hulls_for_link(spec.sdf_path, spec.child_link, lw)
        axis_w, anchor_w = joint_world_axis_anchor(p, lw)
        tq = float(target_q[spec.name]) if spec.name in target_q else authored_target_q(p)
        out[spec.name] = MovingLink(name=spec.name, jtype=spec.type, anchor=anchor_w, axis=axis_w, target_q=tq, hulls=hulls)
    return out

def build_obstacles(scene_root: str, model: str, *, same_room_only: bool=True) -> List[Obstacle]:
    welds = scene_welds(scene_root)
    w = welds.get(model)
    if w is None:
        raise KeyError(f'model {model!r} is not welded in {scene_root} (not in scene_welds)')
    room = w['room']
    obstacles: List[Obstacle] = []
    for om, ow in welds.items():
        if om == model:
            continue
        if same_room_only and ow['room'] != room:
            continue
        sdf = ow['sdf']
        if not sdf:
            continue
        if ow['kind'] == 'geometry':
            hulls = wall_hulls(sdf, ow['link_world'])
            kind = 'wall'
        else:
            hulls = world_hulls_for_model(sdf, ow['link_world'])
            kind = 'object'
        if hulls:
            obstacles.append(Obstacle(name=om, kind=kind, hulls=hulls))
    return obstacles
