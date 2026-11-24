"""Load SceneSmith directives and transform referenced scene geometry."""
from __future__ import annotations
import os
import re
import math
from dataclasses import dataclass, field
from typing import Optional
import xml.etree.ElementTree as ET
import numpy as np
import yaml

class _AngleAxis:

    def __init__(self, angle_deg: float, axis):
        self.angle_deg = float(angle_deg)
        self.axis = [float(a) for a in axis]

def _angleaxis_constructor(loader, node):
    m = loader.construct_mapping(node, deep=True)
    return _AngleAxis(m['angle_deg'], m['axis'])

class _DrakeLoader(yaml.SafeLoader):
    pass
_DrakeLoader.add_constructor('!AngleAxis', _angleaxis_constructor)
_DrakeLoader.add_constructor('!Rpy', lambda l, n: ('Rpy', l.construct_mapping(n)))

def _translate(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = t
    return m

def _rot_axis_angle(axis, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(4)
    a = a / n
    x, y, z = a
    c, s = (math.cos(angle_rad), math.sin(angle_rad))
    C = 1 - c
    r = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    m = np.eye(4)
    m[:3, :3] = r
    return m

def _rot_rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = (math.cos(r), math.sin(r))
    cp, sp = (math.cos(p), math.sin(p))
    cy, sy = (math.cos(y), math.sin(y))
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    rot = rz @ ry @ rx
    m = np.eye(4)
    m[:3, :3] = rot
    return m

def _rot_quat_wxyz(q) -> np.ndarray:
    w, x, y, z = (float(c) for c in q)
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = (w / n, x / n, y / n, z / n)
    m = np.eye(4)
    m[:3, :3] = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return m
GLTF_YUP_TO_ZUP = _rot_axis_angle([1, 0, 0], math.pi / 2)

@dataclass
class MeshInstance:
    name: str
    room: str
    kind: str
    gltf_path: str
    transform: np.ndarray
    model_world: Optional[np.ndarray] = None

@dataclass
class Scene:
    root: str
    wall_height: float = 3.4
    instances: list = field(default_factory=list)
    rooms: dict = field(default_factory=dict)
    exterior_seals: tuple = field(default_factory=tuple)
    ceiling_slabs: tuple = field(default_factory=tuple)
    light_anchors: tuple = field(default_factory=tuple)

def _resolve_package(uri: str, scene_root: str) -> str:
    m = re.match('package://[^/]+/(.*)', uri)
    if m:
        return os.path.normpath(os.path.join(scene_root, m.group(1)))
    return uri

def _parse_sdf_meshes(sdf_path: str):
    out = []
    sdf_dir = os.path.dirname(sdf_path)
    try:
        tree = ET.parse(sdf_path)
    except Exception as e:
        print(f'[scene_loader] WARN cannot parse {sdf_path}: {e}')
        return out
    root = tree.getroot()
    for link in root.iter('link'):
        link_pose = _parse_pose(link.find('pose'))
        for visual in link.findall('visual'):
            mesh = visual.find('./geometry/mesh')
            if mesh is None:
                continue
            uri_el = mesh.find('uri')
            if uri_el is None or not uri_el.text:
                continue
            uri = uri_el.text.strip()
            if not uri.lower().endswith(('.gltf', '.glb')):
                continue
            gltf_abs = os.path.normpath(os.path.join(sdf_dir, uri))
            vpose = _parse_pose(visual.find('pose'))
            scale_el = mesh.find('scale')
            local = link_pose @ vpose
            if scale_el is not None and scale_el.text:
                sx, sy, sz = [float(v) for v in scale_el.text.split()]
                S = np.diag([sx, sy, sz, 1.0])
                local = local @ S
            out.append((gltf_abs, local))
    return out

@dataclass
class CollisionPiece:
    obj_path: str
    link: str
    transform: np.ndarray

def _parse_sdf_collisions(sdf_path: str):
    out = []
    sdf_dir = os.path.dirname(sdf_path)
    try:
        tree = ET.parse(sdf_path)
    except Exception as e:
        print(f'[scene_loader] WARN cannot parse {sdf_path}: {e}')
        return out
    root = tree.getroot()
    for link in root.iter('link'):
        link_name = link.get('name') or ''
        link_pose = _parse_pose(link.find('pose'))
        for collision in link.findall('collision'):
            mesh = collision.find('./geometry/mesh')
            if mesh is None:
                continue
            uri_el = mesh.find('uri')
            if uri_el is None or not uri_el.text:
                continue
            uri = uri_el.text.strip()
            obj_abs = os.path.normpath(os.path.join(sdf_dir, uri))
            cpose = _parse_pose(collision.find('pose'))
            scale_el = mesh.find('scale')
            local = link_pose @ cpose
            if scale_el is not None and scale_el.text:
                sx, sy, sz = [float(v) for v in scale_el.text.split()]
                S = np.diag([sx, sy, sz, 1.0])
                local = local @ S
            out.append(CollisionPiece(obj_path=obj_abs, link=link_name, transform=local))
    return out

def _parse_pose(pose_el) -> np.ndarray:
    if pose_el is None or not (pose_el.text and pose_el.text.strip()):
        return np.eye(4)
    vals = [float(v) for v in pose_el.text.split()]
    x, y, z = vals[0:3]
    r, p, yaw = (vals[3:6] + [0, 0, 0])[:3]
    return _translate([x, y, z]) @ _rot_rpy(r, p, yaw)
_COMBINED_DIRS = ('combined_house_after_furniture', 'combined_house')
_DMD_NAMES = ('house_furniture_welded.dmd.yaml', 'house.dmd.yaml')

def _find_combined(scene_root: str):
    for sub in _COMBINED_DIRS:
        d = os.path.join(scene_root, sub)
        if not os.path.isdir(d):
            continue
        for name in _DMD_NAMES:
            dmd = os.path.join(d, name)
            if os.path.exists(dmd):
                return (d, dmd)
    raise FileNotFoundError(f'no Drake dmd.yaml under {scene_root} (looked in {_COMBINED_DIRS} for {_DMD_NAMES})')
_BACKFILL_TYPES = ('manipuland', 'wall_mounted', 'ceiling_mounted')

def _backfill_placements(o: dict):
    md = o.get('metadata') or {}
    ct = md.get('composite_type')
    if ct == 'filled_container' and md.get('container_asset'):
        subs = [md['container_asset']] + list(md.get('fill_assets') or [])
        return [(s.get('sdf_path'), s.get('transform')) for s in subs]
    if ct in ('stack', 'pile'):
        return [(s.get('sdf_path'), s.get('transform')) for s in md.get('member_assets') or []]
    return [(o.get('sdf_path'), o.get('transform'))]

def _canonicalize_composite_members(scene, house_state):
    from collections import defaultdict
    owners = defaultdict(set)
    for room, rm in (house_state or {}).get('rooms', {}).items():
        for key, obj in (rm.get('objects') or {}).items():
            parent = room + '_' + str(obj.get('object_id') or key)
            metadata = obj.get('metadata') or {}
            if metadata.get('composite_type') not in ('filled_container', 'stack', 'pile'):
                continue
            for member in metadata.get('member_model_names') or ():
                owners[member].add(parent)
    index = defaultdict(list)
    for instance in scene.instances:
        index[instance.name, os.path.realpath(instance.gltf_path)].append(instance)
    kept, events = ([], [])
    for instance in scene.instances:
        parents = owners.get(instance.name, set())
        if len(parents) != 1:
            kept.append(instance)
            if len(parents) > 1:
                events.append(dict(member=instance.name, status='ambiguous_parent_preserved'))
            continue
        parent = next(iter(parents))
        key = (parent, os.path.realpath(instance.gltf_path))
        duplicate = any((np.allclose(instance.transform, other.transform, rtol=0, atol=1e-08) for other in index.get(key, [])))
        events.append(dict(member=instance.name, parent=parent, mesh=instance.gltf_path, status='explicit_member_duplicate_removed' if duplicate else 'explicit_member_realiased'))
        if not duplicate:
            instance.name = parent
            kept.append(instance)
    scene.instances = kept
    scene.composite_identity_events = events

def _resolve_sdf(sdf, scene_root: str, room: str) -> Optional[str]:
    if not sdf:
        return None
    if os.path.isabs(sdf):
        return sdf if os.path.isfile(sdf) else None
    for base in (os.path.join(scene_root, f'room_{room}'), scene_root):
        cand = os.path.normpath(os.path.join(base, sdf))
        if os.path.isfile(cand):
            return cand
    marker = f'{os.path.basename(os.path.normpath(scene_root))}/room_{room}/'
    if marker in sdf:
        tail = sdf.split(marker, 1)[1]
        root = os.path.realpath(os.path.join(scene_root, f'room_{room}'))
        cand = os.path.realpath(os.path.join(root, tail))
        if os.path.commonpath((root, cand)) == root and os.path.isfile(cand):
            return cand
    return None

def load_scene(scene_root: str) -> Scene:
    combined_dir, dmd = _find_combined(scene_root)
    with open(dmd) as f:
        doc = yaml.load(f, Loader=_DrakeLoader)
    directives = doc['directives']
    frames = {'world': np.eye(4)}
    frame_parent = {}
    raw_frames = {}
    for d in directives:
        if 'add_frame' in d:
            fr = d['add_frame']
            name = fr['name']
            xpf = fr['X_PF']
            base = xpf['base_frame']
            t = xpf.get('translation', [0, 0, 0])
            raw_frames[name] = (base, t)

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
    scene = Scene(root=scene_root)
    import json
    house_state = None
    for cand in ('house_state.json', 'house_state.json.bak'):
        try:
            with open(os.path.join(combined_dir, cand)) as f:
                house_state = json.load(f)
            scene.wall_height = (house_state.get('layout') or {}).get('wall_height', scene.wall_height)
            break
        except Exception:
            house_state = None
            continue

    def _pose_matrix(translation, rotation) -> np.ndarray:
        m = _translate(translation or [0, 0, 0])
        if isinstance(rotation, _AngleAxis):
            m = m @ _rot_axis_angle(rotation.axis, math.radians(rotation.angle_deg))
        return m

    def _emit(model_name: str, parent_frame: str, link_pose: np.ndarray, kind: Optional[str]=None):
        if model_name not in models:
            print(f'[scene_loader] WARN model {model_name} referenced but not added')
            return
        parent_world = frame_world(parent_frame)
        link_world = parent_world @ link_pose
        room = parent_frame[:-6] if parent_frame.endswith('_frame') else parent_frame
        room = room[5:] if room.startswith('room_') else room
        if kind is None:
            kind = 'geometry' if model_name.startswith('room_geometry') else 'furniture'
        for gltf_abs, local in _parse_sdf_meshes(models[model_name]):
            world = link_world @ local @ GLTF_YUP_TO_ZUP
            scene.instances.append(MeshInstance(name=model_name, room=room, kind=kind, gltf_path=gltf_abs, transform=world, model_world=link_world))
    for d in directives:
        if 'add_weld' not in d:
            continue
        w = d['add_weld']
        parent = w['parent']
        model_name = w['child'].split('::')[0]
        xpc = w.get('X_PC', {}) or {}
        _emit(model_name, parent, _pose_matrix(xpc.get('translation'), xpc.get('rotation')))
    for d in directives:
        if 'add_model' not in d:
            continue
        am = d['add_model']
        dfbp = am.get('default_free_body_pose')
        if not dfbp:
            continue
        for _link, pose in dfbp.items():
            base = pose.get('base_frame', 'world')
            _emit(am['name'], base, _pose_matrix(pose.get('translation'), pose.get('rotation')), kind='object')
    for rid, rm in (house_state or {}).get('rooms', {}).items():
        rframe = f'room_{rid}_frame'
        room_mat = frame_world(rframe) if rframe in raw_frames else np.eye(4)
        for _oname, o in (rm.get('objects') or {}).items():
            if o.get('object_type') not in _BACKFILL_TYPES:
                continue
            alias = f"{rid}_{o.get('object_id') or _oname}"
            if alias in models:
                continue
            for sdf, tf in _backfill_placements(o):
                sdf_abs = _resolve_sdf(sdf, scene_root, rid)
                if sdf_abs is None:
                    print(f'[scene_loader] WARN backfill {alias}: no sdf for {sdf!r}')
                    continue
                t = tf or {}
                local_obj = _rot_quat_wxyz(t.get('rotation_wxyz', [1, 0, 0, 0]))
                local_obj[:3, 3] = np.asarray(t.get('translation', [0, 0, 0]), float)
                world_mat = room_mat @ local_obj
                for gltf_abs, local in _parse_sdf_meshes(sdf_abs):
                    world = world_mat @ local @ GLTF_YUP_TO_ZUP
                    scene.instances.append(MeshInstance(name=alias, room=rid, kind=o['object_type'], gltf_path=gltf_abs, transform=world, model_world=world_mat))
    _canonicalize_composite_members(scene, house_state)
    from .exterior_seals import seals_from_layout
    layout = (house_state or {}).get('layout') or {}
    scene.exterior_seals = seals_from_layout(layout)
    from .ceiling_lighting import build_ceiling_plan
    scene.ceiling_slabs, scene.light_anchors = build_ceiling_plan(layout, (house_state or {}).get('rooms') or {})
    return scene
if __name__ == '__main__':
    import sys
    from .paths import DEFAULT_SCENE
    root = sys.argv[1] if len(sys.argv) > 1 else str(DEFAULT_SCENE)
    sc = load_scene(root)
    print(f'wall_height={sc.wall_height}')
    print(f'instances={len(sc.instances)}')
    import collections
    print('by kind:', dict(collections.Counter((i.kind for i in sc.instances))))
    missing = [i for i in sc.instances if not os.path.exists(i.gltf_path)]
    print(f'missing gltf files={len(missing)}')
    for i in sc.instances[:8]:
        p = i.transform[:3, 3]
        print(f'  {i.kind:9s} {i.room:16s} {os.path.basename(i.gltf_path):24s} pos=({p[0]:+.2f},{p[1]:+.2f},{p[2]:+.2f})')
    if sc.instances:
        pts = np.array([i.transform[:3, 3] for i in sc.instances])
        print('bounds min', pts.min(0).round(2), 'max', pts.max(0).round(2))
