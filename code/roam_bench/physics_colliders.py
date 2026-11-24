"""Physics colliders utilities for ROAM-Bench."""
from __future__ import annotations
import math
import os
from typing import List, Tuple
import numpy as np
import xml.etree.ElementTree as ET
import roam_bench.scene_loader as sl
AGENT_RADIUS = 0.2
AGENT_HEIGHT = 1.8
RENDER_EXCLUDE_METHOD = os.environ.get('ISAAC_PHYS_RENDER_EXCLUDE', 'guide')

def _pose_matrix(translation, rotation) -> np.ndarray:
    m = sl._translate(translation or [0, 0, 0])
    if isinstance(rotation, sl._AngleAxis):
        m = m @ sl._rot_axis_angle(rotation.axis, math.radians(rotation.angle_deg))
    return m

def parse_wall_boxes(scene_root: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    combined_dir, dmd = sl._find_combined(scene_root)
    import yaml
    doc = yaml.load(open(dmd), Loader=sl._DrakeLoader)
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
        return frame_world(base) @ sl._translate(t)
    models = {}
    for d in directives:
        if 'add_model' in d:
            am = d['add_model']
            models[am['name']] = sl._resolve_package(am['file'], scene_root)
    boxes: List[Tuple[np.ndarray, np.ndarray]] = []
    for d in directives:
        if 'add_weld' not in d:
            continue
        w = d['add_weld']
        name = w['child'].split('::')[0]
        if not name.startswith('room_geometry') or name not in models:
            continue
        xpc = w.get('X_PC', {}) or {}
        root_world = frame_world(w['parent']) @ _pose_matrix(xpc.get('translation'), xpc.get('rotation'))
        root = ET.parse(models[name]).getroot()
        for link in root.iter('link'):
            link_pose = sl._parse_pose(link.find('pose'))
            for col in link.findall('collision'):
                size_el = col.find('./geometry/box/size')
                if size_el is None or not size_el.text:
                    continue
                size = np.array([float(v) for v in size_el.text.split()], float)
                cpose = sl._parse_pose(col.find('pose'))
                boxes.append((root_world @ link_pose @ cpose, size))
    from .exterior_seals import load_exterior_seals
    boxes.extend(((seal.transform.copy(), seal.size.copy()) for seal in load_exterior_seals(scene_root)))
    return boxes

def _set_xform_from_matrix(prim, mat4: np.ndarray):
    from pxr import Gf, UsdGeom
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    gm = Gf.Matrix4d(*[float(v) for v in mat4.T.flatten()])
    xf.AddTransformOp().Set(gm)

def _render_exclude(prim, method: str=RENDER_EXCLUDE_METHOD):
    from pxr import UsdGeom
    if method == 'guide':
        UsdGeom.Imageable(prim).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
    elif method == 'invisible':
        UsdGeom.Imageable(prim).MakeInvisible()
    elif method == 'none':
        pass
    else:
        raise ValueError(f'ISAAC_PHYS_RENDER_EXCLUDE={method!r} unknown (guide|invisible|none)')

def add_wall_colliders(stage, scene_root: str, prim_root: str='/World/phys/walls') -> int:
    from pxr import Usd, UsdGeom, UsdPhysics, Gf
    boxes = parse_wall_boxes(scene_root)
    UsdGeom.Xform.Define(stage, prim_root)
    for i, (world, size) in enumerate(boxes):
        path = f'{prim_root}/box_{i}'
        cube = UsdGeom.Cube.Define(stage, path)
        cube.GetSizeAttr().Set(1.0)
        prim = cube.GetPrim()
        _set_xform_from_matrix(prim, world)
        UsdGeom.Xformable(prim).AddScaleOp().Set(Gf.Vec3f(float(size[0]), float(size[1]), float(size[2])))
        UsdPhysics.CollisionAPI.Apply(prim)
        _render_exclude(prim)
    return len(boxes)

def setup_physics_scene(stage, path: str='/World/physicsScene'):
    from pxr import UsdPhysics, Gf
    scene = UsdPhysics.Scene.Define(stage, path)
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81)
    return scene
_WALK_COLLIDER_KINDS = ('furniture',)
_EXCLUDED_KINDS = ('object', 'manipuland', 'wall_mounted', 'ceiling_mounted', 'geometry')

def _read_obj(path: str):
    pts, faces = ([], [])
    with open(path) as fh:
        for line in fh:
            if line.startswith('v '):
                pts.append([float(x) for x in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(tok.split('/')[0]) - 1 for tok in line.split()[1:]])
    return (np.asarray(pts, float), faces)

def _welded_furniture_placements(scene_root: str):
    combined_dir, dmd = sl._find_combined(scene_root)
    import yaml
    doc = yaml.load(open(dmd), Loader=sl._DrakeLoader)
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
        return frame_world(base) @ sl._translate(t)
    models = {}
    for d in directives:
        if 'add_model' in d:
            am = d['add_model']
            models[am['name']] = sl._resolve_package(am['file'], scene_root)
    out = []
    for d in directives:
        if 'add_weld' not in d:
            continue
        w = d['add_weld']
        name = w['child'].split('::')[0]
        if name.startswith('room_geometry') or name not in models:
            continue
        xpc = w.get('X_PC', {}) or {}
        root_world = frame_world(w['parent']) @ _pose_matrix(xpc.get('translation'), xpc.get('rotation'))
        out.append((root_world, models[name], name))
    return out

def add_furniture_colliders(stage, scene_root: str, prim_root: str='/World/phys/furniture', limit=None) -> dict:
    from pxr import UsdGeom, UsdPhysics, Vt
    UsdGeom.Xform.Define(stage, prim_root)
    placements = _welded_furniture_placements(scene_root)
    print(f'[phys] furniture placements (welded, non-geometry) = {len(placements)}Invalid scene geometry. {_EXCLUDED_KINDS}', flush=True)
    n_pieces = 0
    n_place = 0
    for pi, (root_world, sdf, name) in enumerate(placements):
        pieces = sl._parse_sdf_collisions(sdf)
        if not pieces:
            continue
        n_place += 1
        grp = f'{prim_root}/p{pi}'
        UsdGeom.Xform.Define(stage, grp)
        for ci, cp in enumerate(pieces):
            if limit is not None and n_pieces >= limit:
                print(f'[phys] hit limit={limit} at placement {pi}', flush=True)
                return {'placements': n_place, 'pieces': n_pieces, 'excluded_kinds': _EXCLUDED_KINDS, 'hit_limit': True}
            pts, faces = _read_obj(cp.obj_path)
            if len(pts) < 4:
                continue
            world = root_world @ cp.transform
            mesh = UsdGeom.Mesh.Define(stage, f'{grp}/c{ci}')
            prim = mesh.GetPrim()
            _set_xform_from_matrix(prim, world)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts.astype('float32')))
            counts, idx = ([], [])
            for f in faces:
                counts.append(len(f))
                idx.extend(f)
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(idx))
            UsdPhysics.CollisionAPI.Apply(prim)
            mca = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mca.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
            _render_exclude(prim)
            n_pieces += 1
    print(f'[phys] authored furniture: {n_place} placements, {n_pieces} convexHull pieces', flush=True)
    return {'placements': n_place, 'pieces': n_pieces, 'excluded_kinds': _EXCLUDED_KINDS, 'hit_limit': False}
