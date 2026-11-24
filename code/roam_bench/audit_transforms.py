"""Audit transforms utilities for ROAM-Bench."""
from __future__ import annotations
import argparse
import base64
import json
import os
import struct
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from roam_bench.scene_loader import load_scene, MeshInstance
from roam_bench.paths import DEFAULT_SCENE, PROJECT_ROOT
THOR_JSON = os.environ.get('ROAM_BENCH_THOR_SCENE_JSON', str(PROJECT_ROOT.parent / 'ai2thor' / 'thor-eval' / 'output' / 'scene.json'))
_COMPTYPE = {5120: ('b', 1), 5121: ('B', 1), 5122: ('h', 2), 5123: ('H', 2), 5125: ('I', 4), 5126: ('f', 4)}
_NCOMP = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT2': 4, 'MAT3': 9, 'MAT4': 16}

def _node_matrix(node) -> np.ndarray:
    if 'matrix' in node:
        m = np.array(node['matrix'], dtype=float).reshape(4, 4).T
        return m
    m = np.eye(4)
    if 'scale' in node:
        m = m @ np.diag(list(node['scale']) + [1.0])
    if 'rotation' in node:
        x, y, z, w = node['rotation']
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        r4 = np.eye(4)
        r4[:3, :3] = R
        m = r4 @ m
    if 'translation' in node:
        t = np.eye(4)
        t[:3, 3] = node['translation']
        m = t @ m
    return m

def _accessor_minmax(gltf, acc_idx):
    acc = gltf['accessors'][acc_idx]
    if 'min' in acc and 'max' in acc:
        return (np.array(acc['min'][:3], float), np.array(acc['max'][:3], float))
    return (None, None)

def gltf_local_aabb(gltf_path):
    with open(gltf_path) as f:
        g = json.load(f)
    nodes = g.get('nodes', [])
    meshes = g.get('meshes', [])
    scene_idx = g.get('scene', 0)
    roots = g.get('scenes', [{}])[scene_idx].get('nodes', list(range(len(nodes))))
    corners = []

    def walk(ni, parent):
        node = nodes[ni]
        world = parent @ _node_matrix(node)
        if 'mesh' in node:
            for prim in meshes[node['mesh']].get('primitives', []):
                pos = prim.get('attributes', {}).get('POSITION')
                if pos is None:
                    continue
                lo, hi = _accessor_minmax(g, pos)
                if lo is None:
                    continue
                for cx in (lo[0], hi[0]):
                    for cy in (lo[1], hi[1]):
                        for cz in (lo[2], hi[2]):
                            v = world @ np.array([cx, cy, cz, 1.0])
                            corners.append(v[:3])
        for c in node.get('children', []):
            walk(c, world)
    for r in roots:
        walk(r, np.eye(4))
    if not corners:
        return None
    c = np.array(corners)
    return (c.min(0), c.max(0))

def world_aabb_for_instance(inst: MeshInstance):
    la = gltf_local_aabb(inst.gltf_path)
    if la is None:
        return None
    lo, hi = la
    T = inst.transform
    pts = []
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                v = T @ np.array([cx, cy, cz, 1.0])
                pts.append(v[:3])
    pts = np.array(pts)
    return (pts.min(0), pts.max(0))

def object_world_boxes(scene, kinds=('geometry', 'furniture', 'object')):
    from collections import defaultdict
    groups = defaultdict(list)
    for inst in scene.instances:
        if inst.kind in kinds:
            groups[inst.room, inst.name].append(inst)
    boxes = {}
    for key, insts in groups.items():
        los, his = ([], [])
        for inst in insts:
            wb = world_aabb_for_instance(inst)
            if wb is None:
                continue
            los.append(wb[0])
            his.append(wb[1])
        if not los:
            continue
        lo = np.min(los, 0)
        hi = np.max(his, 0)
        boxes[key] = {'lo': lo, 'hi': hi, 'center': (lo + hi) / 2, 'size': hi - lo, 'n_mesh': len(insts)}
    return boxes

def room_boxes_from_geometry(scene):
    from collections import defaultdict
    g = defaultdict(list)
    for inst in scene.instances:
        if inst.kind == 'geometry':
            wb = world_aabb_for_instance(inst)
            if wb is not None:
                g[inst.room].append(wb)
    out = {}
    for room, boxes in g.items():
        lo = np.min([b[0] for b in boxes], 0)
        hi = np.max([b[1] for b in boxes], 0)
        out[room] = (lo, hi)
    return out
DIRECTION_FLOOR = 0.25
_FIXED_OFFSET_DIR = np.array([0.6, -0.9])

def closeup_camera(center, size, room_aabb, elev=0.6, wall_margin=0.35, clearance=0.4):
    c = np.asarray(center, float)
    sz = np.asarray(size, float)
    lo = np.asarray(room_aabb[0], float)
    hi = np.asarray(room_aabb[1], float)
    span = float(np.max(sz)) or 1.0
    want = span * 2.2 + 0.9
    rc = (lo + hi) / 2.0
    d = rc[:2] - c[:2]
    n = float(np.linalg.norm(d))
    if n < DIRECTION_FLOOR:
        d = _FIXED_OFFSET_DIR / float(np.linalg.norm(_FIXED_OFFSET_DIR))
    else:
        d = d / n
    min_off = 0.5 * float(np.hypot(sz[0], sz[1])) + clearance
    t = max(want, min_off)
    eye_xy = c[:2] + d * t
    span_xy = hi[:2] - lo[:2]
    eff = np.minimum(wall_margin, np.maximum(0.5 * span_xy - 0.001, 0.0))
    eye_xy = np.clip(eye_xy, lo[:2] + eff, hi[:2] - eff)
    eye_z = c[2] + span * elev + 0.7
    eye_z = min(eye_z, hi[2] - wall_margin)
    eye_z = max(eye_z, c[2] + 0.2)
    eye_z = min(max(eye_z, float(lo[2])), float(hi[2]))
    return (np.array([eye_xy[0], eye_xy[1], eye_z]), c)

def load_thor_objects(path=THOR_JSON):
    d = json.load(open(path))
    out = {}

    def walk(objs):
        for ob in objs:
            oid = ob.get('id', '')
            p = ob.get('position', {})
            out[oid] = np.array([p.get('x', 0), p.get('y', 0), p.get('z', 0)])
            if ob.get('children'):
                walk(ob['children'])
    walk(d['objects'])
    return out

def norm_name(room, name):
    if name.startswith(room + '_'):
        base = name[len(room) + 1:]
    else:
        base = name
    return f'{room}|{base}'

def kabsch_similarity(src, dst):
    src = np.asarray(src)
    dst = np.asarray(dst)
    cs = src.mean(0)
    cd = dst.mean(0)
    S = src - cs
    D = dst - cd
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Dm = np.diag([1, 1, d])
    R = Vt.T @ Dm @ U.T
    t = cd - R @ cs
    pred = (R @ src.T).T + t
    per = np.linalg.norm(pred - dst, axis=1)
    return (R, t, float(np.sqrt((per ** 2).mean())), per)

def fit_2d_rigid(src, dst):
    src = np.asarray(src)
    dst = np.asarray(dst)
    cs = src.mean(0)
    cd = dst.mean(0)
    S = src - cs
    D = dst - cd
    H = S.T @ D
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, d]) @ U.T
    t = cd - R @ cs
    pred = (R @ src.T).T + t
    per = np.linalg.norm(pred - dst, axis=1)
    angle = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return (R, t, angle, per)

def per_room_analysis(matched):
    from collections import defaultdict
    by_room = defaultdict(list)
    for nn, dc, tp in matched:
        room = nn.split('|')[0]
        by_room[room].append((nn, dc, tp))
    print('\n=== PER-ROOM intra-room consistency (axis relabel dx->x, dy->z, dz->y) ===')
    print(f"{'room':18s} {'n':>3s} {'horiz_RMSE':>10s} {'horiz_max':>9s} {'rot_deg':>8s} {'height_RMSE':>11s}")
    all_h = []
    for room in sorted(by_room):
        rows = by_room[room]
        if len(rows) < 2:
            print(f'{room:18s} {len(rows):3d}   (too few to fit)')
            continue
        src = np.array([[dc[0], dc[1]] for _, dc, _ in rows])
        dst = np.array([[tp[0], tp[2]] for _, _, tp in rows])
        R, t, ang, per = fit_2d_rigid(src, dst)
        hgt_src = np.array([dc[2] for _, dc, _ in rows])
        hgt_dst = np.array([tp[1] for _, _, tp in rows])
        hoff = np.mean(hgt_dst - hgt_src)
        hrmse = np.sqrt(np.mean((hgt_dst - hgt_src - hoff) ** 2))
        all_h.extend(per.tolist())
        print(f'{room:18s} {len(rows):3d} {np.sqrt((per ** 2).mean()):10.4f} {per.max():9.4f} {ang:8.2f} {hrmse:11.4f}')
    all_h = np.array(all_h)
    if len(all_h):
        print(f'\n  all intra-room horizontal residuals: median={np.median(all_h):.4f}m mean={all_h.mean():.4f}m max={all_h.max():.4f}m')
        print('  (small values here => objects correctly placed *within* each room;')
        print('   room-to-room offsets differ between importers, which is expected)')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', default=str(DEFAULT_SCENE))
    ap.add_argument('--thor-json', default=THOR_JSON, help='AI2-THOR ss5_house.json used for the cross-check')
    ap.add_argument('--seed', type=int, default=1234)
    ap.add_argument('--n', type=int, default=18)
    args = ap.parse_args()
    print(f'=== loading scene {args.scene}')
    scene = load_scene(args.scene)
    print(f'instances={len(scene.instances)}')
    boxes = object_world_boxes(scene)
    print(f'object groups (by room|name)={len(boxes)}')
    thor = load_thor_objects(args.thor_json)
    print(f'THOR objects={len(thor)}')
    matched = []
    unmatched_drake = []
    for (room, name), b in boxes.items():
        nn = norm_name(room, name)
        if nn in thor:
            matched.append((nn, b['center'], thor[nn]))
        else:
            unmatched_drake.append(nn)
    print(f'matched objects (name in both)={len(matched)}')
    if len(matched) >= 4:
        src = np.array([m[1] for m in matched])
        dst = np.array([m[2] for m in matched])
        R, t, rmse, per = kabsch_similarity(src, dst)
        print(f'\n=== best-fit Drake->THOR (rigid) ===')
        print('R=\n', np.round(R, 3))
        print('t=', np.round(t, 3))
        print(f'global RMSE={rmse:.4f} m  median_resid={np.median(per):.4f}  max={per.max():.4f}')
        order = np.argsort(per)[::-1]
        print('\n--- worst 15 residuals (name: resid_m  drake_center  thor_pos) ---')
        for i in order[:15]:
            nn, dc, tp = matched[i]
            print(f'  {per[i]:7.3f}  {nn:48s} drake={np.round(dc, 2)} thor={np.round(tp, 2)}')
        print(f'\n--- residual histogram ---')
        for lo, hi in [(0, 0.02), (0.02, 0.05), (0.05, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1), (1, 99)]:
            c = ((per >= lo) & (per < hi)).sum()
            print(f'  [{lo:5.2f},{hi:5.2f})m : {c}')
        per_room_analysis(matched)
    rng = np.random.default_rng(args.seed)
    keys = sorted(boxes.keys())
    pick = rng.choice(len(keys), size=min(args.n, len(keys)), replace=False)
    print(f'\n=== random per-object detail (seed={args.seed}, n={len(pick)}) ===')
    print(f"{'room|name':46s} {'n':>2s} {'center(X,Y,Z Zup)':>22s} {'size(m)':>20s}")
    for i in sorted(pick):
        room, name = keys[i]
        b = boxes[room, name]
        c = b['center']
        s = b['size']
        print(f"{norm_name(room, name):46s} {b['n_mesh']:2d} ({c[0]:+6.2f},{c[1]:+6.2f},{c[2]:+6.2f}) ({s[0]:5.2f},{s[1]:5.2f},{s[2]:5.2f})")
if __name__ == '__main__':
    main()
