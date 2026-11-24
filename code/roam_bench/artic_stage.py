"""Artic stage utilities for ROAM-Bench."""
from __future__ import annotations
import math
from typing import Dict, List, Tuple
import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf
from roam_bench.joint_articulation import ArticulationPlan

def _mat_to_gf(m: np.ndarray) -> Gf.Matrix4d:
    return Gf.Matrix4d(*[float(m[i][j]) for i in range(4) for j in range(4)])

def _make_link_body(stage, prim_path: str, visuals: List[Tuple[str, np.ndarray]], *, visual_frame_fix: bool=True):
    xf = UsdGeom.Xform.Define(stage, prim_path)
    prim = xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    m = UsdPhysics.MassAPI.Apply(prim)
    m.CreateMassAttr(1.0)
    m.CreateDiagonalInertiaAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    for i, (gltf_abs, local) in enumerate(visuals):
        child_path = '%s/vis_%d' % (prim_path, i)
        child = UsdGeom.Xform.Define(stage, child_path)
        child.GetPrim().GetReferences().AddReference(gltf_abs)
        vis_local = np.asarray(local, float).T if visual_frame_fix else local
        UsdGeom.Xformable(child).MakeMatrixXform().Set(_mat_to_gf(vis_local))
    return prim

def _attach_link_colliders(stage, prim_path: str, pieces) -> int:
    from pxr import Vt
    from roam_bench.physics_colliders import _read_obj, _render_exclude
    n = 0
    for ci, cp in enumerate(pieces):
        pts, faces = _read_obj(cp.obj_path)
        if len(pts) < 4:
            continue
        mesh = UsdGeom.Mesh.Define(stage, '%s/col_%d' % (prim_path, ci))
        prim = mesh.GetPrim()
        UsdGeom.Xformable(prim).MakeMatrixXform().Set(_mat_to_gf(cp.transform))
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
        n += 1
    return n

def _add_drive(joint_prim, instance: str, target: float):
    d = UsdPhysics.DriveAPI.Apply(joint_prim, instance)
    d.CreateTypeAttr('force')
    d.CreateTargetPositionAttr(float(target))
    d.CreateStiffnessAttr(1000000.0)
    d.CreateDampingAttr(100000.0)
    d.CreateMaxForceAttr(1000000000.0)

def author_articulation(stage, plan: ArticulationPlan, link_visuals: Dict[str, List[Tuple[str, np.ndarray]]], link_collisions=None, *, visual_frame_fix: bool=True):
    root_xf = UsdGeom.Xform.Define(stage, plan.instance_root)
    UsdPhysics.ArticulationRootAPI.Apply(root_xf.GetPrim())
    for lp in plan.links:
        _make_link_body(stage, lp.prim_path, link_visuals.get(lp.link_name, []), visual_frame_fix=visual_frame_fix)
    from roam_bench.artic_colliders import plan_link_colliders
    to_author = plan_link_colliders(plan, link_collisions)
    if to_author:
        n_col = sum((_attach_link_colliders(stage, pp, pieces) for pp, pieces in to_author))
        print('Scene validation: %d %d' % (n_col, len(to_author)), flush=True)
    root_link_path = next((lp.prim_path for lp in plan.links if lp.is_root))
    fj = UsdPhysics.FixedJoint.Define(stage, '%s/base_weld' % plan.instance_root)
    fj.CreateBody1Rel().SetTargets([Sdf.Path(root_link_path)])
    drives = {}
    for jp in plan.joints:
        p = jp.params
        if p.drive == 'linear':
            j = UsdPhysics.PrismaticJoint.Define(stage, jp.joint_prim_path)
            instance = 'linear'
        else:
            j = UsdPhysics.RevoluteJoint.Define(stage, jp.joint_prim_path)
            instance = 'angular'
        j.CreateBody0Rel().SetTargets([Sdf.Path(jp.body0_path)])
        j.CreateBody1Rel().SetTargets([Sdf.Path(jp.body1_path)])
        j.CreateAxisAttr(p.axis)
        j.CreateLowerLimitAttr(float(p.lower))
        j.CreateUpperLimitAttr(float(p.upper))
        lp0 = Gf.Vec3f(*[float(v) for v in p.local_pos])
        lr0 = Gf.Quatf(float(p.local_rot_wxyz[0]), Gf.Vec3f(*[float(v) for v in p.local_rot_wxyz[1:]]))
        j.CreateLocalPos0Attr(lp0)
        j.CreateLocalPos1Attr(lp0)
        j.CreateLocalRot0Attr(lr0)
        j.CreateLocalRot1Attr(lr0)
        _add_drive(j.GetPrim(), instance, p.closed_pos)
        drives[jp.joint_name] = (j.GetPrim(), instance)
    return drives

def set_action(drives, plan: ArticulationPlan, action: str, *, targets_override=None):
    if targets_override is None:
        from roam_bench.joint_interact import plan_interact
        ip = plan_interact(plan, action)
        tgt = {jp.joint_name: t.target for jp, t in zip(plan.joints, ip.targets)}
    else:
        tgt = targets_override
    for name, (prim, instance) in drives.items():
        d = UsdPhysics.DriveAPI.Get(prim, instance)
        d.GetTargetPositionAttr().Set(float(tgt[name]))

def open_with_check(stage, drives, plan: ArticulationPlan, action: str, *, moving_links=None, obstacles=None, enable_collision_check: bool=False, rollback: str='start', n_steps: int=20, overlap_tol: float=0.005):
    from roam_bench.artic_open_check import OpenCheckResult, check_open
    if action != 'open' or not enable_collision_check or (not moving_links):
        set_action(drives, plan, action)
        return OpenCheckResult(success=True, error_message='', target_openness=1.0 if action == 'open' else 0.0, final_openness=1.0 if action == 'open' else 0.0, last_safe_openness=1.0 if action == 'open' else 0.0, colliding_object=None, colliding_kind=None, n_steps=0, peak_overlap_frac=0.0, reason='collision check disabled' if not enable_collision_check else 'close/no-links', checked_obstacles=0, detection_enabled=False)
    obstacles = obstacles or []
    worst = None
    for name, link in moving_links.items():
        res = check_open(link, obstacles, target_openness=1.0, n_steps=n_steps, overlap_tol=overlap_tol, rollback=rollback, detection_enabled=True)
        if not res.success:
            worst = res
            break
    if worst is not None:
        set_action(drives, plan, 'close')
        return worst
    set_action(drives, plan, 'open')
    return OpenCheckResult(success=True, error_message='', target_openness=1.0, final_openness=1.0, last_safe_openness=1.0, colliding_object=None, colliding_kind=None, n_steps=n_steps, peak_overlap_frac=0.0, reason='opened, no cross-model collision', checked_obstacles=len(obstacles), detection_enabled=True)

def read_link_pose(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    pos = m.ExtractTranslation()
    q = m.ExtractRotationQuat()
    w = q.GetReal()
    im = q.GetImaginary()
    usd = ((pos[0], pos[1], pos[2]), (w, im[0], im[1], im[2]))
    stale = abs(pos[0]) < 1e-09 and abs(pos[1]) < 1e-09 and (abs(pos[2]) < 1e-09) and (abs(abs(w) - 1.0) < 1e-09)
    if not stale:
        return usd
    try:
        from isaacsim.core.prims import RigidPrim
        rv = RigidPrim(prim_path)
        rv.initialize()
        poss, oris = rv.get_world_poses()
        pos = np.asarray(poss).reshape(-1, 3)[0]
        ori = np.asarray(oris).reshape(-1, 4)[0]
        return ((float(pos[0]), float(pos[1]), float(pos[2])), (float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])))
    except Exception as e:
        print('[artic_stage] RigidPrim read failed: %r' % e, flush=True)
        return usd

def yaw_about_z_deg(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    rot = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z)))
    xr = rot.TransformDir(Gf.Vec3d(1.0, 0.0, 0.0))
    return math.degrees(math.atan2(xr[1], xr[0]))

def rotate_vec(quat_wxyz, v) -> np.ndarray:
    w, x, y, z = quat_wxyz
    r = Gf.Rotation(Gf.Quatd(w, Gf.Vec3d(x, y, z))).TransformDir(Gf.Vec3d(*[float(c) for c in v]))
    return np.array([r[0], r[1], r[2]])

def world_axis(local_rot_wxyz, token: str) -> np.ndarray:
    unit = {'X': (1.0, 0, 0), 'Y': (0, 1.0, 0), 'Z': (0, 0, 1.0)}[token]
    return rotate_vec(local_rot_wxyz, unit)

def signed_angle_about_deg(q_before_wxyz, q_after_wxyz, axis_world) -> float:

    def conj(q):
        return (q[0], -q[1], -q[2], -q[3])

    def mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)
    qr = mul(q_after_wxyz, conj(q_before_wxyz))
    if qr[0] < 0.0:
        qr = (-qr[0], -qr[1], -qr[2], -qr[3])
    ax = np.asarray(axis_world, float)
    n = np.linalg.norm(ax)
    ax = ax / n if n > 1e-12 else ax
    s = float(np.dot(np.asarray(qr[1:]), ax))
    return math.degrees(2.0 * math.atan2(s, qr[0]))
