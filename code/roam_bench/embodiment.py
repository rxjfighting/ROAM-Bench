"""Embodiment utilities for ROAM-Bench."""
from __future__ import annotations
import math
import os
import numpy as np
from .physics_colliders import AGENT_HEIGHT, AGENT_RADIUS
from .paths import CACHE_ROOT
MODES = ('camera', 'mobile_base', 'humanoid')
DEFAULT_EMBODIMENT = 'humanoid'
ROOT_PATH = '/World/EvalAgent'
HUMANOID_REMOTE_ASSET = 'https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd'
HUMANOID_LOCAL_ASSET = str(CACHE_ROOT / 'assets/Isaac/5.1/Isaac/Robots/Unitree/G1/g1.usd')
HUMANOID_ASSET = os.environ.get('ROAM_BENCH_HUMANOID_ASSET', HUMANOID_LOCAL_ASSET if os.path.isfile(HUMANOID_LOCAL_ASSET) else HUMANOID_REMOTE_ASSET)
HUMANOID_SCALE = 1.0
HUMANOID_PELVIS_HEIGHT = 0.79
HUMANOID_CAMERA_MOUNT_OFFSET = -0.08
HUMANOID_MESH_BBOX_MIN = (-0.18158561, -0.15257412, -0.00227286)
HUMANOID_MESH_BBOX_MAX = (0.18157561, 0.33674142, 1.32057261)
HUMANOID_MESH_MARGIN = 0.01
HUMANOID_COLLISION_HEIGHT = HUMANOID_MESH_BBOX_MAX[2] + HUMANOID_MESH_MARGIN
HUMANOID_COLLISION_RADIUS = max(AGENT_RADIUS, max(abs(HUMANOID_MESH_BBOX_MIN[0]), abs(HUMANOID_MESH_BBOX_MAX[0])) + HUMANOID_MESH_MARGIN)
HUMANOID_TURN_CLEARANCE_RADIUS = max((math.hypot(x, y) for x in (HUMANOID_MESH_BBOX_MIN[0], HUMANOID_MESH_BBOX_MAX[0]) for y in (HUMANOID_MESH_BBOX_MIN[1], HUMANOID_MESH_BBOX_MAX[1]))) + HUMANOID_MESH_MARGIN
HUMANOID_COLLISION_FORWARD_OFFSET = 0.15
CAMERA_COWL_CENTER = (0.0, 0.075, 1.25)
CAMERA_COWL_HALF_EXTENTS = (0.14, 0.035, 0.1)

def collision_radius(mode: str) -> float:
    return HUMANOID_COLLISION_RADIUS if mode == 'humanoid' else AGENT_RADIUS

def turn_clearance_radius(mode: str) -> float:
    return HUMANOID_TURN_CLEARANCE_RADIUS if mode == 'humanoid' else collision_radius(mode)

def collision_forward_offsets(mode: str) -> tuple[float, ...]:
    return (0.0, HUMANOID_COLLISION_FORWARD_OFFSET) if mode == 'humanoid' else (0.0,)

def _set_collision_metadata(prim, mode: str) -> None:
    offsets = collision_forward_offsets(mode)
    prim.SetCustomDataByKey('roam_bench:collisionRadiusM', float(collision_radius(mode)))
    height = HUMANOID_COLLISION_HEIGHT if mode == 'humanoid' else AGENT_HEIGHT
    prim.SetCustomDataByKey('roam_bench:collisionHeightM', float(height))
    prim.SetCustomDataByKey('roam_bench:collisionForwardOffsetM', float(max(offsets)))
    prim.SetCustomDataByKey('roam_bench:motion', 'kinematic_mesh_box_sweep' if mode == 'humanoid' else 'kinematic_sweep')
    if mode == 'humanoid':
        prim.SetCustomDataByKey('roam_bench:collisionSource', 'visible_mesh_local_aabbs')
        prim.SetCustomDataByKey('roam_bench:meshBboxMinM', ','.join(map(str, HUMANOID_MESH_BBOX_MIN)))
        prim.SetCustomDataByKey('roam_bench:meshBboxMaxM', ','.join(map(str, HUMANOID_MESH_BBOX_MAX)))

def extract_humanoid_collision_boxes(stage, root_path: str=ROOT_PATH):
    from pxr import Gf, Usd, UsdGeom
    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        return ()
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    out = []
    for mesh_prim in Usd.PrimRange(root):
        if not mesh_prim.IsA(UsdGeom.Mesh):
            continue
        imageable = UsdGeom.Imageable(mesh_prim)
        if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        if imageable.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        local_range = bounds.ComputeLocalBound(mesh_prim).ComputeAlignedRange()
        if local_range.IsEmpty():
            continue
        relative, _reset = xforms.ComputeRelativeTransform(mesh_prim, root)
        lo, hi = (local_range.GetMin(), local_range.GetMax())
        points = []
        for x in (lo[0], hi[0]):
            for y in (lo[1], hi[1]):
                for z in (lo[2], hi[2]):
                    points.append(relative.Transform(Gf.Vec3d(x, y, z)))
        mins = [min((float(p[i]) for p in points)) for i in range(3)]
        maxs = [max((float(p[i]) for p in points)) for i in range(3)]
        mins[2] = max(0.05, mins[2])
        if maxs[2] <= mins[2]:
            continue
        centre = tuple(((mins[i] + maxs[i]) * 0.5 for i in range(3)))
        half = tuple(((maxs[i] - mins[i]) * 0.5 + HUMANOID_MESH_MARGIN for i in range(3)))
        half = (half[0], half[1], min(half[2], centre[2] - 0.05))
        if min(half) <= 0.0:
            continue
        out.append((centre, half, str(mesh_prim.GetPath())))
    return tuple(out)

def pose_matrix(pose) -> np.ndarray:
    angle = -math.radians(float(pose.yaw_deg))
    c, s = (math.cos(angle), math.sin(angle))
    out = np.eye(4)
    out[:3, :3] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    out[:3, 3] = (float(pose.x), float(pose.z), 0.0)
    return out

def _color(prim, rgb):
    from pxr import Gf, UsdGeom
    UsdGeom.Gprim(prim).CreateDisplayColorAttr([Gf.Vec3f(*rgb)])

def author_mobile_base(stage, root_path: str=ROOT_PATH):
    from pxr import Gf, UsdGeom
    root = UsdGeom.Xform.Define(stage, root_path)
    root.MakeMatrixXform()
    chassis = UsdGeom.Cube.Define(stage, root_path + '/Chassis')
    chassis.CreateSizeAttr(1.0)
    api = UsdGeom.XformCommonAPI(chassis)
    api.SetScale(Gf.Vec3f(0.17, 0.14, 0.09))
    api.SetTranslate(Gf.Vec3d(0.0, 0.0, 0.22))
    _color(chassis.GetPrim(), (0.08, 0.22, 0.45))
    for side, x in (('Left', -0.17), ('Right', 0.17)):
        wheel = UsdGeom.Cylinder.Define(stage, f'{root_path}/Wheel{side}')
        wheel.CreateRadiusAttr(0.075)
        wheel.CreateHeightAttr(0.045)
        wheel.CreateAxisAttr(UsdGeom.Tokens.x)
        UsdGeom.XformCommonAPI(wheel).SetTranslate(Gf.Vec3d(x, 0.0, 0.12))
        _color(wheel.GetPrim(), (0.03, 0.03, 0.03))
    mast = UsdGeom.Cylinder.Define(stage, root_path + '/Mast')
    mast.CreateRadiusAttr(0.025)
    mast.CreateHeightAttr(1.05)
    UsdGeom.XformCommonAPI(mast).SetTranslate(Gf.Vec3d(0.0, -0.05, 0.78))
    _color(mast.GetPrim(), (0.55, 0.58, 0.62))
    sensor = UsdGeom.Cube.Define(stage, root_path + '/SensorHead')
    sensor.CreateSizeAttr(1.0)
    sensor_api = UsdGeom.XformCommonAPI(sensor)
    sensor_api.SetScale(Gf.Vec3f(0.075, 0.055, 0.045))
    sensor_api.SetTranslate(Gf.Vec3d(0.0, -0.07, 1.43))
    _color(sensor.GetPrim(), (0.08, 0.22, 0.45))
    cowl = UsdGeom.Cube.Define(stage, root_path + '/CameraCowl')
    cowl.CreateSizeAttr(1.0)
    cowl_api = UsdGeom.XformCommonAPI(cowl)
    cowl_api.SetScale(Gf.Vec3f(*CAMERA_COWL_HALF_EXTENTS))
    cowl_api.SetTranslate(Gf.Vec3d(*CAMERA_COWL_CENTER))
    _color(cowl.GetPrim(), (0.1, 0.32, 0.68))
    prim = root.GetPrim()
    _set_collision_metadata(prim, 'mobile_base')
    prim.SetCustomDataByKey('roam_bench:visualMode', 'mobile_base')
    return root

def _disable_asset_physics(root_prim) -> None:
    from pxr import Usd, UsdPhysics
    for prim in list(Usd.PrimRange(root_prim)):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in Usd.PrimRange(root_prim):
        if prim.IsA(UsdPhysics.Joint):
            UsdPhysics.Joint(prim).CreateJointEnabledAttr(False)
            continue
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)

def author_humanoid(stage, root_path: str=ROOT_PATH, asset_path: str | None=None):
    from pxr import Gf, UsdGeom
    asset_path = asset_path or HUMANOID_ASSET
    root = UsdGeom.Xform.Define(stage, root_path)
    root.MakeMatrixXform()
    mount = UsdGeom.Xform.Define(stage, root_path + '/HumanoidMount')
    mount_api = UsdGeom.XformCommonAPI(mount)
    mount_api.SetTranslate(Gf.Vec3d(0.0, HUMANOID_CAMERA_MOUNT_OFFSET, HUMANOID_PELVIS_HEIGHT))
    mount_api.SetRotate(Gf.Vec3f(0.0, 0.0, 90.0))
    mount_api.SetScale(Gf.Vec3f(HUMANOID_SCALE, HUMANOID_SCALE, HUMANOID_SCALE))
    asset = UsdGeom.Xform.Define(stage, root_path + '/HumanoidMount/G1')
    if not asset.GetPrim().GetReferences().AddReference(asset_path):
        raise RuntimeError(f'failed to reference humanoid USD: {asset_path}')
    _disable_asset_physics(asset.GetPrim())
    prim = root.GetPrim()
    _set_collision_metadata(prim, 'humanoid')
    prim.SetCustomDataByKey('roam_bench:visualMode', 'humanoid')
    prim.SetCustomDataByKey('roam_bench:visualAsset', asset_path)
    prim.SetCustomDataByKey('roam_bench:endEffectorVisual', 'Unitree ThreeFinger hands')
    return root

def sync(stage, pose, mode: str, root_path: str=ROOT_PATH):
    from pxr import Gf, UsdGeom
    if mode not in MODES:
        raise ValueError(f'unknown embodiment {mode!r}; choose one of {MODES}')
    prim = stage.GetPrimAtPath(root_path)
    if mode == 'camera':
        if prim and prim.IsValid():
            UsdGeom.Imageable(prim).MakeInvisible()
        return
    authored_mode = prim.GetCustomDataByKey('roam_bench:visualMode') if prim and prim.IsValid() else None
    if prim and prim.IsValid() and (authored_mode != mode):
        stage.RemovePrim(root_path)
        prim = stage.GetPrimAtPath(root_path)
    if not prim or not prim.IsValid():
        if mode == 'humanoid':
            author_humanoid(stage, root_path)
        else:
            author_mobile_base(stage, root_path)
        prim = stage.GetPrimAtPath(root_path)
    UsdGeom.Imageable(prim).MakeVisible()
    xf = UsdGeom.Xformable(prim)
    ops = xf.GetOrderedXformOps()
    if not ops:
        ops = [xf.AddTransformOp()]
    mat = pose_matrix(pose)
    ops[0].Set(Gf.Matrix4d(*[float(v) for v in mat.T.flatten()]))
