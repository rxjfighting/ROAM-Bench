"""Joint usd map utilities for ROAM-Bench."""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional
from roam_bench.sdf_joints import JointSpec, parse_scene_joints
_AXIS_LETTERS = ('X', 'Y', 'Z')
_CLOSED_POS = 0.0
_ZERO_TOL = 1e-09
_SLACK_TOL = {'angular': 15.0, 'linear': 0.02}

class JointMapError(ValueError):
    pass

@dataclass(frozen=True)
class UsdJointParams:
    drive: str
    axis: str
    lower: float
    upper: float
    closed_pos: float
    open_pos: float
    joint_name: str
    model: str
    interior_zero: bool
    body0: str = ''
    body1: str = ''
    local_pos: tuple = (0.0, 0.0, 0.0)
    local_rot_wxyz: tuple = (1.0, 0.0, 0.0, 0.0)

def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)

def _rpy_to_quat_wxyz(r: float, p: float, y: float) -> tuple:
    cr, sr = (math.cos(r / 2.0), math.sin(r / 2.0))
    cp, sp = (math.cos(p / 2.0), math.sin(p / 2.0))
    cy, sy = (math.cos(y / 2.0), math.sin(y / 2.0))
    qx = (cr, sr, 0.0, 0.0)
    qy = (cp, 0.0, sp, 0.0)
    qz = (cy, 0.0, 0.0, sy)
    return _quat_mul(qz, _quat_mul(qy, qx))

def _quat_to_matrix(q) -> tuple:
    w, x, y, z = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))

def _rpy_to_matrix(r: float, p: float, y: float) -> tuple:
    cr, sr = (math.cos(r), math.sin(r))
    cp, sp = (math.cos(p), math.sin(p))
    cy, sy = (math.cos(y), math.sin(y))
    return ((cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr), (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr), (-sp, cp * sr, cp * cr))

def _canonical_axis(axis_xyz) -> tuple:
    nz = [(i, v) for i, v in enumerate(axis_xyz) if abs(v) > _ZERO_TOL]
    if len(nz) != 1 or abs(abs(nz[0][1]) - 1.0) > 1e-06:
        raise JointMapError(f'non-canonical axis {axis_xyz!r}: UsdPhysics axis token needs a single unit component (X/Y/Z)')
    idx, val = nz[0]
    return (_AXIS_LETTERS[idx], 1.0 if val > 0 else -1.0)

def _conv(drive: str, q: float) -> float:
    return math.degrees(q) if drive == 'angular' else q

def _deconv(drive: str, q: float) -> float:
    return math.radians(q) if drive == 'angular' else q

def map_joint(spec: JointSpec) -> UsdJointParams:
    if spec.type == 'revolute':
        drive = 'angular'
    elif spec.type == 'prismatic':
        drive = 'linear'
    else:
        raise JointMapError(f'joint {spec.name!r}: type {spec.type!r} not drivable')
    lo, hi = (spec.limit_low, spec.limit_high)
    if lo is None or hi is None:
        raise JointMapError(f'joint {spec.name!r}: missing limit(s) ({lo}, {hi})')
    if abs(lo - hi) <= _ZERO_TOL:
        raise JointMapError(f'joint {spec.name!r}: degenerate limits lo==hi=={lo}')
    smin, smax = (lo, hi) if lo <= hi else (hi, lo)
    if not smin - _ZERO_TOL <= 0.0 <= smax + _ZERO_TOL:
        raise JointMapError(f'joint {spec.name!r}: q=0 outside limits [{smin}, {smax}]; the closed=0 assumption is broken -- inspect the asset')
    letter, sign = _canonical_axis(spec.axis_xyz)
    u_lo, u_hi = sorted((sign * _conv(drive, lo), sign * _conv(drive, hi)))
    q_open_sdf = lo if abs(lo) > abs(hi) else hi
    open_pos = sign * _conv(drive, q_open_sdf)
    interior = smin < -_ZERO_TOL and smax > _ZERO_TOL
    if interior:
        small_side = min(abs(_conv(drive, lo)), abs(_conv(drive, hi)))
        if small_side > _SLACK_TOL[drive]:
            unit = 'deg' if drive == 'angular' else 'm'
            raise JointMapError(f'joint {spec.name!r}: interior_zero both-way limits [{_conv(drive, lo):.4g}, {_conv(drive, hi):.4g}] {unit}; argmax|limit| picked open={open_pos:.4g} {unit} but the other side is also real travel (min side {small_side:.4g} {unit} > SLACK_TOL {_SLACK_TOL[drive]} {unit}Invalid scene geometry. ')
    ox = tuple((float(v) for v in spec.origin_xyz or (0.0, 0.0, 0.0)))
    orr, opp, oyy = spec.origin_rpy or (0.0, 0.0, 0.0)
    local_rot = _rpy_to_quat_wxyz(float(orr), float(opp), float(oyy))
    return UsdJointParams(drive=drive, axis=letter, lower=u_lo, upper=u_hi, closed_pos=_CLOSED_POS, open_pos=open_pos, joint_name=spec.name, model=spec.model, interior_zero=interior, body0=spec.parent_link, body1=spec.child_link, local_pos=ox, local_rot_wxyz=local_rot)

def assert_roundtrip(spec: JointSpec, params: UsdJointParams, tol: float=1e-09):
    idx = _AXIS_LETTERS.index(params.axis)
    letter_hat = [0.0, 0.0, 0.0]
    letter_hat[idx] = 1.0
    q_open_sdf = spec.limit_low if abs(spec.limit_low) > abs(spec.limit_high) else spec.limit_high
    sdf_vec = tuple((q_open_sdf * a for a in spec.axis_xyz))
    usd_scalar = _deconv(params.drive, params.open_pos)
    usd_vec = tuple((usd_scalar * h for h in letter_hat))
    for a, b in zip(sdf_vec, usd_vec):
        if abs(a - b) > tol:
            raise AssertionError(f'joint {spec.name!r} open config mismatch: SDF {sdf_vec} vs UsdPhysics {usd_vec} (unit or axis-sign bug)')
    return (sdf_vec, usd_vec)

def assert_frame_roundtrip(spec: JointSpec, params: UsdJointParams, tol: float=1e-09):
    ox = tuple((float(v) for v in spec.origin_xyz or (0.0, 0.0, 0.0)))
    for a, b in zip(ox, params.local_pos):
        if abs(a - b) > tol:
            raise AssertionError(f'joint {spec.name!r} local_pos {params.local_pos} != joint xyz {ox}')
    qn = math.sqrt(sum((c * c for c in params.local_rot_wxyz)))
    if abs(qn - 1.0) > 1e-06:
        raise AssertionError(f'joint {spec.name!r} local_rot not unit: |q|={qn}')
    orr, opp, oyy = spec.origin_rpy or (0.0, 0.0, 0.0)
    m_rpy = _rpy_to_matrix(float(orr), float(opp), float(oyy))
    m_q = _quat_to_matrix(params.local_rot_wxyz)
    for i in range(3):
        for j in range(3):
            if abs(m_rpy[i][j] - m_q[i][j]) > 1e-09:
                raise AssertionError(f'joint {spec.name!r} rpy->quat mismatch at [{i}][{j}]: {m_rpy[i][j]} vs {m_q[i][j]}')
    return (m_rpy, m_q)

def map_scene_joints(scene_root: str) -> list:
    out = []
    for model, _path, specs in parse_scene_joints(scene_root):
        for s in specs:
            p = map_joint(s)
            assert_roundtrip(s, p)
            assert_frame_roundtrip(s, p)
            out.append((model, s, p))
    return out
