"""Align articulation hinge anchors with validated geometry."""
from __future__ import annotations
import dataclasses
import copy
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
_LOG = logging.getLogger('roam_bench.hinge_alignment')
SNAP_THRESHOLD_M = 0.05
PARALLEL_TOL = 0.99
_N_SAMPLES = 12000
BBOX_MATCH_TOL_M = 0.02
ORIENT_MATCH_TOL_M = 0.005
AXIS_CANON_TOL = 0.01
ABSTAIN = 'abstain_no_sibling_reference'
ABSTAIN_TRANSPLANT = 'abstain_no_matching_donor'
ABSTAIN_ORIENT = 'abstain_orientation_mismatch'
INVALID_ANCHOR = 'invalid_pivot_anchor'
CANDIDATE_TRANSPLANT = 'cross_instance_transplant'
ABSTAIN_NO_DONOR = 'abstain_no_registry_donor'
ABSTAIN_AMBIGUOUS = 'abstain_ambiguous_donors'
ABSTAIN_MIDLINE_DONOR = 'abstain_midline_axis_donor'
ABSTAIN_DONOR_Q_MISSING = 'abstain_donor_q_missing'
ABSTAIN_DONOR_ROLE = 'abstain_decorative_donor_role'
DECORATIVE_DONOR_ROLES = frozenset({'knob', 'handle', 'pipe'})
AXIS_MATCH_TOL_COS = 0.99
DONOR_CONSISTENCY_TOL_M = 0.01
MIDLINE_Q_CEIL = 0.5

def _uhat(v) -> np.ndarray:
    v = np.asarray(v, float)
    return v / (np.linalg.norm(v) + 1e-30)

def _perp(pts, P, dh) -> np.ndarray:
    v = np.asarray(pts, float) - np.asarray(P, float)
    return np.linalg.norm(v - np.outer(v @ dh, dh), axis=1)

def _stable_q(pts, P, dh) -> Optional[float]:
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return None
    dh = _uhat(dh)
    c = pts.mean(0)
    cvec = c - np.asarray(P, float)
    c_perp = cvec - cvec @ dh * dh
    r_centre = float(np.linalg.norm(c_perp))
    ext = pts.max(0) - pts.min(0)
    tr = [ext[i] for i in range(3) if abs(dh[i]) < 0.9]
    half = max(tr) / 2.0 if tr else float('nan')
    if not half > 1e-09:
        return None
    return r_centre / half

def _door_ends_xy(pts) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(pts, float)[:, :2]
    c = xy.mean(0)
    _, _, vt = np.linalg.svd(xy - c, full_matrices=False)
    longdir = vt[0]
    t = (xy - c) @ longdir
    return (c + longdir * t.min(), c + longdir * t.max())

def _snap_to_point(P, X, dh) -> np.ndarray:
    v = np.asarray(X, float) - np.asarray(P, float)
    return P + (v - v @ dh * dh)

def _world_to_local(link_world: Optional[np.ndarray], p_world) -> Tuple[float, float, float]:
    p = np.asarray(p_world, float)
    if link_world is None:
        return (float(p[0]), float(p[1]), float(p[2]))
    inv = np.linalg.inv(np.asarray(link_world, float))
    loc = inv @ np.r_[p, 1.0]
    return (float(loc[0]), float(loc[1]), float(loc[2]))

def _model_frame_bbox_extent(pts, link_world: Optional[np.ndarray]) -> Tuple[float, float, float]:
    p = np.asarray(pts, float)
    if link_world is not None:
        inv = np.linalg.inv(np.asarray(link_world, float))
        p = (inv @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    return tuple((float(x) for x in p.max(0) - p.min(0)))

def _model_frame_centroid(pts, link_world: Optional[np.ndarray]) -> np.ndarray:
    p = np.asarray(pts, float)
    if link_world is not None:
        inv = np.linalg.inv(np.asarray(link_world, float))
        p = (inv @ np.c_[p, np.ones(len(p))].T).T[:, :3]
    return p.mean(0)

def _model_frame_dir(vec, link_world: Optional[np.ndarray]) -> np.ndarray:
    v = _uhat(vec)
    if link_world is not None:
        R = np.linalg.inv(np.asarray(link_world, float))[:3, :3]
        v = _uhat(R @ v)
    return v

def _canonical_signed_axis(v) -> Tuple[Tuple[int, int, int], float]:
    v = _uhat(v)
    k = int(np.argmax(np.abs(v)))
    canon = [0, 0, 0]
    canon[k] = 1 if v[k] >= 0 else -1
    resid = float(np.linalg.norm(v - np.asarray(canon, float)))
    return ((canon[0], canon[1], canon[2]), resid)

def _rotate_about_line(pts, P, dh, theta) -> np.ndarray:
    v = np.asarray(pts, float) - np.asarray(P, float)
    k = np.asarray(dh, float)
    vr = v * np.cos(theta) + np.cross(k, v) * np.sin(theta) + np.outer(v @ k, k) * (1.0 - np.cos(theta))
    return np.asarray(P, float) + vr

def which_edge_stationary_under_sweep(pts, anchor, axis, thetas):
    dh = _uhat(axis)
    pts = np.asarray(pts, float)
    lo, hi = _door_ends_xy(pts)
    ilo = int(np.argmin(np.linalg.norm(pts[:, :2] - lo, axis=1)))
    ihi = int(np.argmin(np.linalg.norm(pts[:, :2] - hi, axis=1)))
    P = np.asarray(anchor, float)

    def _disp_max(i):
        p0 = pts[i]
        d = 0.0
        for th in thetas:
            pr = _rotate_about_line(p0[None, :], P, dh, float(th))[0]
            d = max(d, float(np.linalg.norm(pr - p0)))
        return d
    dlo, dhi = (_disp_max(ilo), _disp_max(ihi))
    return ('lo' if dlo < dhi else 'hi', round(dlo, 6), round(dhi, 6))

@dataclass
class PivotOverride:
    joint: str
    action: str
    changed: bool
    reason: str
    anchor_status: str = ''
    which_edge: Optional[str] = None
    sibling_joint: Optional[str] = None
    pre_perp_m: Optional[float] = None
    post_perp_m: Optional[float] = None
    anchor_world_pre: Optional[Tuple[float, float, float]] = None
    anchor_world_post: Optional[Tuple[float, float, float]] = None
    axis_dir_world: Optional[Tuple[float, float, float]] = None
    local_pos_pre: Optional[Tuple[float, float, float]] = None
    local_pos_post: Optional[Tuple[float, float, float]] = None
    extent_gate_pass: Optional[bool] = None
    extent_gap_m: Optional[float] = None
    orient_gate_pass: Optional[bool] = None
    orient_gap_m: Optional[float] = None
    axis_flipped: Optional[bool] = None

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

@dataclass
class TransplantDonor:
    scene: str
    model: str
    joint: str
    local_pos_model: Tuple[float, float, float]
    bbox_extent_model: Tuple[float, float, float]
    centroid_offset_model: Tuple[float, float, float]
    axis_model: Tuple[float, float, float]
    axis_canon_model: Tuple[int, int, int]
    pre_perp_m: float
    sdf_path: str = ''
    donor_q: Optional[float] = None
    n_hull_verts: int = 0
    collision_uri_flavour: str = ''
    sdf_sha256: str = ''
    child_link_role: str = ''
    _TUPLE_FIELDS = ('local_pos_model', 'bbox_extent_model', 'centroid_offset_model', 'axis_model', 'axis_canon_model')

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TransplantDonor':
        d = dict(d)
        for k in cls._TUPLE_FIELDS:
            if k in d and d[k] is not None:
                conv = int if k == 'axis_canon_model' else float
                d[k] = tuple((conv(x) for x in d[k]))
        return cls(**{k: d[k] for k in d if not k.startswith('_')})

def _find_clean_parallel_sibling(links: Dict[str, 'object'], joint: str, dh: np.ndarray, threshold: float, n_samples: int):
    L = links[joint]
    Lpts = _sample_inside(L.hulls, n_samples)
    if len(Lpts) == 0:
        return None
    Lc = Lpts[:, :2].mean(0)
    best = None
    best_d = np.inf
    for nm, S in links.items():
        if nm == joint or getattr(S, 'jtype', None) != 'revolute':
            continue
        sdh = _uhat(S.axis)
        if abs(float(sdh @ dh)) < PARALLEL_TOL:
            continue
        spts = _sample_inside(S.hulls, n_samples)
        if len(spts) == 0:
            continue
        sperp = float(_perp(spts, np.asarray(S.anchor, float), sdh).min())
        if sperp > threshold:
            continue
        Sc = spts[:, :2].mean(0)
        d = float(np.linalg.norm(Sc - Lc))
        if d < best_d:
            best_d = d
            best = (nm, S, spts)
    return best

def compute_override(links: Dict[str, 'object'], joint: str, *, threshold: float=SNAP_THRESHOLD_M, candidate: str='sibling_mirror', n_samples: int=_N_SAMPLES, link_world: Optional[np.ndarray]=None, donor: Optional['TransplantDonor']=None) -> PivotOverride:
    L = links[joint]
    axis_world = tuple((float(c) for c in np.asarray(L.axis, float)))
    if getattr(L, 'jtype', None) != 'revolute':
        return PivotOverride(joint=joint, action='skip_prismatic', changed=False, axis_dir_world=axis_world, reason='Invalid scene geometry. ')
    dh = _uhat(L.axis)
    P = np.asarray(L.anchor, float)
    pts = _sample_inside(L.hulls, n_samples)
    if len(pts) == 0:
        return PivotOverride(joint=joint, action='no_geometry', changed=False, axis_dir_world=axis_world, reason='no collision hulls sampled for this link; cannot measure the anchor -- leaving it UNCHANGED (fail-closed, not snapped to a guess).')
    pre_perp = float(_perp(pts, P, dh).min())
    anchor_pre = (float(P[0]), float(P[1]), float(P[2]))
    if pre_perp <= threshold:
        return PivotOverride(joint=joint, action='no_move_below_threshold', changed=False, pre_perp_m=round(pre_perp, 6), anchor_world_pre=anchor_pre, axis_dir_world=axis_world, reason="clean hinge: anchor is %.4f m from its own door (<= %.2f m) -- NO MOVE, anchor unchanged ." % (pre_perp, threshold))
    if candidate == CANDIDATE_TRANSPLANT:
        return _transplant_override(joint, dh, P, pts, pre_perp, anchor_pre, axis_world, donor, threshold, link_world)
    sib = _find_clean_parallel_sibling(links, joint, dh, threshold, n_samples)
    if sib is None:
        return PivotOverride(joint=joint, action=ABSTAIN, changed=False, anchor_status=INVALID_ANCHOR, pre_perp_m=round(pre_perp, 6), anchor_world_pre=anchor_pre, axis_dir_world=axis_world, reason='off-door hinge (perp %.3f m > %.2f m) but NO clean parallel revolute SIBLING to mirror from. FAIL-CLOSED: NOT snapping. Snap-to-nearest would land on the seam (free) edge and FLIP the opening direction -- the opening direction must remain consistent. Joint kept in %s without changing its anchor.' % (pre_perp, threshold, INVALID_ANCHOR))
    sib_name, S, rpts = sib
    rP = np.asarray(S.anchor, float)
    Rc = rpts[:, :2].mean(0)
    rlo, rhi = _door_ends_xy(rpts)
    r_hinge = rlo if np.linalg.norm(rP[:2] - rlo) < np.linalg.norm(rP[:2] - rhi) else rhi
    r_other = rhi if np.allclose(r_hinge, rlo) else rlo
    Lc = pts[:, :2].mean(0)
    lo, hi = _door_ends_xy(pts)
    r_hinge_is_outer = np.linalg.norm(r_hinge - Lc) > np.linalg.norm(r_other - Lc)
    lo_outer = np.linalg.norm(lo - Rc) > np.linalg.norm(hi - Rc)
    outer_end, seam_end = (lo, hi) if lo_outer else (hi, lo)
    mirror_end = outer_end if r_hinge_is_outer else seam_end
    near_end = lo if np.linalg.norm(P[:2] - lo) < np.linalg.norm(P[:2] - hi) else hi
    chosen_end = mirror_end if candidate == 'sibling_mirror' else near_end
    which = 'outer' if np.allclose(chosen_end, outer_end) else 'seam'
    jm = int(np.argmin(np.linalg.norm(pts[:, :2] - chosen_end, axis=1)))
    P_new = _snap_to_point(P, pts[jm], dh)
    post_perp = float(_perp(pts, P_new, dh).min())
    anchor_post = (float(P_new[0]), float(P_new[1]), float(P_new[2]))
    action = 'snap_sibling_mirror' if candidate == 'sibling_mirror' else 'snap_nearest'
    return PivotOverride(joint=joint, action=action, changed=True, which_edge=which, sibling_joint=sib_name, pre_perp_m=round(pre_perp, 6), post_perp_m=round(post_perp, 6), anchor_world_pre=anchor_pre, anchor_world_post=anchor_post, axis_dir_world=axis_world, local_pos_pre=_world_to_local(link_world, P), local_pos_post=_world_to_local(link_world, P_new), reason='off-door hinge (perp %.3f m -> %.5f m). Snapped the axis line onto the %s edge via %s (sibling %s). N1 (perp->0) is necessary NOT sufficient -- both edges zero it; real-scene tests separately pin the intended edge.' % (pre_perp, post_perp, which, action, sib_name))

def _transplant_override(joint, dh, P, pts, pre_perp, anchor_pre, axis_world, donor, threshold, link_world) -> PivotOverride:

    def _abstain(reason, action=ABSTAIN_TRANSPLANT, **gate):
        return PivotOverride(joint=joint, action=action, changed=False, anchor_status=INVALID_ANCHOR, pre_perp_m=round(pre_perp, 6), anchor_world_pre=anchor_pre, axis_dir_world=axis_world, reason=reason, **gate)
    if donor is None:
        return _abstain('off-door hinge (perp %.3f m > %.2f m) but NO donor supplied for cross_instance_transplant. FAIL-CLOSED: not guessing an anchor and NOT snapping to nearest (single-leaf door, no seam edge; nearest is a near-coin-flip on which edge -> would FLIP the swing). Supply a CLEAN same-model copy via transplant_override_for_joint(donor_scene=, donor_model=). Kept in %s.' % (pre_perp, threshold, INVALID_ANCHOR))
    if donor.pre_perp_m > threshold:
        return _abstain('donor %s/%s:%s is itself NOT clean (its own perp %.4f m > %.2f m): transplanting a bad anchor would just move the defect. FAIL-CLOSED, kept in %s.' % (donor.scene, donor.model, donor.joint, donor.pre_perp_m, threshold, INVALID_ANCHOR))
    if donor.donor_q is None:
        return _abstain('Scene validation: %s %s %s %s' % (donor.scene, donor.model, donor.joint, INVALID_ANCHOR), action=ABSTAIN_DONOR_Q_MISSING)
    if donor.donor_q < MIDLINE_Q_CEIL:
        return _abstain('Scene validation: %s %s %s %.4f %.2f %.4f %s' % (donor.scene, donor.model, donor.joint, donor.donor_q, MIDLINE_Q_CEIL, donor.pre_perp_m, INVALID_ANCHOR), action=ABSTAIN_MIDLINE_DONOR)
    if donor.child_link_role in DECORATIVE_DONOR_ROLES:
        return _abstain('Scene validation: %s %s %s %s %s' % (donor.scene, donor.model, donor.joint, donor.child_link_role, INVALID_ANCHOR), action=ABSTAIN_DONOR_ROLE)
    recip_ext = _model_frame_bbox_extent(pts, link_world)
    ext_gap = float(np.max(np.abs(np.asarray(recip_ext) - np.asarray(donor.bbox_extent_model, float))))
    extent_pass = ext_gap <= BBOX_MATCH_TOL_M
    if not extent_pass:
        return _abstain("SIZE MISMATCH: donor %s/%s model-frame extent %s vs recipient %s differ by %.4f m > %.3f m -- a size VARIANT (e.g. scene_191's Z=0.363 nightstand vs the 0.303 copies), whose clean local_pos is WRONG to transplant. FAIL-CLOSED to the current abstain; NOT 'finding a nearest copy to make do', NOT snapping to nearest. Kept in %s." % (donor.scene, donor.model, tuple((round(x, 4) for x in donor.bbox_extent_model)), tuple((round(x, 4) for x in recip_ext)), ext_gap, BBOX_MATCH_TOL_M, INVALID_ANCHOR), extent_gate_pass=False, extent_gap_m=round(ext_gap, 6))
    local_post = tuple((float(c) for c in np.asarray(donor.local_pos_model, float)))
    recip_centroid = _model_frame_centroid(pts, link_world)
    recip_offset = recip_centroid - np.asarray(local_post, float)
    orient_gap = float(np.max(np.abs(recip_offset - np.asarray(donor.centroid_offset_model, float))))
    recip_axis_model = _model_frame_dir(axis_world, link_world)
    recip_canon, recip_resid = _canonical_signed_axis(recip_axis_model)
    axis_flipped = recip_canon != tuple(donor.axis_canon_model)
    orient_pass = orient_gap <= ORIENT_MATCH_TOL_M and (not axis_flipped) and (recip_resid <= AXIS_CANON_TOL)
    _readout = dict(extent_gate_pass=True, extent_gap_m=round(ext_gap, 6), orient_gate_pass=orient_pass, orient_gap_m=round(orient_gap, 6), axis_flipped=bool(axis_flipped))
    if recip_resid > AXIS_CANON_TOL:
        return _abstain('OFF-AXIS JOINT (unsupported axis geometry): recipient model-frame axis %s is %.4f > %.0e from any canonical +/-{X,Y,Z}, so the signed-axis byte-equality cannot be trusted (joint axes must satisfy canonical-axis checks). FAIL-CLOSED rather than mis-snap. Kept in %s.' % (tuple((round(float(c), 4) for c in recip_axis_model)), recip_resid, AXIS_CANON_TOL, INVALID_ANCHOR), action=ABSTAIN_ORIENT, **_readout)
    if not orient_pass:
        return _abstain('Scene validation: %s %s %.4f %s %s %.4f %.3f %s %s %s %s' % (donor.scene, donor.model, ext_gap, tuple((round(x, 4) for x in donor.centroid_offset_model)), tuple((round(float(c), 4) for c in recip_offset)), orient_gap, ORIENT_MATCH_TOL_M, tuple(donor.axis_canon_model), recip_canon, axis_flipped, INVALID_ANCHOR), action=ABSTAIN_ORIENT, **_readout)
    if link_world is None:
        P_new = np.asarray(local_post, float)
    else:
        P_new = (np.asarray(link_world, float) @ np.r_[np.asarray(local_post, float), 1.0])[:3]
    post_perp = float(_perp(pts, P_new, dh).min())
    anchor_post = (float(P_new[0]), float(P_new[1]), float(P_new[2]))
    lo, hi = _door_ends_xy(pts)
    ilo = int(np.argmin(np.linalg.norm(pts[:, :2] - lo, axis=1)))
    ihi = int(np.argmin(np.linalg.norm(pts[:, :2] - hi, axis=1)))
    plo = float(_perp(pts[ilo][None, :], P_new, dh)[0])
    phi = float(_perp(pts[ihi][None, :], P_new, dh)[0])
    which = 'lo' if plo < phi else 'hi'
    return PivotOverride(joint=joint, action='transplant_cross_instance', changed=True, which_edge=which, sibling_joint='%s/%s:%s' % (donor.scene, donor.model, donor.joint), pre_perp_m=round(pre_perp, 6), post_perp_m=round(post_perp, 6), anchor_world_pre=anchor_pre, anchor_world_post=anchor_post, axis_dir_world=axis_world, local_pos_pre=_world_to_local(link_world, P), local_pos_post=local_post, reason='Scene validation: %.3f %.5f %s %s %s %.5f %.4f %r' % (pre_perp, post_perp, donor.scene, donor.model, donor.joint, donor.pre_perp_m, ext_gap, which), **_readout)

def pivot_override_for_joint(scene_root: str, model: str, joint: str, *, candidate: str='sibling_mirror', threshold: float=SNAP_THRESHOLD_M, n_samples: int=_N_SAMPLES) -> PivotOverride:
    from roam_bench.artic_open_assemble import build_moving_links, scene_welds
    links = build_moving_links(scene_root, model)
    if joint not in links:
        raise KeyError('joint %r not a drivable joint of model %r in %s' % (joint, model, scene_root))
    lw = (scene_welds(scene_root).get(model) or {}).get('link_world')
    return compute_override(links, joint, threshold=threshold, candidate=candidate, n_samples=n_samples, link_world=lw)

def transplant_donor_from_scene(donor_scene: str, donor_model: str, joint: str, *, threshold: float=SNAP_THRESHOLD_M, n_samples: int=_N_SAMPLES) -> TransplantDonor:
    from roam_bench.artic_open_assemble import build_moving_links, scene_welds
    links = build_moving_links(donor_scene, donor_model)
    if joint not in links:
        raise KeyError('donor joint %r not a drivable joint of %r in %s' % (joint, donor_model, donor_scene))
    L = links[joint]
    dh = _uhat(L.axis)
    P = np.asarray(L.anchor, float)
    pts = _sample_inside(L.hulls, n_samples)
    if len(pts) == 0:
        raise ValueError('donor %s/%s:%s has no sampled hull geometry' % (donor_scene, donor_model, joint))
    pre_perp = float(_perp(pts, P, dh).min())
    _w = scene_welds(donor_scene).get(donor_model) or {}
    lw = _w.get('link_world')
    sdf = _w.get('sdf') or ''
    lp = _world_to_local(lw, P)
    centroid = _model_frame_centroid(pts, lw)
    axis_m = _model_frame_dir(L.axis, lw)
    canon, _resid = _canonical_signed_axis(axis_m)
    q = _stable_q(pts, P, dh)
    role = _child_link_role(_child_links_of(donor_scene, donor_model).get(joint, ''))
    return TransplantDonor(scene=donor_scene, model=donor_model, joint=joint, local_pos_model=lp, bbox_extent_model=_model_frame_bbox_extent(pts, lw), centroid_offset_model=tuple((float(c) for c in centroid - np.asarray(lp, float))), axis_model=tuple((float(c) for c in axis_m)), axis_canon_model=canon, pre_perp_m=round(pre_perp, 6), sdf_path=sdf, donor_q=round(q, 5) if q is not None else None, n_hull_verts=int(len(pts)), collision_uri_flavour=_collision_uri_flavour(sdf), sdf_sha256=_sha256_file(sdf) or '', child_link_role=role)

def transplant_override_for_joint(scene_root: str, model: str, joint: str, *, donor_scene: str, donor_model: str, donor_joint: Optional[str]=None, threshold: float=SNAP_THRESHOLD_M, n_samples: int=_N_SAMPLES) -> PivotOverride:
    donor = transplant_donor_from_scene(donor_scene, donor_model, donor_joint or joint, threshold=threshold, n_samples=n_samples)
    from roam_bench.artic_open_assemble import build_moving_links, scene_welds
    links = build_moving_links(scene_root, model)
    if joint not in links:
        raise KeyError('joint %r not a drivable joint of model %r in %s' % (joint, model, scene_root))
    lw = (scene_welds(scene_root).get(model) or {}).get('link_world')
    return compute_override(links, joint, threshold=threshold, candidate=CANDIDATE_TRANSPLANT, n_samples=n_samples, link_world=lw, donor=donor)

@dataclass
class DonorMatch:
    status: str
    donor: Optional['TransplantDonor']
    layer: Optional[str] = None
    n_candidates: int = 0
    recip_extent_model: Optional[Tuple[float, float, float]] = None
    recip_axis_model: Optional[Tuple[float, float, float]] = None
    max_extent_gap_m: Optional[float] = None
    max_localpos_spread_m: Optional[float] = None

class DonorRegistry:

    def __init__(self, donors: Optional[List['TransplantDonor']]=None, provenance: Optional[dict]=None):
        self.donors: List[TransplantDonor] = list(donors or [])
        self.provenance: dict = dict(provenance or {})
        self.registry_drift: List[dict] = []
    _SCHEMA = 2

    def add(self, donor: 'TransplantDonor') -> None:
        self.donors.append(donor)

    def __len__(self) -> int:
        return len(self.donors)

    def match(self, recip_extent_model, recip_axis_model, *, recip_scene: Optional[str]=None, tol_m: float=BBOX_MATCH_TOL_M, tol_cos: float=AXIS_MATCH_TOL_COS) -> DonorMatch:
        re = np.asarray(recip_extent_model, float)
        ra = _uhat(recip_axis_model)
        cands = []
        worst_gap = 0.0
        for d in self.donors:
            gap = float(np.max(np.abs(np.asarray(d.bbox_extent_model, float) - re)))
            if gap > tol_m:
                continue
            if float(_uhat(d.axis_model) @ ra) < tol_cos:
                continue
            cands.append((d, gap))
        base = DonorMatch(status='none', donor=None, recip_extent_model=tuple((float(c) for c in re)), recip_axis_model=tuple((float(c) for c in ra)), n_candidates=len(cands))
        if not cands:
            return base
        worst_gap = max((g for _, g in cands))
        base.max_extent_gap_m = round(worst_gap, 6)
        lps = np.array([d.local_pos_model for d, _ in cands], float)
        spread = float(np.max(np.linalg.norm(lps - lps[0], axis=1))) if len(lps) > 1 else 0.0
        base.max_localpos_spread_m = round(spread, 6)
        if spread > DONOR_CONSISTENCY_TOL_M:
            base.status = 'ambiguous'
            return base
        in_scene = [d for d, _ in cands if recip_scene is not None and d.scene == recip_scene]
        chosen = in_scene[0] if in_scene else cands[0][0]
        base.status = 'ok'
        base.donor = chosen
        base.layer = 'in_scene' if in_scene else 'cross_scene'
        return base

    def to_dict(self) -> dict:
        return {'schema': self._SCHEMA, 'provenance': self.provenance, 'donors': [d.to_dict() for d in self.donors]}

    @classmethod
    def from_dict(cls, d: dict) -> 'DonorRegistry':
        return cls(donors=[TransplantDonor.from_dict(x) for x in d.get('donors', [])], provenance=d.get('provenance') or {})

    def donor_keys(self):
        return {(tuple((int(c) for c in dn.axis_canon_model)), tuple((round(float(c), 4) for c in dn.bbox_extent_model))) for dn in self.donors}

def build_donor_registry(scenes: List[str], *, threshold: float=SNAP_THRESHOLD_M, n_samples: int=_N_SAMPLES) -> DonorRegistry:
    from roam_bench.artic_open_assemble import build_moving_links, scene_welds
    reg = DonorRegistry()
    for scene in scenes:
        welds = scene_welds(scene)
        for model in welds:
            lw = (welds.get(model) or {}).get('link_world')
            sdf = (welds.get(model) or {}).get('sdf') or ''
            try:
                links = build_moving_links(scene, model)
            except Exception as exc:
                _LOG.debug('[donor-registry] skip %s/%s: %s', scene, model, exc)
                continue
            roles = _child_links_of(scene, model)
            for jname, L in links.items():
                if getattr(L, 'jtype', None) != 'revolute':
                    continue
                pts = _sample_inside(L.hulls, n_samples)
                if len(pts) == 0:
                    continue
                dh = _uhat(L.axis)
                P = np.asarray(L.anchor, float)
                if float(_perp(pts, P, dh).min()) > threshold:
                    continue
                axis_m = _model_frame_dir(L.axis, lw)
                _canon, resid = _canonical_signed_axis(axis_m)
                if resid > AXIS_CANON_TOL:
                    continue
                q = _stable_q(pts, P, dh)
                if q is None or q < MIDLINE_Q_CEIL:
                    _LOG.info('[donor-registry] skip MIDLINE %s/%s:%s (q=%s < %.2f): pre_perp clean but axis runs through the door centroid -- not an edge donor.', scene, model, jname, 'None' if q is None else '%.4f' % q, MIDLINE_Q_CEIL)
                    continue
                role = _child_link_role(roles.get(jname, ''))
                if role in DECORATIVE_DONOR_ROLES:
                    _LOG.info('[donor-registry] skip DECORATIVE %s/%s:%s (child_link_role=%s): not a door/drawer donor.', scene, model, jname, role)
                    continue
                lp = _world_to_local(lw, P)
                centroid = _model_frame_centroid(pts, lw)
                reg.add(TransplantDonor(scene=scene, model=model, joint=jname, local_pos_model=lp, bbox_extent_model=_model_frame_bbox_extent(pts, lw), centroid_offset_model=tuple((float(c) for c in centroid - np.asarray(lp, float))), axis_model=tuple((float(c) for c in axis_m)), axis_canon_model=_canon, pre_perp_m=round(float(_perp(pts, P, dh).min()), 6), sdf_path=sdf, donor_q=round(q, 5), n_hull_verts=int(len(pts)), collision_uri_flavour=_collision_uri_flavour(sdf), sdf_sha256=_sha256_file(sdf) or '', child_link_role=role))
    assets = {}
    for dn in reg.donors:
        if dn.sdf_path and dn.sdf_path not in assets:
            assets[dn.sdf_path] = _sha256_file(dn.sdf_path)
    reg.provenance = {'source_scenes': list(scenes), 'n_source_scenes': len(scenes), 'assets': assets}
    _LOG.info('[donor-registry] built from %d scene(s): %d clean revolute donor(s)', len(scenes), len(reg))
    return reg

def _sha256_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 16), b''):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None

def _collision_uri_flavour(sdf_path: str) -> str:
    if not sdf_path:
        return ''
    try:
        with open(sdf_path, encoding='utf-8', errors='replace') as f:
            txt = f.read()
    except OSError:
        return ''
    has_coacd, has_vhacd = ('_coacd' in txt, '_vhacd' in txt)
    if has_coacd and (not has_vhacd):
        return 'coacd'
    if has_vhacd and (not has_coacd):
        return 'vhacd'
    if has_coacd and has_vhacd:
        return 'mixed'
    return ''
_ROLE_KEYS = ('drawer', 'handle', 'door', 'knob', 'pipe', 'lid', 'body')

def _child_link_role(child_link: str) -> str:
    low = (child_link or '').lower()
    for k in _ROLE_KEYS:
        if k in low:
            return k
    return 'other'

def _child_links_of(scene_root: str, model: str) -> Dict[str, str]:
    from roam_bench.artic_open_assemble import map_scene_joints
    out: Dict[str, str] = {}
    for m, spec, _p in map_scene_joints(scene_root):
        if m == model:
            out[spec.name] = getattr(spec, 'child_link', '') or ''
    return out
_DATA_URI = 'data://'

def _portable_data_path(path: str, data_root=None) -> str:
    if not path or path.startswith(_DATA_URI):
        return path
    from .paths import DATA_ROOT
    root = Path(data_root or DATA_ROOT).expanduser().resolve()
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        return path
    try:
        relative = candidate.resolve().relative_to(root)
    except ValueError:
        return path
    return _DATA_URI + relative.as_posix()

def _hydrate_data_path(path: str, data_root=None) -> str:
    if not path or not path.startswith(_DATA_URI):
        return path
    from .paths import DATA_ROOT
    relative = path[len(_DATA_URI):]
    rel = Path(relative)
    if not relative or rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'invalid donor registry data URI: {path!r}')
    return str((Path(data_root or DATA_ROOT).expanduser().resolve() / rel).resolve())

def _portable_registry_dict(registry: 'DonorRegistry', data_root=None) -> dict:
    payload = copy.deepcopy(registry.to_dict())
    payload['schema'] = DonorRegistry._SCHEMA
    provenance = payload.get('provenance') or {}
    provenance['source_scenes'] = [_portable_data_path(value, data_root) for value in provenance.get('source_scenes', ())]
    provenance['assets'] = {_portable_data_path(key, data_root): value for key, value in (provenance.get('assets') or {}).items()}
    for donor in payload.get('donors', ()):
        donor['scene'] = _portable_data_path(donor.get('scene', ''), data_root)
        donor['sdf_path'] = _portable_data_path(donor.get('sdf_path', ''), data_root)
    return payload

def _hydrate_registry_dict(payload: dict, data_root=None) -> dict:
    schema = int(payload.get('schema', 1))
    if schema not in (1, 2):
        raise ValueError(f'unsupported donor registry schema: {schema}')
    if schema == 1:
        return payload
    provenance = payload.get('provenance') or {}
    provenance['source_scenes'] = [_hydrate_data_path(value, data_root) for value in provenance.get('source_scenes', ())]
    provenance['assets'] = {_hydrate_data_path(key, data_root): value for key, value in (provenance.get('assets') or {}).items()}
    for donor in payload.get('donors', ()):
        donor['scene'] = _hydrate_data_path(donor.get('scene', ''), data_root)
        donor['sdf_path'] = _hydrate_data_path(donor.get('sdf_path', ''), data_root)
    return payload

def save_donor_registry(registry: 'DonorRegistry', path: str, *, data_root=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(_portable_registry_dict(registry, data_root), f, indent=2)

def load_donor_registry(path: str, *, data_root=None) -> 'DonorRegistry':
    with open(path) as f:
        payload = json.load(f)
    payload = _hydrate_registry_dict(payload, data_root)
    reg = DonorRegistry.from_dict(payload)
    drift = []
    for sdf_path, stored_sha in (reg.provenance.get('assets') or {}).items():
        current = _sha256_file(sdf_path)
        if current != stored_sha:
            drift.append({'asset': sdf_path, 'stored_sha256': stored_sha, 'current_sha256': current, 'reason': 'missing' if current is None else 'content_changed'})
    reg.registry_drift = drift
    if drift:
        _LOG.warning('[donor-registry] LOADED registry from %s has DRIFT: %d of %d asset(s) changed on disk since build. Registry still usable (NOT fail-closed); recommend rebuild. First: %s', path, len(drift), len(reg.provenance.get('assets') or {}), drift[0]['asset'])
    return reg
_SHIPPED_REGISTRY = os.path.normpath(os.path.join(os.path.dirname(__file__), 'data', 'pivot_donor_registry.json'))

def load_shipped_donor_registry(path: str=_SHIPPED_REGISTRY) -> Optional['DonorRegistry']:
    if not os.path.exists(path):
        _LOG.warning('Scene validation: %s', path)
        return None
    return load_donor_registry(path)

def resolve_registry_donor(links: Dict[str, 'object'], joint: str, *, registry: 'DonorRegistry', link_world: Optional[np.ndarray]=None, recip_scene: Optional[str]=None, n_samples: int=_N_SAMPLES) -> DonorMatch:
    L = links[joint]
    pts = _sample_inside(L.hulls, n_samples)
    if len(pts) == 0:
        return DonorMatch(status='none', donor=None, n_candidates=0)
    ext = _model_frame_bbox_extent(pts, link_world)
    axis_m = _model_frame_dir(L.axis, link_world)
    scene = recip_scene if recip_scene is not None else getattr(L, 'scene', None)
    return registry.match(ext, tuple((float(c) for c in axis_m)), recip_scene=scene)

def align_pivots(plan, scene_root: str, *, enable: bool=True, candidate: str='sibling_mirror', threshold: float=SNAP_THRESHOLD_M, n_samples: int=_N_SAMPLES, links=None, link_world=None, scene_model_key: Optional[str]=None, donors: Optional[Dict[str, 'TransplantDonor']]=None, donor_registry: Optional['DonorRegistry']=None):
    report = {'enabled': bool(enable), 'candidate': candidate, 'threshold_m': threshold, 'flag_default': 'ON', 'model': plan.model, 'overrides': {}, 'n_changed': 0, 'n_abstain': 0, 'n_no_move': 0, 'n_skip_prismatic': 0}
    if not enable:
        report['note'] = 'Invalid scene geometry. '
        return (plan, report)
    if links is None:
        from roam_bench.artic_open_assemble import build_moving_links, scene_welds
        key = scene_model_key or plan.model
        welds = scene_welds(scene_root)
        if key not in welds:
            raise KeyError('Scene validation: %r %s %r %d %s' % (key, scene_root, plan.model, len(welds), ', '.join(sorted(welds)[:6])))
        report['scene_model_key'] = key
        links = build_moving_links(scene_root, key)
        link_world = (welds.get(key) or {}).get('link_world')
    lw = link_world
    if donor_registry is not None:
        report['registry_provenance'] = {'n_donors': len(donor_registry), 'n_source_scenes': donor_registry.provenance.get('n_source_scenes'), 'source_scenes': donor_registry.provenance.get('source_scenes'), 'n_assets': len(donor_registry.provenance.get('assets') or {}), 'registry_drift': list(donor_registry.registry_drift)}
    new_joints: List = []
    for jp in plan.joints:
        if jp.joint_name not in links:
            new_joints.append(jp)
            continue
        ov = compute_override(links, jp.joint_name, threshold=threshold, candidate=candidate, n_samples=n_samples, link_world=lw, donor=(donors or {}).get(jp.joint_name))
        registry_decision = None
        if donor_registry is not None and candidate != CANDIDATE_TRANSPLANT and (ov.action == ABSTAIN):
            dm = resolve_registry_donor(links, jp.joint_name, registry=donor_registry, link_world=lw, recip_scene=scene_model_key, n_samples=n_samples)
            registry_decision = {'status': dm.status, 'layer': dm.layer, 'n_candidates': dm.n_candidates, 'recip_extent_model': dm.recip_extent_model, 'recip_axis_model': dm.recip_axis_model, 'max_extent_gap_m': dm.max_extent_gap_m, 'max_localpos_spread_m': dm.max_localpos_spread_m}
            if dm.status == 'ok':
                ov = compute_override(links, jp.joint_name, threshold=threshold, candidate=CANDIDATE_TRANSPLANT, n_samples=n_samples, link_world=lw, donor=dm.donor)
                registry_decision['donor'] = '%s/%s:%s' % (dm.donor.scene, dm.donor.model, dm.donor.joint)
            elif dm.status == 'none':
                ov = dataclasses.replace(ov, action=ABSTAIN_NO_DONOR, reason='off-door hinge, no clean sibling, AND the donor registry has NO signature match (extent within %.3f m + signed-axis dot >= %.2f over %d donor(s)). FAIL-CLOSED: anchor kept in %s, did NOT guess.' % (BBOX_MATCH_TOL_M, AXIS_MATCH_TOL_COS, len(donor_registry), INVALID_ANCHOR))
            else:
                ov = dataclasses.replace(ov, action=ABSTAIN_AMBIGUOUS, reason='off-door hinge, no clean sibling; the registry matched %d donors on the geometry signature but their clean anchors DISAGREE by %.4f m (> %.3f m) -- transplanting one would write a possibly-wrong number. FAIL-CLOSED: anchor kept in %s.' % (dm.n_candidates, dm.max_localpos_spread_m or 0.0, DONOR_CONSISTENCY_TOL_M, INVALID_ANCHOR))
        ovd = ov.as_dict()
        if registry_decision is not None:
            ovd['donor_registry'] = registry_decision
        report['overrides'][jp.joint_name] = ovd
        if ov.changed:
            new_params = dataclasses.replace(jp.params, local_pos=tuple(ov.local_pos_post))
            new_joints.append(dataclasses.replace(jp, params=new_params))
            report['n_changed'] += 1
        else:
            new_joints.append(jp)
            if ov.action in (ABSTAIN, ABSTAIN_TRANSPLANT, ABSTAIN_ORIENT, ABSTAIN_NO_DONOR, ABSTAIN_AMBIGUOUS):
                report['n_abstain'] += 1
                _LOG.warning('[hinge-alignment ABSTAIN] %s.%s kept AUTHORED anchor (fail-closed, NOT snapped to nearest): %s [anchor_status=%s]', plan.model, jp.joint_name, ov.reason, ov.anchor_status)
            elif ov.action == 'no_move_below_threshold':
                report['n_no_move'] += 1
            elif ov.action == 'skip_prismatic':
                report['n_skip_prismatic'] += 1
    plan_out = dataclasses.replace(plan, joints=tuple(new_joints))
    _summary = '[hinge-alignment] model=%s candidate=%s: %d snapped, %d abstained (invalid_anchor), %d clean/no-move, %d prismatic-skip' % (plan.model, candidate, report['n_changed'], report['n_abstain'], report['n_no_move'], report['n_skip_prismatic'])
    (_LOG.warning if report['n_abstain'] else _LOG.info)(_summary)
    report['summary'] = _summary
    return (plan_out, report)

def _sample_inside(hulls, n):
    from roam_bench.artic_open_check import _sample_inside as _si
    return _si(hulls, n)
