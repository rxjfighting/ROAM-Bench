"""Sdf joints utilities for ROAM-Bench."""
from __future__ import annotations
import collections
import os
import re
from dataclasses import dataclass
from typing import Optional
import xml.etree.ElementTree as ET
import yaml
_DRIVABLE_TYPES = ('revolute', 'prismatic')
_COMBINED_DIRS = ('combined_house_after_furniture', 'combined_house')
_DMD_NAMES = ('house_furniture_welded.dmd.yaml', 'house.dmd.yaml')

@dataclass
class JointSpec:
    name: str
    type: str
    parent_link: str
    child_link: str
    axis_xyz: tuple
    limit_low: Optional[float]
    limit_high: Optional[float]
    origin_xyz: tuple = (0.0, 0.0, 0.0)
    origin_rpy: tuple = (0.0, 0.0, 0.0)
    sdf_path: str = ''
    model: str = ''

def _floats(text) -> list:
    if not text or not text.strip():
        return []
    return [float(v) for v in text.split()]

def _parse_origin(joint_el):
    pose_el = joint_el.find('pose')
    if pose_el is None:
        return ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    vals = _floats(pose_el.text)
    vals = (vals + [0.0] * 6)[:6]
    return (tuple(vals[0:3]), tuple(vals[3:6]))

def _parse_one_joint(j, sdf_path: str, model_name: str) -> Optional[JointSpec]:
    jtype = j.get('type')
    if jtype not in _DRIVABLE_TYPES:
        return None
    axis_vals = _floats(j.findtext('./axis/xyz'))
    axis_xyz = tuple(axis_vals) if len(axis_vals) == 3 else (0.0, 0.0, 0.0)
    lo_txt = j.findtext('./axis/limit/lower')
    hi_txt = j.findtext('./axis/limit/upper')
    limit_low = float(lo_txt) if lo_txt not in (None, '') else None
    limit_high = float(hi_txt) if hi_txt not in (None, '') else None
    origin_xyz, origin_rpy = _parse_origin(j)
    return JointSpec(name=j.get('name') or '', type=jtype, parent_link=(j.findtext('parent') or '').strip(), child_link=(j.findtext('child') or '').strip(), axis_xyz=axis_xyz, limit_low=limit_low, limit_high=limit_high, origin_xyz=origin_xyz, origin_rpy=origin_rpy, sdf_path=sdf_path, model=model_name)

def parse_sdf_joints(sdf_path: str) -> list:
    try:
        tree = ET.parse(sdf_path)
    except (ET.ParseError, FileNotFoundError, OSError) as e:
        print(f'[sdf_joints] WARN cannot parse {sdf_path}: {e}')
        return []
    out = []
    root = tree.getroot()
    for model in root.iter('model'):
        model_name = model.get('name') or ''
        for j in model.findall('joint'):
            spec = _parse_one_joint(j, sdf_path, model_name)
            if spec is not None:
                out.append(spec)
    return out

def _find_dmd(scene_root: str) -> Optional[str]:
    for sub in _COMBINED_DIRS:
        d = os.path.join(scene_root, sub)
        if not os.path.isdir(d):
            continue
        for name in _DMD_NAMES:
            dmd = os.path.join(d, name)
            if os.path.exists(dmd):
                return dmd
    return None

def _resolve_model_file(uri: str, scene_root: str) -> Optional[str]:
    m = re.match('package://[^/]+/(.*)', uri)
    cand = os.path.normpath(os.path.join(scene_root, m.group(1))) if m else uri
    return cand if os.path.isfile(cand) else None

def parse_scene_joints(scene_root: str) -> list:
    dmd = _find_dmd(scene_root)
    if dmd is None:
        raise FileNotFoundError(f'no Drake dmd.yaml under {scene_root} (looked in {_COMBINED_DIRS} for {_DMD_NAMES})')
    from roam_bench.scene_loader import _DrakeLoader
    with open(dmd) as f:
        doc = yaml.load(f, Loader=_DrakeLoader)
    directives = doc.get('directives', []) if isinstance(doc, dict) else []
    out = []
    cache: dict = {}
    for d in directives:
        if not isinstance(d, dict) or 'add_model' not in d:
            continue
        am = d['add_model']
        name = am.get('name')
        uri = am.get('file')
        if not name or not uri:
            continue
        sdf_path = _resolve_model_file(uri, scene_root)
        if sdf_path is None:
            continue
        if sdf_path not in cache:
            cache[sdf_path] = parse_sdf_joints(sdf_path)
        specs = cache[sdf_path]
        if specs:
            inst = [JointSpec(**{**s.__dict__, 'model': name}) for s in specs]
            out.append((name, sdf_path, inst))
    return out

def joint_type_counts(scene_joints) -> collections.Counter:
    c = collections.Counter()
    for _name, _path, specs in scene_joints:
        for s in specs:
            c[s.type] += 1
    return c
