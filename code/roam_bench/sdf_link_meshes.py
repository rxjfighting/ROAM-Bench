"""Sdf link meshes utilities for ROAM-Bench."""
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple
import numpy as np
from roam_bench.scene_loader import _parse_pose, _parse_sdf_collisions, CollisionPiece, GLTF_YUP_TO_ZUP

def parse_link_visuals(sdf_path: str) -> Dict[str, List[Tuple[str, np.ndarray]]]:
    out: Dict[str, List[Tuple[str, np.ndarray]]] = {}
    sdf_dir = os.path.dirname(sdf_path)
    try:
        tree = ET.parse(sdf_path)
    except Exception as e:
        print(f'[sdf_link_meshes] WARN cannot parse {sdf_path}: {e}')
        return out
    root = tree.getroot()
    for link in root.iter('link'):
        link_name = link.get('name') or ''
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
            local = link_pose @ vpose
            scale_el = mesh.find('scale')
            if scale_el is not None and scale_el.text:
                sx, sy, sz = [float(v) for v in scale_el.text.split()]
                local = local @ np.diag([sx, sy, sz, 1.0])
            local = local @ GLTF_YUP_TO_ZUP
            out.setdefault(link_name, []).append((gltf_abs, local))
    return out

def parse_link_collisions(sdf_path: str) -> Dict[str, List[CollisionPiece]]:
    out: Dict[str, List[CollisionPiece]] = {}
    for piece in _parse_sdf_collisions(sdf_path):
        out.setdefault(piece.link, []).append(piece)
    return out
