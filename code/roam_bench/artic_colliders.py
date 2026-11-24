"""Artic colliders utilities for ROAM-Bench."""
from __future__ import annotations
import os
from typing import Dict, List, Tuple
from roam_bench.scene_loader import CollisionPiece
from roam_bench.joint_articulation import ArticulationPlan
COLLIDERS_ENV = 'ISAAC_ARTIC_COLLIDERS'

def env_colliders_enabled() -> bool:
    return os.environ.get(COLLIDERS_ENV, '0').strip().lower() not in ('', '0', 'false', 'no')

def plan_link_colliders(plan: ArticulationPlan, link_collisions: Dict[str, List[CollisionPiece]] | None) -> List[Tuple[str, List[CollisionPiece]]]:
    if not link_collisions:
        return []
    out: List[Tuple[str, List[CollisionPiece]]] = []
    for lp in plan.links:
        pieces = link_collisions.get(lp.link_name) or []
        if pieces:
            out.append((lp.prim_path, list(pieces)))
    return out
