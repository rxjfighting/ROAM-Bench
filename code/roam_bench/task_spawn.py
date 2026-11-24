"""Task spawn utilities for ROAM-Bench."""
from __future__ import annotations
import dataclasses
import hashlib
import math
from collections import Counter
from typing import Iterable, Mapping, Sequence
from roam_bench.protocol.types import Pose
from . import astar_path, embodiment, physics_colliders
POLICY_VERSION = 'diverse_safe_spawn_v1'
DEFAULT_YAWS = (0.0, 90.0, 180.0, 270.0)

def spawn_for(header: Mapping, spec=None) -> dict:
    truth = getattr(spec, 'truth', None) or {}
    raw = truth.get('spawn') or header.get('spawn')
    if not isinstance(raw, Mapping):
        raise ValueError('frozen taskset has no explicit spawn pose')
    required = ('x', 'z', 'yaw_deg')
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f'spawn pose is missing {missing}')
    return dict(raw)

def pose_for(header: Mapping, spec=None) -> Pose:
    raw = spawn_for(header, spec)
    return Pose(x=float(raw['x']), y=float(raw.get('y', 1.5)), z=float(raw['z']), yaw_deg=float(raw['yaw_deg']), horizon_deg=float(raw.get('horizon_deg', 0.0)))

def _stable_rank(task_id: str, candidate: Mapping, seed: int) -> str:
    token = f"{seed}|{task_id}|{candidate['room']}|{candidate['x']:.4f}|{candidate['z']:.4f}|{candidate['yaw_deg']:.1f}"
    return hashlib.sha256(token.encode('utf-8')).hexdigest()

def _spread_cells(cells: Sequence[tuple[float, float]], center: tuple[float, float], limit: int, min_separation: float) -> list[tuple[float, float]]:
    remaining = sorted(set(cells))
    if not remaining:
        return []
    first = min(remaining, key=lambda p: ((p[0] - center[0]) ** 2 + (p[1] - center[1]) ** 2, p))
    chosen = [first]
    remaining.remove(first)
    while remaining and len(chosen) < limit:
        ranked = sorted(((min((math.dist(cell, old) for old in chosen)), cell) for cell in remaining), key=lambda item: (-item[0], item[1]))
        distance, cell = ranked[0]
        if distance < min_separation:
            break
        chosen.append(cell)
        remaining.remove(cell)
    return chosen

def candidate_spawns(sim, *, positions_per_room: int=6, min_position_separation: float=1.25, yaws: Iterable[float]=DEFAULT_YAWS) -> list[dict]:
    scene_root = getattr(sim, 'scene_root', '')
    wall_boxes = physics_colliders.parse_wall_boxes(scene_root) if scene_root else []
    planner = astar_path.PathPlanner(sim._nav, wall_boxes=wall_boxes, agent_radius=embodiment.collision_radius(embodiment.DEFAULT_EMBODIMENT) + 0.02, furniture_radius=embodiment.collision_radius(embodiment.DEFAULT_EMBODIMENT))
    turn_radius = embodiment.turn_clearance_radius(embodiment.DEFAULT_EMBODIMENT)
    result = []
    for room in sorted(sim.facts.room_types):
        bounds = sim.facts.room_bounds.get(room)
        if not bounds:
            continue
        safe = [cell for cell in sim._nav.walkable_cells(room) if planner.point_clear(cell, turn_radius) and sim.facts.room_localization_point(*cell) == room]
        center = ((bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2)
        for x, z in _spread_cells(safe, center, positions_per_room, min_position_separation):
            for yaw in yaws:
                result.append({'x': round(float(x), 4), 'y': 1.5, 'z': round(float(z), 4), 'yaw_deg': float(yaw), 'horizon_deg': 0.0, 'room': room})
    if not result:
        raise RuntimeError('scene has no collision-safe per-task spawn candidates')
    return result

def _visible_by_candidate(sim, candidates: Sequence[Mapping], min_pixel_frac: float) -> list[frozenset[str]]:
    original_pose = sim._pose
    original_room = sim._room
    visible = []
    try:
        for raw in candidates:
            sim._pose = Pose(float(raw['x']), float(raw.get('y', 1.5)), float(raw['z']), float(raw['yaw_deg']), float(raw.get('horizon_deg', 0.0)), True)
            sim._room = str(raw['room'])
            visible.append(frozenset(sim.visible_instances(min_pixel_frac)))
    finally:
        sim._pose = original_pose
        sim._room = original_room
    return visible

def _safe_for_spec(spec, visible: frozenset[str]) -> bool:
    evidence = set(spec.evidence)
    answers = set(spec.answer_instances)
    answer_visible = bool(answers & visible)
    single_frame_sufficient = bool(evidence) and evidence <= visible
    return not answer_visible and (not single_frame_sufficient)

def assign_diverse_spawns(specs: Sequence, sim, *, seed: int=0, positions_per_room: int=6, min_position_separation: float=1.25, min_pixel_frac: float=0.002):
    candidates = candidate_spawns(sim, positions_per_room=positions_per_room, min_position_separation=min_position_separation)
    visible = _visible_by_candidate(sim, candidates, min_pixel_frac)
    room_use, position_use, heading_use, pose_use = (Counter(), Counter(), Counter(), Counter())
    assigned, rejected = ([], [])
    for spec in sorted(specs, key=lambda value: value.task.task_id):
        valid = [(candidate, seen) for candidate, seen in zip(candidates, visible) if _safe_for_spec(spec, seen)]
        if not valid:
            rejected.append({'task_id': spec.task.task_id, 'family': spec.family, 'reason': 'no_non_leaking_collision_safe_spawn'})
            continue
        evidence_rooms = {sim.facts.by_alias[alias].room_id for alias in spec.evidence if alias in sim.facts.by_alias}
        has_outside = any((candidate['room'] not in evidence_rooms for candidate, _seen in valid))

        def key(item):
            candidate, _seen = item
            position = (candidate['room'], candidate['x'], candidate['z'])
            outside_penalty = int(has_outside and candidate['room'] in evidence_rooms)
            return (outside_penalty, room_use[candidate['room']], position_use[position], heading_use[candidate['yaw_deg']], pose_use[position, candidate['yaw_deg']], _stable_rank(spec.task.task_id, candidate, seed))
        chosen, seen = min(valid, key=key)
        position = (chosen['room'], chosen['x'], chosen['z'])
        room_use[chosen['room']] += 1
        position_use[position] += 1
        heading_use[chosen['yaw_deg']] += 1
        pose_use[position, chosen['yaw_deg']] += 1
        truth = dict(spec.truth or {})
        truth['spawn'] = dict(chosen)
        truth['spawn_policy'] = {'version': POLICY_VERSION, 'seed': int(seed), 'answer_visible_at_spawn': False, 'solvable_from_initial_frame': False, 'visible_instances': sorted(seen)}
        assigned.append(dataclasses.replace(spec, truth=truth))
    order = {spec.task.task_id: index for index, spec in enumerate(specs)}
    assigned.sort(key=lambda spec: order[spec.task.task_id])
    return (assigned, rejected)

def spawn_distribution(specs: Sequence) -> dict:
    rows = [dict((spec.truth or {}).get('spawn') or {}) for spec in specs]
    positions = {(row.get('room'), row.get('x'), row.get('z')) for row in rows if row}
    return {'tasks': len(rows), 'unique_poses': len({(row.get('x'), row.get('z'), row.get('yaw_deg')) for row in rows if row}), 'unique_positions': len(positions), 'rooms_used': sorted({row.get('room') for row in rows if row.get('room')}), 'rooms_withs': dict(sorted(Counter((row.get('room') for row in rows if row)).items())), 'yaw_counts': dict(sorted(Counter((str(row.get('yaw_deg')) for row in rows if row)).items()))}
