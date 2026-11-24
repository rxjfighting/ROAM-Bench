"""Exterior seals utilities for ROAM-Bench."""
from __future__ import annotations
from dataclasses import dataclass
import json
import math
import os
import numpy as np
SEAL_THICKNESS = 0.1

@dataclass(frozen=True)
class ExteriorSeal:
    opening_id: str
    room_id: str
    wall_id: str
    transform: np.ndarray
    size: np.ndarray

def _state_path(scene_root: str) -> str | None:
    candidates = (os.path.join(scene_root, 'combined_house', 'house_state.json'), os.path.join(scene_root, 'combined_house', 'house_state.json.bak'), os.path.join(scene_root, 'house_state.json'), os.path.join(scene_root, 'house_state.json.bak'))
    return next((path for path in candidates if os.path.isfile(path)), None)

def exterior_door_ids(layout: dict) -> frozenset[str]:
    return frozenset((str(door['id']) for door in layout.get('doors', ()) if door.get('id') is not None and (door.get('door_type') == 'exterior' or door.get('room_b') is None)))

def seals_from_layout(layout: dict, thickness: float=SEAL_THICKNESS) -> tuple[ExteriorSeal, ...]:
    exterior = exterior_door_ids(layout)
    if not exterior:
        return ()
    seals = []
    seen = set()
    for room in layout.get('placed_rooms', ()):
        room_id = str(room.get('room_id') or 'unknown')
        for wall in room.get('walls', ()):
            start, end = (wall.get('start_point'), wall.get('end_point'))
            if not start or not end:
                continue
            dx, dy = (float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
            length = math.hypot(dx, dy)
            if length <= 1e-09:
                continue
            ux, uy = (dx / length, dy / length)
            outward = (-uy, ux)
            position = room.get('position') or ()
            if len(position) >= 2 and room.get('width') and room.get('depth'):
                room_centre = (float(position[0]) + float(room['width']) / 2.0, float(position[1]) + float(room['depth']) / 2.0)
                wall_mid = ((float(start[0]) + float(end[0])) / 2.0, (float(start[1]) + float(end[1])) / 2.0)
                if (wall_mid[0] - room_centre[0]) * outward[0] + (wall_mid[1] - room_centre[1]) * outward[1] < 0:
                    outward = (-outward[0], -outward[1])
            for opening in wall.get('openings', ()):
                opening_id = str(opening.get('opening_id') or '')
                if opening_id not in exterior or opening_id in seen:
                    continue
                if opening.get('opening_type', 'door') != 'door':
                    continue
                width = float(opening.get('width', 0.0))
                height = float(opening.get('height', 0.0))
                sill = float(opening.get('sill_height', 0.0))
                if width <= 0.0 or height <= 0.0:
                    continue
                along = float(opening.get('position_along_wall', 0.0)) + width / 2.0
                cx = float(start[0]) + ux * along
                cy = float(start[1]) + uy * along
                world = np.eye(4)
                tx, ty = (outward[1], -outward[0])
                world[:3, :3] = np.array(((tx, outward[0], 0.0), (ty, outward[1], 0.0), (0.0, 0.0, 1.0)))
                world[:3, 3] = (cx, cy, sill + height / 2.0)
                seals.append(ExteriorSeal(opening_id=opening_id, room_id=room_id, wall_id=str(wall.get('wall_id') or ''), transform=world, size=np.asarray((width, float(thickness), height), dtype=float)))
                seen.add(opening_id)
    return tuple(sorted(seals, key=lambda seal: seal.opening_id))

def load_exterior_seals(scene_root: str) -> tuple[ExteriorSeal, ...]:
    path = _state_path(scene_root)
    if path is None:
        return ()
    with open(path, encoding='utf-8') as handle:
        state = json.load(handle)
    return seals_from_layout(state.get('layout') or {})

def seal_box_mesh(size, uv_scale: float=0.5):
    sx, sy, sz = (float(value) for value in size)
    hx, hy, hz = (sx / 2.0, sy / 2.0, sz / 2.0)
    faces = (((-hx, -hy, -hz), (hx, -hy, -hz), (hx, -hy, hz), (-hx, -hy, hz), sx, sz), ((hx, hy, -hz), (-hx, hy, -hz), (-hx, hy, hz), (hx, hy, hz), sx, sz), ((-hx, hy, -hz), (-hx, -hy, -hz), (-hx, -hy, hz), (-hx, hy, hz), sy, sz), ((hx, -hy, -hz), (hx, hy, -hz), (hx, hy, hz), (hx, -hy, hz), sy, sz), ((-hx, -hy, hz), (hx, -hy, hz), (hx, hy, hz), (-hx, hy, hz), sx, sy), ((-hx, hy, -hz), (hx, hy, -hz), (hx, -hy, -hz), (-hx, -hy, -hz), sx, sy))
    points = tuple((point for face in faces for point in face[:4]))
    uvs = []
    for face in faces:
        u, v = (face[4] / uv_scale, face[5] / uv_scale)
        uvs.extend(((0.0, 0.0), (u, 0.0), (u, v), (0.0, v)))
    return (points, (4,) * 6, tuple(range(24)), tuple(uvs))
