"""Ceiling lighting utilities for ROAM-Bench."""
from __future__ import annotations
from dataclasses import dataclass
import math
CEILING_THICKNESS = 0.04
MAX_LIGHTS_PER_ROOM = 8
AMBIENT_REFERENCE_AREA = 4.2 * 4.0
MIN_AMBIENT_INTENSITY = 900.0
MAX_AMBIENT_INTENSITY = 14400.0
_LIGHT_WORDS = ('light', 'lamp', 'led', 'downlight', 'spotlight', 'pendant', 'chandelier', 'luminaire', 'flush mount', 'flush-mount', 'track lighting')
_NON_LIGHT_WORDS = ('detector', 'sensor', 'sprinkler', 'camera', 'speaker', 'vent', 'projector', 'smoke alarm', 'ceiling fan')

@dataclass(frozen=True)
class CeilingSlab:
    room_id: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

@dataclass(frozen=True)
class LightAnchor:
    room_id: str
    alias: str
    position: tuple[float, float, float]
    fallback: bool = False

def fixture_light_position(anchor_position, bbox_min, bbox_max, mode: str, clearance: float=0.02) -> tuple[float, float, float]:
    if mode not in ('center', 'bottom'):
        raise ValueError("fixture light position must be 'center' or 'bottom'")
    if bbox_min is None or bbox_max is None:
        return tuple((float(value) for value in anchor_position))
    lo, hi = (tuple(map(float, bbox_min)), tuple(map(float, bbox_max)))
    centre = tuple(((a + b) / 2.0 for a, b in zip(lo, hi)))
    if mode == 'center':
        return centre
    return (centre[0], centre[1], lo[2] - float(clearance))

def fixture_light_radius(bbox_min, bbox_max, shape: str) -> float:
    if shape not in ('sphere', 'disk'):
        raise ValueError("fixture light shape must be 'sphere' or 'disk'")
    if bbox_min is None or bbox_max is None:
        return 0.2 if shape == 'disk' else 0.08
    width = max(0.0, float(bbox_max[0]) - float(bbox_min[0]))
    depth = max(0.0, float(bbox_max[1]) - float(bbox_min[1]))
    dimensions = [value for value in (width, depth) if value > 1e-06]
    if not dimensions:
        return 0.2 if shape == 'disk' else 0.08
    footprint = min(dimensions)
    if shape == 'disk':
        return min(0.3, max(0.04, footprint * 0.45))
    return min(0.1, max(0.025, footprint * 0.18))

def room_fill_intensity(width: float, depth: float, reference_intensity: float=3600.0) -> float:
    area = max(0.0, float(width)) * max(0.0, float(depth))
    area_ratio = area / AMBIENT_REFERENCE_AREA
    scaled = float(reference_intensity) * area_ratio * area_ratio
    return min(MAX_AMBIENT_INTENSITY, max(MIN_AMBIENT_INTENSITY, scaled))

def room_main_light_budget(width: float, depth: float, reference_budget: float=24000.0) -> float:
    area = max(0.0, float(width)) * max(0.0, float(depth))
    if area <= 5.0:
        return float(reference_budget)
    return float(reference_budget) * max(1.0, area / AMBIENT_REFERENCE_AREA)

def room_fill_grid(width: float, depth: float) -> tuple[tuple[float, float], ...]:
    width, depth = (max(0.0, float(width)), max(0.0, float(depth)))
    if width <= 0 or depth <= 0:
        return ()
    if width * depth <= 5.0:
        return ((0.0, 0.0),)
    nx = max(1, math.ceil(width / 4.2))
    ny = max(1, math.ceil(depth / 4.0))
    return tuple((((ix + 0.5) * width / nx - width / 2.0, (iy + 0.5) * depth / ny - depth / 2.0) for iy in range(ny) for ix in range(nx)))

def is_light_fixture(obj: dict) -> bool:
    if obj.get('object_type') != 'ceiling_mounted':
        return False
    text = ' '.join((str(obj.get(key) or '') for key in ('object_id', 'name', 'description'))).lower()
    return not any((word in text for word in _NON_LIGHT_WORDS)) and any((word in text for word in _LIGHT_WORDS))

def _spread(points: list[LightAnchor], limit: int) -> tuple[LightAnchor, ...]:
    if len(points) <= limit:
        return tuple(sorted(points, key=lambda item: item.alias))
    ordered = sorted(points, key=lambda item: item.alias)
    cx = sum((p.position[0] for p in ordered)) / len(ordered)
    cy = sum((p.position[1] for p in ordered)) / len(ordered)
    first = min(ordered, key=lambda p: ((p.position[0] - cx) ** 2 + (p.position[1] - cy) ** 2, p.alias))
    selected = [first]
    remaining = [p for p in ordered if p is not first]
    while remaining and len(selected) < limit:
        pick = max(remaining, key=lambda p: (min((math.dist(p.position[:2], q.position[:2]) for q in selected)), p.alias))
        selected.append(pick)
        remaining.remove(pick)
    return tuple(selected)

def build_ceiling_plan(layout: dict, rooms: dict, *, thickness: float=CEILING_THICKNESS, max_lights: int=MAX_LIGHTS_PER_ROOM) -> tuple[tuple[CeilingSlab, ...], tuple[LightAnchor, ...]]:
    wall_height = float(layout.get('wall_height', 3.4))
    slabs, anchors = ([], [])
    for room in sorted(layout.get('placed_rooms', ()), key=lambda item: str(item.get('room_id') or '')):
        rid = str(room.get('room_id') or '')
        pos = room.get('position') or (0.0, 0.0)
        width, depth = (float(room.get('width', 0)), float(room.get('depth', 0)))
        if not rid or width <= 0 or depth <= 0:
            continue
        cx, cy = (float(pos[0]) + width / 2, float(pos[1]) + depth / 2)
        slabs.append(CeilingSlab(rid, (cx, cy, wall_height + thickness / 2), (width, depth, float(thickness))))
        candidates = []
        objects = (rooms.get(rid) or {}).get('objects') or {}
        for key, obj in objects.items():
            if not is_light_fixture(obj):
                continue
            translation = (obj.get('transform') or {}).get('translation') or ()
            if len(translation) < 3:
                continue
            alias = f"{rid}_{obj.get('object_id') or key}"
            candidates.append(LightAnchor(rid, alias, (cx + float(translation[0]), cy + float(translation[1]), min(wall_height - 0.06, float(translation[2]) - 0.06))))
        chosen = _spread(candidates, max(1, int(max_lights)))
        if not chosen:
            chosen = (LightAnchor(rid, f'{rid}_fallback_ceiling_light', (cx, cy, wall_height - 0.12), True),)
        anchors.extend(chosen)
    return (tuple(slabs), tuple(anchors))
