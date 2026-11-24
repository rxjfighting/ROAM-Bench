"""Candidate quality utilities for ROAM-Bench."""
from __future__ import annotations
import hashlib
import math
from collections import Counter
from typing import Iterable
FETCH_FAMILY = 'relocation'
FETCH_MIN_AUTO_CAP = 4

def _family(spec) -> str:
    return str(getattr(spec, 'family', ''))

def automatic_relocation_cap(specs: Iterable) -> int:
    specs = list(specs)
    room_localization = sum((_family(spec) == 'room_localization' for spec in specs))
    sources_by_room, destinations_by_source_room = ({}, {})
    for spec in specs:
        if _family(spec) != FETCH_FAMILY:
            continue
        truth = getattr(spec, 'truth', {}) or {}
        source_room = str(truth.get('from_room_id') or truth.get('from_room') or '')
        destination_room = str(truth.get('to_room_id') or truth.get('to_room') or '')
        sources_by_room.setdefault(source_room, set()).add(str(truth.get('object') or ''))
        destinations_by_source_room.setdefault(source_room, set()).add(destination_room)
    coverage_floor = sum((max(len(sources), len(destinations_by_source_room.get(room, ()))) for room, sources in sources_by_room.items()))
    return max(FETCH_MIN_AUTO_CAP, room_localization, coverage_floor)

def _kind(facts, alias: str) -> str:
    inst = getattr(facts, 'by_alias', {}).get(alias)
    return str(getattr(inst, 'kind', None) or alias)

def _area(facts, alias: str) -> float | None:
    footprint = getattr(facts, 'footprints', {}).get(alias)
    if not footprint or len(footprint) < 4:
        return None
    try:
        value = abs(float(footprint[2]) * float(footprint[3]))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None

def _relocation_record(spec, facts) -> dict:
    truth = getattr(spec, 'truth', {}) or {}
    source = str(truth.get('object') or '')
    destination = str(truth.get('dest') or '')
    source_room = str(truth.get('from_room_id') or truth.get('from_room') or '')
    destination_room = str(truth.get('to_room_id') or truth.get('to_room') or '')
    source_kind, destination_kind = (_kind(facts, source), _kind(facts, destination))
    source_area, destination_area = (_area(facts, source), _area(facts, destination))
    fit_ratio = None
    if source_area is not None and destination_area is not None:
        fit_ratio = source_area / destination_area
    requires_open = bool(getattr(spec, 'evidence_bound', None) and getattr(spec.evidence_bound, 'must_open', ()))
    features = (('source_instance', source), ('destination_instance', destination), ('source_room', source_room), ('destination_room', destination_room), ('room_pair', source_room, destination_room), ('source_kind', source_kind), ('destination_kind', destination_kind), ('kind_pair', source_kind, destination_kind), ('requires_open', requires_open), ('difficulty', 'L3' if requires_open else 'L2'))
    stable = hashlib.sha256(str(spec.task.task_id).encode('utf-8')).hexdigest()
    return {'spec': spec, 'source': source, 'destination': destination, 'source_room': source_room, 'destination_room': destination_room, 'features': features, 'fit_ratio': fit_ratio, 'stable': stable}

def select_relocation_candidates(specs: Iterable, facts, cap: int | None=None):
    specs = list(specs)
    relocation = [_relocation_record(spec, facts) for spec in specs if _family(spec) == FETCH_FAMILY]
    wanted = automatic_relocation_cap(specs) if cap in (None, 0) else int(cap)
    if wanted < 0:
        raise ValueError('relocation cap must be non-negative')
    wanted = min(wanted, len(relocation))
    selected, used = ([], set())
    feature_counts, source_counts, destination_counts = (Counter(), Counter(), Counter())
    weights = {'room_pair': 8, 'source_instance': 6, 'destination_instance': 6, 'source_room': 5, 'destination_room': 5, 'kind_pair': 4, 'source_kind': 3, 'destination_kind': 3, 'requires_open': 1, 'difficulty': 1}

    def choose(row):
        selected.append(row['spec'])
        used.add(row['stable'])
        source_counts[row['source']] += 1
        destination_counts[row['destination']] += 1
        feature_counts.update(row['features'])

    def secondary_rank(row):
        ratio = row['fit_ratio']
        fit = -ratio if ratio is not None else -float('inf')
        return (-destination_counts[row['destination']], fit, tuple((-ord(ch) for ch in row['stable'])))
    if cap in (None, 0):
        for source_room in sorted({row['source_room'] for row in relocation}):
            group = [row for row in relocation if row['source_room'] == source_room]
            sources = sorted({row['source'] for row in group})
            destination_rooms = sorted({row['destination_room'] for row in group})
            for index in range(max(len(sources), len(destination_rooms))):
                source = sources[index % len(sources)]
                destination_room = destination_rooms[index % len(destination_rooms)]
                choices = [row for row in group if row['stable'] not in used and row['source'] == source and (row['destination_room'] == destination_room)]
                choose(max(choices, key=secondary_rank))
    while len(selected) < wanted:
        candidates = [row for row in relocation if row['stable'] not in used]
        if not candidates:
            break

        def rank(row):
            gain = sum((weights[feature[0]] for feature in row['features'] if feature_counts[feature] == 0))
            ratio = row['fit_ratio']
            fit = -ratio if ratio is not None else -float('inf')
            room_pair = next((feature for feature in row['features'] if feature[0] == 'room_pair'))
            return (feature_counts[room_pair] == 0, gain, -source_counts[row['source']], -destination_counts[row['destination']], fit, tuple((-ord(ch) for ch in row['stable'])))
        choose(max(candidates, key=rank))
    selected_ids = {spec.task.task_id for spec in selected}
    output = [spec for spec in specs if _family(spec) != FETCH_FAMILY or spec.task.task_id in selected_ids]
    coverage = {name: len({feature[1:] for row in relocation for feature in row['features'] if feature[0] == name}) for name in ('source_instance', 'destination_instance', 'source_room', 'destination_room', 'room_pair', 'source_kind', 'destination_kind', 'kind_pair', 'difficulty')}
    kept_coverage = {name: len({feature[1:] for row in relocation if row['stable'] in used for feature in row['features'] if feature[0] == name}) for name in coverage}
    return (output, {'policy': 'relocation-balanced-coverage-v1', 'cap': wanted, 'cap_mode': 'automatic_room_localization_or_room_pair_max' if cap in (None, 0) else 'explicit', 'before': len(relocation), 'after': len(selected), 'dropped': len(relocation) - len(selected), 'coverage_before': coverage, 'coverage_after': kept_coverage, 'all_room_pairs_retained': kept_coverage['room_pair'] == coverage['room_pair'], 'all_source_instances_retained': len(source_counts) == len({row['source'] for row in relocation}), 'all_destination_instances_retained': len(destination_counts) == len({row['destination'] for row in relocation}), 'max_per_source': max(source_counts.values(), default=0), 'max_per_destination': max(destination_counts.values(), default=0)})
