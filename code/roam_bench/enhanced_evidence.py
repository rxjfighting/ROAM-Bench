"""Verify task-specific minimum and stronger spatial evidence."""
from __future__ import annotations
import math
from typing import Iterable, Mapping
POLICY_VERSION = 'roam-evidence-v1'
PASS, FAIL, UNKNOWN = ('pass', 'fail', 'unknown')
CLEAR_MIN_PIXELS = 16
CLEAR_MIN_FRACTION = None
POSITION_ERROR_FLOOR_M = 0.15
DEPTH_VALID_MIN_FRACTION = 0.5

def _get(obj, name: str, default=None):
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)

def _set(value: Iterable | None) -> set[str]:
    return {str(v) for v in value or () if str(v)}

def _receipt(status: str, reasons: Iterable[str], **details) -> dict:
    return {'status': status, 'reasons': list(reasons), **details}

def _combine(parts: Iterable[dict]) -> dict:
    parts = list(parts)
    statuses = {p.get('status') for p in parts}
    status = FAIL if FAIL in statuses else UNKNOWN if UNKNOWN in statuses else PASS
    return _receipt(status, [r for p in parts for r in p.get('reasons', ())], parts=parts)

def _binary_receipt(value):
    if isinstance(value, list):
        return [_binary_receipt(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: _binary_receipt(item) for key, item in value.items()}
    if out.get('status') == UNKNOWN:
        out['status'] = FAIL
        reasons = list(out.get('reasons') or ())
        if 'insufficient_evidence_to_verify' not in reasons:
            reasons.append('insufficient_evidence_to_verify')
        out['reasons'] = reasons
        out['failure_kind'] = 'unverified'
    return out

def _truth(spec) -> Mapping:
    return _get(spec, 'truth', {}) or {}

def _bound(spec):
    return _get(spec, 'evidence_bound')

def _bound_values(spec, field: str) -> set[str]:
    return _set(_get(_bound(spec), field, ()))

def _observations(trace: Mapping) -> list[Mapping]:
    rows = trace.get('observations')
    return rows if isinstance(rows, list) else []

def _objects(obs: Mapping) -> list[Mapping]:
    rows = obs.get('visible_objects')
    return rows if isinstance(rows, list) else []

def _clear_measurement(row: Mapping) -> bool:
    pixels = row.get('pixel_count')
    try:
        return int(pixels) >= CLEAR_MIN_PIXELS
    except (TypeError, ValueError):
        return False

def _evidence_objects(obs: Mapping) -> list[Mapping]:
    segmented = obs.get('segmentation_objects')
    if not isinstance(segmented, list):
        return _objects(obs)
    by_alias = {str(row.get('alias')): dict(row) for row in segmented if isinstance(row, Mapping) and row.get('alias') and _clear_measurement(row)}
    for row in _objects(obs):
        alias = str(row.get('alias') or '')
        if alias in by_alias:
            by_alias[alias].update(row)
    return list(by_alias.values())

def _visible_aliases(obs: Mapping) -> set[str]:
    return _set(obs.get('visible_aliases')) | {str(row.get('alias')) for row in _objects(obs) if row.get('alias')}

def _evidence_aliases(obs: Mapping) -> set[str]:
    if not isinstance(obs.get('segmentation_objects'), list):
        return _visible_aliases(obs)
    return {str(row.get('alias')) for row in _evidence_objects(obs) if row.get('alias')}

def _seen(trace: Mapping) -> set[str]:
    return {alias for obs in _observations(trace) for alias in _evidence_aliases(obs)}

def _visited(trace: Mapping) -> set[str]:
    return {str(obs.get('room_id')) for obs in _observations(trace) if obs.get('room_id') is not None}

def _index(obs: Mapping, fallback: int) -> int:
    try:
        return int(obs.get('observation_index', fallback))
    except (TypeError, ValueError):
        return fallback

def _object_rows(trace: Mapping, alias: str) -> list[tuple[int, Mapping]]:
    out = []
    for fallback, obs in enumerate(_observations(trace)):
        idx = _index(obs, fallback)
        for row in _evidence_objects(obs):
            if str(row.get('alias', '')) == alias:
                out.append((idx, row))
    return out

def _first_visible(trace: Mapping, alias: str) -> int | None:
    for fallback, obs in enumerate(_observations(trace)):
        if alias in _evidence_aliases(obs):
            return _index(obs, fallback)
    return None

def _aliases(required: set[str], trace: Mapping, reason: str) -> dict:
    if not isinstance(trace.get('observations'), list):
        return _receipt(UNKNOWN, ['observation_trace_unavailable'])
    seen = _seen(trace)
    missing = required - seen
    return _receipt(FAIL if missing else PASS, [f'missing_{reason}'] if missing else [f'all_{reason}_seen'], required=sorted(required), observed=sorted(required & seen), missing=sorted(missing))

def _scope(spec, *, exhaustive: bool=True) -> set[str]:
    search = _bound_values(spec, 'search_regions')
    required = _bound_values(spec, 'must_visit_regions')
    if exhaustive and search:
        return search
    if required:
        return required
    rooms = _truth(spec).get('rooms')
    return _set(rooms) or search

def _coverage(spec, trace: Mapping, *, exhaustive: bool=True) -> dict:
    scope = _scope(spec, exhaustive=exhaustive)
    if not scope:
        return _receipt(UNKNOWN, ['search_scope_not_recorded'])
    if not isinstance(trace.get('observations'), list):
        return _receipt(UNKNOWN, ['observation_trace_unavailable'])
    visited = _visited(trace) & scope
    missing = scope - visited
    return _receipt(FAIL if missing else PASS, ['required_rooms_not_visited'] if missing else ['declared_scope_covered'], required=sorted(scope), observed=sorted(visited), missing=sorted(missing))

def _visual_room_coverage(spec, trace: Mapping, *, exhaustive: bool=True) -> dict:
    scope = _scope(spec, exhaustive=exhaustive)
    if not scope:
        return _receipt(UNKNOWN, ['search_scope_not_recorded'])
    if not isinstance(trace.get('observations'), list):
        return _receipt(UNKNOWN, ['observation_trace_unavailable'])
    covered = _visited(trace) & scope
    for obs in _observations(trace):
        for row in _evidence_objects(obs):
            room = str(row.get('object_room_id') or '')
            if room not in scope:
                continue
            if _clear_measurement(row):
                covered.add(room)
    missing = scope - covered
    return _receipt(FAIL if missing else PASS, ['required_rooms_not_visually_covered'] if missing else ['declared_scope_visually_covered'], required=sorted(scope), observed=sorted(covered), missing=sorted(missing))

def _viewpoint_coverage(spec, trace: Mapping) -> dict:
    if not isinstance(trace.get('observations'), list):
        return _receipt(UNKNOWN, ['observation_trace_unavailable'])
    minimum = int(_get(_bound(spec), 'min_distinct_viewpoints', 0) or 0)
    if minimum <= 0:
        return _receipt(PASS, ['no_viewpoint_minimum_declared'], required=0, observed=0)
    from .evidence import viewpoint_key
    viewpoints = {key for key in map(viewpoint_key, _observations(trace)) if key is not None}
    if not viewpoints and _observations(trace):
        return _receipt(UNKNOWN, ['viewpoint_pose_unavailable'], required=minimum)
    observed = len(viewpoints)
    return _receipt(PASS if observed >= minimum else FAIL, ['viewpoint_count_met'] if observed >= minimum else ['insufficient_viewpoints'], required=minimum, observed=observed)

def _relations(trace: Mapping) -> dict[str, Mapping]:
    rows = trace.get('placement_relations')
    if isinstance(rows, Mapping):
        return {str(k): v for k, v in rows.items() if isinstance(v, Mapping)}
    if isinstance(rows, list):
        return {str(v.get('child')): v for v in rows if isinstance(v, Mapping) and v.get('child')}
    return {}

def _events(trace: Mapping, verbs=()) -> list[Mapping]:
    rows = trace.get('interaction_events')
    if not isinstance(rows, list):
        return []
    wanted = {str(v) for v in verbs}
    return [row for row in rows if not wanted or str(row.get('verb', '')) in wanted]

def _successful_event(trace: Mapping, verb: str, target: str, before_index=None) -> list[Mapping]:
    out = []
    for event in _events(trace, (verb,)):
        event_target = str(event.get('target') or event.get('source') or '')
        if event_target != target or event.get('success') is not True:
            continue
        try:
            idx = int(event.get('after_observation_index', -1))
        except (TypeError, ValueError):
            idx = -1
        if before_index is None or idx < before_index:
            out.append(event)
    return out

def _inside_open_before_visible(spec, trace: Mapping, aliases: set[str], cutoff_indices: Mapping[str, int] | None=None) -> dict:
    if 'placement_relations' not in trace:
        return _receipt(UNKNOWN, ['placement_relations_unavailable'])
    if 'interaction_events' not in trace:
        return _receipt(UNKNOWN, ['interaction_events_unavailable'])
    relations = _relations(trace)
    required, missing, unknown = ({}, [], [])
    for alias in sorted(aliases):
        rel = relations.get(alias)
        if not rel or rel.get('relation') == 'unknown':
            unknown.append(alias)
            continue
        if rel.get('relation') != 'inside':
            continue
        host = str(rel.get('host') or '')
        first = (cutoff_indices or {}).get(alias) if cutoff_indices is not None else _first_visible(trace, alias)
        required[alias] = host
        host_seen_open = any((idx <= first and row.get('is_open') is True for idx, row in _object_rows(trace, host))) if first is not None else False
        if not host or first is None or (not host_seen_open and (not _successful_event(trace, 'open', host, first))):
            missing.append(alias)
    if missing:
        return _receipt(FAIL, ['container_not_opened_before_internal_target_visible'], required=required, missing=missing)
    if unknown:
        return _receipt(UNKNOWN, ['target_placement_relation_unknown'], required=required, unknown=unknown)
    return _receipt(PASS, ['internal_targets_opened_before_visibility'], required=required)

def _room_localization_open_before_visible(spec, trace: Mapping, target: str) -> dict:
    bound = _get(spec, 'evidence_bound', None)
    absent = object()
    must_open = _get(bound, 'must_open', absent)
    if must_open is not absent and (not must_open):
        return _receipt(PASS, ['no_container_open_required'])
    return _inside_open_before_visible(spec, trace, {target})

def _room_localization_target_witness(spec, trace: Mapping, target: str) -> dict:
    exact = _aliases({target}, trace, 'target')
    if exact.get('status') == PASS:
        exact['witnesses'] = [target]
        exact['exact_target'] = True
        return exact
    truth = _truth(spec)
    kind = str(truth.get('kind') or '')
    room = str(truth.get('room_id') or truth.get('room') or '')
    witnesses = sorted({str(row.get('alias') or '') for obs in _observations(trace) for row in _evidence_objects(obs) if kind and room and (str(row.get('kind') or '') == kind) and (str(row.get('object_room_id') or '') == room) and str(row.get('alias') or '')})
    if witnesses:
        return _receipt(PASS, ['answer_equivalent_target_seen'], required={'kind': kind, 'room': room}, witnesses=witnesses, exact_target=False)
    return exact

def _clear(alias: str, trace: Mapping) -> bool | None:
    rows = _object_rows(trace, alias)
    if not rows:
        return False if 'observations' in trace else None
    measured = False
    for _idx, row in rows:
        pixels = row.get('pixel_count')
        if pixels is None:
            continue
        measured = True
        if _clear_measurement(row):
            return True
    return False if measured else None

def _clear_all(aliases: set[str], trace: Mapping) -> dict:
    values = {alias: _clear(alias, trace) for alias in aliases}
    if any((v is False for v in values.values())):
        return _receipt(FAIL, ['targets_not_clearly_visible'], failed=sorted((a for a, v in values.items() if v is False)))
    if any((v is None for v in values.values())):
        return _receipt(UNKNOWN, ['visibility_measurements_unavailable'], unknown=sorted((a for a, v in values.items() if v is None)))
    return _receipt(PASS, ['all_targets_clearly_visible'], thresholds={'min_pixels': CLEAR_MIN_PIXELS, 'min_fraction': CLEAR_MIN_FRACTION})

def _room_binding(expected: Mapping[str, str], trace: Mapping) -> dict:
    missing, wrong, unknown = ([], {}, [])
    for alias, room in expected.items():
        rows = _object_rows(trace, alias)
        if not rows:
            missing.append(alias)
            continue
        rooms = {str(row.get('object_room_id')) for _, row in rows if row.get('object_room_id') is not None}
        if not rooms:
            unknown.append(alias)
        elif rooms != {room}:
            wrong[alias] = sorted(rooms)
    if missing or wrong:
        return _receipt(FAIL, ['object_room_binding_failed'], missing=missing, wrong=wrong)
    if unknown:
        return _receipt(UNKNOWN, ['object_room_id_unavailable'], unknown=unknown)
    return _receipt(PASS, ['object_room_bindings_confirmed'], expected=expected)

def _room_grouping(targets: set[str], trace: Mapping, *, expected_rooms: set[str] | None=None, expected_counts: Mapping[str, int] | None=None) -> dict:
    groups, missing, unknown, inconsistent = ({}, [], [], {})
    for alias in sorted(targets):
        rows = _object_rows(trace, alias)
        if not rows:
            missing.append(alias)
            continue
        rooms = {str(row.get('object_room_id')) for _, row in rows if row.get('object_room_id') is not None}
        if not rooms:
            unknown.append(alias)
        elif len(rooms) != 1:
            inconsistent[alias] = sorted(rooms)
        else:
            groups[alias] = next(iter(rooms))
    if missing or inconsistent:
        return _receipt(FAIL, ['object_room_grouping_failed'], missing=missing, inconsistent=inconsistent)
    if unknown:
        return _receipt(UNKNOWN, ['object_room_id_unavailable'], unknown=unknown)
    counts = {room: sum((value == room for value in groups.values())) for room in set(groups.values())}
    wanted_rooms = set(expected_rooms or ())
    wrong_rooms = sorted(set(counts) - wanted_rooms) if wanted_rooms else []
    missing_rooms = sorted(wanted_rooms - set(counts))
    count_mismatch = {}
    if expected_counts is not None:
        normalized = {str(room): int(value) for room, value in expected_counts.items()}
        count_mismatch = {room: {'expected': wanted, 'observed': counts.get(room, 0)} for room, wanted in normalized.items() if counts.get(room, 0) != wanted}
        wrong_rooms = sorted(set(counts) - set(normalized))
    if wrong_rooms or missing_rooms or count_mismatch:
        return _receipt(FAIL, ['object_room_groups_do_not_match_truth'], observed_counts=counts, wrong_rooms=wrong_rooms, missing_rooms=missing_rooms, count_mismatch=count_mismatch)
    return _receipt(PASS, ['object_room_groups_confirmed'], observed_counts=counts, bindings=groups)

def _both_in_room_check(aliases: set[str], trace: Mapping, kind_x: str, kind_y: str, answer_room: str, *, reject_alternative: bool) -> dict:
    if not isinstance(trace.get('observations'), list):
        return _receipt(UNKNOWN, ['observation_trace_unavailable'])
    by_kind = {kind_x: {}, kind_y: {}}
    observed_aliases, missing_kind, missing_room = (set(), set(), set())
    for alias in sorted(aliases):
        rows = _object_rows(trace, alias)
        if not rows:
            continue
        observed_aliases.add(alias)
        typed = False
        for _idx, row in rows:
            kind = row.get('kind')
            room = row.get('object_room_id')
            if kind is None:
                continue
            typed = True
            kind = str(kind)
            if kind not in by_kind:
                continue
            if room is None:
                missing_room.add(alias)
                continue
            by_kind[kind].setdefault(str(room), set()).add(alias)
        if not typed:
            missing_kind.add(alias)
    if missing_kind or missing_room:
        return _receipt(UNKNOWN, ['typed_object_room_telemetry_unavailable'], missing_kind=sorted(missing_kind), missing_room=sorted(missing_room))
    room_count_both = set(by_kind[kind_x]) & set(by_kind[kind_y])
    if answer_room not in room_count_both:
        return _receipt(FAIL, ['answer_room_pair_not_observed'], answer_room=answer_room, room_count_both=sorted(room_count_both), observed_aliases=sorted(observed_aliases))
    alternatives = sorted(room_count_both - {answer_room})
    if reject_alternative and alternatives:
        return _receipt(FAIL, ['second_positive_pair_room_observed'], answer_room=answer_room, alternative_rooms=alternatives, room_count_both=sorted(room_count_both))
    witnesses = sorted(by_kind[kind_x][answer_room] | by_kind[kind_y][answer_room])
    return _receipt(PASS, ['unique_answer_room_pair_confirmed'] if reject_alternative else ['answer_room_pair_confirmed'], answer_room=answer_room, room_count_both=sorted(room_count_both), witnesses=witnesses)

def _point(row: Mapping):
    value = row.get('world_point_from_depth')
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        point = tuple((float(v) for v in value))
    except (TypeError, ValueError):
        return None
    return point if all((math.isfinite(v) for v in point)) else None

def _best_point(alias: str, trace: Mapping):
    candidates = [(row.get('pixel_count', 0) or 0, _point(row), row) for _idx, row in _object_rows(trace, alias) if _point(row) is not None]
    return max(candidates, default=(0, None, None), key=lambda item: item[0])[1:]

def _spatial_points(aliases: set[str], trace: Mapping) -> tuple[dict, dict]:
    points, missing = ({}, [])
    for alias in aliases:
        point, row = _best_point(alias, trace)
        if point is None:
            missing.append(alias)
        else:
            points[alias] = {'xyz': point, 'error_m': float(row.get('position_error_m', POSITION_ERROR_FLOOR_M))}
    status = PASS if not missing else UNKNOWN
    return (points, _receipt(status, ['depth_positions_available'] if not missing else ['depth_position_unavailable'], missing=sorted(missing)))

def _distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])

def _family(spec, trace: Mapping) -> tuple[dict, dict]:
    family = str(_get(spec, 'family', ''))
    truth = _truth(spec)
    evidence = _set(_get(spec, 'evidence', ()))
    answers = _set(_get(spec, 'answer_instances', ()))
    if family == 'room_localization':
        target = str(truth.get('alias', ''))
        target_witness = _room_localization_target_witness(spec, trace, target)
        open_check = _room_localization_open_before_visible(spec, trace, target) if target_witness.get('exact_target') is not False else _receipt(PASS, ['answer_equivalent_target_needs_no_exact_open'])
        necessary = _combine([target_witness, open_check])
        expected = {target: str(truth.get('room_id') or truth.get('room') or '')}
        sufficient = _combine([necessary, _room_binding(expected, trace), _clear_all({target}, trace)])
    elif family == 'surface_contents':
        host = str(truth.get('host', ''))
        children = {str(c.get('alias', '')) for c in truth.get('children', ())}
        necessary = _aliases({host} | children, trace, 'surface_contents')
        if 'placement_relations' not in trace:
            relation_check = _receipt(UNKNOWN, ['placement_relations_unavailable'])
        else:
            relations = _relations(trace)
            bad, unknown = ([], [])
            for child in children:
                observed_supports = [row.get('support') for _idx, row in _object_rows(trace, child) if 'support' in row]
                if observed_supports:
                    known_supports = {str(v) for v in observed_supports if v is not None}
                    if known_supports - {host}:
                        bad.append(child)
                    elif host not in known_supports:
                        unknown.append(child)
                    continue
                rel = relations.get(child)
                if not rel or rel.get('relation') == 'unknown':
                    unknown.append(child)
                elif rel.get('relation') != 'on' or str(rel.get('host')) != host:
                    bad.append(child)
            relation_check = _receipt(FAIL, ['child_not_supported_by_host'], failed=bad) if bad else _receipt(UNKNOWN, ['child_support_unknown'], unknown=unknown) if unknown else _receipt(PASS, ['all_children_supported_by_host'])
        sufficient = _combine([necessary, relation_check, _clear_all(children | {host}, trace)])
    elif family == 'relocation':
        source, destination = (str(truth.get('object', '')), str(truth.get('dest', '')))
        picks = []
        if 'interaction_events' not in trace:
            structured = _receipt(UNKNOWN, ['interaction_events_unavailable'])
        else:
            interactions = _events(trace)
            picks = _successful_event(trace, 'pick', source)
            places = [e for e in _events(trace, ('place',)) if e.get('success') is True and str(e.get('source') or e.get('held') or '') == source and (str(e.get('destination') or e.get('support') or '') == destination)]
            pick_ids, place_ids = ({id(event) for event in picks}, {id(event) for event in places})
            pick_positions = [i for i, event in enumerate(interactions) if id(event) in pick_ids]
            place_positions = [i for i, event in enumerate(interactions) if id(event) in place_ids]
            ordered = any((pick < place for pick in pick_positions for place in place_positions))
            structured = _receipt(PASS if ordered else FAIL, ['structured_pick_and_place_confirmed'] if ordered else ['structured_pick_or_place_missing'])
        pick_cutoff = {}
        if picks:
            try:
                pick_cutoff[source] = min((int(e.get('after_observation_index', -1)) for e in picks))
            except (TypeError, ValueError):
                pass
        necessary = _combine([structured, _inside_open_before_visible(spec, trace, {source}, cutoff_indices=pick_cutoff)])
        stable = [e for e in _events(trace, ('place',)) if e.get('success') is True and str(e.get('source') or e.get('held') or '') == source and (str(e.get('support') or '') == destination) and (e.get('stable_after_settle') is True)]
        stability = _receipt(UNKNOWN, ['stable_placement_readback_unavailable']) if 'interaction_events' not in trace or not any(('stable_after_settle' in e for e in _events(trace, ('place',)))) else _receipt(PASS if stable else FAIL, ['stable_destination_placement'] if stable else ['placement_not_stable_on_destination'])
        sufficient = _combine([necessary, stability])
    elif family in {'object_count', 'count_in_room', 'room_count', 'room_comparison'}:
        targets = evidence or answers
        physical_coverage = _coverage(spec, trace)
        necessary_parts = [_aliases(targets, trace, 'counted_instances'), _visual_room_coverage(spec, trace)]
        grouping = None
        if family in {'room_count', 'count_in_room'}:
            rooms = _set(truth.get('rooms'))
            grouping = _room_grouping(targets, trace, expected_rooms=rooms) if rooms else None
        elif family == 'room_comparison':
            per_room = truth.get('per_room') or {}
            grouping = _room_grouping(targets, trace, expected_counts=per_room) if per_room else None
        necessary = _combine(necessary_parts + ([grouping] if grouping else []))
        sufficient = _combine([necessary, physical_coverage, _clear_all(targets, trace)])
    elif family == 'both_in_room':
        answer_room = str(truth.get('answer', ''))
        kx, ky = (str(truth.get('kind_x', '')), str(truth.get('kind_y', '')))
        necessary = _both_in_room_check(answers, trace, kx, ky, answer_room, reject_alternative=False)
        all_candidates = evidence or answers
        candidate_visibility = _aliases(all_candidates, trace, 'both_in_room_candidates')
        exclusion = _both_in_room_check(all_candidates, trace, kx, ky, answer_room, reject_alternative=True)
        witnesses = _set(necessary.get('witnesses'))
        sufficient = _combine([necessary, _coverage(spec, trace), candidate_visibility, exclusion, _clear_all(witnesses, trace) if witnesses else _receipt(UNKNOWN, ['answer_pair_witnesses_unavailable'])])
    elif family in {'count_absent', 'unanswerable'}:
        necessary = _visual_room_coverage(spec, trace)
        physical_coverage = _coverage(spec, trace)
        containers = _set(_get(spec, 'search_containers', ()))
        if not containers:
            sufficient = _combine([necessary, physical_coverage])
        elif 'interaction_events' not in trace:
            sufficient = _combine([necessary, physical_coverage, _receipt(UNKNOWN, ['interaction_events_unavailable'])])
        else:
            opened = {str(e.get('target', '')) for e in _events(trace, ('open',)) if e.get('success') is True}
            missing = containers - opened
            sufficient = _combine([necessary, physical_coverage, _receipt(FAIL if missing else PASS, ['declared_search_containers_missing'] if missing else ['declared_search_containers_checked'], missing=sorted(missing))])
    elif family == 'hidden_count':
        raw_answer_instances = _get(spec, 'answer_instances', None)
        targets = _set(raw_answer_instances) if raw_answer_instances is not None else evidence
        necessary = _combine([_aliases(targets, trace, 'hidden_count_targets'), _coverage(spec, trace), _inside_open_before_visible(spec, trace, targets)])
        containers = _set(_get(spec, 'search_containers', ()))
        if not containers:
            container_check = _receipt(PASS, ['no_search_containers_declared'])
        elif 'interaction_events' not in trace:
            container_check = _receipt(UNKNOWN, ['interaction_events_unavailable'])
        else:
            opened = {str(e.get('target', '')) for e in _events(trace, ('open',)) if e.get('success') is True}
            missing = containers - opened
            container_check = _receipt(FAIL if missing else PASS, ['declared_search_containers_missing'] if missing else ['declared_search_containers_checked'], missing=sorted(missing))
        sufficient = _combine([necessary, _clear_all(targets, trace), container_check])
    elif family == 'state_count':
        targets = evidence or answers
        visible = _aliases(targets, trace, 'state_candidates')
        if 'interaction_events' not in trace:
            initial = _receipt(UNKNOWN, ['interaction_events_unavailable'])
        else:
            changed = {str(e.get('target', '')) for e in _events(trace, ('open', 'close')) if e.get('success') is True and str(e.get('target', '')) in targets}
            missing_state = [a for a in targets if not any((row.get('is_open') is not None for _idx, row in _object_rows(trace, a)))]
            initial = _receipt(FAIL, ['candidate_state_was_modified'], changed=sorted(changed)) if changed else _receipt(UNKNOWN, ['initial_state_observation_unavailable'], missing=sorted(missing_state)) if missing_state else _receipt(PASS, ['initial_states_seen_before_any_change'])
        necessary = _combine([visible, _coverage(spec, trace), initial])
        sufficient = _combine([necessary, _clear_all(targets, trace)])
    elif family in {'relative_direction', 'object_distance', 'nearer_object_comparison', 'gap_width'}:
        targets = evidence
        necessary = _aliases(targets, trace, 'spatial_referents')
        points, available = _spatial_points(targets, trace)
        spatial = available
        if available['status'] == PASS and family == 'object_distance':
            a, b = (str(truth.get('a', '')), str(truth.get('b', '')))
            measured = _distance(points[a]['xyz'], points[b]['xyz'])
            expected = float(truth.get('metres', measured))
            tol = float(truth.get('tol_m', _get(spec, 'answer_tolerance', 0.0)) or 0.0)
            spatial = _receipt(PASS if abs(measured - expected) <= tol else FAIL, ['reconstructed_distance_matches'] if abs(measured - expected) <= tol else ['reconstructed_distance_mismatch'], measured_m=measured, expected_m=expected, tolerance_m=tol)
        elif available['status'] == PASS and family == 'nearer_object_comparison':
            origin, winner = (str(truth.get('x', '')), str(truth.get('winner', '')))
            distances = {a: _distance(points[origin]['xyz'], points[a]['xyz']) for a in _set(truth.get('choices'))}
            ranked = sorted(distances, key=distances.get)
            if not ranked:
                spatial = _receipt(UNKNOWN, ['nearer_object_comparison_choices_unavailable'])
            else:
                best = ranked[0]
                overlap = False
                if len(ranked) > 1:
                    a, b = ranked[:2]
                    ea = points[origin]['error_m'] + points[a]['error_m']
                    eb = points[origin]['error_m'] + points[b]['error_m']
                    overlap = distances[a] + ea >= distances[b] - eb
                spatial = _receipt(UNKNOWN, ['nearest_error_intervals_overlap'], distances=distances) if overlap else _receipt(PASS if best == winner else FAIL, ['reconstructed_nearest_matches'] if best == winner else ['reconstructed_nearest_mismatch'], distances=distances)
        elif available['status'] == PASS and family == 'relative_direction':
            o, f, t = (str(truth.get(k, '')) for k in ('origin', 'forward', 'target'))

            def bearing(a, b):
                return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1]))
            deg = (bearing(points[o]['xyz'], points[t]['xyz']) - bearing(points[o]['xyz'], points[f]['xyz'])) % 360.0
            expected = float(truth.get('sector_deg', 0.0)) % 360.0
            error = abs((deg - expected + 180.0) % 360.0 - 180.0)
            tol = float(_get(spec, 'answer_tolerance', truth.get('tol_deg', 22.5)) or 22.5)
            spatial = _receipt(PASS if error <= tol else FAIL, ['reconstructed_direction_matches'] if error <= tol else ['reconstructed_direction_mismatch'], measured_deg=deg, expected_deg=expected, tolerance_deg=tol)
        elif family == 'gap_width' and available['status'] == PASS:
            a, b = (str(truth.get('a', '')), str(truth.get('b', '')))
            measurements = []
            for obs in _observations(trace):
                rows = {str(row.get('alias', '')): row for row in _evidence_objects(obs)}
                if not {a, b} <= set(rows):
                    continue

                def edges(row):
                    out = []
                    for value in (row.get('edge_world_points_from_depth') or {}).values():
                        if isinstance(value, (list, tuple)) and len(value) == 3:
                            try:
                                point = tuple((float(v) for v in value))
                            except (TypeError, ValueError):
                                continue
                            if all((math.isfinite(v) for v in point)):
                                out.append(point)
                    return out
                ea, eb = (edges(rows[a]), edges(rows[b]))
                if ea and eb:
                    measurements.append(min((_distance(x, y) for x in ea for y in eb)) * 100.0)
            if not measurements:
                spatial = _receipt(UNKNOWN, ['same_frame_depth_edges_unavailable'])
            else:
                measured = min(measurements)
                expected = float(truth.get('cm', measured))
                tol = float(truth.get('tol_cm', _get(spec, 'answer_tolerance', 0.0)) or 0.0)
                matches = abs(measured - expected) <= tol
                spatial = _receipt(PASS if matches else FAIL, ['reconstructed_gap_matches'] if matches else ['reconstructed_gap_mismatch'], measured_cm=measured, expected_cm=expected, tolerance_cm=tol)
        sufficient = _combine([necessary, spatial, _clear_all(targets, trace)])
    elif family == 'path_integration':
        necessary = _aliases(evidence, trace, 'landmarks')
        sufficient = _combine([necessary, _coverage(spec, trace, exhaustive=False), _viewpoint_coverage(spec, trace)])
    elif family == 'change_detect':
        candidates = _set(truth.get('choices'))
        target, op = (str(truth.get('target', '')), str(truth.get('op', '')))
        if 'mutations' not in trace:
            necessary = sufficient = _receipt(UNKNOWN, ['mutation_telemetry_unavailable'])
        else:
            mutations = [e for e in trace.get('mutations', ()) if str(e.get('alias', '')) == target and str(e.get('op', '')) == op and (e.get('success') is True)]
            sequence = _receipt(UNKNOWN, ['mutation_boundary_unavailable'])
            if len(mutations) != 1:
                mutation = _receipt(FAIL, ['expected_one_successful_target_mutation'])
            else:
                boundary = int(mutations[0].get('after_observation_index', -1))
                prior = next((obs for i, obs in enumerate(_observations(trace)) if _index(obs, i) == boundary), None)
                hidden = prior is not None and target not in _visible_aliases(prior)
                others = [e for e in trace.get('mutations', ()) if e.get('success') is True and e is not mutations[0]]
                mutation = _receipt(PASS if hidden else FAIL, ['mutation_hidden_from_actual_model_frame'] if hidden else ['mutation_target_visible_in_model_frame'], extra_mutations=len(others))
                before, after = (set(), set())
                for i, obs in enumerate(_observations(trace)):
                    (before if _index(obs, i) <= boundary else after).update(_visible_aliases(obs) & candidates)
                missing_before, missing_after = (candidates - before, candidates - after)
                sequence = _receipt(FAIL if missing_before or missing_after else PASS, ['candidates_missing_before_or_after'] if missing_before or missing_after else ['all_candidates_seen_before_and_after'], missing_before=sorted(missing_before), missing_after=sorted(missing_after))
            necessary = _combine([sequence, mutation])
            all_mutations = [e for e in trace.get('mutations', ()) if e.get('success') is True]
            sufficient = _combine([necessary, _clear_all(candidates, trace), _receipt(PASS if len(all_mutations) == 1 else FAIL, ['no_extra_mutations'] if len(all_mutations) == 1 else ['extra_successful_mutations'])])
    elif family == 'room_area':
        room = str(truth.get('room', ''))
        if not isinstance(trace.get('observations'), list):
            necessary = sufficient = _receipt(UNKNOWN, ['observation_trace_unavailable'])
        else:
            necessary = _receipt(PASS if room in _visited(trace) else FAIL, ['target_room_visited'] if room in _visited(trace) else ['target_room_not_visited'])
            room_observations = [obs for obs in _observations(trace) if str(obs.get('room_id')) == room]
            measured_depth = [obs for obs in room_observations if isinstance(obs.get('depth_summary'), Mapping)]
            if not measured_depth:
                boundary = _receipt(UNKNOWN, ['boundary_depth_telemetry_unavailable'])
            else:
                sectors = {int(float((obs.get('pose') or {}).get('yaw_deg')) % 360 // 90) for obs in measured_depth if isinstance(obs.get('pose'), Mapping) and (obs.get('pose') or {}).get('yaw_deg') is not None and (float((obs.get('depth_summary') or {}).get('valid_fraction', 0.0) or 0.0) >= DEPTH_VALID_MIN_FRACTION) and ((obs.get('depth_summary') or {}).get('median_m') is not None)}
                boundary = _receipt(PASS if len(sectors) >= 3 else FAIL, ['multiple_depth_backed_boundary_directions_observed'] if len(sectors) >= 3 else ['insufficient_depth_backed_boundary_directions'], direction_sectors=sorted(sectors), min_depth_valid_fraction=DEPTH_VALID_MIN_FRACTION)
            sufficient = _combine([necessary, boundary])
    elif family == 'reach_blocked':
        winner = str(truth.get('winner', ''))
        candidates = {winner} | _set(truth.get('choices'))
        if 'approaches' not in trace:
            attempts = _receipt(UNKNOWN, ['structured_approaches_unavailable'])
        else:
            rows = {alias: [e for e in trace.get('approaches', ()) if str(e.get('alias', '')) == alias] for alias in candidates}
            missing = sorted((alias for alias, events in rows.items() if not events))
            winner_failed = any((e.get('success') is False for e in rows.get(winner, ())))
            winner_succeeded = any((e.get('success') is True for e in rows.get(winner, ())))
            free_failed = sorted((alias for alias, events in rows.items() if alias != winner and (not any((e.get('success') is True for e in events)))))
            ok = not missing and winner_failed and (not winner_succeeded) and (not free_failed)
            attempts = _receipt(PASS if ok else FAIL, ['all_candidates_physically_tested'] if ok else ['candidate_approach_outcomes_incomplete'], missing=missing, distractors_without_success=free_failed, blocked_target=winner)
        necessary = _combine([_aliases(candidates, trace, 'reach_candidates'), attempts])
        sufficient = _combine([necessary, _clear_all(candidates, trace)])
    elif family == 'topo_route':
        necessary = _receipt(UNKNOWN, ['topo_route_semantics_not_frozen'], observed_edges=trace.get('observed_edges', []))
        sufficient = necessary
    else:
        necessary = sufficient = _receipt(UNKNOWN, ['family_policy_not_defined'])
    return (necessary, sufficient)

def judge(spec, trace: Mapping | None, answer_success: bool) -> dict:
    base = {'policy_version': POLICY_VERSION, 'answer_success': bool(answer_success), 'thresholds': {'clear_min_pixels': CLEAR_MIN_PIXELS, 'clear_min_fraction': CLEAR_MIN_FRACTION, 'position_error_floor_m': POSITION_ERROR_FLOOR_M, 'depth_valid_min_fraction': DEPTH_VALID_MIN_FRACTION}}
    if not isinstance(trace, Mapping):
        unknown = _receipt(UNKNOWN, ['enhanced_trace_unavailable'])
        return _binary_receipt({**base, 'status': 'scored', 'necessary_evidence': unknown, 'sufficient_evidence': unknown, 'strict_success': False})
    necessary, sufficient = _family(spec, trace)
    strict = bool(answer_success and necessary['status'] == PASS)
    return _binary_receipt({**base, 'status': 'scored', 'necessary_evidence': necessary, 'sufficient_evidence': sufficient, 'strict_success': strict})
