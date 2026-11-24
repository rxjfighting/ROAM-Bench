"""Tour optimizer utilities for ROAM-Bench."""
from __future__ import annotations
import contextlib
from contextvars import ContextVar
_VISIT_EVIDENCE = ContextVar('tour_visit_evidence', default=None)

@contextlib.contextmanager
def visit_evidence(provider):
    token = _VISIT_EVIDENCE.set(provider)
    try:
        yield
    finally:
        _VISIT_EVIDENCE.reset(token)

def optimized_tour_stops(baselines, inv, targets, enclosed, standing=None):
    facts = {str(f.alias): f for f in inv}
    target_stops, seen = ([], set())
    for alias in targets:
        stop = str(enclosed.get(alias, alias))
        if stop not in seen:
            seen.add(stop)
            target_stops.append(stop)
    enclosed_stops = {str(stop) for stop in enclosed.values()}
    target_stops.sort(key=lambda stop: (stop in enclosed_stops, str(facts.get(stop).room_id or facts.get(stop).room if facts.get(stop) is not None else ''), stop))
    ordinary_stops = [stop for stop in target_stops if stop not in enclosed_stops]
    deferred_containers = [stop for stop in target_stops if stop in enclosed_stops]
    stops, covered_rooms = (list(ordinary_stops), set())
    for alias in targets:
        stop = str(enclosed.get(alias, alias))
        fact = facts.get(stop) or facts.get(str(alias))
        if fact is not None:
            room = fact.room_id or fact.room
            if room:
                covered_rooms.add(room)
    handles = baselines.room_tour_handles(inv, standing=standing)
    for room, aliases in sorted(handles.items()):
        if not aliases:
            continue
        alias = aliases[0]
        if room in covered_rooms or alias in seen:
            continue
        seen.add(alias)
        stops.append(alias)
    stops.extend(deferred_containers)
    return stops

def optimized_sweep_here(baselines, sim, rec, want, min_pixel_frac=None, yaws=None, pitches=None):
    if min_pixel_frac is None:
        min_pixel_frac = baselines.COUNT_PIXEL_FRAC
    if yaws is None:
        yaws = baselines.COUNT_YAWS
    if pitches is None:
        pitches = baselines.COUNT_PITCHES
    want = set(want)
    if want:
        inventory = getattr(sim, 'inventory', None)
        facts = {str(f.alias): f for f in inventory()} if callable(inventory) else {}
        high_tokens = ('ceiling', 'pendant', 'track_light', 'flush_mount', 'downlight', 'smoke_detector')
        if any((any((token in f"{alias}_{getattr(facts.get(alias), 'kind', '')}".lower() for token in high_tokens)) for alias in want)):
            pitches = tuple(dict.fromkeys((*pitches, -60.0)))
    seen = set(sim.visible_instances(min_pixel_frac)) & want
    yaw_now = 0.0
    start_pitch = pitch_now = baselines._horizon_of(sim)
    done = bool(want and want <= seen)
    for index, yaw in enumerate(yaws):
        if done:
            break
        if abs(yaw - yaw_now) > 0.5:
            if not baselines._rec_step(sim, rec, 'turn', yaw - yaw_now).success:
                continue
            yaw_now = yaw
        if pitch_now is None:
            seen |= set(sim.visible_instances(min_pixel_frac)) & want
            done = bool(want and want <= seen)
            continue
        ordered_pitches = pitches if index % 2 == 0 else tuple(reversed(pitches))
        for pitch in ordered_pitches:
            if abs(pitch - pitch_now) > 0.5:
                if not baselines._rec_step(sim, rec, 'look', pitch_now - pitch).success:
                    continue
                pitch_now = pitch
            seen |= set(sim.visible_instances(min_pixel_frac)) & want
            if want and want <= seen:
                done = True
                break
    if pitch_now is not None and abs(pitch_now - start_pitch) > 0.5:
        baselines._rec_step(sim, rec, 'look', pitch_now - start_pitch)
    return seen

@contextlib.contextmanager
def optimize_tours(enabled: bool=True):
    if not enabled:
        yield
        return
    from roam_bench.protocol import baselines
    original_stops = baselines._tour_stops
    original_sweep = baselines._sweep_here
    original_look_inside = baselines._look_inside
    last_sweep_seen = set()

    def patched_stops(inv, targets, enclosed, standing=None):
        return optimized_tour_stops(baselines, inv, targets, enclosed, standing)

    def patched_sweep(sim, rec, want, min_pixel_frac=baselines.COUNT_PIXEL_FRAC, yaws=baselines.COUNT_YAWS, pitches=baselines.COUNT_PITCHES):
        if want and tuple(pitches) == tuple(baselines.COUNT_PITCHES):
            pitches = tuple(pitches) + (-30.0,)
        prior = set()
        provider = _VISIT_EVIDENCE.get()
        if want and callable(provider):
            try:
                prior = set(map(str, provider())) & set(map(str, want))
            except (AttributeError, TypeError, ValueError):
                prior = set()
        if want and set(map(str, want)) <= prior:
            seen = set(map(str, want))
        else:
            seen = optimized_sweep_here(baselines, sim, rec, want, min_pixel_frac, yaws, pitches)
        last_sweep_seen.clear()
        last_sweep_seen.update(map(str, seen))
        return seen

    def patched_look_inside(sim, rec, stop, want, min_pixel_frac=baselines.COUNT_PIXEL_FRAC, targets_here=None):
        local = None if targets_here is None else set(targets_here)
        if local is not None and local <= last_sweep_seen:
            return set()
        remaining = set(want) - last_sweep_seen
        remaining_local = None if local is None else local - last_sweep_seen
        return original_look_inside(sim, rec, stop, remaining, min_pixel_frac, remaining_local)
    baselines._tour_stops = patched_stops
    baselines._sweep_here = patched_sweep
    baselines._look_inside = patched_look_inside
    try:
        yield
    finally:
        baselines._tour_stops = original_stops
        baselines._sweep_here = original_sweep
        baselines._look_inside = original_look_inside
