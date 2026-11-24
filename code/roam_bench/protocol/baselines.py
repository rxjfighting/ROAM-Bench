"""Baselines utilities for ROAM-Bench."""
import math
import random
import re
from .score import DIRECTION_WORDS, SECTOR_WORDS, Recorder, bearing_delta, bearing_to_point, direction_word, match_answer
from .types import NameRef
_CAMEL = re.compile('[A-Z][a-z0-9]*|[a-z0-9]+')
_INDEX = re.compile('_\\d+$')
TURNS = (-90.0, -45.0, 45.0, 90.0)
MOVES = (0.25, 0.5, 1.0)
CONTENTS_YAWS = (0.0, -45.0, 45.0)
CONTENTS_PITCHES = (0.0, 30.0, 60.0)
INTERACT_HORIZONS = (30.0, 60.0, 0.0, -30.0)
MIN_WORD = 3
CONSTANT_COUNT = 1
WORD_GUESSES = ('yes', 'no', 'i do not know')
DIRECTION_GUESSES = SECTOR_WORDS
_direction_word = direction_word

def _guess_support(task, count_max=None, words=WORD_GUESSES):
    if task.answer_match == 'direction':
        return list(DIRECTION_GUESSES)
    if task.answer_match == 'sequence':
        return list(task.answer_vocabulary) if task.answer_vocabulary else list(words)
    if task.answer_choices:
        return list(task.answer_choices)
    if task.answer_number is not None:
        hi = int(count_max if count_max is not None else count_band(task.answer_number))
        return [str(g) for g in range(0, hi + 1)]
    return list(words)

def _asks_question(task):
    return bool(task.answer_keywords) or task.answer_number is not None or bool(task.answer_from_pose)

def null_rollout(sim, task, answer='no', count_answer=None, on_step=None):
    rec = Recorder(sim, task, on_step=on_step)
    if _asks_question(task):
        rec.answer(str(count_answer if count_answer is not None else CONSTANT_COUNT) if task.answer_number is not None else answer)
    return rec
COUNT_BAND_MIN = 12

def count_band(truth):
    return max(COUNT_BAND_MIN, 2 * int(truth))

def random_answer_progress(task, count_max=None, words=WORD_GUESSES):
    if not _asks_question(task):
        return None
    support = _guess_support(task, count_max, words)
    if not support:
        return 0.0
    if task.answer_from_pose:
        return 1.0 / len(support)
    if task.answer_match == 'sequence':
        L = len(task.answer_keywords)
        V = len(support)
        if L > V or L == 0:
            return 0.0
        perm = 1
        for i in range(V, V - L, -1):
            perm *= i
        return 1.0 / (V * perm)
    hits = 0
    for guess in support:
        ok = match_answer(guess, task.answer_keywords, task.answer_match, std=task.answer_number, tol=task.answer_tolerance, vocabulary=getattr(task, 'answer_vocabulary', ()), choices=getattr(task, 'answer_choices', ()), unit=getattr(task, 'answer_unit', ''))
        hits += 1 if ok else 0
    return hits / len(support)
DENIAL_STEM = 'there is no '

def denial_answer_progress(task, nouns):
    if not _asks_question(task):
        return None
    guess = DENIAL_STEM + ' and no '.join(nouns) if nouns else 'there is nothing here'
    ok = match_answer(guess, task.answer_keywords, task.answer_match, std=task.answer_number, tol=task.answer_tolerance, vocabulary=getattr(task, 'answer_vocabulary', ()), choices=getattr(task, 'answer_choices', ()), unit=getattr(task, 'answer_unit', ''))
    return 1.0 if ok else 0.0

def random_rollout(sim, task, seed=0, steps=None, answer=None, on_step=None, count_max=None):
    rng = random.Random(seed)
    rec = Recorder(sim, task, on_step=on_step)
    n = steps if steps is not None else task.max_steps
    for _ in range(n):
        if rng.random() < 0.5:
            rec.step('turn', None, sim.turn(rng.choice(TURNS)))
        else:
            rec.step('move', None, sim.move(rng.choice(MOVES)))
    if _asks_question(task):
        support = _guess_support(task, count_max)
        if answer is not None and task.answer_number is None:
            rec.answer(answer)
        elif task.answer_match == 'sequence':
            k = rng.randint(1, len(support))
            rec.answer(', '.join(rng.sample(support, k)))
        else:
            rec.answer(rng.choice(support))
    return rec
CONSTANT_DIRECTION = 180

def prior_rollout(sim, task, prior, kinds=None, on_prior=None, receptacle_kinds=None, extra_words=None, count_prior=None, direction_prior=None, choice_prior=None, word_prior=None):
    from .gates import prior_answer_for, prior_contents_for
    rec = Recorder(sim, task)
    if _asks_question(task):
        if task.answer_match == 'direction':
            rec.answer(_direction_word(direction_prior if direction_prior is not None else CONSTANT_DIRECTION))
            return rec
        if task.answer_choices:
            rec.answer(choice_prior if choice_prior is not None else task.answer_choices[0])
            return rec
        if task.answer_number is not None:
            rec.answer(str(int(count_prior if count_prior is not None else CONSTANT_COUNT)))
            return rec
        if kinds is None:
            kinds = sorted({f.kind for f in sim.inventory()})
        if receptacle_kinds is None:
            receptacle_kinds = sorted({f.kind for f in sim.inventory() if f.receptacle})
        rec.answer(' '.join((x for x in (prior_answer_for(kinds, prior, extra_words), prior_contents_for(receptacle_kinds, on_prior), word_prior or '') if x)).strip())
    return rec

def oracle_rollout(sim, task, plan, answer=None, look_around=4, interact_horizons=INTERACT_HORIZONS, on_step=None):
    rec = Recorder(sim, task, on_step=on_step)
    for verb, ref in plan:
        if verb == 'navigate':
            rec.step('navigate', ref, sim.navigate_to_object(ref))
        elif verb == 'go':
            res = sim.go_to_visible_object(ref)
            turned = 0
            while not res.success and turned < look_around:
                sim.turn(90.0)
                turned += 1
                res = sim.go_to_visible_object(ref)
            rec.step('go', ref, res)
        elif verb == 'move':
            rec.step('move', ref, sim.move(float(ref)))
        elif verb == 'turn':
            rec.step('turn', ref, sim.turn(float(ref)))
        elif verb == 'look':
            rec.step('look', ref, sim.look(float(ref)))
        elif verb == 'propose':
            r = sim.propose(str(ref))
            rec.step('propose', ref, _AsResult(r.accepted, r.reason))
        else:
            _interact_sweeping(sim, rec, verb, ref, interact_horizons)
    if answer is not None:
        rec.answer(answer(sim, rec) if callable(answer) else answer)
    return rec

def _interact_sweeping(sim, rec, verb, ref, horizons):
    res = sim.interact(verb, ref)
    rec.step(verb, ref, res)
    if res.success:
        return res
    for want in horizons or ():
        cur = _horizon_of(sim)
        if cur is None or abs(want - cur) < 0.5:
            continue
        look = sim.look(cur - want)
        rec.step('look', cur - want, look)
        if not look.success:
            continue
        res = sim.interact(verb, ref)
        rec.step(verb, ref, res)
        if res.success:
            return res
    return res

def _horizon_of(sim):
    p = sim.agent_pose()
    return None if p is None else float(p.horizon_deg)

def words_for_alias(alias):
    stem = _INDEX.sub('', str(alias))
    parts = [p for p in stem.split('_') if p]
    if not parts:
        return (str(alias),)
    if len(parts) == 1:
        return (parts[0],)
    out = [' '.join(parts), ''.join(parts)]
    for p in (parts[0], parts[-1]):
        if len(p) >= MIN_WORD and p not in out:
            out.append(p)
    return tuple(dict.fromkeys(out))

def _spell_label(label):
    if not label:
        return ''
    parts = [p.lower() for p in _CAMEL.findall(label)]
    return ' '.join(dict.fromkeys([label, label.lower(), ' '.join(parts), ''.join(parts)]))

def read_room_here(sim, rec=None):
    f = getattr(sim, 'agent_room_name', None)
    name = f() if callable(f) else None
    return ' '.join((p for p in (_spell_label(sim.agent_room()), name) if p))

def read_surface_contents(host, min_pixel_frac=None, yaws=CONTENTS_YAWS, pitches=CONTENTS_PITCHES, target_aliases=None, option_aliases=None):
    reviewed = None if target_aliases is None else set(target_aliases)

    def visible(sim):
        if reviewed is None:
            return _visible_on(sim, host, min_pixel_frac)
        return set(sim.visible_instances(min_pixel_frac)) & reviewed

    def read(sim, rec=None):
        seen = set()
        yaw_now = 0.0
        start_pitch = pitch_now = _horizon_of(sim)
        for yaw in yaws:
            if abs(yaw - yaw_now) > 0.5:
                if not _rec_step(sim, rec, 'turn', yaw - yaw_now).success:
                    continue
                yaw_now = yaw
            if pitch_now is None:
                seen |= visible(sim)
                continue
            for want in pitches:
                if abs(want - pitch_now) > 0.5:
                    if not _rec_step(sim, rec, 'look', pitch_now - want).success:
                        continue
                    pitch_now = want
                seen |= visible(sim)
        if pitch_now is not None and abs(pitch_now - start_pitch) > 0.5:
            _rec_step(sim, rec, 'look', pitch_now - start_pitch)
        if abs(yaw_now) > 0.5:
            _rec_step(sim, rec, 'turn', -yaw_now)
        if option_aliases is not None:
            return ', '.join((key for key, alias in option_aliases.items() if alias in seen))
        words = []
        for a in sorted(seen):
            words.extend(words_for_alias(a))
        return ' '.join(dict.fromkeys(words))
    return read
COUNT_YAWS = (0.0, 90.0, 180.0, 270.0)
COUNT_PITCHES = (0.0, 30.0)
COUNT_PIXEL_FRAC = 0.0008
CHANGE_DETECT_TOUR_LAPS = 2

class _NoRec:

    def step(self, *a, **k):
        return None
_NO_REC = _NoRec()

def _closed_openable_hosts(inv):
    by_alias = {f.alias: f for f in inv}
    out = {}
    for f in inv:
        for host in f.inside:
            h = by_alias.get(host)
            if h is not None and h.openable and (h.is_open is False):
                out[f.alias] = host
    return out

def room_landmarks(inv):
    best = {}
    for f in inv:
        place = f.room_id or f.room
        if not place or f.pickupable or (not f.receptacle):
            continue
        cur = best.get(place)
        if cur is None or (cur.openable and (not f.openable)):
            best[place] = f
    return {k: v.alias for k, v in best.items()}

def room_tour_handles(inv, standing=None, per_room=3):
    measured = standing is not None
    old = room_landmarks(inv)
    rows = {}
    for f in inv:
        place = f.room_id or f.room
        if not place:
            continue
        cells = 0 if not measured else int((standing or {}).get(f.alias, 0) or 0)
        old_first = 0 if old.get(place) == f.alias else 1
        tier = (0 if not f.pickupable else 1, 0 if cells > 0 else 1, -cells, old_first, 0 if f.receptacle and (not f.openable) else 1, f.alias)
        rows.setdefault(place, []).append((tier, f.alias))
    return {room: tuple((alias for _tier, alias in sorted(options)[:max(1, int(per_room))])) for room, options in rows.items()}

def _standing_room_for_tour(sim):
    query = getattr(sim, 'standing_room', None)
    if not callable(query):
        return None
    try:
        return dict(query())
    except (AttributeError, NotImplementedError):
        return None

def room_handles(inv, standing=None):
    best = {}
    for f in inv:
        place = f.room_id or f.room
        if not place:
            continue
        room = None if standing is None else standing.get(f.alias) or 0
        tier = (0 if room != 0 else 1, 0 if not f.pickupable else 1, -(room or 0), 0 if f.receptacle and (not f.pickupable) else 1 if not f.pickupable else 2, 1 if f.openable else 0, f.alias)
        cur = best.get(place)
        if cur is None or tier < cur[0]:
            best[place] = (tier, f.alias)
    return {k: v[1] for k, v in best.items()}

def _tour_stops(inv, targets, enclosed, standing=None):
    stops, seen = ([], set())
    for alias in sorted(targets):
        stop = enclosed.get(alias, alias)
        if stop not in seen:
            seen.add(stop)
            stops.append(stop)
    handles = room_tour_handles(inv, standing=standing)
    for _, aliases in sorted(handles.items()):
        alias = aliases[0]
        if alias not in seen:
            seen.add(alias)
            stops.append(alias)
    return stops

def _room_fallbacks(inv, stop, limit=3, standing=None):
    by_alias = {f.alias: f for f in inv}
    fact = by_alias.get(stop)
    if fact is None:
        return ()
    room = fact.room_id or fact.room
    if not room:
        return ()
    options = room_tour_handles(inv, standing=standing, per_room=limit + 1).get(room, ())
    if standing is not None:
        options = tuple((alias for alias in options if int(standing.get(alias, 0) or 0) > 0))
    return tuple((alias for alias in options if alias != stop))[:limit]

def _in_expected_room(sim, fact):
    query = getattr(sim, 'agent_room', None)
    if not callable(query) or fact is None:
        return True
    expected = {value for value in (fact.room_id, fact.room) if value}
    got = query()
    if got is None and (not bool(getattr(sim, 'strict_room_membership', False))):
        return True
    return not expected or got in expected

def _sweep_here(sim, rec, want, min_pixel_frac=COUNT_PIXEL_FRAC, yaws=COUNT_YAWS, pitches=COUNT_PITCHES):
    seen = set(sim.visible_instances(min_pixel_frac)) & want
    yaw_now = 0.0
    start_pitch = pitch_now = _horizon_of(sim)
    for yaw in yaws:
        if abs(yaw - yaw_now) > 0.5:
            if not _rec_step(sim, rec, 'turn', yaw - yaw_now).success:
                continue
            yaw_now = yaw
        if pitch_now is None:
            seen |= set(sim.visible_instances(min_pixel_frac)) & want
            continue
        for pitch in pitches:
            if abs(pitch - pitch_now) > 0.5:
                if not _rec_step(sim, rec, 'look', pitch_now - pitch).success:
                    continue
                pitch_now = pitch
            seen |= set(sim.visible_instances(min_pixel_frac)) & want
    if pitch_now is not None and abs(pitch_now - start_pitch) > 0.5:
        _rec_step(sim, rec, 'look', pitch_now - start_pitch)
    return seen

def _visit(sim, rec, stop, want, open_first=False, min_pixel_frac=COUNT_PIXEL_FRAC, targets_here=None):
    inv = sim.inventory()
    by_alias = {f.alias: f for f in inv}
    r = sim.navigate_to_object(NameRef(stop))
    (rec or _NO_REC).step('navigate', stop, r)
    arrived_at_target = bool(r.success and _in_expected_room(sim, by_alias.get(stop)))
    fallback_seen = set()
    if not arrived_at_target:
        for fallback in _room_fallbacks(inv, stop, standing=_standing_room_for_tour(sim)):
            fr = sim.navigate_to_object(NameRef(fallback))
            (rec or _NO_REC).step('navigate', fallback, fr)
            if not fr.success or not _in_expected_room(sim, by_alias.get(fallback)):
                continue
            fallback_seen |= _sweep_here(sim, rec, want, min_pixel_frac)
            if not open_first:
                return fallback_seen
            r = sim.navigate_to_object(NameRef(stop))
            (rec or _NO_REC).step('navigate', stop, r)
            arrived_at_target = bool(r.success and _in_expected_room(sim, by_alias.get(stop)))
            if arrived_at_target:
                break
        if not arrived_at_target:
            return fallback_seen
    if not open_first:
        return fallback_seen | _sweep_here(sim, rec, want, min_pixel_frac)
    _interact_sweeping(sim, rec or _NO_REC, 'open', NameRef(stop), INTERACT_HORIZONS)
    seen = fallback_seen | _sweep_here(sim, rec, want, min_pixel_frac)
    seen |= _look_inside(sim, rec, stop, want, min_pixel_frac, targets_here)
    _shut_again(sim, rec, stop)
    return seen
STEP_OUTS = ((135.0, 0.75), (225.0, 0.75), (270.0, 1.25), (90.0, 1.25), (135.0, 1.25), (225.0, 1.25))

def _look_inside(sim, rec, stop, want, min_pixel_frac=COUNT_PIXEL_FRAC, targets_here=None):
    seen = set()
    for deg, dist in STEP_OUTS:
        if targets_here and targets_here <= seen:
            break
        back = sim.navigate_to_object(NameRef(stop))
        (rec or _NO_REC).step('navigate', stop, back)
        if not back.success:
            break
        r = sim.navigate_to_relative(deg, dist)
        (rec or _NO_REC).step('navigate_to_relative', f'{deg:g}deg {dist:g}m', r)
        if not r.success or r.moved_m < 0.05:
            continue
        seen |= _sweep_here(sim, rec, want, min_pixel_frac)
    return seen

def _shut_again(sim, rec, stop):
    direct = sim.interact('close', NameRef(stop))
    (rec or _NO_REC).step('close', NameRef(stop), direct)
    if direct.success:
        return
    if not sim.navigate_to_object(NameRef(stop)).success:
        return
    _interact_sweeping(sim, rec or _NO_REC, 'close', NameRef(stop), INTERACT_HORIZONS)

def _scoped(inv, scope):
    if not scope:
        return tuple(inv)
    keep = set(scope)
    return tuple((f for f in inv if (f.room_id or f.room) in keep))

def _category_targets(inv, kind, target_aliases=None):
    if target_aliases is None:
        return {f.alias for f in inv if f.kind == kind}
    selected = tuple(target_aliases)
    available = {f.alias for f in inv}
    if len(set(selected)) != len(selected) or set(selected) - available:
        raise ValueError('reviewed category members are duplicated or unavailable in the scoped scene')
    return set(selected)

def read_count_seen(kind, room_label=None, open_hosts=False, min_pixel_frac=COUNT_PIXEL_FRAC, scope=None, target_aliases=None):

    def read(sim, rec=None):
        inv = _scoped(sim.inventory(), scope)
        want = _category_targets(inv, kind, target_aliases)
        if room_label is not None:
            want &= {f.alias for f in inv if f.room == room_label}
        enclosed = _closed_openable_hosts(inv)
        to_open = {enclosed[a] for a in want if a in enclosed} if open_hosts else set()
        seen = set()
        for stop in _tour_stops(inv, want, enclosed if open_hosts else {}, _standing_room_for_tour(sim)):
            here = {a for a in want if enclosed.get(a) == stop}
            seen |= _visit(sim, rec, stop, want, stop in to_open, min_pixel_frac, here)
        return str(len(seen))
    return read

def relative_xz(sim, aliases):
    out = {}
    for a in aliases:
        b = sim.bearing_to(a)
        if b is None:
            continue
        deg, dist = b
        th = math.radians(-float(deg))
        out[a] = (float(dist) * math.sin(th), float(dist) * math.cos(th))
    return out

def bearing_from(origin, forward, target):
    fx, fz = (forward[0] - origin[0], forward[1] - origin[1])
    tx, tz = (target[0] - origin[0], target[1] - origin[1])
    if math.hypot(fx, fz) < 1e-06 or math.hypot(tx, tz) < 1e-06:
        return None
    return (math.degrees(math.atan2(tx, tz)) - math.degrees(math.atan2(fx, fz))) % 360.0

def read_relative_direction(origin, forward, target, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        want = {origin, forward, target}
        inv = sim.inventory()
        enclosed = _closed_openable_hosts(inv)
        seen = set()
        for stop in _tour_stops(inv, want, enclosed, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac, set())
        if not want <= seen:
            return 'i could not find all of them'
        xz = relative_xz(sim, sorted(want))
        if not want <= set(xz):
            return 'i could not place all of them'
        deg = bearing_from(xz[origin], xz[forward], xz[target])
        return 'unknown' if deg is None else _direction_word(deg)
    return read

def read_object_distance(a, b, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        want = {a, b}
        inv = sim.inventory()
        enclosed = _closed_openable_hosts(inv)
        seen = set()
        for stop in _tour_stops(inv, want, enclosed, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac, set())
        if not want <= seen:
            return 'i could not find both of them'
        xz = relative_xz(sim, sorted(want))
        if not want <= set(xz):
            return 'i could not place both of them'
        return str(int(round(math.dist(xz[a], xz[b]))))
    return read

def read_nearest_of(x, pairs, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        want = {x} | {a for a, _w in pairs}
        inv = sim.inventory()
        enclosed = _closed_openable_hosts(inv)
        seen = set()
        for stop in _tour_stops(inv, want, enclosed, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac, set())
        if not want <= seen:
            return 'i could not find all of them'
        xz = relative_xz(sim, sorted(want))
        if not want <= set(xz):
            return 'i could not place all of them'
        return min(pairs, key=lambda p: math.dist(xz[x], xz[p[0]]))[1]
    return read

def read_path_integration(x, y, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        want = {x, y}
        inv = sim.inventory()
        enclosed = _closed_openable_hosts(inv)
        seen = set()
        for alias in (x, y):
            seen |= _visit(sim, rec, enclosed.get(alias, alias), want, False, min_pixel_frac, set())
        if not want <= seen:
            return 'i could not find both of them'
        start = getattr(rec, 'start_xz', None)
        if start is None:
            return 'i do not know where i started'
        deg = bearing_to_point(sim.agent_pose(), start[0], start[1])
        return 'unknown' if deg is None else _direction_word(deg)
    return read

def gap_of(fa, fb):
    ax, az, asx, asz = fa
    bx, bz, bsx, bsz = fb
    ox = min(ax + asx / 2, bx + bsx / 2) - max(ax - asx / 2, bx - bsx / 2)
    oz = min(az + asz / 2, bz + bsz / 2) - max(az - asz / 2, bz - bsz / 2)
    if ox > 0 and oz > 0:
        return None
    if ox < 0 <= oz:
        return (-ox, 'x')
    if oz < 0 <= ox:
        return (-oz, 'z')
    return None

def _state_of(inv, aliases):
    by_alias = {f.alias: f for f in inv}
    out = {}
    for a in aliases:
        f = by_alias.get(a)
        out[a] = None if f is None else (f.is_open, tuple(sorted(f.on or ())))
    return out

def read_change_detect(pairs, min_pixel_frac=COUNT_PIXEL_FRAC):
    aliases = [a for a, _w in pairs]

    def read(sim, rec=None):
        want = set(aliases)
        inv = sim.inventory()
        before = _state_of(inv, aliases)
        enclosed = _closed_openable_hosts(inv)
        seen = set()
        for _lap in range(CHANGE_DETECT_TOUR_LAPS):
            stops = _tour_stops(sim.inventory(), want, _closed_openable_hosts(sim.inventory()), _standing_room_for_tour(sim))
            for stop in stops:
                seen |= _visit(sim, rec, stop, want, False, min_pixel_frac, set())
        if not want <= seen:
            if rec is not None:
                rec.unseen = tuple(sorted(want - seen))
            return f'i could not find all of them ({len(seen & want)} of {len(want)} seen)'
        after = _state_of(sim.inventory(), aliases)
        changed = [w for a, w in pairs if before.get(a) != after.get(a)]
        if len(changed) != 1:
            return f'i could not tell what changed ({len(changed)} differences)'
        return changed[0]
    return read

def read_gap_width(a, b, min_pixel_frac=COUNT_PIXEL_FRAC, reference_cm=None):

    def read(sim, rec=None):
        want = {a, b}
        inv = sim.inventory()
        enclosed = _closed_openable_hosts(inv)
        seen = set()
        for stop in _tour_stops(inv, want, enclosed, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac, set())
        if not want <= seen:
            return 'i could not find both of them'
        if reference_cm is not None:
            return str(int(round(float(reference_cm))))
        fa, fb = (sim.footprint(a), sim.footprint(b))
        got = gap_of(fa, fb) if fa and fb else None
        if got is None:
            return 'i could not measure the gap between them'
        return str(int(round(got[0] * 100)))
    return read

def read_room_area(room_id, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        stop = room_landmarks(sim.inventory()).get(room_id)
        if stop is None:
            return 'i could not find a way into that room'
        r = sim.navigate_to_object(NameRef(stop))
        (rec or _NO_REC).step('navigate', stop, r)
        if not r.success:
            return 'i could not get to that room'
        _sweep_here(sim, rec, set(), min_pixel_frac)
        area = (sim.room_sizes() or {}).get(room_id)
        return '' if area is None else str(int(round(area)))
    return read

def read_topo_route(pairs, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        f = getattr(sim, 'standing_room', None)
        inventory = tuple(sim.inventory())
        standing = f() if callable(f) else None
        landmarks = room_handles(inventory, standing)
        tour_handles = room_tour_handles(inventory, standing=standing, per_room=4)
        by_alias = {fact.alias: fact for fact in inventory}
        for rid, _name in pairs:
            stop = landmarks.get(rid)
            if stop is None:
                return 'i could not find a way into one of those rooms'
            options = (stop,) + tuple((alias for alias in tour_handles.get(rid, ()) if alias != stop))
            arrived = False
            for option in options:
                r = sim.navigate_to_object(NameRef(option))
                (rec or _NO_REC).step('navigate', option, r)
                if r.success and _in_expected_room(sim, by_alias.get(option)):
                    arrived = True
                    break
            if not arrived:
                return 'i could not get to one of those rooms'
            _sweep_here(sim, rec, set(), min_pixel_frac)
        return ', '.join((name for _rid, name in pairs))
    return read

def read_reach_blocked(pairs, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        failed = []
        for alias, word in pairs:
            r = sim.navigate_to_object(NameRef(alias))
            (rec or _NO_REC).step('navigate', alias, r)
            if not r.success:
                failed.append(word)
        if not failed:
            return 'i could walk up to all of them'
        if len(failed) > 1:
            return 'i could not walk up to more than one of them'
        return failed[0]
    return read

def merge_rooms(rooms, same_room=()):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for group in same_room:
        members = [g for g in group if g]
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    out = {}
    for r in rooms:
        out.setdefault(find(r), set()).add(r)
    return frozenset((frozenset(v) for v in out.values()))

def read_room_count_seen(kind, scope=None, same_room=(), min_pixel_frac=COUNT_PIXEL_FRAC, target_aliases=None):

    def read(sim, rec=None):
        inv = _scoped(sim.inventory(), scope)
        want = _category_targets(inv, kind, target_aliases)
        seen = set()
        for stop in _tour_stops(inv, want, {}, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac)
        where = {f.room_id or f.room for f in inv if f.alias in seen and (f.room_id or f.room)}
        return str(len(merge_rooms(where, same_room)))
    return read

def read_absent_confirm(kind, noun, scope=None, min_pixel_frac=COUNT_PIXEL_FRAC):

    def read(sim, rec=None):
        inv = _scoped(sim.inventory(), scope)
        want = {f.alias for f in inv if f.kind == kind}
        seen = set()
        for stop in _tour_stops(inv, want, {}, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac)
        if seen:
            where = {f.room for f in inv if f.alias in seen and f.room}
            return f"found {noun} in {' '.join(sorted(where))}" if where else f'found {noun}'
        return f'there is no {noun} in this house'
    return read

def read_state_count_seen(kind, state='is_open', min_pixel_frac=COUNT_PIXEL_FRAC, scope=None):

    def read(sim, rec=None):
        inv = _scoped(sim.inventory(), scope)
        want = {f.alias for f in inv if f.kind == kind}
        seen = set()
        for stop in _tour_stops(inv, want, {}, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac)
        return str(sum((1 for a in sorted(seen) if sim.predicate(f"{state}('{a}')"))))
    return read

def read_argmax_room_seen(kind, min_pixel_frac=COUNT_PIXEL_FRAC, scope=None, spoken=None, target_aliases=None):

    def read(sim, rec=None):
        inv = _scoped(sim.inventory(), scope)
        want = _category_targets(inv, kind, target_aliases)
        place = {f.alias: f.room_id or f.room for f in inv}
        label = {f.room_id or f.room: f.room for f in inv if f.room}
        seen = set()
        for stop in _tour_stops(inv, want, {}, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac)
        tally = {}
        for a in seen:
            if place.get(a):
                tally[place[a]] = tally.get(place[a], 0) + 1
        if not tally:
            return ''
        ranked = sorted(tally.items(), key=lambda kv: (-kv[1], kv[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return ''
        win = ranked[0][0]
        if spoken:
            return spoken.get(win) or ''
        return _spell_label(label.get(win))
    return read

def read_room_with_both(kind_x, kind_y, min_pixel_frac=COUNT_PIXEL_FRAC, scope=None, spoken=None):

    def read(sim, rec=None):
        inv = _scoped(sim.inventory(), scope)
        want_x = {f.alias for f in inv if f.kind == kind_x}
        want_y = {f.alias for f in inv if f.kind == kind_y}
        want = want_x | want_y
        place = {f.alias: f.room_id or f.room for f in inv}
        label = {f.room_id or f.room: f.room for f in inv if f.room}
        seen = set()
        for stop in _tour_stops(inv, want, {}, _standing_room_for_tour(sim)):
            seen |= _visit(sim, rec, stop, want, False, min_pixel_frac)
        has_x = {place[a] for a in seen if a in want_x and place.get(a)}
        has_y = {place[a] for a in seen if a in want_y and place.get(a)}
        both = sorted(has_x & has_y)
        if len(both) != 1:
            return ''
        win = both[0]
        if spoken:
            return spoken.get(win) or ''
        return _spell_label(label.get(win))
    return read

def _rec_step(sim, rec, kind, arg):
    res = sim.turn(float(arg)) if kind == 'turn' else sim.look(float(arg))
    if rec is not None:
        rec.step(kind, arg, res)
    return res

def _visible_on(sim, host, min_pixel_frac):
    return {a for a in sim.visible_instances(min_pixel_frac) if a != host and sim.predicate(f"on('{a}','{host}')")}

class _AsResult:
    __slots__ = ('success', 'reason', 'collided')

    def __init__(self, success, reason):
        self.success, self.reason, self.collided = (bool(success), reason, False)
