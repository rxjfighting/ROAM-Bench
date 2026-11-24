"""Generate and serialize task instructions, acceptance rules, and evidence requirements."""
import collections
import itertools
import math
import re
from dataclasses import dataclass, field, replace
from typing import Optional
from . import baselines, gates, houses, score
from .score import COUNT_UNITS, Goal, Mutation, Task, count_prior_of, count_unit
from .types import NameRef
ROOM_WORDS = {'Kitchen': ('kitchen',), 'LivingRoom': ('living room', 'livingroom', 'lounge', 'sitting room'), 'Bedroom': ('bedroom', 'bed room'), 'Bathroom': ('bathroom', 'bath room', 'restroom', 'washroom')}
_CAMEL = re.compile('[A-Z][a-z0-9]*|[a-z0-9]+')

def sole_words(name, room_id, house_rooms):
    taken = {w for rid in house_rooms if rid != room_id for w in (houses.room_name(rid) or '').split()}
    return tuple((w for w in name.split() if w not in taken))

def room_answer(room_id, label, house_rooms):
    words, synonyms = ([], ROOM_WORDS.get(label, ()))
    name = houses.room_name(room_id)
    if name:
        words.append(name)
        words.extend((w for w in sole_words(name, room_id, house_rooms) if w not in ('room',) and len(w) >= MIN_WORD))
        if any((_normalise(s) in _normalise(name) for s in synonyms)):
            words.extend(synonyms)
    else:
        words.extend(synonyms)
    seen, out = (set(), [])
    for w in words:
        if w and w not in seen:
            seen.add(w)
            out.append(w)
    return (tuple(out), out[0] if out else None)

def _normalise(s):
    return ''.join((ch for ch in s.lower() if ch.isalnum()))

def _where(f):
    return f.room_id or f.room
MIN_WORD = 3

def kind_words(kind):
    parts = [p.lower() for p in _CAMEL.findall(kind)]
    if not parts:
        return (kind.lower(),)
    if len(parts) == 1:
        return (parts[0],)
    out = [' '.join(parts), kind.lower()]
    for p in (parts[0], parts[-1]):
        if len(p) >= MIN_WORD and p not in out:
            out.append(p)
    return tuple(dict.fromkeys(out))

def answer_groups(kinds):
    groups = [kind_words(k) for k in sorted(set(kinds))]
    seen, shared = (set(), set())
    for g in groups:
        for w in g:
            (shared if w in seen else seen).add(w)
    out = []
    for g in groups:
        kept = tuple((w for w in g if w not in shared))
        out.append(kept or (g[0],))
    return tuple(out)

@dataclass(frozen=True)
class EvidenceBound:
    must_visit_regions: frozenset = frozenset()
    must_open: frozenset = frozenset()
    must_see: frozenset = frozenset()
    min_distinct_viewpoints: int = 1
    search_regions: frozenset = frozenset()

    def to_dict(self):
        return {'must_visit_regions': sorted(self.must_visit_regions), 'must_open': sorted(self.must_open), 'must_see': sorted(self.must_see), 'min_distinct_viewpoints': self.min_distinct_viewpoints, 'search_regions': sorted(self.search_regions)}

    @staticmethod
    def from_dict(d):
        if not d:
            return None
        return EvidenceBound(must_visit_regions=frozenset(d.get('must_visit_regions', ())), must_open=frozenset(d.get('must_open', ())), must_see=frozenset(d.get('must_see', ())), min_distinct_viewpoints=int(d.get('min_distinct_viewpoints', 1)), search_regions=frozenset(d.get('search_regions', d.get('must_visit_regions', ()))))

def _jsonable(v):
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v

@dataclass
class TaskSpec:
    task: Task
    family: str
    kinds: tuple = ()
    receptacle_kinds: tuple = ()
    evidence: tuple = ()
    answer_instances: tuple = ()
    referents: tuple = ()
    plan: tuple = ()
    oracle_answer: Optional[str] = None
    oracle_read: Optional[tuple] = None
    evidence_bound: Optional[EvidenceBound] = None
    truth: dict = field(default_factory=dict)
    scene_variant: dict = field(default_factory=dict)

    def to_dict(self):
        t = self.task
        return _jsonable({'task_id': t.task_id, 'family': self.family, 'instruction': t.instruction, 'goal_images': [list(g) for g in t.goal_images], 'referents': list(self.referents), 'goals': [{'expr': g.expr, 'when': g.when, 'label': g.label, 'weight': g.weight} for g in t.goals], 'answer_keywords': [list(g) for g in t.answer_keywords], 'answer_choices': list(t.answer_choices), 'answer_choice_mode': t.answer_choice_mode, 'answer_choice_labels': list(t.answer_choice_labels), 'answer_from_pose': list(t.answer_from_pose), 'mutation': t.mutation.to_dict() if t.mutation else None, 'answer_vocabulary': list(t.answer_vocabulary), 'answer_unit': t.answer_unit, 'answer_match': t.answer_match, 'max_steps': t.max_steps, 'answer_number': t.answer_number, 'answer_tolerance': t.answer_tolerance, 'evidence_bound': self.evidence_bound.to_dict() if self.evidence_bound else None, 'kinds': list(self.kinds), 'receptacle_kinds': list(self.receptacle_kinds), 'evidence': list(self.evidence), 'answer_instances': list(self.answer_instances), 'plan': [[v, getattr(r, 'name', r)] for v, r in self.plan], 'oracle_answer': self.oracle_answer, 'oracle_read': list(self.oracle_read) if self.oracle_read else None, 'truth': self.truth, 'scene_variant': self.scene_variant})

def oracle_answer_for(spec):
    kind = (spec.oracle_read or (None,))[0]
    membership = (spec.truth or {}).get('category_membership')
    target_aliases = None
    if membership is not None:
        if membership.get('schema') != 'reviewed-category-members-v1' or not isinstance(membership.get('aliases'), list):
            raise ValueError('invalid reviewed category membership')
        target_aliases = membership['aliases']
        selected = set(target_aliases)
        if len(selected) != len(target_aliases) or selected != set(spec.evidence) or selected != set(spec.answer_instances) or (len(spec.kinds) == 1 and membership.get('category') != spec.kinds[0]):
            raise ValueError('reviewed category, answer instances and evidence must agree')
        if spec.family == 'object_count' and spec.task.answer_number != len(selected):
            raise ValueError('reviewed count differs from physical members')
    if kind == 'room_here':
        return _map_choice_reader(spec.task, baselines.read_room_here)
    if kind == 'surface_contents':
        reviewed = (spec.truth or {}).get('reviewed_contents')
        if reviewed is not None:
            children = [row['alias'] for row in spec.truth.get('children', ())]
            options = {row['id']: row['alias'] for row in spec.truth.get('answer_options', ())}
            if reviewed.get('schema') != 'visually-reviewed-contents-v1' or set(reviewed.get('aliases', ())) != set(children) or len(children) != len(set(children)) or (set(spec.answer_instances) != set(children)) or (set(spec.evidence) != set(children) | {spec.truth['host']}):
                raise ValueError('invalid visually reviewed contents binding')
            return baselines.read_surface_contents(spec.oracle_read[1], target_aliases=children, option_aliases=options)
        reader = baselines.read_surface_contents(spec.oracle_read[1])
        if getattr(spec.task, 'answer_choice_mode', 'single') == 'multiple':
            return _map_multiple_choice_reader(spec.task, reader)
        return reader
    if kind == 'count_seen':
        return baselines.read_count_seen(spec.oracle_read[1], scope=_read_scope(spec.oracle_read, 2), target_aliases=target_aliases)
    if kind == 'count_seen_opening':
        return baselines.read_count_seen(spec.oracle_read[1], open_hosts=True, scope=_read_scope(spec.oracle_read, 2))
    if kind == 'room_count_seen':
        return baselines.read_room_count_seen(spec.oracle_read[1], scope=_read_scope(spec.oracle_read, 2), same_room=tuple((tuple(g) for g in (spec.oracle_read[3] if len(spec.oracle_read) > 3 else ()))), target_aliases=target_aliases)
    if kind == 'count_seen_in_label':
        return baselines.read_count_seen(spec.oracle_read[1], room_label=spec.oracle_read[2], scope=_read_scope(spec.oracle_read, 3))
    if kind == 'state_count_seen':
        return baselines.read_state_count_seen(spec.oracle_read[1], scope=_read_scope(spec.oracle_read, 2))
    if kind == 'argmax_room_seen':
        reader = baselines.read_argmax_room_seen(spec.oracle_read[1], scope=_read_scope(spec.oracle_read, 2), spoken=dict(spec.oracle_read[3]) if len(spec.oracle_read) > 3 else None, target_aliases=target_aliases)
        return _map_choice_reader(spec.task, reader)
    if kind == 'room_with_both':
        reader = baselines.read_room_with_both(spec.oracle_read[1], spec.oracle_read[2], scope=_read_scope(spec.oracle_read, 3), spoken=dict(spec.oracle_read[4]) if len(spec.oracle_read) > 4 else None)
        return _map_choice_reader(spec.task, reader)
    if kind == 'absent_confirm':
        reader = baselines.read_absent_confirm(spec.oracle_read[1], spec.oracle_read[2], scope=_read_scope(spec.oracle_read, 3))
        if 'NOT_PRESENT' in tuple(getattr(spec.task, 'answer_choices', ()) or ()):
            return _map_room_or_absent_reader(spec.task, reader)
        return reader
    if kind == 'relative_direction':
        return baselines.read_relative_direction(spec.oracle_read[1], spec.oracle_read[2], spec.oracle_read[3])
    if kind == 'object_distance':
        return baselines.read_object_distance(spec.oracle_read[1], spec.oracle_read[2])
    if kind == 'nearest_of':
        return baselines.read_nearest_of(spec.oracle_read[1], tuple((tuple(p) for p in spec.oracle_read[2])))
    if kind == 'path_integration':
        return baselines.read_path_integration(spec.oracle_read[1], spec.oracle_read[2])
    if kind == 'change_detect':
        return baselines.read_change_detect(tuple((tuple(p) for p in spec.oracle_read[1])))
    if kind == 'gap_width':
        return baselines.read_gap_width(spec.oracle_read[1], spec.oracle_read[2], reference_cm=spec.truth.get('reference_gap_cm'))
    if kind == 'room_area':
        return baselines.read_room_area(spec.oracle_read[1])
    if kind == 'topo_route':
        return baselines.read_topo_route(tuple((tuple(p) for p in spec.oracle_read[1])))
    if kind == 'reach_blocked':
        return baselines.read_reach_blocked(tuple((tuple(p) for p in spec.oracle_read[1])))
    return spec.oracle_answer

def _map_choice_reader(task, reader):
    choices = tuple(getattr(task, 'answer_choices', ()) or ())
    labels = tuple(getattr(task, 'answer_choice_labels', ()) or ())
    if not choices or len(choices) != len(labels):
        return reader
    by_label = dict(zip(labels, choices))

    def mapped(sim, rec=None):
        answer = reader(sim, rec)
        return by_label.get(answer, answer)
    return mapped

def _map_multiple_choice_reader(task, reader):
    choices = tuple(getattr(task, 'answer_choices', ()) or ())
    labels = tuple(getattr(task, 'answer_choice_labels', ()) or ())
    if not choices or len(choices) != len(labels):
        return reader

    def mapped(sim, rec=None):
        answer = reader(sim, rec)
        selected = [choice for choice, label in zip(choices, labels) if score.match_answer(str(answer), ((label,),), 'word')]
        return ', '.join(selected)
    return mapped

def _map_room_or_absent_reader(task, reader):
    choices = tuple(getattr(task, 'answer_choices', ()) or ())
    labels = tuple(getattr(task, 'answer_choice_labels', ()) or ())
    by_label = dict(zip(labels, choices))

    def mapped(sim, rec=None):
        answer = str(reader(sim, rec))
        normal = score.normalise_answer(answer)
        if any((token in normal.split() for token in ('no', 'none', 'absent', 'missing'))):
            return 'NOT_PRESENT'
        return by_label.get(answer, answer)
    return mapped

def _read_scope(read, at):
    return tuple(read[at]) if len(read) > at and read[at] else None

def _read_from_json(read):
    return tuple((_read_from_json(x) if isinstance(x, list) else x for x in read))

def spec_from_dict(d):
    family = d.get('family', '')
    default_unit = {'object_distance': 'm', 'gap_width': 'cm', 'room_area': 'm2'}.get(family, '')
    evidence_bound = EvidenceBound.from_dict(d.get('evidence_bound'))
    from ..scene_variants import validate_variant
    protected = set(d.get('referents', ())) | set(d.get('evidence', ())) | set(d.get('answer_instances', ()))
    if evidence_bound:
        protected.update(evidence_bound.must_see)
    scene_variant = validate_variant(d.get('scene_variant'), protected)
    if family == 'surface_contents' and evidence_bound is not None:
        evidence_bound = replace(evidence_bound, must_open=frozenset())
    return TaskSpec(task=Task(task_id=d['task_id'], instruction=d['instruction'], goals=tuple((Goal(expr=g['expr'], label=g.get('label', ''), weight=g.get('weight', 1.0), when=g.get('when', 'final')) for g in d.get('goals', ()))), answer_keywords=tuple((tuple(g) for g in d.get('answer_keywords', ()))), answer_match=d.get('answer_match', 'word'), answer_number=d.get('answer_number'), answer_tolerance=d.get('answer_tolerance'), goal_images=tuple((tuple(g) for g in d.get('goal_images', ()))), answer_choices=tuple(d.get('answer_choices', ())), answer_choice_mode=d.get('answer_choice_mode', 'single'), answer_choice_labels=tuple(d.get('answer_choice_labels', ())), answer_from_pose=tuple(d.get('answer_from_pose', ())), mutation=Mutation.from_dict(d.get('mutation')), answer_vocabulary=tuple(d.get('answer_vocabulary', ())), answer_unit=d.get('answer_unit', default_unit), max_steps=d.get('max_steps', 60)), family=family, referents=tuple(d.get('referents', ())), kinds=tuple(d.get('kinds', ())), receptacle_kinds=tuple(d.get('receptacle_kinds', ())), evidence=tuple(d.get('evidence', ())), answer_instances=tuple(d.get('answer_instances', ())), plan=tuple(((v, NameRef(r)) for v, r in d.get('plan', ()))), oracle_answer=d.get('oracle_answer'), oracle_read=_read_from_json(d['oracle_read']) if d.get('oracle_read') else None, evidence_bound=evidence_bound, truth=dict(d.get('truth', {})), scene_variant=scene_variant)

def room_localization_tasks(inv, max_steps=80, per_room_cap=40):
    by_alias = {f.alias: f for f in inv}
    search = frozenset(_scope_rooms(inv))
    out, per_room = ([], {})
    for f in inv:
        if not f.pickupable:
            continue
        words, oracle = room_answer(f.room_id, f.room, search)
        if not words:
            continue
        instruction = _room_search_instruction(_noun(f.kind))
        haystack = instruction.lower().replace('_', ' ')
        if any((re.search(f'\\b{re.escape(w)}\\b', haystack) for w in words)):
            continue
        if per_room_cap is not None and per_room.get(oracle, 0) >= per_room_cap:
            continue
        per_room[oracle] = per_room.get(oracle, 0) + 1
        out.append(TaskSpec(task=Task(task_id=f'room_localization__{f.alias}', instruction=instruction, answer_keywords=(words,), max_steps=max_steps), family='room_localization', kinds=(f.kind,), referents=(f.alias,), evidence=(f.alias,), answer_instances=(f.alias,), plan=(('navigate', NameRef(f.alias)),), oracle_answer=oracle, oracle_read=('room_here',), evidence_bound=EvidenceBound(must_visit_regions=frozenset({_where(f)} - {None}), must_open=_must_open_for((f.alias,), by_alias), must_see=frozenset({f.alias}), min_distinct_viewpoints=1, search_regions=search), truth={'alias': f.alias, 'kind': f.kind, 'room': f.room, 'room_id': f.room_id, 'room_name': houses.room_name(f.room_id)}))
    return out

def surface_contents_tasks(inv, max_steps=80, min_children=1):
    children = {}
    for f in inv:
        for host in f.on:
            if host in f.inside:
                continue
            children.setdefault(host, []).append(f)
    by_alias = {f.alias: f for f in inv}
    search = frozenset(_scope_rooms(inv))
    out = []
    object_vocabulary = tuple(sorted({_noun(f.kind) for f in inv if f.pickupable}))
    for host, kids in sorted(children.items()):
        h = by_alias.get(host)
        if h is None or h.pickupable or len(kids) < min_children:
            continue
        kids = sorted(kids, key=lambda f: f.alias)
        out.append(TaskSpec(task=Task(task_id=f'surface_contents__{host}', instruction=f'Go to the {_noun(h.kind)} and tell me the name of every object resting on it.', answer_keywords=answer_groups([k.kind for k in kids]), answer_match='set', answer_vocabulary=object_vocabulary, max_steps=max_steps), family='surface_contents', kinds=tuple((k.kind for k in kids)), receptacle_kinds=(h.kind,), referents=(host,), evidence=(host,) + tuple((k.alias for k in kids)), answer_instances=tuple((k.alias for k in kids)), plan=(('navigate', NameRef(host)),), oracle_answer=', '.join((g[0] for g in answer_groups([k.kind for k in kids]))), oracle_read=('surface_contents', host), evidence_bound=EvidenceBound(must_visit_regions=frozenset({_where(h)} - {None}), must_open=frozenset(), must_see=frozenset([host] + [k.alias for k in kids]), min_distinct_viewpoints=2 if len(kids) > 1 else 1, search_regions=search), truth={'host': host, 'host_kind': h.kind, 'room': h.room, 'children': [{'alias': k.alias, 'kind': k.kind} for k in kids]}))
    return out

def relocation_tasks(inv, max_steps=140, cross_room_only=True):
    picks = [f for f in inv if f.pickupable and f.room]
    dests = [f for f in inv if f.receptacle and (not f.pickupable) and f.room]
    by_alias = {f.alias: f for f in inv}
    search = frozenset(_scope_rooms(inv))
    out = []
    for p in picks:
        for d in dests:
            if cross_room_only and _where(p) == _where(d):
                continue
            if d.openable:
                continue
            out.append(TaskSpec(task=Task(task_id=f'relocation__{p.alias}__to__{d.alias}', instruction=f'Find the {_noun(p.kind)}, pick it up, and put it on the {_noun(d.kind)}.', goals=(Goal(f"holding('{p.alias}')", label='picked up', when='ever'), Goal(f"on('{p.alias}','{d.alias}')", label='placed')), max_steps=max_steps), family='relocation', referents=(p.alias, d.alias), evidence=(p.alias, d.alias), plan=(('navigate', NameRef(p.alias)), ('pick', NameRef(p.alias)), ('navigate', NameRef(d.alias)), ('place', NameRef(d.alias))), evidence_bound=EvidenceBound(must_visit_regions=frozenset({_where(p), _where(d)} - {None}), must_open=_must_open_for((p.alias,), by_alias), must_see=frozenset({p.alias, d.alias}), min_distinct_viewpoints=2, search_regions=search), truth={'object': p.alias, 'from_room': p.room, 'from_room_id': p.room_id, 'dest': d.alias, 'to_room': d.room, 'to_room_id': d.room_id}))
    return out
_UNCOUNTABLE_SUFFIX = ('s', 'x', 'z', 'ch', 'sh')

def _plural(word):
    w = str(word)
    if w.endswith('fe'):
        return w[:-2] + 'ves'
    if w.endswith('f'):
        return w[:-1] + 'ves'
    if w.endswith('y') and len(w) > 1 and (w[-2] not in 'aeiou'):
        return w[:-1] + 'ies'
    if w.endswith(_UNCOUNTABLE_SUFFIX):
        return w + 'es'
    return w + 's'

def _a(word):
    w = str(word)
    return f'an {w}' if w[:1].lower() in 'aeio' else f'a {w}'

def _noun(kind):
    return kind_words(kind)[0]

def _children_by_host(inv):
    out = {}
    for f in inv:
        for host in f.on:
            out.setdefault(host, []).append(f)
    return out

def _rooms_of(instances):
    return {_where(f) for f in instances if _where(f)}

def _must_open_for(aliases, by_alias):
    hosts = set()
    for alias in aliases:
        f = by_alias.get(alias)
        if f is None or not f.pickupable:
            continue
        for host in f.inside:
            h = by_alias.get(host)
            if h is not None and h.openable and (h.is_open is False):
                hosts.add(host)
    return frozenset(hosts)

def _reader_would_miss(aliases, by_alias):
    return bool(_must_open_for(aliases, by_alias))

def _landmarks_per_room(inv):
    best = {}
    for f in inv:
        place = _where(f)
        if not place or f.pickupable or (not f.receptacle):
            continue
        cur = best.get(place)
        if cur is None or (cur.openable and (not f.openable)):
            best[place] = f
    return {k: v.alias for k, v in best.items()}
HOUSE_SCOPE = 'this house'

def _scope(scope):
    if not scope or scope == HOUSE_SCOPE:
        return (HOUSE_SCOPE, '')
    return (scope, ' Parts of this house cannot be reached from where you start; anything in them does not count.')

def _scope_rooms(inv):
    return tuple(sorted({f.room_id or f.room for f in inv if f.room_id or f.room}))

def _object_count_instruction(where, plural, caveat=''):
    return f'Search {where} and count the {plural}. There is more than one room, and there may be {plural} in more than one of them.{caveat} When you have finished searching, tell me how many {plural} there are in total, as a number.'

def _room_search_instruction(noun, where='this house', caveat=''):
    return f'Find the {noun} in {where}, then tell me which room it is in. If there is no {noun} here, choose NOT_PRESENT instead of guessing a room.{caveat}'

def alpha_object_count(inv, max_steps=240, min_rooms=2, min_total=2, max_total=12, scope=HOUSE_SCOPE):
    by_alias = {f.alias: f for f in inv}
    by_kind = {}
    for f in inv:
        by_kind.setdefault(f.kind, []).append(f)
    where, caveat = _scope(scope)
    out = []
    for kind, insts in sorted(by_kind.items()):
        rooms = _rooms_of(insts)
        n = len(insts)
        if len(rooms) < min_rooms or not min_total <= n <= max_total:
            continue
        noun, plural = (_noun(kind), _plural(_noun(kind)))
        aliases = tuple(sorted((f.alias for f in insts)))
        if _reader_would_miss(aliases, by_alias):
            continue
        out.append(TaskSpec(task=Task(task_id=f'object_count__{kind}', instruction=_object_count_instruction(where, plural, caveat), answer_match='count', answer_number=n, max_steps=max_steps), family='object_count', kinds=(kind,), evidence=aliases, answer_instances=aliases, plan=(), oracle_answer=str(n), oracle_read=('count_seen', kind, _scope_rooms(inv)), evidence_bound=EvidenceBound(must_visit_regions=frozenset(rooms), must_see=frozenset(aliases), min_distinct_viewpoints=len(rooms), search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'total': n, 'noun': noun, 'rooms': sorted(rooms), 'per_room': {r: sum((1 for f in insts if _where(f) == r)) for r in sorted(rooms)}}))
    return out

def alpha_count_conditional(inv, max_steps=240, min_inside=1, min_outside=1, max_total=12, scope=HOUSE_SCOPE):
    by_alias = {f.alias: f for f in inv}
    by_label = {}
    for f in inv:
        if f.room and _where(f):
            by_label.setdefault(f.room, set()).add(_where(f))
    multi = {lab for lab, places in by_label.items() if len(places) >= 2}
    by_kind = {}
    for f in inv:
        by_kind.setdefault(f.kind, []).append(f)
    out = []
    for kind, insts in sorted(by_kind.items()):
        if len(insts) > max_total:
            continue
        for label in sorted(multi):
            if label not in ROOM_WORDS:
                continue
            inside = [f for f in insts if f.room == label]
            outside = [f for f in insts if f.room != label]
            if len(inside) < min_inside or len(outside) < min_outside:
                continue
            rooms = _rooms_of(inside)
            if len(rooms) < 2:
                continue
            plural = _plural(_noun(kind))
            word = ROOM_WORDS[label][0]
            aliases = tuple(sorted((f.alias for f in inside)))
            if _reader_would_miss(aliases, by_alias):
                continue
            out.append(TaskSpec(task=Task(task_id=f'count_in__{kind}__{label}', instruction=f'This house has more than one {word}. Count the {plural} that are in {_a(word)} -- there are {plural} elsewhere as well, and those do not count.{_scope(scope)[1]} Tell me the number.', answer_match='count', answer_number=len(inside), max_steps=max_steps), family='count_in_room', kinds=(kind,), evidence=aliases, answer_instances=aliases, plan=(), oracle_answer=str(len(inside)), oracle_read=('count_seen_in_label', kind, label, _scope_rooms(inv)), evidence_bound=EvidenceBound(must_visit_regions=frozenset(rooms), must_see=frozenset(aliases), min_distinct_viewpoints=len(rooms), search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'room_label': label, 'n_inside': len(inside), 'n_outside': len(outside), 'rooms': sorted(rooms)}))
    return out

def alpha_room_count(inv, max_steps=240, min_rooms=2, max_rooms=12, min_absent=1, scope=HOUSE_SCOPE, same_room=()):
    by_alias = {f.alias: f for f in inv}
    by_kind = {}
    for f in inv:
        by_kind.setdefault(f.kind, []).append(f)
    all_places = baselines.merge_rooms(_scope_rooms(inv), same_room)
    where, caveat = _scope(scope)
    out = []
    for kind, insts in sorted(by_kind.items()):
        aliases = tuple(sorted((f.alias for f in insts)))
        if _reader_would_miss(aliases, by_alias):
            continue
        places = baselines.merge_rooms(_rooms_of(insts), same_room)
        n = len(places)
        if not min_rooms <= n <= max_rooms:
            continue
        if len(all_places) - n < min_absent:
            continue
        noun, plural = (_noun(kind), _plural(_noun(kind)))
        out.append(TaskSpec(task=Task(task_id=f'room_count__{kind}', instruction=f'Search {where} and find out which rooms contain at least one {noun}. Count two places as different rooms if you have to go through a doorway to get from one to the other. Some rooms contain no {plural} at all, and a room with several {plural} still counts once.{caveat} Tell me how many rooms contain at least one {noun}, as a number.', answer_match='count', answer_number=n, max_steps=max_steps), family='room_count', kinds=(kind,), evidence=aliases, answer_instances=aliases, plan=(), oracle_answer=str(n), oracle_read=('room_count_seen', kind, _scope_rooms(inv), tuple((tuple(sorted(g)) for g in sorted(map(sorted, same_room))))), evidence_bound=EvidenceBound(must_visit_regions=frozenset(_rooms_of(insts)), must_see=frozenset(aliases), min_distinct_viewpoints=n, search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'n_rooms': n, 'noun': noun, 'rooms': sorted(_rooms_of(insts)), 'places': sorted((sorted(g) for g in places)), 'n_places_in_scope': len(all_places), 'total': len(insts)}))
    return out

def alpha_compare_extremum(inv, max_steps=240, min_rooms=2, margin=1, scope=HOUSE_SCOPE):
    by_alias = {f.alias: f for f in inv}
    spoken = {}
    house = frozenset(_scope_rooms(inv))
    for f in inv:
        place = _where(f)
        if place is not None and place not in spoken:
            spoken[place] = room_answer(place, f.room, house)
    place_label = {_where(f): f.room for f in inv if _where(f) and f.room}
    by_kind = {}
    for f in inv:
        by_kind.setdefault(f.kind, []).append(f)
    out = []
    for kind, insts in sorted(by_kind.items()):
        per = {}
        for f in insts:
            if _where(f):
                per.setdefault(_where(f), []).append(f)
        if len(per) < min_rooms:
            continue
        ranked = sorted(per.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        (win_place, win_insts), (_, second) = (ranked[0], ranked[1])
        if len(win_insts) - len(second) < margin:
            continue
        label = place_label.get(win_place)
        words, display = spoken.get(win_place, ((), None))
        if not words or not display:
            continue
        if any((p != win_place and sp and score.match_answer(sp, (words,), 'word') for p, (_, sp) in spoken.items())):
            continue
        plural = _plural(_noun(kind))
        aliases = tuple(sorted((f.alias for f in insts)))
        if _reader_would_miss(aliases, by_alias):
            continue
        out.append(TaskSpec(task=Task(task_id=f'room_comparison__{kind}', instruction=f'Different rooms hold different numbers of {plural}. Search {_scope(scope)[0]}, then tell me which single room has the most {plural} in it.{_scope(scope)[1]}', answer_keywords=(words,), max_steps=max_steps), family='room_comparison', kinds=(kind,), evidence=aliases, answer_instances=tuple(sorted((f.alias for f in win_insts))), plan=(), oracle_answer=display, oracle_read=('argmax_room_seen', kind, _scope_rooms(inv), {p: sp for p, (_, sp) in sorted(spoken.items()) if sp}), evidence_bound=EvidenceBound(must_visit_regions=frozenset(per), must_see=frozenset(aliases), min_distinct_viewpoints=len(per), search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'winner': win_place, 'winner_label': label, 'winner_spoken': display, 'per_room': {p: len(v) for p, v in sorted(per.items())}, 'margin': len(win_insts) - len(second)}))
    return out

def _one_clue_room(rooms):
    rs = sorted(rooms)
    return rs[0] if len(rs) == 1 else None

def _kind_tables(inv):
    rooms_of_kind, insts_of_kind = ({}, {})
    for f in inv:
        place = _where(f)
        if place is None:
            continue
        rooms_of_kind.setdefault(f.kind, set()).add(place)
        insts_of_kind.setdefault(f.kind, []).append(f)
    return (rooms_of_kind, insts_of_kind)
GATE_SHARE = 'X and Y share'
GATE_X_ALONE = 'X alone names the answer room'
GATE_Y_ALONE = 'Y alone names the answer room'

def both_in_room_answer(rooms_x, rooms_y):
    inter = sorted(set(rooms_x) & set(rooms_y))
    if len(inter) != 1:
        return (None, f'{GATE_SHARE} {len(inter)} rooms, not one')
    answer = inter[0]
    if _one_clue_room(rooms_x) == answer:
        return (None, f'{GATE_X_ALONE}; Y is never needed')
    if _one_clue_room(rooms_y) == answer:
        return (None, f'{GATE_Y_ALONE}; X is never needed')
    return (answer, None)

def both_in_room_gate_line(inv):
    rooms_of_kind, _ = _kind_tables(inv)
    kinds = sorted(rooms_of_kind)
    reasons, admitted = (collections.Counter(), 0)
    for i, kx in enumerate(kinds):
        for ky in kinds[i + 1:]:
            answer, why = both_in_room_answer(rooms_of_kind[kx], rooms_of_kind[ky])
            if answer is not None:
                admitted += 1
            elif why.startswith(GATE_X_ALONE) or why.startswith(GATE_Y_ALONE):
                reasons['单线索'] += 1
            elif why.startswith(GATE_SHARE):
                reasons['交集不是一间房'] += 1
            else:
                reasons['未分类'] += 1
    pairs = admitted + sum(reasons.values())
    if not pairs:
        return f'both_in_room 准入门: 这栋房子没有可判的 kind 对 (0 对, {len(kinds)} 个有位置的 kind) -- 门没有被行使，不是门没有拒绝'
    single = reasons['单线索']
    bit = f'其中单线索 {single} 对' if single else '**单线索那一档 0 对 -- 门在本房未触发**，它会咬是 test_taskset_a_class 证的，不是这一行'
    return f'both_in_room 准入门: {pairs} 对候选中准入 {admitted} 对、拒 {pairs - admitted} 对 {dict(reasons.most_common())}；{bit}（门之后还有 words/同词/_reader_would_miss/per_room_cap/limit 几道，所以准入数是发货数的上界，两者不是一个数）'

def alpha_both_in_room(inv, max_steps=240, scope=HOUSE_SCOPE, limit=8, per_room_cap=2):
    by_alias = {f.alias: f for f in inv}
    spoken = {}
    house = frozenset(_scope_rooms(inv))
    for f in inv:
        place = _where(f)
        if place is not None and place not in spoken:
            spoken[place] = room_answer(place, f.room, house)
    rooms_of_kind, insts_of_kind = _kind_tables(inv)
    where, caveat = _scope(scope)
    out, per_room = ([], {})
    kinds = sorted(rooms_of_kind)
    for i, kx in enumerate(kinds):
        for ky in kinds[i + 1:]:
            if len(out) >= limit:
                return out
            answer, _why = both_in_room_answer(rooms_of_kind[kx], rooms_of_kind[ky])
            if answer is None:
                continue
            words, display = spoken.get(answer, ((), None))
            if not words or not display:
                continue
            if any((p != answer and sp and score.match_answer(sp, (words,), 'word') for p, (_, sp) in spoken.items())):
                continue
            wx, wy = (kind_words(kx), kind_words(ky))
            if not wx or not wy or set(wx) & set(wy):
                continue
            aliases_x = tuple(sorted((f.alias for f in insts_of_kind[kx])))
            aliases_y = tuple(sorted((f.alias for f in insts_of_kind[ky])))
            if _reader_would_miss(aliases_x + aliases_y, by_alias):
                continue
            if per_room.get(display, 0) >= per_room_cap:
                continue
            per_room[display] = per_room.get(display, 0) + 1
            nx, ny = (_noun(kx), _noun(ky))
            evidence = tuple(sorted(set(aliases_x + aliases_y)))
            answer_here = tuple(sorted((f.alias for f in insts_of_kind[kx] + insts_of_kind[ky] if _where(f) == answer)))
            regions = frozenset(rooms_of_kind[kx] | rooms_of_kind[ky])
            out.append(TaskSpec(task=Task(task_id=f'both_in_room__{kx}__{ky}', instruction=f'Exactly one room in {where} contains both {_a(nx)} and {_a(ny)}; no other room has both. Search until you find that room, then tell me which room it is.{caveat}', answer_keywords=(words,), max_steps=max_steps), family='both_in_room', kinds=(kx, ky), evidence=evidence, answer_instances=answer_here, plan=(), oracle_answer=display, oracle_read=('room_with_both', kx, ky, _scope_rooms(inv), {p: sp for p, (_, sp) in sorted(spoken.items()) if sp}), evidence_bound=EvidenceBound(must_visit_regions=regions, must_see=frozenset(evidence), min_distinct_viewpoints=len(regions), search_regions=frozenset(_scope_rooms(inv))), truth={'kind_x': kx, 'kind_y': ky, 'answer': answer, 'answer_spoken': display, 'rooms_x': sorted(rooms_of_kind[kx]), 'rooms_y': sorted(rooms_of_kind[ky])}))
    return out
PLAUSIBLE_KINDS = ('Apple', 'Book', 'Bowl', 'Bread', 'Candle', 'CellPhone', 'Cup', 'Fork', 'HandTowel', 'Kettle', 'Knife', 'Laptop', 'Mug', 'Newspaper', 'Pan', 'Pen', 'Pencil', 'Plate', 'Pot', 'RemoteControl', 'SaltShaker', 'SoapBar', 'Spoon', 'Statue', 'Television', 'TissueBox', 'ToiletPaper', 'Vase', 'WateringCan', 'WineBottle')

def alpha_exist_negation(inv, max_steps=240, candidates=PLAUSIBLE_KINDS, limit=6, scope=HOUSE_SCOPE):
    present = {f.kind for f in inv}
    landmarks = _landmarks_per_room(inv)
    rooms = frozenset(landmarks)
    where, caveat = _scope(scope)
    out = []
    for kind in candidates:
        if kind in present:
            continue
        if len(out) >= limit:
            break
        plural = _plural(_noun(kind))
        out.append(TaskSpec(task=Task(task_id=f'count_absent__{kind}', instruction=_object_count_instruction(where, plural, caveat), answer_match='count', answer_number=0, max_steps=max_steps), family='count_absent', kinds=(kind,), evidence=tuple(sorted(landmarks.values())), answer_instances=(), plan=(), oracle_answer='0', oracle_read=('count_seen', kind, _scope_rooms(inv)), evidence_bound=EvidenceBound(must_visit_regions=rooms, must_see=frozenset(landmarks.values()), min_distinct_viewpoints=len(rooms), search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'total': 0, 'rooms': sorted(rooms)}))
    return out
DENIAL_WORDS = ('no', 'not', 'none', 'nothing', 'absent', 'without', 'isnt', 'arent', 'does not', 'do not', 'did not', 'could not', 'cannot', 'couldnt', 'havent', 'have not', 'lacks', 'lacking', 'empty', 'zero', 'never')

def unanswerable_tasks(inv, max_steps=240, candidates=PLAUSIBLE_KINDS, limit=4, scope=HOUSE_SCOPE):
    present = {f.kind for f in inv}
    landmarks = _landmarks_per_room(inv)
    rooms = frozenset(landmarks)
    where, caveat = _scope(scope)
    out = []
    for kind in candidates:
        if kind in present:
            continue
        if len(out) >= limit:
            break
        noun = _noun(kind)
        out.append(TaskSpec(task=Task(task_id=f'unanswerable__{kind}', instruction=_room_search_instruction(noun, where, caveat), answer_keywords=(DENIAL_WORDS, kind_words(kind)), max_steps=max_steps), family='unanswerable', kinds=(kind,), evidence=tuple(sorted(landmarks.values())), answer_instances=(), plan=(), oracle_answer=f'there is no {noun} in this house', oracle_read=('absent_confirm', kind, noun, _scope_rooms(inv)), evidence_bound=EvidenceBound(must_visit_regions=rooms, must_see=frozenset(landmarks.values()), min_distinct_viewpoints=len(rooms), search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'present': False, 'rooms': sorted(rooms)}))
    return out

def _question_nouns(text, vocabulary):
    low = text.lower()
    return [n for n in sorted(vocabulary, key=len, reverse=True) if re.search(f'\\b{re.escape(n)}\\b', low)]

def denial_baseline_progress(specs):
    vocab = {_noun(k) for k in PLAUSIBLE_KINDS}
    vocab |= {_noun(k) for s in specs for k in s.kinds or ()}
    scored = []
    for s in specs:
        p = baselines.denial_answer_progress(s.task, _question_nouns(s.task.instruction, vocab))
        if p is not None:
            scored.append(p)
    return sum(scored) / len(scored) if scored else 0.0

def epsilon_hidden_count(inv, max_steps=300, min_hidden=1, min_total=2, max_total=12, scope=HOUSE_SCOPE):
    children = _children_by_host(inv)
    by_alias = {f.alias: f for f in inv}
    by_kind = {}
    for f in inv:
        by_kind.setdefault(f.kind, []).append(f)
    out = []
    for kind, insts in sorted(by_kind.items()):
        n = len(insts)
        if not min_total <= n <= max_total:
            continue
        hidden, hosts = ([], set())
        for f in insts:
            for host in f.inside:
                h = by_alias.get(host)
                if h is not None and h.openable and (h.is_open is False) and f.pickupable:
                    hidden.append(f.alias)
                    hosts.add(host)
        if len(hidden) < min_hidden:
            continue
        rooms = _rooms_of(insts)
        plural = _plural(_noun(kind))
        aliases = tuple(sorted((f.alias for f in insts)))
        out.append(TaskSpec(task=Task(task_id=f'hidden_count__{kind}', instruction=f'Count every {_noun(kind)} in {_scope(scope)[0]}, including any that are shut inside cabinets, drawers, fridges or boxes. Open what you need to open.{_scope(scope)[1]} Tell me the total number of {plural}, as a number.', answer_match='count', answer_number=n, max_steps=max_steps), family='hidden_count', kinds=(kind,), receptacle_kinds=tuple(sorted({by_alias[h].kind for h in hosts})), evidence=aliases + tuple(sorted(hosts)), answer_instances=aliases, plan=(), oracle_answer=str(n), oracle_read=('count_seen_opening', kind, _scope_rooms(inv)), evidence_bound=EvidenceBound(must_visit_regions=frozenset(rooms), must_open=frozenset(hosts), must_see=frozenset(aliases), min_distinct_viewpoints=max(1, len(rooms)), search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'total': n, 'n_hidden': len(hidden), 'hidden': sorted(hidden), 'must_open': sorted(hosts), 'containment_semantics': 'inside_v1', 'rooms': sorted(rooms)}))
    return out

def epsilon_state_count(inv, max_steps=240, min_total=3, max_total=12, scope=HOUSE_SCOPE):
    by_kind = {}
    for f in inv:
        if f.openable and f.is_open is not None and (not f.pickupable):
            by_kind.setdefault(f.kind, []).append(f)
    out = []
    for kind, insts in sorted(by_kind.items()):
        n = len(insts)
        if not min_total <= n <= max_total:
            continue
        rooms = _rooms_of(insts)
        if len(rooms) < 2:
            continue
        n_open = sum((1 for f in insts if f.is_open))
        plural = _plural(_noun(kind))
        aliases = tuple(sorted((f.alias for f in insts)))
        out.append(TaskSpec(task=Task(task_id=f'state_count__{kind}__open', instruction=f'There are several {plural} in {_scope(scope)[0]}. Without opening or closing anything, count how many of them you find standing open, and tell me that number.{_scope(scope)[1]}', answer_match='count', answer_number=n_open, max_steps=max_steps), family='state_count', receptacle_kinds=(kind,), evidence=aliases, answer_instances=tuple(sorted((f.alias for f in insts if f.is_open))), plan=(), oracle_answer=str(n_open), oracle_read=('state_count_seen', kind, _scope_rooms(inv)), evidence_bound=EvidenceBound(must_visit_regions=frozenset(rooms), must_see=frozenset(aliases), min_distinct_viewpoints=len(rooms), search_regions=frozenset(_scope_rooms(inv))), truth={'kind': kind, 'total': n, 'n_open': n_open, 'open': sorted((f.alias for f in insts if f.is_open)), 'rooms': sorted(rooms)}))
    return out
ANGLE_TOL = 22
CENTRE_TOL = 12
SECTORS = tuple(range(0, 360, 45))

def _snap_sector(deg):
    best = min(SECTORS, key=lambda c: score.bearing_delta(c, deg))
    return (best, score.bearing_delta(best, deg))

def _pairs_by_room(inv, geom):
    kinds = collections.Counter((f.kind for f in inv))
    looks = collections.Counter((f.look for f in inv if f.look))
    out = {}
    for f in sorted(inv, key=lambda f: f.alias):
        place = _where(f)
        if not place or f.pickupable or f.alias not in geom:
            continue
        if not (f.receptacle or f.openable):
            continue
        if f.look:
            if looks[f.look] != 1:
                continue
        elif kinds[f.kind] != 1:
            continue
        out.setdefault(place, []).append(f)
    return out

def spatial_relative_direction(inv, geom=None, max_steps=300, limit=12, min_leg_m=1.5, scope=HOUSE_SCOPE):
    geom = dict(geom or {})
    if not geom:
        return []
    by_room = _pairs_by_room(inv, geom)
    rooms = sorted(by_room)
    out = []
    for a_room in rooms:
        here = by_room[a_room]
        if len(here) < 2:
            continue
        for A in here:
            for C in here:
                if C.alias == A.alias:
                    continue
                if math.dist(geom[A.alias], geom[C.alias]) < min_leg_m:
                    continue
                for b_room in rooms:
                    if b_room == a_room:
                        continue
                    for B in by_room[b_room]:
                        if math.dist(geom[A.alias], geom[B.alias]) < min_leg_m:
                            continue
                        deg = baselines.bearing_from(geom[A.alias], geom[C.alias], geom[B.alias])
                        if deg is None:
                            continue
                        centre, off = _snap_sector(deg)
                        if off > CENTRE_TOL:
                            continue
                        out.append(_relative_direction_spec(A, B, C, centre, off, deg, geom, max_steps, scope))
    out.sort(key=lambda s: (round(s.truth['offset_deg'], 3), s.task.task_id))
    return out[:limit]

def _relative_direction_spec(A, B, C, centre, off, deg, geom, max_steps, scope):
    return TaskSpec(task=Task(task_id=f'relative_direction__{A.alias}__{C.alias}__{B.alias}', instruction=f'Find the {_noun(A.kind)} and the {_noun(C.kind)}; they are in the same room. Also find the {_noun(B.kind)}, which is somewhere else in {_scope(scope)[0]}. Then answer this: if you stood at the {_noun(A.kind)} and faced the {_noun(C.kind)}, which direction would the {_noun(B.kind)} be in? Answer with one of: ahead, ahead-right, right, behind-right, behind, behind-left, left, ahead-left.{_scope(scope)[1]}', answer_match='direction', answer_number=int(centre), answer_tolerance=ANGLE_TOL, max_steps=max_steps), family='relative_direction', kinds=(A.kind, B.kind, C.kind), referents=(A.alias, C.alias, B.alias), evidence=(A.alias, B.alias, C.alias), answer_instances=(), plan=(), oracle_answer=str(int(centre)), oracle_read=('relative_direction', A.alias, C.alias, B.alias), evidence_bound=EvidenceBound(must_visit_regions=frozenset({_where(A), _where(B)}), must_open=frozenset(), must_see=frozenset({A.alias, B.alias, C.alias}), min_distinct_viewpoints=2, search_regions=frozenset(_scope_rooms([A, B, C]))), truth={'origin': A.alias, 'forward': C.alias, 'target': B.alias, 'bearing_deg': round(float(deg), 2), 'sector_deg': int(centre), 'offset_deg': round(float(off), 2), 'origin_room': _where(A), 'target_room': _where(B), 'leg_m': round(math.dist(geom[A.alias], geom[B.alias]), 2)})

def spatial_object_distance(inv, geom=None, max_steps=300, limit=12, min_m=3.0, tol_m=1, scope=HOUSE_SCOPE):
    geom = dict(geom or {})
    if not geom:
        return []
    by_room = _pairs_by_room(inv, geom)
    rooms = sorted(by_room)
    out = []
    for i, ra in enumerate(rooms):
        for rb in rooms[i + 1:]:
            for A in by_room[ra]:
                for B in by_room[rb]:
                    d = math.dist(geom[A.alias], geom[B.alias])
                    if d < min_m:
                        continue
                    out.append(_distance_spec(A, B, d, max_steps, tol_m, scope))
    out.sort(key=lambda s: (round(abs(s.truth['metres'] - s.task.answer_number), 3), s.task.task_id))
    return out[:limit]

def _distance_spec(A, B, d, max_steps, tol_m, scope):
    n = int(round(d))
    return TaskSpec(task=Task(task_id=f'object_distance__{A.alias}__{B.alias}', instruction=f'Find the {_noun(A.kind)} and the {_noun(B.kind)} in {_scope(scope)[0]}; they are in different rooms. Then tell me roughly how far apart they are, in metres, as a number. Measure the straight line between them, not the distance you walked.{_scope(scope)[1]}', answer_match='count', answer_number=n, answer_tolerance=int(tol_m), answer_unit='m', max_steps=max_steps), family='object_distance', kinds=(A.kind, B.kind), referents=(A.alias, B.alias), evidence=(A.alias, B.alias), answer_instances=(), plan=(), oracle_answer=str(n), oracle_read=('object_distance', A.alias, B.alias), evidence_bound=EvidenceBound(must_visit_regions=frozenset({_where(A), _where(B)}), must_open=frozenset(), must_see=frozenset({A.alias, B.alias}), min_distinct_viewpoints=2, search_regions=frozenset(_scope_rooms([A, B]))), truth={'a': A.alias, 'b': B.alias, 'metres': round(float(d), 2), 'a_room': _where(A), 'b_room': _where(B), 'tol_m': int(tol_m)})
PATH_ANGLE_TOL = 22.5
DYNAMIC_TRUTH = ('start',)
GEOM_ORIGIN = (0.0, 0.0)

def spatial_path_integration(inv, geom=None, max_steps=400, limit=12, min_m=4.0, start=None, min_end_m=3.0, scope=HOUSE_SCOPE):
    geom = dict(geom or {})
    if not geom:
        return []
    by_room = _pairs_by_room(inv, geom)
    rooms = sorted(by_room)
    by_end = {}
    for x_room in rooms:
        for y_room in rooms:
            if y_room == x_room:
                continue
            for X in by_room[x_room]:
                for Y in by_room[y_room]:
                    d = math.dist(geom[X.alias], geom[Y.alias])
                    if d < min_m:
                        continue
                    if start is not None and math.dist(geom[Y.alias], start) < min_end_m:
                        continue
                    by_end.setdefault(Y.alias, []).append(_path_integration_spec(X, Y, d, max_steps, scope))
    for group in by_end.values():
        group.sort(key=lambda s: (-s.truth['leg_m'], s.task.task_id))
    out = [group[0] for group in by_end.values()]
    out.sort(key=lambda s: (-s.truth['leg_m'], s.task.task_id))
    return out[:limit]

def _path_integration_spec(X, Y, d, max_steps, scope):
    return TaskSpec(task=Task(task_id=f'path_integration__{X.alias}__{Y.alias}', instruction=f'Find the {_noun(X.kind)} and then the {_noun(Y.kind)}; they are in different rooms of {_scope(scope)[0]}. When you have seen both, stay where you are -- do not walk back -- and answer this: from where you are standing and facing at that moment, which direction is the spot where you started this episode? Answer with one of: ahead, ahead-right, right, behind-right, behind, behind-left, left, ahead-left.{_scope(scope)[1]}', answer_match='direction', answer_tolerance=PATH_ANGLE_TOL, answer_from_pose=DYNAMIC_TRUTH, max_steps=max_steps), family='path_integration', kinds=(X.kind, Y.kind), referents=(X.alias, Y.alias), evidence=(X.alias, Y.alias), answer_instances=(), plan=(), oracle_answer='', oracle_read=('path_integration', X.alias, Y.alias), evidence_bound=EvidenceBound(must_visit_regions=frozenset({_where(X), _where(Y)}), must_open=frozenset(), must_see=frozenset({X.alias, Y.alias}), min_distinct_viewpoints=2, search_regions=frozenset(_scope_rooms([X, Y]))), truth={'x': X.alias, 'y': Y.alias, 'x_room': _where(X), 'y_room': _where(Y), 'leg_m': round(float(d), 2), 'dynamic': 'answer_time_pose', 'tol_deg': PATH_ANGLE_TOL})

def _choices_are_separable(words):
    for i, a in enumerate(words):
        for j, b in enumerate(words):
            if i != j and score.match_answer(a, ((b,),), 'word'):
                return False
    return True

def spatial_nearer_object_comparison(inv, geom=None, max_steps=300, limit=12, n_choices=3, margin_m=1.5, ratio=1.25, scope=HOUSE_SCOPE):
    geom = dict(geom or {})
    if not geom:
        return []
    by_room = _pairs_by_room(inv, geom)
    landmarks = [f for fs in by_room.values() for f in fs]
    out = []
    for X in landmarks:
        pool = sorted(((math.dist(geom[X.alias], geom[g.alias]), g.alias, g) for g in landmarks if g.alias != X.alias and _where(g) != _where(X)))
        if len(pool) < n_choices:
            continue
        d_win, _wa, winner = pool[0]
        runners = [(d, a, g) for d, a, g in pool[1:] if d >= d_win + margin_m and d >= d_win * ratio]
        if len(runners) < n_choices - 1:
            continue
        chosen = [pool[0]] + runners[:n_choices - 1]
        words = sorted((_noun(g.kind) for _d, _a, g in chosen))
        if not _choices_are_separable(words):
            continue
        out.append(_nearer_object_comparison_spec(X, winner, chosen, words, max_steps, scope))
    out.sort(key=lambda s: (-s.truth['lead_m'], s.task.task_id))
    return balance_choice_attacks(out, limit)

def _nearer_object_comparison_spec(X, winner, chosen, words, max_steps, scope):
    aliases = [a for _d, a, _g in chosen]
    rooms = {_where(g) for _d, _a, g in chosen} | {_where(X)}
    listed = ', '.join((f'the {w}' for w in words[:-1])) + f' and the {words[-1]}'
    return TaskSpec(task=Task(task_id=f'nearer_object_comparison__{X.alias}', instruction=f'Somewhere in {_scope(scope)[0]} there are {listed}. None of them is in the same room as the {_noun(X.kind)}. Find out which one of them is closest to the {_noun(X.kind)}, and answer with just that one thing. Compare the straight-line distances, not how far you had to walk.{_scope(scope)[1]}', answer_keywords=((_noun(winner.kind),),), answer_match='word', answer_choices=tuple(words), max_steps=max_steps), family='nearer_object_comparison', kinds=tuple((g.kind for _d, _a, g in chosen)) + (X.kind,), referents=(X.alias,) + tuple(aliases), evidence=(X.alias,) + tuple(aliases), answer_instances=(), plan=(), oracle_answer=_noun(winner.kind), oracle_read=('nearest_of', X.alias, tuple(((a, _noun(g.kind)) for _d, a, g in chosen))), evidence_bound=EvidenceBound(must_visit_regions=frozenset(rooms), must_open=frozenset(), must_see=frozenset([X.alias] + aliases), min_distinct_viewpoints=len(rooms), search_regions=frozenset(_scope_rooms([X] + [g for _d, _a, g in chosen]))), truth={'x': X.alias, 'winner': winner.alias, 'choices': list(aliases), 'words': list(words), 'metres': {a: round(float(d), 2) for d, a, _g in chosen}, 'lead_m': round(float(chosen[1][0] - chosen[0][0]), 2), 'x_room': _where(X)})

def spatial_change_detect(inv, move_targets=None, agent_visible=None, max_steps=400, limit=8, n_choices=5, scope=HOUSE_SCOPE):
    move_targets = dict(move_targets or {})
    house_kinds = collections.Counter((f.kind for f in inv))
    cand = []
    for f in sorted(inv, key=lambda g: g.alias):
        if house_kinds[f.kind] != 1 or not _where(f):
            continue
        if agent_visible is not None and f.alias not in agent_visible:
            continue
        if f.openable and f.is_open is not None:
            cand.append((f, 'open', not f.is_open))
        elif f.alias in move_targets:
            cand.append((f, 'move', tuple(move_targets[f.alias])))
    cand = cand[:limit]
    words = sorted((_noun(f.kind) for f, _op, _a in cand))
    rooms = sorted({_where(f) for f, _op, _a in cand})
    if len(cand) < n_choices or len(rooms) < 2 or (not _choices_are_separable(words)):
        return []
    openables = [f for f, op, _a in cand if op == 'open']
    out = []
    for f, op, arg in cand:
        if op == 'open' and (not any((g.alias != f.alias and g.is_open for g in openables))):
            continue
        out.append(_change_detect_spec(f, op, arg, cand, words, rooms, max_steps, scope, agent_visible is not None))
    return out

def _change_detect_spec(target, op, arg, cand, words, rooms, max_steps, scope, vis_measured):
    pairs = tuple(((f.alias, _noun(f.kind)) for f, _op, _a in cand))
    listed = ', '.join((f'the {w}' for w in words[:-1])) + f' and the {words[-1]}'
    return TaskSpec(task=Task(task_id=f'change_detect__{target.alias}', instruction=f'In {_scope(scope)[0]} there are {listed}. Walk through the house and look at all {len(words)} of them, then walk through and look at all {len(words)} of them a second time. Exactly one of them is different the second time you see it -- it has been opened, closed, or moved somewhere else in its room. Nothing else changes. Tell me which one changed, and answer with just that one thing.{_scope(scope)[1]}', answer_keywords=((_noun(target.kind),),), answer_match='word', answer_choices=tuple(words), mutation=Mutation(op=op, alias=target.alias, arg=arg, room=_where(target), after_rooms=tuple(rooms)), max_steps=max_steps), family='change_detect', kinds=tuple((f.kind for f, _op, _a in cand)), referents=tuple((a for a, _w in pairs)), evidence=tuple((a for a, _w in pairs)), answer_instances=(), plan=(), oracle_answer=None, oracle_read=('change_detect', pairs), evidence_bound=EvidenceBound(must_visit_regions=frozenset(rooms), must_open=frozenset(), must_see=frozenset((a for a, _w in pairs)), min_distinct_viewpoints=2 * len(rooms), search_regions=frozenset(_scope_rooms([f for f, _op, _a in cand]))), truth={'target': target.alias, 'op': op, 'arg': list(arg) if op == 'move' else arg, 'target_room': _where(target), 'rooms': list(rooms), 'choices': [a for a, _w in pairs], 'words': list(words), 'agent_visible_measured': bool(vis_measured), 'open_at_start': {f.alias: f.is_open for f, o, _a in cand if o == 'open'}})

def _gap_clear(fa, fb, others, axis, clear=0.05):
    ax, az, asx, asz = fa
    bx, bz, bsx, bsz = fb
    if axis == 'x':
        lo, hi = (min(ax + asx / 2, bx + bsx / 2), max(ax - asx / 2, bx - bsx / 2))
        plo, phi = (max(az - asz / 2, bz - bsz / 2), min(az + asz / 2, bz + bsz / 2))
    else:
        lo, hi = (min(az + asz / 2, bz + bsz / 2), max(az - asz / 2, bz - bsz / 2))
        plo, phi = (max(ax - asx / 2, bx - bsx / 2), min(ax + asx / 2, bx + bsx / 2))
    for kx, kz, ksx, ksz in others:
        kg, ksg = (kx, ksx) if axis == 'x' else (kz, ksz)
        kp, ksp = (kz, ksz) if axis == 'x' else (kx, ksx)
        if kg - ksg / 2 < hi + clear and kg + ksg / 2 > lo - clear and (kp - ksp / 2 < phi + clear) and (kp + ksp / 2 > plo - clear):
            return False
    return True

def spatial_gap_width(inv, foot=None, max_steps=300, limit=12, band=(50, 150), tol_cm=10, scope=HOUSE_SCOPE):
    foot = dict(foot or {})
    if not foot:
        return []
    by_room = _pairs_by_room(inv, foot)
    lo, hi = band
    out = []
    for room, marks in sorted(by_room.items()):
        for A, B in itertools.combinations(marks, 2):
            fa, fb = (foot.get(A.alias), foot.get(B.alias))
            if fa is None or fb is None:
                continue
            got = baselines.gap_of(fa, fb)
            if got is None:
                continue
            metres, axis = got
            cm = int(round(metres * 100))
            if not lo <= cm <= hi:
                continue
            others = [foot[m.alias] for m in marks if m.alias not in (A.alias, B.alias) and m.alias in foot]
            if not _gap_clear(fa, fb, others, axis):
                continue
            out.append(_gap_spec(A, B, cm, max_steps, tol_cm, scope))
    out.sort(key=lambda s: (-s.task.answer_number, s.task.task_id))
    return out[:limit]

def _gap_spec(A, B, cm, max_steps, tol_cm, scope):
    return TaskSpec(task=Task(task_id=f'gap_width__{A.alias}__{B.alias}', instruction=f'Find the {_noun(A.kind)} and the {_noun(B.kind)} in {_scope(scope)[0]}; they are in the same room, standing apart. Then tell me roughly how wide the clear gap between them is, in centimetres, as a number. Measure the empty floor from the edge of one to the edge of the other, not the distance between their centres.{_scope(scope)[1]}', answer_match='count', answer_number=cm, answer_tolerance=int(tol_cm), answer_unit='cm', max_steps=max_steps), family='gap_width', kinds=(A.kind, B.kind), referents=(A.alias, B.alias), evidence=(A.alias, B.alias), answer_instances=(), plan=(), oracle_answer=str(cm), oracle_read=('gap_width', A.alias, B.alias), evidence_bound=EvidenceBound(must_visit_regions=frozenset({_where(A)}), must_open=frozenset(), must_see=frozenset({A.alias, B.alias}), min_distinct_viewpoints=1, search_regions=frozenset(_scope_rooms([A, B]))), truth={'a': A.alias, 'b': B.alias, 'cm': cm, 'room': _where(A), 'tol_cm': int(tol_cm)})

def spatial_room_area(inv, areas=None, max_steps=400, limit=12, band=(14, 200), tol_frac=0.12, min_tol=3, scope=HOUSE_SCOPE):
    areas = dict(areas or {})
    if not areas:
        return []
    label_of = {}
    for f in inv:
        rid = _where(f)
        if rid is not None:
            label_of.setdefault(rid, f.room)
    landmarks = baselines.room_landmarks(inv)
    lo, hi = band
    house = frozenset(label_of) | frozenset(areas)
    cand = []
    for rid, area in areas.items():
        if rid not in label_of or rid not in landmarks:
            continue
        if not lo <= area <= hi:
            continue
        words, display = room_answer(rid, label_of[rid], house)
        if not words or not display:
            continue
        cand.append((rid, area, display))
    spelled = collections.Counter((_normalise(d) for _, _, d in cand))
    cand = [c for c in cand if spelled[_normalise(c[2])] == 1]

    def tol_of(area):
        return max(int(min_tol), int(round(tol_frac * area)))
    out, right = ([], None)
    for rid, area, display in sorted(cand, key=lambda c: (round(c[1] + tol_of(c[1]), 6), c[0])):
        tol = tol_of(area)
        if right is not None and area - tol <= right:
            continue
        right = area + tol
        out.append(_room_area_spec(rid, display, inv, area, tol, max_steps, scope))
    return out[:limit]

def _room_area_spec(rid, display, inv, area, tol, max_steps, scope):
    m2 = int(round(area))
    return TaskSpec(task=Task(task_id=f'room_area__{rid}', instruction=f'Find the {display} in {_scope(scope)[0]}, then tell me roughly how big its floor is, in square metres, as a number. I want the floor area of that one room, measured inside its walls -- not the whole house, and not the other rooms.{_scope(scope)[1]}', answer_match='count', answer_number=m2, answer_tolerance=int(tol), answer_unit='m2', max_steps=max_steps), family='room_area', kinds=(), referents=(), evidence=(), answer_instances=(), plan=(), oracle_answer=str(m2), oracle_read=('room_area', rid), evidence_bound=EvidenceBound(must_visit_regions=frozenset({rid}), must_open=frozenset(), must_see=frozenset(), min_distinct_viewpoints=1, search_regions=frozenset(_scope_rooms(inv))), truth={'room': rid, 'name': display, 'm2': m2, 'tol_m2': int(tol)})

def spatial_topo_route(inv, graph=None, max_steps=500, limit=12, min_middle=1, max_share=0.25, scope=HOUSE_SCOPE, standing=None):
    graph = {r: tuple(ns) for r, ns in (graph or {}).items()}
    if len(graph) < 2:
        return []
    label_of = {}
    for f in inv:
        rid = _where(f)
        if rid is not None:
            label_of.setdefault(rid, f.room)
    landmarks = baselines.room_handles(inv, standing)
    named = {}
    house = frozenset(label_of) | frozenset(graph)
    for rid in graph:
        words, display = room_answer(rid, label_of.get(rid), house)
        if words and display:
            named[rid] = (tuple(words), display)
    usable = set(named)
    for a, (_wa, da) in named.items():
        for b, (kb, _db) in named.items():
            if a != b and score.match_answer(da, (kb,), 'word'):
                usable.discard(a)
                usable.discard(b)
    usable = {r for r in usable if r in landmarks}
    if standing:
        usable = {r for r in usable if standing.get(landmarks[r], 1) > 0}
    vocabulary = tuple(sorted((d for _k, d in named.values())))
    if len(usable) < 2:
        return []
    routes = []
    for src, dst, path in _shortest_room_paths(graph):
        if len(path) < min_middle + 2:
            continue
        if src not in usable or dst not in usable:
            continue
        if any((r not in usable for r in path[1:-1])):
            continue
        routes.append(path)
    routes.sort(key=lambda p: (-len(p), tuple((named[r][1] for r in p))))
    specs = [_topo_route_spec(p, named, vocabulary, inv, max_steps, scope) for p in routes]
    return _balance_by_attack(specs, limit, max_share)

def _shortest_room_paths(graph):
    out = []
    for src in graph:
        dist, preds = ({src: 0}, collections.defaultdict(list))
        q = collections.deque([src])
        while q:
            n = q.popleft()
            for m in graph.get(n, ()):
                if m not in dist:
                    dist[m] = dist[n] + 1
                    preds[m].append(n)
                    q.append(m)
                elif dist[m] == dist[n] + 1:
                    preds[m].append(n)
        for dst in graph:
            if dst == src or dst not in dist:
                continue
            paths, stack = ([], [(dst, (dst,))])
            while stack and len(paths) < 2:
                n, tail = stack.pop()
                if n == src:
                    paths.append(tail)
                    continue
                for p in preds[n]:
                    stack.append((p, (p,) + tail))
            if len(paths) == 1:
                out.append((src, dst, paths[0]))
    return out

def _balance_by_attack(specs, limit, max_share, distinct=True):
    if not specs:
        return []
    middles = {gates.topo_route_middle(s.task) for s in specs}
    middles.add(())
    hits = {m: [gates.topo_route_attack_hits(m, s.task) for s in specs] for m in middles}
    for n in range(min(limit, len(specs)), 0, -1):
        cap = int(max_share * n)
        if cap < 1:
            break
        if distinct:
            cap = 1
        picked, count = ([], collections.Counter())
        for i, s in enumerate(specs):
            if len(picked) >= n:
                break
            if any((count[m] + 1 > cap for m in middles if hits[m][i])):
                continue
            for m in middles:
                if hits[m][i]:
                    count[m] += 1
            picked.append(s)
        if len(picked) == n:
            return picked
    return []

def choice_attack_hits(word, task):
    return bool(task.answer_choices) and bool(score.match_answer(word, task.answer_keywords, task.answer_match))

def balance_choice_attacks(specs, limit):
    if not specs:
        return []
    words = {c for s in specs for c in s.task.answer_choices or ()}
    hits = {w: [choice_attack_hits(w, s.task) for s in specs] for w in words}
    picked, count = ([], collections.Counter())
    for i, s in enumerate(specs):
        if len(picked) >= limit:
            break
        if any((count[w] >= 1 for w in words if hits[w][i])):
            continue
        for w in words:
            if hits[w][i]:
                count[w] += 1
        picked.append(s)
    return picked

def _topo_route_spec(path, named, vocabulary, inv, max_steps, scope):
    rooms = [named[r][1] for r in path]
    a, b = (rooms[0], rooms[-1])
    return TaskSpec(task=Task(task_id=f'topo_route__{path[0]}__{path[-1]}', instruction=f'Somewhere in {_scope(scope)[0]} there is {_a(a)} and {_a(b)}. Work out the shortest way to walk from the {a} to the {b}, and tell me which rooms you go through, in order. Start your list with the {a} and end it with the {b}. Name only the rooms on that route -- if you name a room that is not on it, the answer is wrong.{_scope(scope)[1]}', answer_keywords=tuple((named[r][0] for r in path)), answer_match='sequence', answer_vocabulary=vocabulary, max_steps=max_steps), family='topo_route', kinds=(), referents=(), evidence=(), answer_instances=(), plan=(), oracle_answer=', '.join(rooms), oracle_read=('topo_route', tuple(((r, named[r][1]) for r in path))), evidence_bound=EvidenceBound(must_visit_regions=frozenset(path), must_open=frozenset(), must_see=frozenset(), min_distinct_viewpoints=len(path), search_regions=frozenset(_scope_rooms(inv))), truth={'from': path[0], 'to': path[-1], 'route': list(path), 'rooms': rooms, 'middle': rooms[1:-1]})

def gamma_reach_blocked(inv, standing=None, reachable=None, max_steps=400, limit=12, n_choices=5, min_cells=8, scope=HOUSE_SCOPE):
    standing = dict(standing or {})
    if not standing:
        return []
    reachable = None if reachable is None else {str(alias): bool(value) for alias, value in dict(reachable).items()}
    measured_reachable = lambda alias: True if reachable is None else reachable.get(alias) is True
    measured_blocked = lambda alias: True if reachable is None else reachable.get(alias) is False
    by_room = _pairs_by_room(inv, standing)
    marks = [f for fs in by_room.values() for f in fs]
    live_rooms = {room for room, fs in by_room.items() if any((standing.get(f.alias, 0) >= min_cells and measured_reachable(f.alias) for f in fs))}
    free = sorted((f for f in marks if standing.get(f.alias, 0) >= min_cells and measured_reachable(f.alias)), key=lambda f: (-standing[f.alias], f.alias))
    out = []
    for w in sorted((f for f in marks if standing.get(f.alias) == 0 and _where(f) in live_rooms and measured_blocked(f.alias)), key=lambda f: f.alias):
        chosen, words, rooms_used = ([], [_noun(w.kind)], {_where(w)})
        for pool in (True, False):
            for f in free:
                if len(chosen) >= n_choices - 1:
                    break
                if f.alias == w.alias or f in chosen:
                    continue
                if pool and _where(f) in rooms_used:
                    continue
                word = _noun(f.kind)
                if not _choices_are_separable(words + [word]):
                    continue
                chosen.append(f)
                words.append(word)
                rooms_used.add(_where(f))
        if len(chosen) < n_choices - 1:
            continue
        out.append(_reach_blocked_spec(w, chosen, sorted(words), standing, max_steps, scope, reachable=reachable))
    out.sort(key=lambda s: (-s.truth['margin_cells'], s.task.task_id))
    return balance_choice_attacks(out, limit)

def _reach_blocked_spec(w, chosen, words, standing, max_steps, scope, reachable=None):
    aliases = [f.alias for f in chosen]
    rooms = {_where(f) for f in chosen} | {_where(w)}
    listed = ', '.join((f'the {x}' for x in words[:-1])) + f' and the {words[-1]}'
    return TaskSpec(task=Task(task_id=f'reach_blocked__{w.alias}', instruction=f'Somewhere in {_scope(scope)[0]} there are {listed}. Four of them you can walk right up to, close enough to reach out and touch. One of them you cannot: you can get into its room, but not up to the thing itself -- the way in is too tight, and the closest you can get leaves you out of reach. Go and try each of them, then tell me which one you could not get up to, and answer with just that one thing. Judge it by trying to walk there, not by how it looks.{_scope(scope)[1]}', answer_keywords=((_noun(w.kind),),), answer_match='word', answer_choices=tuple(words), max_steps=max_steps), family='reach_blocked', kinds=tuple((f.kind for f in chosen)) + (w.kind,), referents=(w.alias,) + tuple(aliases), evidence=(w.alias,) + tuple(aliases), answer_instances=(w.alias,), plan=(), oracle_answer=_noun(w.kind), oracle_read=('reach_blocked', tuple(sorted([(w.alias, _noun(w.kind))] + [(f.alias, _noun(f.kind)) for f in chosen]))), evidence_bound=EvidenceBound(must_visit_regions=frozenset(rooms), must_open=frozenset(), must_see=frozenset([w.alias] + aliases), min_distinct_viewpoints=len(rooms), search_regions=frozenset(_scope_rooms([w] + list(chosen)))), truth={'winner': w.alias, 'choices': list(aliases), 'words': list(words), 'cells': {a: int(standing.get(a, 0)) for a in [w.alias] + aliases}, 'margin_cells': min((int(standing.get(a, 0)) for a in aliases)), 'winner_room': _where(w), **({'legal_navigation_reachable': {a: bool(reachable[a]) for a in [w.alias] + aliases}} if reachable is not None else {})})
FAMILIES = {'room_localization': room_localization_tasks, 'surface_contents': surface_contents_tasks, 'relocation': relocation_tasks, 'object_count': alpha_object_count, 'count_in_room': alpha_count_conditional, 'room_count': alpha_room_count, 'room_comparison': alpha_compare_extremum, 'both_in_room': alpha_both_in_room, 'count_absent': alpha_exist_negation, 'hidden_count': epsilon_hidden_count, 'state_count': epsilon_state_count, 'relative_direction': spatial_relative_direction, 'object_distance': spatial_object_distance, 'nearer_object_comparison': spatial_nearer_object_comparison, 'path_integration': spatial_path_integration, 'change_detect': spatial_change_detect, 'gap_width': spatial_gap_width, 'room_area': spatial_room_area, 'topo_route': spatial_topo_route, 'reach_blocked': gamma_reach_blocked, 'unanswerable': unanswerable_tasks}
GEOMETRIC = ('relative_direction', 'object_distance', 'nearer_object_comparison', 'path_integration')
FOOTPRINT = ('gap_width',)
AREAS = ('room_area',)
GRAPH = ('topo_route',)
STANDING = ('topo_route', 'reach_blocked')
TIER_REQUIRED = {'room_localization': 'plan', 'surface_contents': 'plan', 'relocation': 'plan', 'object_count': 'plan', 'count_in_room': 'plan', 'room_comparison': 'plan', 'both_in_room': 'plan', 'count_absent': 'plan', 'room_count': 'plan', 'hidden_count': 'plan', 'state_count': 'plan', 'relative_direction': 'spatial', 'object_distance': 'spatial', 'nearer_object_comparison': 'spatial', 'gap_width': 'spatial', 'path_integration': 'spatial', 'change_detect': 'spatial', 'room_area': 'spatial', 'topo_route': 'spatial', 'reach_blocked': 'spatial', 'unanswerable': 'plan'}
A_CLASS = ('object_count', 'count_in_room', 'room_count', 'room_comparison', 'both_in_room', 'count_absent', 'hidden_count', 'state_count', 'unanswerable')
CAPABILITY_AXIS = {'room_localization': 'G1', 'surface_contents': 'G1', 'room_comparison': 'G1', 'relocation': 'G5', 'object_count': 'G2', 'count_in_room': 'G2', 'count_absent': 'G2', 'room_count': 'G2', 'both_in_room': 'G2', 'hidden_count': 'G2', 'state_count': 'G2', 'relative_direction': 'G4', 'object_distance': 'G3', 'nearer_object_comparison': 'G3', 'gap_width': 'G3', 'room_area': 'G3', 'topo_route': 'G4', 'path_integration': 'G4', 'change_detect': 'G4', 'reach_blocked': 'G3', 'unanswerable': 'G1'}
_L4_AXES = ('G3', 'G4')

def difficulty_level(spec):
    axis = CAPABILITY_AXIS.get(getattr(spec, 'family', None))
    b = getattr(spec, 'evidence_bound', None)
    detail = {'axis': axis, 'family': getattr(spec, 'family', None), 'R': None, 'O': None, 'V': None, 'search_regions': None}
    if b is None:
        return (None, detail)
    R = len(b.must_visit_regions)
    O = len(b.must_open)
    detail.update(R=R, O=O, V=b.min_distinct_viewpoints, search_regions=len(b.search_regions))
    if axis in _L4_AXES:
        return ('L4', detail)
    if R >= 4 and O >= 1:
        return ('L4', detail)
    if R >= 4 or O >= 1:
        return ('L3', detail)
    if R >= 2:
        return ('L2', detail)
    return ('L1', detail)

def difficulty_histogram(specs):
    out = collections.Counter()
    for s in specs:
        level, detail = difficulty_level(s)
        out[detail['axis'], level] += 1
    return dict(out)

def generate(sim, families=None, inv=None, **kw):
    inv = sim.inventory() if inv is None else tuple(inv)
    names = list(families or FAMILIES)
    geom = {}
    if any((n in GEOMETRIC for n in names)):
        geom = geometry_of(sim, inv)
    foot = {}
    if any((n in FOOTPRINT for n in names)):
        foot = footprints_of(sim, inv)
    areas = {}
    if any((n in AREAS for n in names)):
        areas = areas_of(sim)
    graph = {}
    if any((n in GRAPH for n in names)):
        graph = room_graph_of(sim)
    standing = {}
    if any((n in STANDING for n in names)):
        standing = standing_room_localization(sim)
    out = []
    for name in names:
        opts = dict(kw.get(name, {}))
        if name in GEOMETRIC:
            opts.setdefault('geom', geom)
        if name in FOOTPRINT:
            opts.setdefault('foot', foot)
        if name in AREAS:
            opts.setdefault('areas', areas)
        if name in GRAPH:
            opts.setdefault('graph', graph)
        if name in STANDING:
            opts.setdefault('standing', standing)
        out += FAMILIES[name](inv, **opts)
    return out

def geometry_of(sim, inv):
    if not hasattr(sim, 'bearing_to'):
        return {}
    return baselines.relative_xz(sim, [f.alias for f in inv])

def footprints_of(sim, inv):
    if not hasattr(sim, 'footprint'):
        return {}
    out = {}
    for f in inv:
        box = sim.footprint(f.alias)
        if box is not None:
            out[f.alias] = box
    return out

def areas_of(sim):
    f = getattr(sim, 'room_sizes', None)
    return dict(f()) if callable(f) else {}

def room_graph_of(sim):
    f = getattr(sim, 'room_graph', None)
    return dict(f()) if callable(f) else {}

def standing_room_localization(sim):
    f = getattr(sim, 'standing_room', None)
    return dict(f()) if callable(f) else {}

def require_geometry(sim, families=None):
    names = list(families or FAMILIES)
    wanted = [n for n in names if n in GEOMETRIC]
    if wanted and (not hasattr(sim, 'bearing_to')):
        return f"几何任务未生成：{', '.join(wanted)} 需要几何量，而这个 sim 不提供 bearing_to —— 本次候选里没有度量题"
    wanted = [n for n in names if n in FOOTPRINT]
    if wanted and (not hasattr(sim, 'footprint')):
        return f"几何任务未生成：{', '.join(wanted)} 需要物体的包围盒，而这个 sim 不提供 footprint —— 本次候选里没有间隙题"
    wanted = [n for n in names if n in AREAS]
    if wanted and (not hasattr(sim, 'room_sizes')):
        return f"几何任务未生成：{', '.join(wanted)} 需要房间的地板多边形，而这个 sim 不提供 room_sizes —— 本次候选里没有面积题"
    wanted = [n for n in names if n in STANDING and n != 'topo_route']
    if wanted and (not hasattr(sim, 'standing_room')):
        return f"可达性任务未生成：{', '.join(wanted)} 需要每件家具旁边可站的格点数，而这个 sim 不提供 standing_room —— 本次候选里没有可达性题"
    return None

def _counting(specs):
    return [s for s in specs if s.task.answer_number is not None and s.task.answer_match == 'count']

def direction_prior_for(specs):
    truths = [s.task.answer_number for s in specs if s.task.answer_match == 'direction' and s.task.answer_number is not None]
    if not truths:
        return None
    best, best_hits = (None, -1)
    for cand in SECTORS:
        hits = 0
        for s in specs:
            if s.task.answer_match != 'direction' or s.task.answer_number is None:
                continue
            tol = s.task.answer_tolerance
            if score.bearing_delta(cand, s.task.answer_number) <= (tol if tol is not None else 0):
                hits += 1
        if hits > best_hits:
            best, best_hits = (cand, hits)
    return best

def choice_prior_for(specs):
    choice = [s for s in specs if s.task.answer_choices]
    if not choice:
        return None
    pool = sorted({w for s in choice for w in s.task.answer_choices})
    best, best_hits = (None, -1)
    for cand in pool:
        hits = sum((1 for s in choice if score.match_answer(cand, s.task.answer_keywords, s.task.answer_match)))
        if hits > best_hits:
            best, best_hits = (cand, hits)
    return best

def count_priors_for(specs):
    units = {count_unit(s) for s in _counting(specs)}
    return {u: count_prior_for(specs, u) for u in sorted(units)}

def count_prior_for(specs, unit='count'):
    counting = [s for s in _counting(specs) if count_unit(s) == unit]
    truths = [s.task.answer_number for s in counting]
    if not truths:
        return None
    best, best_hits = (None, -1)
    for cand in sorted(set(truths)):
        hits = 0
        for s in counting:
            std = s.task.answer_number
            tol = s.task.answer_tolerance
            hits += 1 if (abs(cand - std) <= tol if tol is not None else cand == std) else 0
        if hits > best_hits:
            best, best_hits = (cand, hits)
    return best

def count_channel_loss(specs, unit='count'):
    prior = count_prior_for(specs, unit)
    if prior is None:
        return (None, 0, [])
    hit = []
    for s in _counting(specs):
        if count_unit(s) != unit:
            continue
        std = s.task.answer_number
        tol = s.task.answer_tolerance
        if abs(prior - std) <= tol if tol is not None else prior == std:
            hit.append(s.task.task_id)
    return (prior, len(hit), hit)
_ALL_FAMILIES = object()

def _word_channel(specs, family=_ALL_FAMILIES):
    out = [s for s in specs if gates.g1_channel(s.task) == 'word']
    if family is not _ALL_FAMILIES:
        out = [s for s in out if getattr(s, 'family', None) == family]
    return out

def word_prior_for(specs, family):
    word = _word_channel(specs, family)
    if not word:
        return None
    pool = sorted({w for s in word for g in s.task.answer_keywords for w in g})
    best, best_hits = (None, -1)
    for cand in pool:
        hits = sum((1 for s in word if score.match_answer(cand, s.task.answer_keywords, s.task.answer_match)))
        if hits > best_hits:
            best, best_hits = (cand, hits)
    return best

def word_priors_for(specs):
    fams = {getattr(s, 'family', None) for s in _word_channel(specs)}
    return {f: word_prior_for(specs, f) for f in sorted(fams, key=lambda x: str(x))}

def word_channel_loss(specs, family):
    word = _word_channel(specs, family)
    prior = word_prior_for(specs, family)
    if prior is None:
        return (None, 0, [], 0, None)
    hit = [s.task.task_id for s in word if score.match_answer(prior, s.task.answer_keywords, s.task.answer_match)]
    n = len(word)
    return (prior, len(hit), hit, n, gates.word_prior_floor(n))

def word_channel_line(specs, family):
    prior, n_hit, hit_ids, n, floor = word_channel_loss(specs, family)
    if prior is None or not n:
        return ''
    rate = n_hit / n
    cost = f'G1 会毙掉 {n_hit}/{n}: ' + ', '.join(sorted(hit_ids)) if hit_ids else f'毙不掉任何题 (0/{n})'
    if n < gates.G1_WORD_MIN_N:
        why = f'n<{gates.G1_WORD_MIN_N}，命中率读不出结论（下限 1/n 本身就 > RANDOM_MAX {gates.RANDOM_MAX}）'
    elif rate > floor:
        why = '答案聚集（高于下限），损失是结构性的，不是一次碰撞'
    elif rate < floor:
        why = '低于下限：答案是合取的，单词常量对这一族结构上无效'
    else:
        why = '正好在下限上：损失是常量与它自己那道题的碰撞，答案是分散的'
    return f'word channel ({family}): 常量 {prior!r} -- {cost} (命中率 {rate:.2f}, 下限 {floor:.2f}); {why}'

def _choice_channel(specs, family=_ALL_FAMILIES):
    out = [s for s in specs if gates.g1_channel(s.task) == 'choice']
    if family is not _ALL_FAMILIES:
        out = [s for s in out if getattr(s, 'family', None) == family]
    return out

def choice_prior_for_family(specs, family):
    choice = _choice_channel(specs, family)
    if not choice:
        return None
    pool = sorted({w for s in choice for w in s.task.answer_choices})
    best, best_hits = (None, -1)
    for cand in pool:
        hits = sum((1 for s in choice if score.match_answer(cand, s.task.answer_keywords, s.task.answer_match)))
        if hits > best_hits:
            best, best_hits = (cand, hits)
    return best

def choice_channel_loss(specs, family):
    choice = _choice_channel(specs, family)
    prior = choice_prior_for_family(specs, family)
    if prior is None:
        return (None, 0, [], 0, None)
    hit = [s.task.task_id for s in choice if score.match_answer(prior, s.task.answer_keywords, s.task.answer_match)]
    n = len(choice)
    return (prior, len(hit), hit, n, gates.modal_constant_floor(n))
_SETLEVEL_CHANNELS = {'word': word_prior_for, 'choice': choice_prior_for_family}
SETLEVEL_ENFORCED_CHANNELS = ('choice', 'word')

def setlevel_constant_pass(specs, alive_ids, enforced=SETLEVEL_ENFORCED_CHANNELS):
    alive = [s for s in specs if s.task.task_id in set(alive_ids)]
    out = {}
    for channel, prior_fn in sorted(_SETLEVEL_CHANNELS.items()):
        fams = {getattr(s, 'family', None) for s in specs if gates.g1_channel(s.task) == channel}
        for family in sorted(fams, key=lambda x: str(x)):
            same = [s for s in alive if getattr(s, 'family', None) == family and gates.g1_channel(s.task) == channel]
            n = len(same)
            if not n:
                continue
            best = prior_fn(alive, family)
            if best is None:
                continue
            hit_ids = sorted((s.task.task_id for s in same if score.match_answer(best, s.task.answer_keywords, s.task.answer_match)))
            rate = len(hit_ids) / n
            floor = gates.modal_constant_floor(n)
            before = [s for s in specs if getattr(s, 'family', None) == family and gates.g1_channel(s.task) == channel]
            n_before = len(before)
            best_before = prior_fn(specs, family)
            hit_before = sum((1 for s in before if best_before is not None and score.match_answer(best_before, s.task.answer_keywords, s.task.answer_match)))
            if rate <= gates.RANDOM_MAX:
                verdict = 'ok'
                why = f'命中率 {rate:.2f} <= RANDOM_MAX {gates.RANDOM_MAX}' + (f'，但高于下限 {floor:.2f}（净空 +{rate - floor:.2f}），离阈值不远' if rate > floor else f'，且不高于下限 {floor:.2f}' + ('；低于下限说明答案是合取的，单常量对这一族结构上无效' if rate < floor else ''))
            elif n < gates.G1_MODAL_MIN_N:
                verdict = 'too_few'
                why = f'命中率 {rate:.2f} > RANDOM_MAX {gates.RANDOM_MAX}，而存活 {n} < {gates.G1_MODAL_MIN_N}，下限 1/n = {floor:.2f} 本身就在阈值之上：任何答案分布都救不了，要修的是供给不是配平'
            else:
                verdict = 'clustered'
                why = f'命中率 {rate:.2f} > RANDOM_MAX {gates.RANDOM_MAX}，而下限只有 {floor:.2f}（净空 +{rate - floor:.2f}）：n 够大，所以这不是族太小，是答案聚集，要修的是生成时配平'
            enforced_here = channel in enforced
            drop = verdict != 'ok' and enforced_here
            out[family] = {'family': family, 'channel': channel, 'n': n, 'best': best, 'hits': len(hit_ids), 'hit_ids': hit_ids, 'rate': round(rate, 3), 'floor': round(floor, 3), 'headroom': round(rate - floor, 3), 'threshold': gates.RANDOM_MAX, 'min_n': gates.G1_MODAL_MIN_N, 'n_before': n_before, 'best_before': best_before, 'rate_before': round(hit_before / n_before, 3) if n_before else None, 'verdict': verdict, 'enforced': enforced_here, 'dropped': drop, 'veto_ids': sorted((s.task.task_id for s in same)) if drop else [], 'reason': why, 'task_ids': sorted((s.task.task_id for s in same))}
    return out

def setlevel_constant_line(report):
    if not report:
        return ''
    r = report
    arrow = ''
    if r.get('rate_before') is not None and r['n_before'] != r['n']:
        arrow = f"，拒稿前是 {r['rate_before']:.2f} over {r['n_before']} (常量 {r['best_before']!r})"
    if r['verdict'] == 'ok':
        tail = '-> 过'
    elif r['dropped']:
        tail = f"-> 毙掉全族 {r['n']} 道"
    else:
        tail = f"-> **测到了但没执行**（{r['channel']} 通道不在 SETLEVEL_ENFORCED_CHANNELS 里），这一族照原样发货"
    return f"G1 任务质量检查（同类任务） ({r['family']}, {r['channel']} 通道): 存活 {r['n']} 道，最优常量 {r['best']!r} 命中 {r['rate']:.3f}，下限 {r['floor']:.3f}{arrow}；{r['verdict']}：{r['reason']} {tail}"
FULLSET_CONSTANT_ENFORCED = False

def fullset_constant_pass(specs, alive_ids, enforced=FULLSET_CONSTANT_ENFORCED):
    alive = [s for s in specs if s.task.task_id in set(alive_ids)]
    covered = [s for s in alive if gates.g1_channel(s.task) in _SETLEVEL_CHANNELS]
    if not covered:
        return {}

    def _pooled_best(pop):
        pool = set()
        for s in pop:
            if gates.g1_channel(s.task) == 'choice':
                pool.update(s.task.answer_choices)
            else:
                for group in s.task.answer_keywords:
                    pool.update(group)
        best, best_hits = (None, -1)
        for cand in sorted(pool):
            hits = sum((1 for s in pop if score.match_answer(cand, s.task.answer_keywords, s.task.answer_match)))
            if hits > best_hits:
                best, best_hits = (cand, hits)
        return (best, max(best_hits, 0))
    n = len(covered)
    best, _ = _pooled_best(covered)
    hit_ids = sorted((s.task.task_id for s in covered if score.match_answer(best, s.task.answer_keywords, s.task.answer_match)))
    rate = len(hit_ids) / n
    floor = gates.modal_constant_floor(n)
    before = [s for s in specs if gates.g1_channel(s.task) in _SETLEVEL_CHANNELS]
    n_before = len(before)
    best_before, hits_before = _pooled_best(before) if before else (None, 0)
    counts = collections.Counter((getattr(s, 'family', None) for s in covered))
    top_family, top_n = counts.most_common(1)[0]
    if rate <= gates.RANDOM_MAX:
        verdict = 'ok'
        why = f'全集最优常量命中 {rate:.2f} <= RANDOM_MAX {gates.RANDOM_MAX}' + (f'，但高于下限 {floor:.2f}（净空 +{rate - floor:.2f}）' if rate > floor else f'，且不高于下限 {floor:.2f}')
    elif n < gates.G1_MODAL_MIN_N:
        verdict = 'too_few'
        why = f'命中 {rate:.2f} > RANDOM_MAX {gates.RANDOM_MAX}，而可攻击的题只有 {n} < {gates.G1_MODAL_MIN_N}，下限 1/n = {floor:.2f} 本身就在阈值之上：这是集合太小，不是答案聚集，配平救不了'
    else:
        verdict = 'clustered'
        why = f'命中 {rate:.2f} > RANDOM_MAX {gates.RANDOM_MAX}，而下限只有 {floor:.2f}（净空 +{rate - floor:.2f}）：n 够大，一句话就能拿走全集这么多分'
    drop = verdict != 'ok' and bool(enforced)
    return {'n': n, 'n_alive': len(alive), 'best': best, 'hits': len(hit_ids), 'hit_ids': hit_ids, 'rate': round(rate, 3), 'floor': round(floor, 3), 'headroom': round(rate - floor, 3), 'share_of_set': round(len(hit_ids) / len(alive), 3) if alive else None, 'threshold': gates.RANDOM_MAX, 'min_n': gates.G1_MODAL_MIN_N, 'n_before': n_before, 'best_before': best_before, 'rate_before': round(hits_before / n_before, 3) if n_before else None, 'families': dict(sorted(counts.items(), key=lambda kv: str(kv[0]))), 'top_family': top_family, 'family_share': round(top_n / n, 3), 'verdict': verdict, 'enforced': bool(enforced), 'dropped': drop, 'veto_ids': hit_ids if drop else [], 'reason': why}

def fullset_constant_line(report):
    if not report:
        return ''
    r = report
    arrow = ''
    if r.get('rate_before') is not None and r['n_before'] != r['n']:
        arrow = f"，拒稿前是 {r['rate_before']:.2f} over {r['n_before']} (常量 {r['best_before']!r})"
    if r['verdict'] == 'ok':
        tail = '-> 过'
    elif r['dropped']:
        tail = f"-> 毙掉被它命中的 {r['hits']} 道"
    else:
        tail = '-> **测到了但没执行**（FULLSET_CONSTANT_ENFORCED 是 False），这一套照原样发货'
    return f"G1 任务质量检查（完整题集）: 可攻击 {r['n']}/{r['n_alive']} 道，最优常量 {r['best']!r} 命中 {r['rate']:.3f}（占全集 {r['share_of_set']}），下限 {r['floor']:.3f}{arrow}；最大族 {r['top_family']} 占 {r['family_share']:.2f}（{len(r['families'])} 族）；{r['verdict']}：{r['reason']} {tail}"

def validate_spec(sim, spec, priors, null_progress=None, prior_progress=None, random_progress=None, oracle_progress=None, oracle_steps=None, oracle_invalid=None, oracle_answer=None, min_pixel_frac=None):
    return gates.measure(sim, spec.task, kinds=spec.kinds, prior=(priors or {}).get('prior'), receptacle_kinds=spec.receptacle_kinds, on_prior=(priors or {}).get('on_prior'), count_prior=count_prior_of(priors, spec), direction_prior=(priors or {}).get('direction_prior'), choice_prior=(priors or {}).get('choice_prior'), word_prior=score.word_prior_of(priors, spec), evidence=spec.evidence, answer_instances=spec.answer_instances, null_progress=null_progress, prior_progress=prior_progress, random_progress=random_progress, oracle_progress=oracle_progress, oracle_steps=oracle_steps, oracle_invalid=oracle_invalid, oracle_answer=oracle_answer, min_pixel_frac=min_pixel_frac)

def named_aliases(text, inv):
    if not text:
        return ()
    low = text.lower()
    out = []
    for f in inv:
        a = str(f.alias)
        if a.lower() in low and re.search(f'\\b{re.escape(a.lower())}\\b', low):
            out.append(a)
    return tuple(sorted(set(out)))

def ambiguous_referents(spec, inv):
    declared = tuple(getattr(spec, 'referents', ()) or ())
    named = declared or named_aliases(getattr(spec.task, 'instruction', ''), inv)
    if not named:
        return ()
    by_alias = {str(f.alias): f for f in inv}
    by_kind = {}
    for f in inv:
        by_kind.setdefault(f.kind, []).append(f)
    looks = collections.Counter((f.look for f in inv if f.look))
    bad = []
    for a in named:
        f = by_alias.get(a)
        if f is None:
            continue
        peers = by_kind.get(f.kind, ())
        if len(peers) <= 1:
            continue
        if spec.family == 'room_localization' and len(_rooms_of(peers)) == 1:
            continue
        if declared and f.look and (looks[f.look] == 1):
            continue
        bad.append((a, f.kind, len(peers)))
    return tuple(bad)

def discriminable(spec, inv):
    return not ambiguous_referents(spec, inv)
OVERSIZE_RATIO = 0.5

def oversized_children(spec, inv):
    if spec.family != 'surface_contents':
        return ((), True)
    by_alias = {str(f.alias): f for f in inv}
    truth = getattr(spec, 'truth', None) or {}
    host = by_alias.get(str(truth.get('host') or ''))
    kids = [by_alias.get(str(k.get('alias'))) for k in truth.get('children') or ()]
    kids = [k for k in kids if k is not None]
    if host is None or not kids:
        return ((), False)
    if not host.footprint:
        return ((), False)
    offenders, measured = ([], True)
    for k in kids:
        if not k.footprint:
            measured = False
            continue
        ratio = k.footprint / host.footprint
        if ratio >= OVERSIZE_RATIO:
            offenders.append((str(k.alias), k.kind, ratio))
    return (tuple(offenders), measured)

def partition(specs, facts_by_id):
    accepted, rejected = ([], [])
    for s in specs:
        f = facts_by_id.get(s.task.task_id)
        if f is None:
            rejected.append((s, ['未测：没有 TaskFacts']))
            continue
        problems, _ = gates.problems_with_coverage(f)
        (accepted if not problems else rejected).append(s if not problems else (s, problems))
    return (accepted, rejected)
