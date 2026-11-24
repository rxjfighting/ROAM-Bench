"""Score discrete, numeric, pose-dependent, and scene-state outcomes."""
import math
import re
from dataclasses import dataclass, field
from typing import Optional
SCORER_VERSION = 'roam-outcome-v1'

@dataclass(frozen=True)
class Goal:
    expr: str
    label: str = ''
    weight: float = 1.0
    when: str = 'final'

    def __post_init__(self):
        if self.when not in ('final', 'ever'):
            raise ValueError(f"Goal.when must be 'final' or 'ever', got {self.when!r}")

@dataclass(frozen=True)
class Mutation:
    op: str
    alias: str
    arg: object
    room: str
    after_rooms: tuple = ()

    def __post_init__(self):
        if self.op not in ('open', 'move'):
            raise ValueError(f"Mutation.op must be 'open' or 'move', got {self.op!r}")
        if self.op == 'move' and (not isinstance(self.arg, (tuple, list)) or len(self.arg) != 3):
            raise ValueError(f"Mutation 'move' needs an (x, y, z), got {self.arg!r}")
        if self.op == 'open' and (not isinstance(self.arg, bool)):
            raise ValueError(f"Mutation 'open' needs a bool, got {self.arg!r}")

    def to_dict(self):
        return {'op': self.op, 'alias': self.alias, 'arg': list(self.arg) if self.op == 'move' else self.arg, 'room': self.room, 'after_rooms': list(self.after_rooms)}

    @staticmethod
    def from_dict(d):
        if not d:
            return None
        return Mutation(op=d['op'], alias=d['alias'], arg=tuple(d['arg']) if d['op'] == 'move' else bool(d['arg']), room=d['room'], after_rooms=tuple(d.get('after_rooms', ())))

@dataclass(frozen=True)
class Task:
    task_id: str
    instruction: str
    goals: tuple = ()
    answer_keywords: tuple = ()
    answer_match: str = 'word'
    answer_number: Optional[float] = None
    answer_tolerance: Optional[float] = None
    max_steps: int = 60
    target_sets: dict = field(default_factory=dict)
    goal_images: tuple = ()
    answer_choices: tuple = ()
    answer_choice_mode: str = 'single'
    answer_from_pose: tuple = ()
    mutation: Optional['Mutation'] = None
    answer_vocabulary: tuple = ()
    answer_unit: str = ''
    answer_choice_labels: tuple = ()

@dataclass(frozen=True)
class Click:
    u: int
    v: int
    targets: tuple
    landed_on: Optional[str]
    strict_hit: bool
    dilated_hit: bool
    dilation: int

@dataclass(frozen=True)
class Referring:
    clicks: int = 0
    strict_hits: int = 0
    dilated_hits: int = 0
    first_click_hit: Optional[bool] = None
    recovered: int = 0
    missed_first: int = 0

    @property
    def strict_rate(self):
        return self.strict_hits / self.clicks if self.clicks else None

    @property
    def dilated_rate(self):
        return self.dilated_hits / self.clicks if self.clicks else None

    @property
    def error_rate(self):
        return 1.0 - self.strict_rate if self.clicks else None

    @property
    def edge_gap(self):
        if not self.clicks:
            return None
        return (self.dilated_hits - self.strict_hits) / self.clicks

    @property
    def self_correction_rate(self):
        return self.recovered / self.missed_first if self.missed_first else None

@dataclass(frozen=True)
class Score:
    task_id: str
    success: bool
    progress: float
    goals: tuple = ()
    answer_ok: Optional[bool] = None
    answer_given: Optional[str] = None
    steps: int = 0
    invalid_actions: int = 0
    collisions: int = 0
    referring: Optional[Referring] = None
    mutation_done: Optional[bool] = None

    def to_dict(self):
        d = {'task_id': self.task_id, 'success': self.success, 'progress': round(self.progress, 4), 'goals': [{'label': g[0], 'expr': g[1], 'when': g[2], 'ok': g[3]} for g in self.goals], 'answer_ok': self.answer_ok, 'answer_given': self.answer_given, 'steps': self.steps, 'invalid_actions': self.invalid_actions, 'collisions': self.collisions}
        if self.mutation_done is not None:
            d['mutation_done'] = self.mutation_done
        if self.referring:
            r = self.referring
            d['referring'] = {'clicks': r.clicks, 'strict_hits': r.strict_hits, 'dilated_hits': r.dilated_hits, 'first_click_hit': r.first_click_hit, 'strict_rate': r.strict_rate, 'dilated_rate': r.dilated_rate, 'edge_gap': r.edge_gap, 'self_correction_rate': r.self_correction_rate, 'missed_first': r.missed_first, 'recovered': r.recovered}
        return d
_PUNCT = re.compile('[^\\w\\s]+|_+', re.UNICODE)
_SPACE = re.compile('\\s+')

def normalise_answer(text):
    if not text:
        return ''
    return _SPACE.sub(' ', _PUNCT.sub(' ', str(text).lower())).strip()
_INFLECTION = '(?:e?s|ed|ing)?'
_HAS_LATIN = re.compile('[a-z0-9]')
_CARDINAL = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90}
_SCALE = {'hundred': 100, 'thousand': 1000}
_GLUED_UNIT = re.compile('(\\d)([a-z])')
_DECIMAL_POINT = re.compile('(?<=[0-9])\\.(?=[0-9])')
_DECIMAL_MARKER = 'decimalpointmarker'
_ARABIC_NUMBER = re.compile('[0-9]+(?:\\.[0-9]+)?')
_UNIT_FOLLOWERS = frozenset(('metres', 'meters', 'metre', 'meter', 'm', 'centimetres', 'centimeters', 'centimetre', 'centimeter', 'cm', 'millimetres', 'millimeters', 'millimetre', 'millimeter', 'mm', 'km', 'm2', 'square', 'sq'))

def _read_number_at(words, i):
    if i >= len(words):
        return (None, i)
    w = words[i]
    if _ARABIC_NUMBER.fullmatch(w):
        return (float(w) if '.' in w else int(w), i + 1)
    if w not in _CARDINAL and w not in _SCALE:
        return (None, i)
    total = current = 0
    j = i
    while j < len(words):
        w = words[j]
        if w in _CARDINAL:
            current += _CARDINAL[w]
        elif w in _SCALE:
            current = (current or 1) * _SCALE[w]
            total += current
            current = 0
        else:
            break
        j += 1
    return (total + current, j)

def _iter_numbers(hay):
    words = _GLUED_UNIT.sub('\\1 \\2', hay).split()
    i = 0
    while i < len(words):
        val, nxt = _read_number_at(words, i)
        if val is None:
            i += 1
            continue
        yield (val, nxt < len(words) and words[nxt] in _UNIT_FOLLOWERS)
        i = nxt
_LENGTH_UNITS = {'m': ('length', 1.0), 'meter': ('length', 1.0), 'meters': ('length', 1.0), 'metre': ('length', 1.0), 'metres': ('length', 1.0), 'cm': ('length', 0.01), 'centimeter': ('length', 0.01), 'centimeters': ('length', 0.01), 'centimetre': ('length', 0.01), 'centimetres': ('length', 0.01), 'mm': ('length', 0.001), 'millimeter': ('length', 0.001), 'millimeters': ('length', 0.001), 'millimetre': ('length', 0.001), 'millimetres': ('length', 0.001), 'km': ('length', 1000.0)}
_AREA_UNITS = {'m2': ('area', 1.0), 'm²': ('area', 1.0), 'cm2': ('area', 0.0001), 'cm²': ('area', 0.0001)}
_EXPECTED_UNIT = {'m': ('length', 1.0), 'cm': ('length', 0.01), 'm2': ('area', 1.0)}

def _quantity_unit(words, at):
    if at >= len(words):
        return None
    word = words[at]
    if word in _LENGTH_UNITS:
        return _LENGTH_UNITS[word]
    if word in _AREA_UNITS:
        return _AREA_UNITS[word]
    if word in ('square', 'sq') and at + 1 < len(words):
        base = _LENGTH_UNITS.get(words[at + 1])
        if base:
            return ('area', base[1] ** 2)
    return None

def answer_quantity(text, expected_unit):
    if expected_unit not in _EXPECTED_UNIT and expected_unit != 'count':
        return answer_number(text)
    if not text:
        return None
    protected = _DECIMAL_POINT.sub(_DECIMAL_MARKER, str(text).lower())
    hay = normalise_answer(protected).replace(_DECIMAL_MARKER, '.')
    words = _GLUED_UNIT.sub('\\1 \\2', hay).split()
    expected_dim, expected_factor = _EXPECTED_UNIT.get(expected_unit, ('count', 1.0))
    first_bare = None
    saw_incompatible = False
    i = 0
    while i < len(words):
        value, nxt = _read_number_at(words, i)
        if value is None:
            i += 1
            continue
        parsed_unit = _quantity_unit(words, nxt)
        if parsed_unit:
            dim, factor = parsed_unit
            if dim == expected_dim:
                return float(value) * factor / expected_factor
            saw_incompatible = True
        elif first_bare is None:
            first_bare = value
        i = nxt
    return None if saw_incompatible else first_bare

def answer_number(text):
    if not text:
        return None
    protected = _DECIMAL_POINT.sub(_DECIMAL_MARKER, str(text).lower())
    hay = normalise_answer(protected).replace(_DECIMAL_MARKER, '.')
    first = None
    for val, unit_follows in _iter_numbers(hay):
        if unit_follows:
            return val
        if first is None:
            first = val
    return first
DIRECTION_WORDS = {'ahead': 0, 'forward': 0, 'in front': 0, 'straight ahead': 0, 'twelve o clock': 0, 'ahead right': 45, 'forward right': 45, 'front right': 45, 'right ahead': 45, 'right': 90, 'to the right': 90, 'starboard': 90, 'behind right': 135, 'back right': 135, 'right behind': 135, 'behind': 180, 'back': 180, 'backwards': 180, 'to the rear': 180, 'behind left': 225, 'back left': 225, 'left behind': 225, 'left': 270, 'to the left': 270, 'port': 270, 'ahead left': 315, 'forward left': 315, 'front left': 315, 'left ahead': 315}
_DIRECTION_KEYS = tuple(sorted(DIRECTION_WORDS, key=len, reverse=True))
_CLOCK = re.compile('\\b(\\d{1,2})\\s*(?:o\\s*clock|oclock)\\b')
_DEGREES = re.compile('\\b(\\d{1,3})\\s*(?:degrees?|deg)\\b')

def direction_bearing(text):
    if not text:
        return None
    hay = normalise_answer(text)
    if not hay:
        return None
    m = _DEGREES.search(hay)
    if m:
        mag = int(m.group(1)) % 360
        if re.search('(?<!\\w)(left|port|anticlockwise|counterclockwise)(?!\\w)', hay):
            return -mag % 360
        if re.search('(?<!\\w)(right|starboard|clockwise)(?!\\w)', hay):
            return mag
        return None
    m = _CLOCK.search(hay)
    if m:
        return int(m.group(1)) % 12 * 30
    for key in _DIRECTION_KEYS:
        if re.search('(?<!\\w)' + re.escape(key) + '(?!\\w)', hay):
            return DIRECTION_WORDS[key]
    return None
SECTOR_WORDS = ('ahead', 'ahead right', 'right', 'behind right', 'behind', 'behind left', 'left', 'ahead left')

def direction_word(deg):
    return min(SECTOR_WORDS, key=lambda w: bearing_delta(DIRECTION_WORDS[w], deg))

def bearing_delta(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return 360.0 - d if d > 180.0 else d
MIN_PATH_DISPLACEMENT_M = 0.75

def bearing_to_point(pose, x, z):
    dx, dz = (float(x) - float(pose.x), float(z) - float(pose.z))
    if math.hypot(dx, dz) < MIN_PATH_DISPLACEMENT_M:
        return None
    return (math.degrees(math.atan2(dx, dz)) - float(pose.yaw_deg)) % 360.0
COUNT_UNITS = {'object_distance': 'm', 'gap_width': 'cm', 'room_area': 'm2'}

def count_unit(spec):
    return COUNT_UNITS.get(getattr(spec, 'family', None), 'count')

def count_prior_of(priors, spec):
    cp = (priors or {}).get('count_prior')
    return cp.get(count_unit(spec)) if isinstance(cp, dict) else cp

def word_prior_of(priors, spec):
    wp = (priors or {}).get('word_prior')
    return wp.get(getattr(spec, 'family', None)) if isinstance(wp, dict) else wp

def _first_positions(hay, group):
    best = None
    for kw in (group,) if isinstance(group, str) else tuple(group):
        k = normalise_answer(kw)
        if not k:
            continue
        if not _HAS_LATIN.search(k):
            i = hay.find(k)
        else:
            m = re.search('(?<!\\w)' + re.escape(k) + _INFLECTION + '(?!\\w)', hay)
            i = m.start() if m else -1
        if i >= 0 and (best is None or i < best):
            best = i
    return best

def match_sequence(text, groups, vocabulary=()):
    hay = normalise_answer(text)
    if not hay or not groups:
        return None if not groups else False
    pos = [_first_positions(hay, g) for g in groups]
    if any((p is None for p in pos)):
        return False
    if pos != sorted(pos):
        return False
    wanted = {normalise_answer(w) for g in groups for w in ((g,) if isinstance(g, str) else g)}
    for group in groups:
        spans = []
        for kw in (group,) if isinstance(group, str) else tuple(group):
            k = normalise_answer(kw)
            if not k:
                continue
            if not _HAS_LATIN.search(k):
                spans.extend(((m.start(), m.end()) for m in re.finditer(re.escape(k), hay)))
            else:
                spans.extend(((m.start(), m.end()) for m in re.finditer('(?<!\\w)' + re.escape(k) + _INFLECTION + '(?!\\w)', hay)))
        merged = []
        for start, end in sorted(spans):
            if merged and start < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                merged.append((start, end))
        if len(merged) != 1:
            return False
    for w in vocabulary:
        if normalise_answer(w) in wanted:
            continue
        if _first_positions(hay, (w,)) is not None:
            return False
    return True

def match_answer(text, groups, mode='word', std=None, tol=None, vocabulary=(), choices=(), unit=''):
    if mode == 'sequence':
        return match_sequence(text, groups, vocabulary)
    if mode == 'direction':
        if std is None:
            return None
        got = direction_bearing(text)
        if got is None:
            return False
        return bearing_delta(got, std) <= (float(tol) if tol is not None else 0.0)
    if mode == 'count':
        if std is None:
            return None
        got = answer_quantity(text, unit) if unit else answer_number(text)
        if got is None:
            return False
        return std - tol <= got <= std + tol if tol is not None else got == std
    if not groups:
        return None
    hay = normalise_answer(text)
    if not hay:
        return False
    for group in groups:
        alts = (group,) if isinstance(group, str) else tuple(group)
        hit = False
        for kw in alts:
            k = normalise_answer(kw)
            if not k:
                continue
            if mode == 'substring' or not _HAS_LATIN.search(k):
                hit = k in hay
            else:
                hit = re.search('(?<!\\w)' + re.escape(k) + _INFLECTION + '(?!\\w)', hay) is not None
            if hit:
                break
        if not hit:
            return False
    if choices and mode != 'set':
        hits = sum((1 for choice in choices if _first_positions(hay, (choice,)) is not None))
        if hits != 1:
            return False
    if mode == 'set':
        wanted = {normalise_answer(w) for g in groups for w in ((g,) if isinstance(g, str) else tuple(g))}
        for word in vocabulary:
            if normalise_answer(word) not in wanted and _first_positions(hay, (word,)) is not None:
                return False
    return True

def judge_click(sim, u, v, targets, dilation=2):
    targets = tuple(targets)
    exact = sim.instance_at_pixel(u, v)
    near = sim.instances_in_patch(u, v, dilation) if dilation else {exact} if exact else set()
    tset = set(targets)
    return Click(u=int(u), v=int(v), targets=targets, landed_on=exact, strict_hit=exact in tset and exact is not None, dilated_hit=bool(tset & set(near)), dilation=int(dilation))

class Recorder:

    def __init__(self, sim, task, dilation=2, on_step=None):
        self.sim = sim
        self.task = task
        self.dilation = dilation
        self.on_step = on_step
        self.steps = []
        self.clicks = []
        self.answer_given = None
        self.collisions = 0
        self._ever = set()
        self.start_xz = None
        if task.answer_from_pose:
            p = sim.agent_pose()
            self.start_xz = (p.x, p.z)
        self.resolved_truth_deg = None
        self.resolved_pose = None
        self._visited = set()
        self.mutation_done = None
        self.mutation_failed = False
        self.unseen = ()
        if task.mutation is not None:
            self.mutation_done = False
            here = self._room_now()
            if here is not None:
                self._visited.add(here)
        self.poll()

    def poll(self):
        for g in self.task.goals:
            if g.when == 'ever' and g.expr not in self._ever:
                if self._safe(g.expr):
                    self._ever.add(g.expr)

    def step(self, kind, arg, result):
        self.steps.append((kind, _label(arg), bool(result.success), getattr(result, 'reason', '')))
        if getattr(result, 'collided', False):
            self.collisions += 1
        self.poll()
        self._maybe_mutate()
        if self.on_step is not None:
            try:
                self.on_step(kind, arg, result)
            except Exception:
                pass
        return result

    def _room_now(self):
        get = getattr(self.sim, 'agent_room_id', None)
        rid = get() if callable(get) else None
        return rid or self.sim.agent_room()

    def _maybe_mutate(self):
        m = self.task.mutation
        if m is None or self.mutation_done:
            return
        here = self._room_now()
        if here is not None:
            self._visited.add(here)
        if not set(m.after_rooms) <= self._visited or here == m.room:
            return
        ok = False
        try:
            ok = bool(self.sim.mutate(m.op, m.alias, m.arg))
        except Exception:
            ok = False
        self.mutation_done = ok
        self.mutation_failed = not ok

    def click(self, u, v, target_key):
        targets = self.task.target_sets.get(target_key, ())
        c = judge_click(self.sim, u, v, targets, self.dilation)
        self.clicks.append((target_key, c))
        return c

    def answer(self, text):
        self.answer_given = text
        if self.task.answer_from_pose and self.start_xz is not None:
            self.resolved_pose = self.sim.agent_pose()
            self.resolved_truth_deg = bearing_to_point(self.resolved_pose, self.start_xz[0], self.start_xz[1])
        return text

    def score(self):
        rows, got, total = ([], 0.0, 0.0)
        for g in self.task.goals:
            ok = g.expr in self._ever if g.when == 'ever' else self._safe(g.expr)
            rows.append((g.label or g.expr, g.expr, g.when, bool(ok)))
            total += g.weight
            got += g.weight if ok else 0.0
        asked = bool(self.task.answer_keywords) or self.task.answer_number is not None or bool(self.task.answer_from_pose)
        if self.task.mutation is not None and (not self.mutation_done):
            return self._score_with(rows, got, total, False)
        std = self.task.answer_number
        if self.task.answer_from_pose:
            std = self.resolved_truth_deg
            if std is None:
                return self._score_with(rows, got, total, False)
        answer_ok = match_answer(self.answer_given, self.task.answer_keywords, self.task.answer_match, std=std, tol=self.task.answer_tolerance, vocabulary=self.task.answer_vocabulary, choices=self.task.answer_choices, unit=self.task.answer_unit or ('count' if self.task.answer_match == 'count' else '')) if asked else None
        if asked and answer_ok is None:
            answer_ok = False
        return self._score_with(rows, got, total, answer_ok)

    def _score_with(self, rows, got, total, answer_ok):
        goals_ok = all((r[3] for r in rows)) if rows else True
        success = goals_ok and answer_ok is not False
        if not rows and answer_ok is None:
            success = False
        progress = got / total if total else 1.0 if answer_ok else 0.0
        return Score(task_id=self.task.task_id, success=bool(success), progress=progress, goals=tuple(rows), answer_ok=answer_ok, answer_given=self.answer_given, steps=len(self.steps), invalid_actions=sum((1 for s in self.steps if not s[2])), collisions=self.collisions, mutation_done=self.mutation_done, referring=self._referring())

    def _referring(self):
        if not self.clicks:
            return None
        first, missed_first, recovered = ({}, set(), set())
        for key, c in self.clicks:
            if key not in first:
                first[key] = c.strict_hit
                if not c.strict_hit:
                    missed_first.add(key)
            elif key in missed_first and c.strict_hit:
                recovered.add(key)
        firsts = [first[k] for k in first]
        return Referring(clicks=len(self.clicks), strict_hits=sum((1 for _, c in self.clicks if c.strict_hit)), dilated_hits=sum((1 for _, c in self.clicks if c.dilated_hit)), first_click_hit=all(firsts) if firsts else None, missed_first=len(missed_first), recovered=len(recovered))

    def _safe(self, expr):
        try:
            return bool(self.sim.predicate(expr))
        except Exception:
            return False

def _label(arg):
    if hasattr(arg, 'name'):
        return str(arg.name)
    if hasattr(arg, 'u'):
        return f'({arg.u},{arg.v})'
    if hasattr(arg, 'deg'):
        return f'{arg.deg:+.0f}deg/{arg.dist_m:.2f}m'
    return '' if arg is None else str(arg)

def aggregate(scores):
    scores = [s for s in scores if s is not None]
    n = len(scores)
    if not n:
        return {'n': 0}
    answered = [s for s in scores if s.answer_ok is not None]
    clicked = [s for s in scores if s.referring and s.referring.clicks]
    missed = [s for s in clicked if s.referring.missed_first]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None
    out = {'n': n, 'success_rate': sum((1 for s in scores if s.success)) / n, 'mean_progress': sum((s.progress for s in scores)) / n, 'mean_steps': sum((s.steps for s in scores)) / n, 'invalid_action_rate': sum((s.invalid_actions for s in scores)) / max(1, sum((s.steps for s in scores))), 'answer_accuracy': sum((1 for s in answered if s.answer_ok)) / len(answered) if answered else None, 'first_click_hit_rate': sum((1 for s in clicked if s.referring.first_click_hit)) / len(clicked) if clicked else None, 'referring_error_rate': mean([s.referring.error_rate for s in clicked]), 'edge_gap': mean([s.referring.edge_gap for s in clicked]), 'self_correction_rate': sum((s.referring.recovered for s in missed)) / sum((s.referring.missed_first for s in missed)) if missed else None}
    return out
