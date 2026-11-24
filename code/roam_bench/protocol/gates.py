"""Gates utilities for ROAM-Bench."""
import math
import re
from dataclasses import dataclass, field
from typing import Optional
from .score import SECTOR_WORDS, count_prior_of, direction_word, match_answer, match_sequence, word_prior_of
_CAMEL = re.compile('[A-Z][a-z0-9]*|[a-z0-9]+')
NULL_MAX = 0.0
RANDOM_MAX = 0.25
ORACLE_MIN = 1.0

@dataclass(frozen=True)
class TaskFacts:
    task_id: str
    gt_requires_observation: bool
    solvable_from_initial_frame: bool
    answer_visible_at_spawn: bool
    judge_uses_llm: bool
    null_progress: Optional[float] = None
    prior_progress: Optional[float] = None
    random_progress: Optional[float] = None
    oracle_progress: Optional[float] = None
    oracle_steps: Optional[int] = None
    oracle_invalid: Optional[int] = None
    oracle_answer: Optional[str] = None
    prior_answer: Optional[str] = None
    evidence: tuple = ()
    answer_instances: tuple = ()
    visible_at_spawn: tuple = ()
    notes: tuple = field(default_factory=tuple)

    def to_dict(self):
        return {'task_id': self.task_id, 'gt_requires_observation': self.gt_requires_observation, 'solvable_from_initial_frame': self.solvable_from_initial_frame, 'answer_visible_at_spawn': self.answer_visible_at_spawn, 'judge_uses_llm': self.judge_uses_llm, 'null_progress': self.null_progress, 'prior_progress': self.prior_progress, 'random_progress': self.random_progress, 'oracle_progress': self.oracle_progress, 'oracle_steps': self.oracle_steps, 'oracle_invalid': self.oracle_invalid, 'oracle_answer': self.oracle_answer, 'prior_answer': self.prior_answer, 'evidence': list(self.evidence), 'answer_instances': list(self.answer_instances), 'visible_at_spawn': list(self.visible_at_spawn), 'notes': list(self.notes)}
ORACLE_PLAN_REFUSED = '计划全被拒'
ORACLE_NAV_PARTIAL = '导航部分失败'
ORACLE_NO_ANSWER = '读数器无答案'
ORACLE_ANSWER_MISMATCH = '答案不一致'

def oracle_cause(task):
    if getattr(task, 'oracle_progress', None) is not None and task.oracle_progress >= ORACLE_MIN:
        return (None, None)
    steps = getattr(task, 'oracle_steps', None)
    invalid = getattr(task, 'oracle_invalid', None)
    if steps is None or invalid is None:
        return (None, None)
    if steps and invalid >= steps:
        return (ORACLE_PLAN_REFUSED, f'计划里 {steps} 个动作全部被拒，先查计划和适配层，不是题')
    if invalid:
        return (ORACLE_NAV_PARTIAL, f'{steps} 个动作里 {invalid} 个被拒，先怀疑适配层（M14 §5.7），不是题')
    answer = getattr(task, 'oracle_answer', None)
    if answer in (None, ''):
        return (ORACLE_NO_ANSWER, f'走完 {steps} 步没有动作被拒，但上界没给出答案，查读数器')
    return (ORACLE_ANSWER_MISMATCH, f'走完 {steps} 步没有动作被拒、读出「{answer}」仍不满分，查真值与读数器是否一致')

def validate(task) -> list:
    problems = []
    if not task.gt_requires_observation:
        problems.append('G1: 常识可猜')
    if task.solvable_from_initial_frame:
        problems.append('G2: 单帧可解')
    if task.answer_visible_at_spawn:
        problems.append('G3: 起点即可见')
    if task.judge_uses_llm:
        problems.append('G4: 主观判分')
    prior_p = getattr(task, 'prior_progress', None)
    if prior_p is not None and prior_p > NULL_MAX:
        problems.append(f'G1-先验基线: 照常识猜也得 {prior_p:.2f} 分')
    null_p = getattr(task, 'null_progress', None)
    if null_p is not None and null_p > NULL_MAX:
        problems.append(f'G3-空基线: 什么都不做也得 {null_p:.2f} 分')
    rand_p = getattr(task, 'random_progress', None)
    if rand_p is not None and rand_p > RANDOM_MAX:
        problems.append(f'G3-随机基线: 随机走得 {rand_p:.2f} 分 (> {RANDOM_MAX})')
    oracle_p = getattr(task, 'oracle_progress', None)
    if oracle_p is not None:
        _tag, why = oracle_cause(task)
        if oracle_p <= 0.0:
            problems.append(f'S-上界基线: 完美感知没解出 — {why}' if why else 'S-上界基线: 完美感知也解不出，任务本身不可能')
        elif oracle_p < ORACLE_MIN:
            problems.append(f'S-上界基线: 完美感知只做到 {oracle_p:.2f} (< {ORACLE_MIN})，任务有做不到的部分' + (f' — {why}' if why else ''))
    return problems

def problems_with_coverage(facts) -> tuple:
    checked = ['G1', 'G2', 'G3', 'G4']
    for name, attr in (('G1-先验基线', 'prior_progress'), ('G3-空基线', 'null_progress'), ('G3-随机基线', 'random_progress'), ('S-上界基线', 'oracle_progress')):
        if getattr(facts, attr, None) is not None:
            checked.append(name)
    return (validate(facts), checked)

def prior_answer_for(kinds, prior, room_label_words=None):
    words = []
    for k in kinds:
        p = (prior or {}).get(k)
        if not p or p.get('tied_with'):
            continue
        room = p['room']
        words.append(room)
        words.append(room.replace('_', ' '))
        words.append(room.replace('_', ''))
    if room_label_words:
        words += list(room_label_words)
    return ' '.join(words)

def prior_contents_for(receptacle_kinds, on_prior, top_n=8, min_rate=0.05):
    words = []
    for rk in receptacle_kinds:
        row = (on_prior or {}).get(rk)
        if not row:
            continue
        for ch in row.get('children', [])[:top_n]:
            if ch.get('rate', 0.0) >= min_rate:
                words.append(ch['kind'])
    out = []
    for w in words:
        out.append(w)
        parts = _CAMEL.findall(w)
        if len(parts) > 1:
            out.append(' '.join(parts))
            out.append(parts[-1])
    return ' '.join(out)

def g1_channel(task):
    if task.answer_from_pose:
        return 'dynamic'
    if task.answer_match == 'sequence':
        return 'sequence'
    if task.answer_match == 'direction':
        return 'direction'
    if task.answer_choices:
        return 'choice'
    if task.answer_number is not None:
        return 'count'
    if not task.answer_keywords:
        return 'no_answer'
    return 'word'
G1_CHANNELS = ('dynamic', 'sequence', 'direction', 'choice', 'count', 'no_answer', 'word')

def _attack_missed(task, text, groups, mode, std=None, tol=None):
    verdict = match_answer(text, groups, mode, std=std, tol=tol)
    if verdict is None:
        raise ValueError(f"G1 的常量攻击无法判定，这是出题侧的字段自相矛盾，不是「常识可猜」：task_id={getattr(task, 'task_id', '?')!r} answer_match={mode!r} std={std!r} groups={groups!r}。count/direction 通道需要 answer_number（或 answer_from_pose 冻结的真值），word/sequence 通道需要非空 answer_keywords。match_answer 返回 None 时 `not None` 会变成一次「已测通过」，所以这里不折叠。")
    return not verdict

def measure_g1(task, kinds=(), prior=None, receptacle_kinds=(), on_prior=None, extra_words=None, count_prior=None, direction_prior=None, choice_prior=None, word_prior=None):
    ch = g1_channel(task)
    if ch == 'dynamic':
        return (True, None, 'G1 未测：本题真值在回答那一刻才产生，静态常数攻击无从算起（改由 measure_g1_dynamic 在跑完基线后按实测真值测）')
    if ch == 'sequence':
        return (True, None, 'G1 未测：本题问路线，端点写在题面里，真正的攻击是「填众数中间段」，那是集合级的（改由 measure_g1_topo_route 在建集时测）')
    if ch == 'direction':
        if direction_prior is None:
            return (True, None, 'G1 未测：本题问方向，但没有给 direction_prior')
        guess = direction_word(direction_prior)
        ok = _attack_missed(task, guess, (), 'direction', std=task.answer_number, tol=task.answer_tolerance)
        return (ok, guess, '')
    if ch == 'choice':
        if choice_prior is None:
            return (True, None, 'G1 未测：本题是几选一，但没有给 choice_prior')
        ok = _attack_missed(task, choice_prior, task.answer_keywords, task.answer_match)
        return (ok, choice_prior, '')
    if ch == 'count':
        if count_prior is None:
            return (True, None, 'G1 未测：本题问数量，但没有给 count_prior')
        guess = str(int(count_prior))
        ok = _attack_missed(task, guess, (), 'count', std=task.answer_number, tol=task.answer_tolerance)
        return (ok, guess, '')
    if ch == 'no_answer':
        return (True, None, 'G1 不适用：本题不问问题，无答案可猜')
    guess = ' '.join((x for x in (prior_answer_for(kinds, prior, extra_words), prior_contents_for(receptacle_kinds, on_prior), word_prior or '') if x)).strip()
    if not guess:
        return (True, None, 'G1 未测：两个先验对本题都没有意见（集合级常量也未提供）')
    ok = _attack_missed(task, guess, task.answer_keywords, task.answer_match)
    if ok and word_prior is None:
        return (ok, guess, 'G1 只测了跨房子语料先验：没有给 word_prior，本套题集自己的最强常量（集合级常量攻击）未跑')
    return (ok, guess, '')

def measure_g1_dynamic(truths, tol):
    vals = [t for t in truths if t is not None]
    if not vals:
        return (None, None)
    best_word, best_hits = (None, -1)
    for word in SECTOR_WORDS:
        hits = sum((1 for t in vals if match_answer(word, (), 'direction', std=t, tol=tol)))
        if hits > best_hits:
            best_word, best_hits = (word, hits)
    return (best_word, best_hits / len(vals))

def g1_dynamic_floor(n):
    if n <= 0:
        return None
    return -(-n // len(SECTOR_WORDS)) / n
G1_DYNAMIC_MIN_N = next((n for n in range(1, len(SECTOR_WORDS) + 1) if g1_dynamic_floor(n) <= RANDOM_MAX))

def modal_constant_floor(n):
    if not n:
        return None
    return 1.0 / n

def word_prior_floor(n):
    return modal_constant_floor(n)
G1_MODAL_MIN_N = math.ceil(1.0 / RANDOM_MAX)
G1_WORD_MIN_N = G1_MODAL_MIN_N

def g1_dynamic_reason(n, hit, best, full_n=None, full_hit=None):
    floor = g1_dynamic_floor(n)
    if floor is None:
        return '没有真值可攻击'
    if n >= G1_DYNAMIC_MIN_N:
        return '这套房子的路线拐得不够'
    head = f'n={n} < {G1_DYNAMIC_MIN_N}，命中率下界就是 {floor:.3f}，真值再分散也过不了'
    if full_hit is not None and full_n:
        if full_hit > RANDOM_MAX:
            return head + f'；但把跑批丢掉的路线也算回来（n={full_n}）实测 {full_hit:.3f} 仍然过不了，所以补供给没用，是这套房子的路线本身拐得不够'
        return head + f'；把跑批丢掉的路线算回来（n={full_n}）实测 {full_hit:.3f} 就能过，所以该查的是那几条路线为什么没走到这一步，不是户型'
    if hit is not None and hit > floor + 1e-09:
        return head + f'；另外这一组实测 {hit:.3f} 已是下界的 {hit / floor:.1f} 倍，{best!r} 这个方向上真值本身也挤在一起，两件事都成立'
    return head + '；这一组真值已是这个 n 能有的最分散，所以这里没有任何关于户型的结论'

def topo_route_middle(task):
    return tuple((g if isinstance(g, str) else g[0] for g in task.answer_keywords[1:-1]))

def topo_route_attack_hits(middle, task):
    ends = task.answer_keywords
    if len(ends) < 2:
        return False
    first = ends[0] if isinstance(ends[0], str) else ends[0][0]
    last = ends[-1] if isinstance(ends[-1], str) else ends[-1][0]
    said = ', '.join([first] + list(middle) + [last])
    return bool(match_sequence(said, task.answer_keywords, task.answer_vocabulary))

def measure_g1_topo_route(tasks):
    tasks = [t for t in tasks if len(t.answer_keywords) >= 2]
    if not tasks:
        return (None, None)
    middles = {topo_route_middle(t) for t in tasks}
    middles.add(())
    best_mid, best_hits = (None, -1)
    for mid in sorted(middles):
        hits = sum((1 for t in tasks if topo_route_attack_hits(mid, t)))
        if hits > best_hits:
            best_mid, best_hits = (mid, hits)
    return (best_mid, best_hits / len(tasks))

def free_problems(spec, priors=None):
    task = spec.task
    priors = priors or {}
    problems = []
    g1, _guess, _note = measure_g1(task, getattr(spec, 'kinds', ()) or (), priors.get('prior'), getattr(spec, 'receptacle_kinds', ()) or (), priors.get('on_prior'), None, count_prior_of(priors, spec), priors.get('direction_prior'), priors.get('choice_prior'), word_prior=word_prior_of(priors, spec))
    if not g1:
        problems.append('G1: 常识可猜')
    from . import baselines
    rand_p = baselines.random_answer_progress(task)
    if rand_p is not None and rand_p > RANDOM_MAX:
        problems.append(f'G3-随机基线: 随机走得 {rand_p:.2f} 分 (> {RANDOM_MAX})')
    return problems

def measure(sim, task, kinds=(), prior=None, evidence=(), answer_instances=(), receptacle_kinds=(), on_prior=None, null_progress=None, prior_progress=None, random_progress=None, oracle_progress=None, oracle_steps=None, oracle_invalid=None, oracle_answer=None, extra_prior_words=None, min_pixel_frac=None, count_prior=None, direction_prior=None, choice_prior=None, word_prior=None):
    visible = set(sim.visible_instances(min_pixel_frac))
    evidence, answer_instances = (tuple(evidence), tuple(answer_instances))
    notes = []
    g1, guess, g1_note = measure_g1(task, kinds, prior, receptacle_kinds, on_prior, extra_prior_words, count_prior, direction_prior, choice_prior, word_prior)
    if g1_note:
        notes.append(g1_note)
    if evidence:
        g2 = all((a in visible for a in evidence))
    else:
        g2 = False
        notes.append('G2 未测：任务未声明 evidence')
    g3 = bool(set(answer_instances) & visible)
    if not answer_instances:
        notes.append('G3 仅结构测量：任务未声明 answer_instances，只能靠基线判定')
    if null_progress is None or random_progress is None:
        notes.append('G3 仅结构测量：基线未跑')
    return TaskFacts(task_id=task.task_id, gt_requires_observation=g1, solvable_from_initial_frame=g2, answer_visible_at_spawn=g3, judge_uses_llm=False, null_progress=null_progress, prior_progress=prior_progress, random_progress=random_progress, oracle_progress=oracle_progress, oracle_steps=oracle_steps, oracle_invalid=oracle_invalid, oracle_answer=oracle_answer, prior_answer=guess, evidence=evidence, answer_instances=answer_instances, visible_at_spawn=tuple(sorted(visible)), notes=tuple(notes))

def report(facts_list) -> dict:
    rows, per_gate = ([], {})
    for f in facts_list:
        problems, checked = problems_with_coverage(f)
        rows.append({'task_id': f.task_id, 'ok': not problems, 'problems': problems, 'checked': checked, 'facts': f.to_dict()})
        for p in problems:
            per_gate[p.split(':')[0]] = per_gate.get(p.split(':')[0], 0) + 1
    return {'n': len(rows), 'n_ok': sum((1 for r in rows if r['ok'])), 'rejected_by_gate': per_gate, 'tasks': rows, 'thresholds': {'NULL_MAX': NULL_MAX, 'RANDOM_MAX': RANDOM_MAX, 'ORACLE_MIN': ORACLE_MIN}}
