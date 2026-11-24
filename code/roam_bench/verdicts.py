"""Verdicts utilities for ROAM-Bench."""
from __future__ import annotations
import contextlib
import json
import os
import time
from pathlib import Path
from typing import Callable, Optional
SCHEMA_VERSION = 3
_COMPATIBLE_SCHEMAS = frozenset({1, 2, 3})
SOLVE_THRESHOLD = 0.5

def semantics_probe() -> dict:
    import inspect
    from roam_bench.protocol import score
    Mut = getattr(score, 'Mutation', None)
    try:
        src = inspect.getsource(score)
    except OSError:
        src = ''
    return {'vocabulary_param': 'vocabulary' in inspect.signature(score.match_answer).parameters, 'mutation_open_needs_bool': Mut is not None and 'needs a bool' in inspect.getsource(Mut), 'mc_five_way_floor_0p40': '0.40' in src and 'five-way' in src}

def resolve_scene_reference(scene: str, *, scenesmith_root=None, data_root=None) -> str:
    raw = str(scene)
    if '://' not in raw:
        return raw
    from .paths import DATA_ROOT, SCENESMITH_ROOT
    roots = {'scenesmith': Path(scenesmith_root or SCENESMITH_ROOT), 'data': Path(data_root or DATA_ROOT)}
    scheme, relative = raw.split('://', 1)
    if scheme not in roots:
        raise ValueError(f'unsupported frozen scene URI scheme {scheme!r}: {raw!r}')
    rel = Path(relative)
    if not relative or rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'invalid frozen scene URI path: {raw!r}')
    return str((roots[scheme] / rel).resolve())

def freeze(specs, *, scene: str, checkout: str, note: str='', **metadata) -> dict:
    reserved = {'schema_version', 'scene', 'code_checkout', 'semantics', 'note', 'specs'}
    overlap = reserved & set(metadata)
    if overlap:
        raise ValueError(f'frozen metadata cannot replace reserved keys: {sorted(overlap)}')
    return {'schema_version': SCHEMA_VERSION, 'scene': scene, 'code_checkout': checkout, 'semantics': semantics_probe(), 'note': note, **metadata, 'specs': [s.to_dict() for s in specs]}

def write_frozen(path: str, specs, *, scene: str, checkout: str, note: str='', **metadata) -> dict:
    doc = freeze(specs, scene=scene, checkout=checkout, note=note, **metadata)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
    return doc

def load_frozen(path: str):
    from roam_bench.protocol import taskset
    with open(path) as f:
        doc = json.load(f)
    if doc.get('schema_version') not in _COMPATIBLE_SCHEMAS:
        raise ValueError(f"frozen taskset {path} is schema v{doc.get('schema_version')}, not in this code's compatible set {sorted(_COMPATIBLE_SCHEMAS)}; regenerate or migrate rather than compare across shapes")
    specs = [taskset.spec_from_dict(d) for d in doc['specs']]
    header = {k: value for k, value in doc.items() if k not in ('schema_version', 'specs')}
    header['scene'] = resolve_scene_reference(header['scene'])
    return (header, specs)

def _pose_dict(sim) -> Optional[dict]:
    try:
        p = sim.agent_pose()
    except Exception:
        return None
    if p is None:
        return None
    return {'x': round(float(p.x), 2), 'y': round(float(p.y), 2), 'z': round(float(p.z), 2), 'yaw_deg': round(float(p.yaw_deg), 1), 'horizon_deg': round(float(p.horizon_deg), 1)}

def _classify(exc: BaseException) -> str:
    return 'not_implemented' if isinstance(exc, (NotImplementedError, AttributeError)) else 'error'

def _reason(rec: dict) -> str:
    if rec.get('oracle_error'):
        if rec.get('oracle_failure_kind') == 'not_implemented':
            return f"backend has not implemented a method this task needs ({rec['oracle_error']}) -- unbuilt code, not an unsolvable task"
        return f"oracle raised {rec['oracle_error']}"
    os_ = rec.get('oracle_success')
    if os_ is None:
        return 'oracle did not run'
    if os_ >= SOLVE_THRESHOLD:
        return f'oracle solved it (success={os_:.2f})'
    return f'oracle ran but scored {os_:.2f} < {SOLVE_THRESHOLD:.2f}: the plan could not be carried out on this backend, or the object is absent here'

@contextlib.contextmanager
def _capture_tour(acc: dict, *, optimize: bool=False, repeat_observations: bool=False):
    from roam_bench.protocol import baselines
    orig = getattr(baselines, '_visit', None)
    if orig is None:
        yield acc
        return

    def _visit_logged(sim, rec, stop, want, *a, **k):
        original_want = set(map(str, want))
        acc['want'].update(original_want)
        evidence_start = None
        evidence_query = getattr(sim, 'exploration_evidence', None)
        if optimize and callable(evidence_query):
            try:
                evidence_start = len(evidence_query().get('observations') or ())
            except (AttributeError, TypeError):
                evidence_start = None
        shortcut_seen = acc['seen']
        covered_key = 'covered_target_rooms'
        if repeat_observations:
            lap_stops = acc.setdefault('lap_stops', set())
            if str(stop) in lap_stops:
                lap_stops.clear()
                acc.setdefault('lap_seen', set()).clear()
                acc.setdefault('lap_covered_target_rooms', set()).clear()
            lap_stops.add(str(stop))
            shortcut_seen = acc.setdefault('lap_seen', set())
            covered_key = 'lap_covered_target_rooms'
        open_first = bool(k.get('open_first', a[0] if a else False))
        if optimize:
            evidence_frac = float(getattr(getattr(sim, '_episode', None), 'view_min_pixel_fraction', 0.002))
            if 'min_pixel_frac' in k:
                k['min_pixel_frac'] = max(float(k['min_pixel_frac']), evidence_frac)
            elif len(a) >= 2:
                a = list(a)
                a[1] = max(float(a[1]), evidence_frac)
                a = tuple(a)
            else:
                k['min_pixel_frac'] = evidence_frac
        effective_want = want
        stop_room = None
        if optimize and original_want and (not open_first):
            remaining = original_want - shortcut_seen
            inventory = tuple(sim.inventory())
            by_alias = {str(f.alias): f for f in inventory}
            stop_fact = by_alias.get(str(stop))
            stop_room = None if stop_fact is None else stop_fact.room_id or stop_fact.room
            local_remaining = {alias for alias in remaining if alias in by_alias and (by_alias[alias].room_id or by_alias[alias].room) == stop_room}
            if local_remaining:
                effective_want = local_remaining
            elif str(stop) in original_want:
                covered = acc.setdefault(covered_key, set())
                if stop_room and stop_room in covered:
                    return set()
                effective_want = remaining
            else:
                effective_want = set()
        original_nav = getattr(sim, 'navigate_to_object', None)
        nav_attempts = []

        def tracked_nav(*args, **kwargs):
            result = original_nav(*args, **kwargs)
            nav_attempts.append(result)
            return result
        installed = callable(original_nav)
        had_instance_attr = hasattr(sim, '__dict__') and 'navigate_to_object' in sim.__dict__
        old_instance_attr = sim.__dict__.get('navigate_to_object') if had_instance_attr else None
        if installed:
            try:
                setattr(sim, 'navigate_to_object', tracked_nav)
            except (AttributeError, TypeError):
                installed = False
        acc['tour_nav_depth'] = int(acc.get('tour_nav_depth', 0)) + 1

        def current_visit_seen():
            if evidence_start is None:
                return set()
            observations = evidence_query().get('observations') or ()
            seen = set()
            for observation in observations[evidence_start:]:
                seen.update(map(str, observation.get('visible_aliases') or ()))
            return seen
        try:
            from .tour_optimizer import visit_evidence
            with visit_evidence(current_visit_seen if evidence_start is not None else None):
                r = orig(sim, rec, stop, effective_want, *a, **k)
        finally:
            acc['tour_nav_depth'] -= 1
            if installed:
                if had_instance_attr:
                    setattr(sim, 'navigate_to_object', old_instance_attr)
                else:
                    delattr(sim, 'navigate_to_object')
        r = set(map(str, r))
        if evidence_start is not None:
            try:
                observations = evidence_query().get('observations') or ()
                for observation in observations[evidence_start:]:
                    r.update(set(map(str, observation.get('visible_aliases') or ())) & original_want)
            except (AttributeError, TypeError):
                pass
        acc['stops'] += 1
        if nav_attempts:
            if any((getattr(result, 'success', False) for result in nav_attempts)):
                acc['reached'] = int(acc.get('reached', 0)) + 1
                if optimize and stop_room and (str(stop) in original_want):
                    acc.setdefault(covered_key, set()).add(stop_room)
            else:
                acc['nav_failed'] = int(acc.get('nav_failed', 0)) + 1
        acc['seen'].update(map(str, r))
        if repeat_observations:
            acc.setdefault('lap_seen', set()).update(map(str, r))
        return r
    baselines._visit = _visit_logged
    try:
        yield acc
    finally:
        baselines._visit = orig

def _nav_fail_counter(acc: dict):

    def on_step(kind, arg, result):
        if kind == 'navigate' and (not acc.get('tour_nav_depth', 0)) and (not getattr(result, 'success', False)):
            acc['nav_failed'] += 1
    return on_step

def _oracle_evidence_observer(sim, acc: dict):
    count_failure = _nav_fail_counter(acc)

    def on_step(kind, arg, result):
        count_failure(kind, arg, result)
        sim.observe()
    return on_step

def verdict_for(sim, spec, *, scene: str, episode_factory: Callable, seed: int=0, run_null: bool=True, oracle_tier: str='high', oracle_refer: str='name') -> dict:
    from roam_bench.protocol import baselines, taskset
    t = spec.task
    started = time.perf_counter()
    rec: dict = {'task_id': t.task_id, 'family': spec.family, 'instruction': t.instruction, 'oracle_tier': oracle_tier, 'oracle_refer': oracle_refer}

    def measure_spawn_visibility() -> None:
        if 'answer_visible_at_spawn' in rec or 'spawn_visibility_error' in rec:
            return
        try:
            visible = set(sim.visible_instances(0.002))
            evidence = set(spec.evidence)
            answers = set(spec.answer_instances)
            rec['visible_at_spawn'] = sorted(visible)
            rec['answer_visible_at_spawn'] = bool(answers & visible)
            rec['solvable_from_initial_frame'] = bool(evidence) and evidence <= visible
        except Exception as exc:
            rec['spawn_visibility_error'] = f'{type(exc).__name__}: {exc}'
    if run_null:
        try:
            sim.reset(episode_factory(t.task_id, scene))
            rec['spawn_pose'] = _pose_dict(sim)
            measure_spawn_visibility()
            rec['null_success'] = baselines.null_rollout(sim, t).score().success
        except Exception as e:
            rec['null_success'] = None
            rec['null_error'] = f'{type(e).__name__}: {e}'
            rec['null_failure_kind'] = _classify(e)
    else:
        rec['null_success'] = None
        rec['null_skipped'] = True
    tour = {'stops': 0, 'reached': 0, 'want': set(), 'seen': set(), 'nav_failed': 0, 'tour_nav_depth': 0}
    try:
        sim.reset(episode_factory(t.task_id, scene))
        if rec.get('spawn_pose') is None:
            rec['spawn_pose'] = _pose_dict(sim)
        measure_spawn_visibility()
        read = taskset.oracle_answer_for(spec)
        from .tour_optimizer import optimize_tours
        with _capture_tour(tour, optimize=oracle_tier != 'high', repeat_observations=spec.family == 'change_detect'), optimize_tours(enabled=oracle_tier != 'high'):
            observer = _oracle_evidence_observer(sim, tour)
            if oracle_tier == 'high':
                if oracle_refer != 'name':
                    raise ValueError('high oracle only supports the name referent channel')
                rec['oracle_executor_version'] = 'privileged-canonical'
                rollout = baselines.oracle_rollout(sim, t, plan=spec.plan, answer=read, on_step=observer)
            else:
                from .legal_oracle import VERSION, prepare_plan, rollout as legal_rollout
                rec['oracle_executor_version'] = VERSION
                bound = getattr(spec, 'evidence_bound', None)
                executable_plan = prepare_plan(spec.plan, must_open=getattr(bound, 'must_open', ()) if bound else (), reader_managed_open=spec.family == 'hidden_count')
                rec['oracle_plan'] = [{'verb': str(verb), 'target': getattr(ref, 'name', str(ref))} for verb, ref in executable_plan]
                rollout = legal_rollout(sim, t, plan=executable_plan, answer=read, tier=oracle_tier, refer=oracle_refer, on_step=observer, max_steps=int(t.max_steps))
            s = rollout.score()
        rec['oracle_success'] = s.success
        rec['oracle_answer'] = s.answer_given
        rec['score'] = s.to_dict()
        rec['actions'] = [{'verb': kind, 'argument': arg, 'success': success, 'reason': reason} for kind, arg, success, reason in rollout.steps]
        if hasattr(sim, 'placement_events'):
            rec['placement_events'] = sim.placement_events()
        rec['scoring_details'] = {'answer_match': getattr(t, 'answer_match', None), 'goal_results': list(getattr(s, 'goals', ()) or ())}
        rec['reader_callable'] = callable(read)
        from .enhanced_evidence import judge as judge_evidence
        trace = sim.exploration_evidence() if hasattr(sim, 'exploration_evidence') else None
        rec['oracle_approaches'] = list((trace or {}).get('approaches') or ())
        failed_action = next((row for row in rec['actions'] if row.get('success') is not True), None)
        failed_approach = next((row for row in rec['oracle_approaches'] if row.get('success') is not True), None)
        incomplete_goals = [str(goal.get('label')) for goal in (rec.get('score') or {}).get('goals', ()) if goal.get('ok') is not True]
        rec['oracle_failure'] = {'first_failed_action': failed_action, 'first_failed_approach': failed_approach, 'incomplete_goals': incomplete_goals}
        receipt = judge_evidence(spec, trace, bool(s.success >= 1.0))
        rec['evidence'] = receipt
        rec['strict_success'] = receipt['strict_success']
        rec['exploration_trace'] = trace
    except Exception as e:
        if getattr(e, 'oracle_budget_exceeded', False):
            partial = e.recorder
            rec['oracle_success'] = 0.0
            rec['oracle_budget_terminated'] = True
            rec['actions'] = [{'verb': kind, 'argument': arg, 'success': success, 'reason': reason} for kind, arg, success, reason in partial.steps]
            rec['score'] = {'success': 0.0, 'goals': []}
            rec['strict_success'] = False
        else:
            rec['oracle_success'] = None
            rec['oracle_error'] = f'{type(e).__name__}: {e}'
            rec['oracle_failure_kind'] = _classify(e)
            rec['strict_success'] = False
    rec['n_sweep_stops'] = tour['stops']
    rec['n_tour_stops_reached'] = tour['reached']
    rec['tour_seen'] = sorted(tour['seen'])
    rec['tour_want_unseen'] = sorted(tour['want'] - tour['seen'])
    rec['n_navigate_failed'] = tour['nav_failed']
    rec['live_stance_success'] = bool(tour['nav_failed'] == 0 and (not tour['want'] - tour['seen']))
    rec['timing'] = {'total_wall_seconds': max(0.0, time.perf_counter() - started)}
    rec['solvable'] = (rec.get('oracle_success') or 0.0) >= SOLVE_THRESHOLD
    rec['reason'] = _reason(rec)
    return rec

def run_side(sim, specs, *, scene: str, side: str, checkout: str, episode_factory: Callable, seed: int=0) -> dict:
    rows = [verdict_for(sim, s, scene=scene, episode_factory=episode_factory, seed=seed) for s in specs]
    return {'schema_version': SCHEMA_VERSION, 'side': side, 'scene': scene, 'code_checkout': checkout, 'semantics': semantics_probe(), 'n_tasks': len(rows), 'n_solvable': sum((1 for r in rows if r['solvable'])), 'verdicts': rows}

def _comparable(a: dict, b: dict) -> Optional[str]:
    va, vb = (a.get('schema_version'), b.get('schema_version'))
    if not {va, vb} <= _COMPATIBLE_SCHEMAS:
        return f'schema versions are not mutually comparable ({va} vs {vb}; compatible set {sorted(_COMPATIBLE_SCHEMAS)})'
    if a.get('scene') != b.get('scene'):
        return f"different scenes ({a.get('scene')!r} vs {b.get('scene')!r})"
    if a.get('semantics') != b.get('semantics'):
        return f"different scoring semantics ({a.get('semantics')} vs {b.get('semantics')}); the scoring configuration must resolve the same checkout on both sides before verdicts are comparable"
    return None

def _dim_value(rec: dict, dim: str):
    if dim == 'null':
        v = rec.get('null_success')
        return None if v is None else round(v, 3)
    if dim == 'oracle':
        return rec['solvable'] if rec.get('oracle_success') is not None else None
    raise ValueError(f'unknown dimension {dim!r}')

def _compare_dimension(a_by, b_by, side_a, side_b, dim):
    agree = disagree = 0
    disagreements = []
    not_comparable_rows = []
    for tid in sorted(set(a_by) & set(b_by)):
        ra, rb = (a_by[tid], b_by[tid])
        va, vb = (_dim_value(ra, dim), _dim_value(rb, dim))
        if va is None or vb is None:
            missing = side_a if va is None else side_b if vb is None else 'both'
            reason = (ra if va is None else rb).get('reason') or f'{dim} not run on {missing}'
            not_comparable_rows.append({'task_id': tid, 'family': ra['family'], 'missing_on': missing, 'reason': reason})
            continue
        if va == vb:
            agree += 1
        else:
            disagree += 1
            if dim == 'oracle':
                worse = side_b if va and (not vb) else side_a
                reason = (rb if va and (not vb) else ra).get('reason')
            else:
                worse = None
                reason = f'null success differs: {side_a}={va} vs {side_b}={vb}'
            disagreements.append({'task_id': tid, 'family': ra['family'], f'{side_a}_value': va, f'{side_b}_value': vb, 'worse_side': worse, 'reason': reason})
    return {'n_comparable': agree + disagree, 'agree': agree, 'disagree': disagree, 'not_comparable': len(not_comparable_rows), 'disagreements': disagreements, 'not_comparable_rows': not_comparable_rows}

def _compare_spawn(a_by, b_by, side_a, side_b) -> dict:
    same = diff = unknown = 0
    examples = []
    for tid in sorted(set(a_by) & set(b_by)):
        pa, pb = (a_by[tid].get('spawn_pose'), b_by[tid].get('spawn_pose'))
        if pa is None or pb is None:
            unknown += 1
            continue
        if pa == pb:
            same += 1
        else:
            diff += 1
            if len(examples) < 5:
                examples.append({'task_id': tid, side_a: pa, side_b: pb})
    return {'n_same': same, 'n_different': diff, 'n_unknown': unknown, 'different_examples': examples, 'note': "the two backends did not necessarily start every task from the same pose (Isaac forces one verified open pose; FakeSim takes each episode's default). null answers without moving and the oracle navigates to its targets, so n_different > 0 is expected and does not by itself invalidate a comparison -- but a disagreement on a spawn-sensitive family should be checked against these poses before it is read as a backend difference"}
_REQUIRED_ROW_FIELDS = ('task_id', 'family', 'solvable')

def _validate_rows(doc: dict, side_label: str) -> None:
    problems = []
    for i, r in enumerate(doc.get('verdicts') or ()):
        missing = [f for f in _REQUIRED_ROW_FIELDS if f not in r]
        if missing:
            where = r.get('task_id', f'#{i}')
            problems.append(f"{side_label} row {where}: missing {', '.join(missing)}")
    if problems:
        shown = problems[:10]
        more = f' (+{len(problems) - len(shown)} more)' if len(problems) > len(shown) else ''
        raise ValueError('verdict rows are missing fields compare needs: ' + '; '.join(shown) + more)

def compare(side_a: dict, side_b: Optional[dict], dimensions=('null', 'oracle')) -> dict:
    _validate_rows(side_a, side_a.get('side') or 'side_a')
    a_by = {r['task_id']: r for r in side_a['verdicts']}
    if side_b is None:
        return {'isaac_available': False, 'note': 'only one side present; showing its self-consistency. Run the other backend over the SAME frozen taskset and re-compare.', 'side_a': side_a.get('side'), 'n': len(a_by), 'n_solvable_a': side_a.get('n_solvable')}
    blocker = _comparable(side_a, side_b)
    if blocker:
        return {'comparable': False, 'refused_because': blocker, 'side_a': side_a.get('side'), 'side_b': side_b.get('side')}
    _validate_rows(side_b, side_b.get('side') or 'side_b')
    b_by = {r['task_id']: r for r in side_b['verdicts']}
    sa, sb = (side_a.get('side'), side_b.get('side'))
    only_a = sorted(set(a_by) - set(b_by))
    only_b = sorted(set(b_by) - set(a_by))
    dims = {d: _compare_dimension(a_by, b_by, sa, sb, d) for d in dimensions}
    return {'comparable': True, 'isaac_available': True, 'side_a': sa, 'side_b': sb, 'n_tasks': len(set(a_by) | set(b_by)), 'only_in_a': only_a, 'only_in_b': only_b, 'spawn': _compare_spawn(a_by, b_by, sa, sb), 'dimensions': dims}
