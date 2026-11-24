"""Robustness utilities for ROAM-Bench."""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
SCHEMA_VERSION = 1
AXES = ('camera_height', 'lighting', 'material', 'layout', 'action_noise', 'language')
LEVELS = ('identity', 'mild', 'strong')
_PARAMS = {'camera_height': {'identity': {'camera_height_offset_m': 0.0}, 'mild': {'camera_height_offset_m': -0.12}, 'strong': {'camera_height_offset_m': 0.2}}, 'lighting': {'identity': {'lighting_scale': 1.0}, 'mild': {'lighting_scale': 0.75}, 'strong': {'lighting_scale': 1.35}}, 'material': {'identity': {'material_strength': 0.0}, 'mild': {'material_strength': 0.2}, 'strong': {'material_strength': 0.4}}, 'layout': {'identity': {'layout_jitter_m': 0.0}, 'mild': {'layout_jitter_m': 0.05}, 'strong': {'layout_jitter_m': 0.1}}, 'action_noise': {'identity': {'move_sigma_m': 0.0, 'turn_sigma_deg': 0.0}, 'mild': {'move_sigma_m': 0.02, 'turn_sigma_deg': 1.5}, 'strong': {'move_sigma_m': 0.05, 'turn_sigma_deg': 4.0}}, 'language': {'identity': {'language_variant': 'identity'}, 'mild': {'language_variant': 'neutral_reframe_v1'}, 'strong': {'language_variant': 'concise_reframe_v1'}}}

def canonical_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))

def digest(value) -> str:
    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()

def make_profile(axis: str, level: str, seed: int) -> dict:
    if axis not in AXES:
        raise ValueError(f'unknown robustness axis {axis!r}')
    if level not in LEVELS:
        raise ValueError(f'unknown robustness level {level!r}')
    return {'schema': 'ROAM-Bench-perturbation', 'schema_version': SCHEMA_VERSION, 'axis': axis, 'level': level, 'seed': int(seed), 'params': dict(_PARAMS[axis][level])}

def validate_profile(profile: Mapping) -> None:
    axis, level = (profile.get('axis'), profile.get('level'))
    if axis not in AXES or level not in LEVELS:
        raise ValueError('profile must name one supported axis and level')
    if profile.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported perturbation schema')
    if not isinstance(profile.get('seed'), int) or isinstance(profile.get('seed'), bool):
        raise ValueError('perturbation seed must be an integer')
    expected = _PARAMS[axis][level]
    if profile.get('params') != expected:
        raise ValueError(f'parameters do not match frozen {axis}/{level} policy')

def instruction_variant(instruction: str, profile: Mapping) -> str:
    validate_profile(profile)
    if profile['axis'] != 'language' or profile['level'] == 'identity':
        return instruction
    text = str(instruction).strip()
    if profile['params']['language_variant'] == 'neutral_reframe_v1':
        return f'Explore as needed, then complete this spatial task: {text}'
    return f'Spatial task — {text} Give only the requested result after exploring.'

def protected_aliases(spec: Mapping) -> list[str]:
    aliases = set()
    for key in ('referents', 'evidence', 'answer_instances'):
        aliases.update((str(v) for v in spec.get(key) or () if isinstance(v, str)))
    bound = spec.get('evidence_bound') or {}
    for key in ('must_see', 'must_open'):
        aliases.update((str(v) for v in bound.get(key) or () if isinstance(v, str)))

    def walk(value):
        if isinstance(value, str):
            if '_' in value and ' ' not in value:
                aliases.add(value)
        elif isinstance(value, Mapping):
            for child in value.values():
                walk(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                walk(child)
    walk(spec.get('truth') or {})
    for goal in spec.get('goals') or ():
        walk(goal)
    return sorted(aliases)

def deterministic_gaussian(seed: int, step: int, channel: str, sigma: float) -> float:
    if sigma <= 0:
        return 0.0
    token = hashlib.sha256(f'{int(seed)}:{int(step)}:{channel}'.encode()).digest()
    rng = random.Random(int.from_bytes(token[:8], 'big'))
    return rng.gauss(0.0, float(sigma))

def truth_payload(spec: Mapping) -> dict:
    keys = ('task_id', 'family', 'goals', 'answer_keywords', 'answer_choices', 'answer_from_pose', 'answer_match', 'answer_number', 'answer_tolerance', 'answer_unit', 'truth', 'oracle_answer')
    return {key: spec.get(key) for key in keys}

def build_manifest(frozen_sha256: str, scene: str, specs: Iterable[Mapping], *, axes: Iterable[str]=AXES, level: str='mild', seed: int=0, scorer_version: str, budget_cap: int | None=None, config_version: str='robustness-v1') -> dict:
    if level == 'identity':
        selected_level = 'identity'
    else:
        selected_level = level
    cases = []
    for axis in axes:
        control = make_profile(axis, 'identity', seed)
        treatment = make_profile(axis, selected_level, seed)
        for spec in specs:
            task_id = str(spec.get('task_id') or '')
            if not task_id:
                raise ValueError('robustness candidate lacks task_id')
            frozen_budget = int(spec.get('max_steps', 0))
            budget = min(frozen_budget, int(budget_cap)) if budget_cap is not None else frozen_budget
            invariant = {'source_frozen_sha256': frozen_sha256, 'scene': scene, 'task_id': task_id, 'budget': budget, 'scorer_version': scorer_version, 'truth_sha256': digest(truth_payload(spec)), 'seed': int(seed)}
            pair_id = digest({'axis': axis, **invariant})[:20]
            protected = protected_aliases(spec)
            for arm, profile in (('control', control), ('treatment', treatment)):
                cases.append({'pair_id': pair_id, 'arm': arm, 'axis': axis, 'profile': profile, 'protected_aliases': protected, 'instruction': instruction_variant(str(spec.get('instruction') or ''), profile), **invariant})
    return {'schema': 'ROAM-Bench-robustness-manifest', 'schema_version': SCHEMA_VERSION, 'config_version': config_version, 'one_axis_at_a_time': True, 'axes': list(axes), 'level': level, 'seed': int(seed), 'cases': cases}

def validate_manifest(doc: Mapping) -> list[dict]:
    errors = []
    cases = doc.get('cases') or ()
    if not cases:
        errors.append({'code': 'EMPTY_COHORT'})
    for field in ('source_frozen_sha256', 'scene', 'scorer_version', 'seed'):
        values = {canonical_json(row.get(field)) for row in cases}
        if len(values) > 1:
            errors.append({'code': 'COHORT_INVARIANT_DRIFT', 'field': field})
    declared_axes = set(doc.get('axes') or ())
    actual_axes = {row.get('axis') for row in cases}
    if declared_axes != actual_axes or not actual_axes <= set(AXES):
        errors.append({'code': 'AXIS_SET', 'declared': sorted(declared_axes), 'actual': sorted((str(v) for v in actual_axes))})
    grouped = defaultdict(list)
    for row in cases:
        grouped[str(row.get('pair_id'))].append(row)
    for pair_id, rows in grouped.items():
        if len(rows) != 2 or {r.get('arm') for r in rows} != {'control', 'treatment'}:
            errors.append({'code': 'PAIR_CARDINALITY', 'pair_id': pair_id})
            continue
        control = next((r for r in rows if r['arm'] == 'control'))
        treatment = next((r for r in rows if r['arm'] == 'treatment'))
        invariant_keys = ('source_frozen_sha256', 'scene', 'task_id', 'budget', 'scorer_version', 'truth_sha256', 'seed', 'axis')
        drift = [key for key in invariant_keys if control.get(key) != treatment.get(key)]
        if drift:
            errors.append({'code': 'PAIR_INVARIANT_DRIFT', 'pair_id': pair_id, 'fields': drift})
        try:
            validate_profile(control.get('profile') or {})
            validate_profile(treatment.get('profile') or {})
        except ValueError as exc:
            errors.append({'code': 'PROFILE', 'pair_id': pair_id, 'detail': str(exc)})
            continue
        if control['profile']['level'] != 'identity':
            errors.append({'code': 'CONTROL_NOT_IDENTITY', 'pair_id': pair_id})
        if control['profile']['axis'] != control.get('axis') or treatment['profile']['axis'] != treatment.get('axis'):
            errors.append({'code': 'MULTI_AXIS', 'pair_id': pair_id})
    return errors

def _success(row: Mapping) -> bool | None:
    value = row.get('strict_success')
    if isinstance(value, bool):
        return value
    value = row.get('answer_success')
    if isinstance(value, bool):
        return value
    score = row.get('score') or {}
    return score.get('success') if isinstance(score.get('success'), bool) else None

def failure_attribution(row: Mapping) -> str:
    actions = row.get('actions') or ()
    score = row.get('score') if isinstance(row.get('score'), Mapping) else {}
    execution_words = ('blocked', 'collision', 'could not', 'too far', 'failed')
    if float(score.get('collisions') or 0) > 0 or float(score.get('invalid_actions') or 0) > 0 or any((action.get('success') is False or action.get('collided') is True or any((word in str(action.get('raw_reason') or '').lower() for word in execution_words)) for action in actions if isinstance(action, Mapping))):
        return 'execution'
    evidence = row.get('evidence') or {}
    missing = evidence.get('missing') or {}
    missing_text = canonical_json(missing).lower()
    if isinstance(missing, Mapping) and (missing.get('must_see') or missing.get('viewpoints')) or 'see' in missing_text or 'viewpoint' in missing_text or (row.get('frame_dark') is True):
        return 'perception'
    return 'planning'

def _bootstrap_ci(values: list[float], seed: int, samples: int=4000) -> list[float] | None:
    if not values:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum((values[rng.randrange(len(values))] for _ in values)) / len(values))
    means.sort()
    return [round(means[int(0.025 * (samples - 1))], 6), round(means[int(0.975 * (samples - 1))], 6)]

def paired_report(manifest: Mapping, rows: Iterable[Mapping]) -> dict:
    manifest_errors = validate_manifest(manifest)
    by_key = {}
    duplicate = []
    result_mismatches = []
    for row in rows:
        key = (str(row.get('pair_id')), str(row.get('arm')))
        if key in by_key:
            duplicate.append(key)
        by_key[key] = row
    expected_keys = {(str(row.get('pair_id')), str(row.get('arm'))) for row in manifest.get('cases') or ()}
    unexpected = sorted(set(by_key) - expected_keys)
    groups = defaultdict(list)
    missing = []
    attribution = Counter()
    identity_mismatch = []
    for case in manifest.get('cases') or ():
        if case.get('arm') != 'control':
            continue
        pair_id, axis = (str(case['pair_id']), str(case['axis']))
        c, t = (by_key.get((pair_id, 'control')), by_key.get((pair_id, 'treatment')))
        treatment_case = next((r for r in manifest['cases'] if r['pair_id'] == pair_id and r['arm'] == 'treatment'))
        for expected, actual in ((case, c), (treatment_case, t)):
            if actual is None:
                continue
            drift = [key for key in ('pair_id', 'arm', 'task_id', 'budget', 'source_frozen_sha256', 'scorer_version') if actual.get(key) != expected.get(key)]
            actual_profile = actual.get('robustness') or {}
            for key in ('axis', 'level', 'seed', 'params'):
                if actual_profile.get(key) != expected['profile'].get(key):
                    drift.append(f'profile.{key}')
            if drift:
                result_mismatches.append({'pair_id': pair_id, 'arm': expected['arm'], 'fields': drift})
        cs, ts = (_success(c or {}), _success(t or {}))
        if cs is None or ts is None:
            missing.append(pair_id)
            continue
        delta = float(ts) - float(cs)
        groups[axis].append(delta)
        if cs and (not ts):
            attribution[axis, failure_attribution(t)] += 1
        if treatment_case['profile']['level'] == 'identity' and cs != ts:
            identity_mismatch.append(pair_id)
    axes = {}
    for axis, values in sorted(groups.items()):
        axes[axis] = {'n_pairs': len(values), 'accuracy_delta': round(sum(values) / len(values), 6), 'paired_bootstrap_95': _bootstrap_ci(values, int(manifest.get('seed', 0)) + AXES.index(axis)), 'regression_attribution': {cause: attribution[axis, cause] for cause in ('perception', 'planning', 'execution')}}
    status = 'fail' if manifest_errors or duplicate or unexpected or result_mismatches or identity_mismatch else 'pending' if missing else 'pass'
    cases = list(manifest.get('cases') or ())
    return {'schema': 'ROAM-Bench-robustness-report', 'schema_version': SCHEMA_VERSION, 'manifest_sha256': digest(manifest), 'source_frozen_sha256': next(iter({r.get('source_frozen_sha256') for r in cases}), None), 'scene': next(iter({r.get('scene') for r in cases}), None), 'scorer_version': next(iter({r.get('scorer_version') for r in cases}), None), 'task_ids': sorted({str(r.get('task_id')) for r in cases}), 'tested_axes': sorted({str(r.get('axis')) for r in cases}), 'status': status, 'manifest_errors': manifest_errors, 'duplicate_results': duplicate, 'unexpected_results': unexpected, 'missing_pairs': sorted(set(missing)), 'result_mismatches': result_mismatches, 'identity_mismatches': sorted(set(identity_mismatch)), 'axes': axes}
