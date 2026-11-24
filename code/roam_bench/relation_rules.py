"""Resolve verified support relations for scene instances."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

def corrected_relation(relation, *, scene, house_sha256, overrides):
    result = dict(relation)
    if result.get('placement_method') != 'fill_container':
        return result
    key = scene + '/' + str(result.get('child', ''))
    approved = overrides.get(key)
    if approved:
        if approved['house_state_sha256'] != house_sha256:
            raise ValueError('Reviewed relation source changed: ' + key)
        original = approved['original_relation']
        if result != original and result != dict(original, relation='unknown'):
            raise ValueError('Reviewed relation identity or host changed: ' + key)
        result['relation'] = 'on'
    else:
        result['relation'] = 'unknown'
    return result

def corrected_trace(trace, *, scene, house_sha256, overrides):
    result = deepcopy(trace)
    relations = result.get('placement_relations')
    if isinstance(relations, dict):
        result['placement_relations'] = {alias: corrected_relation(row, scene=scene, house_sha256=house_sha256, overrides=overrides) for alias, row in relations.items()}
    elif isinstance(relations, list):
        result['placement_relations'] = [corrected_relation(row, scene=scene, house_sha256=house_sha256, overrides=overrides) for row in relations]
    return result

def reviewed_scene_relation(relation, scene_root, house_path):
    path = Path(__file__).with_name('reviewed_relation_overrides.json')
    overrides = json.loads(path.read_text())
    sha = hashlib.sha256(Path(house_path).read_bytes()).hexdigest()
    from .scene_variants import relation_source
    scene, source_sha = relation_source(scene_root, house_path, sha)
    return corrected_relation(relation, scene=scene, house_sha256=source_sha, overrides=overrides)

def restore_articulations(backend, set_action, plan_interact, pose_within):
    arts = getattr(backend, '_articulations', {}) or {}
    context = getattr(backend, '_sim_ctx', None)
    if not arts or context is None:
        return
    states = getattr(backend, '_open_state', {}) or {}
    dirty = False
    for alias, art in arts.items():
        try:
            rows = backend._read_articulation(art, 'close', plan_interact, pose_within)
        except Exception:
            rows = []
        if not rows or not all((row.get('reached_target') is True for row in rows)) or states.get(alias, False) or (art.get('last_action') == 'open'):
            dirty = True
    if not dirty:
        return
    for art in arts.values():
        set_action(art['drives'], art['plan'], 'close')
    context.reset()
    for _ in range(backend._interact_settle):
        context.step(render=False)
    failures = []
    for alias, art in arts.items():
        rows = backend._read_articulation(art, 'close', plan_interact, pose_within)
        art['last_readouts'] = rows
        reached = sum((row.get('reached_target') is True for row in rows))
        if not rows or reached != len(rows):
            failures.append(f'{alias}: {reached}/{len(rows)} joints')
    if failures:
        raise RuntimeError('could not restore articulation state: ' + '; '.join(failures))
    for alias, art in arts.items():
        art['last_action'] = 'close'
        backend._open_state[alias] = False
