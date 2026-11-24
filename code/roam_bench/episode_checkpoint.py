"""Episode checkpoint utilities for ROAM-Bench."""
import base64
import copy
import hashlib
import json
import math
import os
import dataclasses
import re
import html
import shutil
from pathlib import Path
SCHEMA = 'roam-episode-checkpoint-v1'

def identity(task, configuration):
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__), root / 'roam_bench/harness.py', root / 'roam_bench/cli.py', root / 'roam_bench/backend.py', root / 'roam_bench/protocol/llm.py', root / 'roam_bench/protocol/retrieval_memory.py', root / 'roam_bench/protocol/intermediate_memory.py', root / 'roam_bench/protocol/unified_memory.py', root / 'roam_bench/self_trajectory.py', root / 'roam_bench/enhanced_evidence.py', root / 'roam_bench/protocol/visibility.py', root / 'roam_bench/protocol/types.py']
    return {'task': jsonable(dataclasses.asdict(task)), 'configuration': configuration, 'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}

def atomic(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(f'.json.{os.getpid()}.tmp')
    temp.write_text(json.dumps(doc, ensure_ascii=False))
    temp.replace(path)

def image_url(path, sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('resume archive image SHA differs')
    mime = 'image/png' if Path(path).suffix == '.png' else 'image/jpeg'
    return 'data:' + mime + ';base64,' + base64.b64encode(raw).decode()

def restore_archive(case):
    directory = Path(case) / 'memory_archive'
    manifest = json.loads((directory / 'manifest.json').read_text())
    archive = copy.deepcopy(manifest['rounds'])
    for entry in archive:
        entry['image'] = image_url(directory / entry['image_file'], entry['image_sha256'])
        for frame in entry.get('intermediates', []):
            frame['image'] = image_url(directory / frame['image_file'], frame['image_sha256'])
    return (archive, manifest)

def reconstruct_agent(agent, log, case):
    from .protocol.llm import _user_content
    archive, manifest = restore_archive(case)
    if hasattr(agent, 'intermediate_events') and agent.archive_dir:
        source_audit = Path(case) / 'memory_archive/candidate_audit'
        target_audit = Path(agent.archive_dir) / 'candidate_audit'
        if source_audit.resolve() != target_audit.resolve():
            shutil.copytree(source_audit, target_audit, dirs_exist_ok=True)
    if [r['round'] for r in archive] != [r['round'] for r in log['rounds']]:
        raise ValueError('resume log/archive rounds differ')
    for entry, turn in zip(archive, log['rounds']):
        content = _user_content(turn['prompt'], entry['image'])
        if hasattr(agent, 'intermediate_events'):
            event = turn['intermediate_selection']
            prefix = [{'type': 'text', 'text': f"Intermediate selection: shown={event['generated_ids']}, retained={event['retained_ids']}. CURRENT FINAL is the last image in this observation; use it for action pixels."}]
            if entry.get('intermediates'):
                prefix.append({'type': 'text', 'text': 'Retained intermediate memory:'})
                for f in entry['intermediates']:
                    prefix.extend([{'type': 'text', 'text': f['id']}, {'type': 'image_url', 'image_url': {'url': f['image']}}])
            content = prefix + content
        agent.messages.extend([{'role': 'user', 'content': content}, {'role': 'assistant', 'content': turn['raw']}])
    agent._trim_images()
    agent.rounds = copy.deepcopy(log['rounds'])
    agent.archive = archive
    agent.retrieval_events = copy.deepcopy(log['meta']['retrieval_events'])
    agent.images_sent = log['meta']['frames_shown']
    agent.image_bytes = log['meta']['frame_bytes']
    agent._since_compact = len(agent.rounds)
    if hasattr(agent, 'intermediate_events'):
        agent.intermediate_events = copy.deepcopy(log['meta']['intermediate_events'])
        last = agent.intermediate_events[-1] if agent.intermediate_events else {}
        audit = Path(case) / 'memory_archive/candidate_audit'
        pending_ids = [p.stem for p in sorted(audit.glob(f'R{len(agent.rounds) + 1}.I*.json'))]
        pending = []
        for frame_id in pending_ids:
            p = Path(case) / 'memory_archive/candidate_audit' / (frame_id + '.json')
            f = json.loads(p.read_text())
            f['image'] = image_url(p.parent / f['image_file'], f['image_sha256'])
            pending.append(f)
        agent.pending = pending
        agent.intermediate_events = [e for e in agent.intermediate_events if e['round'] <= len(agent.rounds)]
    return manifest

def pose_close(a, b, tolerance=1e-06):
    return all((abs(float(a.get(k, 0)) - float(b.get(k, 0))) <= tolerance for k in ('x', 'z', 'yaw_deg', 'horizon_deg')))

def set_pose(sim, pose):
    from .protocol.types import Pose
    sim._pose = Pose(x=pose['x'], y=pose.get('y', 1.5), z=pose['z'], yaw_deg=pose['yaw_deg'], horizon_deg=pose['horizon_deg'])
    sim._sync_embodiment_pose()

def restore_steps(space, turns, score, case, scale):
    from .protocol import llm
    from .protocol.score import _label
    rows = re.findall('<tr class="(ok|bad)">(.*?)</tr>', (Path(case) / 'report.html').read_text(), re.S)
    steps = []
    for turn in turns:
        if turn['verb'] is None:
            continue
        kind, arg, _ = llm.bind(space, turn['verb'], tuple(turn['args']), turn.get('kwargs', {}), pixel_scale=scale)
        if kind is None:
            continue
        matching = [status for status, cells in rows if html.escape(turn['raw']) in cells and html.escape(turn.get('raw_reason', '')) in cells]
        if len(matching) != 1:
            raise ValueError('trajectory report lacks unambiguous action outcome')
        steps.append((kind, _label(arg), matching[0] == 'ok', turn.get('raw_reason', '')))
    if len(steps) != score['steps'] or sum((not s[2] for s in steps)) != score['invalid_actions']:
        raise ValueError('trajectory action aggregates differ from source score')
    return steps

def replay_restore(sim, rec, agent, log, row, case):
    from .protocol import llm
    turns = log['rounds']
    if log['meta'].get('compactions'):
        raise ValueError('trajectory compacted history needs an exact state checkpoint')
    observations = log['exploration_trace']['observations']
    final_observations = [o for o in observations if not o.get('frame_id')]
    if row.get('token_budget_exceeded') is not True:
        raise ValueError('trajectory continuation requires a budget-terminated source')
    if log['exploration_trace'].get('mutations'):
        raise ValueError('trajectory mutation restoration needs a state checkpoint')
    if len(final_observations) < 2 * len(turns) + 1:
        raise ValueError('trajectory source lacks the final observation')
    prefix = final_observations[:2 * len(turns)]
    if any((prefix[2 * i]['pose'] != prefix[2 * i + 1]['pose'] for i in range(len(turns)))):
        raise ValueError('trajectory film/model observation pairs differ')
    scale = ((sim.width - 1) / (agent.frame_size[0] - 1), (sim.height - 1) / (agent.frame_size[1] - 1))
    rec.start_xz = (row['spawn']['x'], row['spawn']['z']) if rec.task.answer_from_pose else None
    replayed = []
    for i, turn in enumerate(turns):
        if turn['verb'] not in {'pick', 'place', 'open', 'close', 'toggle'}:
            continue
        before = prefix[2 * i]
        set_pose(sim, before['pose'])
        sim._step_idx = before['step_idx']
        sim.observe()
        kind, arg, thunk = llm.bind(agent.space, turn['verb'], tuple(turn['args']), turn.get('kwargs', {}), pixel_scale=scale)
        if kind is None:
            continue
        result = thunk()
        expected_reason = turn.get('raw_reason')
        if expected_reason is not None and result.reason != expected_reason:
            raise ValueError(f"trajectory manipulation outcome differs at round {turn['round']}")
        replayed.append({'round': turn['round'], 'verb': kind, 'success': result.success})
    last = final_observations[-1]
    set_pose(sim, last['pose'])
    sim._step_idx = last['step_idx']
    sim._evidence_observations = copy.deepcopy(observations)
    sim._evidence_opened = set(log['exploration_trace'].get('opened', []))
    sim._evidence_mutations = copy.deepcopy(log['exploration_trace'].get('mutations', []))
    sim._evidence_approaches = copy.deepcopy(log['exploration_trace'].get('approaches', []))
    sim._evidence_interactions = copy.deepcopy(log['exploration_trace'].get('interaction_events', []))
    steps, invalid = (row['score']['steps'], row['score']['invalid_actions'])
    rec.steps = restore_steps(agent.space, turns, row['score'], case, scale)
    rec.collisions = row['score']['collisions']
    rec._ever = {g['expr'] for g in row['score'].get('goals', []) if g['when'] == 'ever' and g['ok']}
    sim._seen = set().union(*(set(o['visible_aliases']) for o in observations))
    wm = getattr(sim, '_world_model', None)
    if wm is not None:
        for observation in observations:
            wm.observe(visible=observation['visible_aliases'], holding=sim._held)
        wm.holding = sim._held
    reconstruct_agent(agent, log, case)
    repeated_preview = {f['id'] for f in getattr(agent, 'pending', [])}
    observations = [o for o in observations if o.get('frame_id') not in repeated_preview]
    sim._evidence_observations = copy.deepcopy(observations)
    sim.observe()
    actual = sim.exploration_evidence()['observations'][-1]
    if not pose_close(last['pose'], actual['pose']):
        raise ValueError('restored endpoint pose differs')
    original_aliases, actual_aliases = (set(last['visible_aliases']), set(actual['visible_aliases']))
    union = original_aliases | actual_aliases
    jaccard = len(original_aliases & actual_aliases) / len(union) if union else 1
    if jaccard < 0.9:
        raise ValueError(f'restored endpoint segmentation differs: Jaccard={jaccard:.3f}')
    sim._evidence_observations = copy.deepcopy(observations)
    return {'mode': 'trajectory-direct-pose-and-manipulation-state', 'model_calls_for_prefix': 0, 'navigation_steps_replayed': 0, 'manipulations_restored': replayed, 'endpoint_segmentation_jaccard': jaccard, 'next_round': len(turns) + 1, 'repeated_preview_ids': sorted(repeated_preview), 'source_partial_intermediate_events': [e for e in log['meta'].get('intermediate_events', []) if e['round'] > len(turns)], 'source_steps': steps}
BACKEND_FIELDS = ('_step_idx', '_seen', '_held', '_placed_on', '_placed_points', '_placement_events', '_room_overrides', '_open_state', '_toggled_state', '_position_overrides', '_evidence_observations', '_evidence_opened', '_evidence_mutations', '_evidence_approaches', '_evidence_interactions')
AGENT_FIELDS = ('messages', 'rounds', 'archive', 'retrieval_events', 'images_sent', 'image_bytes', '_since_compact', 'intermediate_events', 'pending', '_retained', 'image_limit_events', 'vision_resolution_events', 'vision_packing_events', 'compactions')
AGENT_FIELDS = AGENT_FIELDS + ('trajectory_points',)
REC_FIELDS = ('steps', 'collisions', '_ever', 'start_xz', 'answer_given', '_visited', 'mutation_done', 'mutation_failed', 'resolved_truth_deg', 'resolved_pose', 'clicks')

def jsonable(value):
    if isinstance(value, set):
        return {'__set__': [jsonable(v) for v in value]}
    if isinstance(value, tuple):
        return {'__tuple__': [jsonable(v) for v in value]}
    if dataclasses.is_dataclass(value):
        return {'__dataclass__': type(value).__name__, 'fields': jsonable(dataclasses.asdict(value))}
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value

def decoded(value):
    if isinstance(value, dict):
        if set(value) == {'__set__'}:
            return set((decoded(v) for v in value['__set__']))
        if set(value) == {'__tuple__'}:
            return tuple((decoded(v) for v in value['__tuple__']))
        if '__dataclass__' in value:
            from .protocol.types import Pose
            from .protocol.score import Click
            constructors = {'Pose': Pose, 'Click': Click}
            return constructors[value['__dataclass__']](**decoded(value['fields']))
        return {k: decoded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [decoded(v) for v in value]
    return value

def save(path, sim, rec, agent, next_round, receipt, malformed, refused, identity, usage, terminal=None, pixel_clicks=()):
    from pxr import UsdGeom
    transforms = {}
    changed = set(sim._placed_points) | ({sim._held} if sim._held else set())
    for alias in changed:
        transforms[alias] = []
        for prim_path, baseline in sim._alias_prim_matrices.get(alias, ()):
            ops = UsdGeom.Xformable(sim._stage.GetPrimAtPath(prim_path)).GetOrderedXformOps()
            if ops:
                transforms[alias].append([prim_path, [[float(q) for q in row] for row in ops[0].Get()]])
    state = {'schema': SCHEMA, 'identity': identity, 'next_round': next_round, 'receipt': receipt, 'malformed': malformed, 'refused': refused, 'pose': vars(sim._pose), 'backend': {k: getattr(sim, k) for k in BACKEND_FIELDS}, 'transforms': transforms, 'held_visual_aabb': sim._held_visual_aabb, 'agent': {k: getattr(agent, k) for k in AGENT_FIELDS if hasattr(agent, k)}, 'recorder': {k: getattr(rec, k) for k in REC_FIELDS}, 'usage': usage, 'world_model': vars(sim._world_model) if sim._world_model is not None else None, 'terminal': terminal, 'pixel_clicks': pixel_clicks}
    atomic(path, jsonable(state))

def load(path, sim, rec, agent, identity):
    from pxr import UsdGeom, Gf
    import numpy as np
    state = decoded(json.loads(Path(path).read_text()))
    if state['schema'] != SCHEMA or jsonable(state['identity']) != identity:
        raise ValueError('episode checkpoint identity differs')
    for alias, value in state['backend']['_open_state'].items():
        if sim._open_state.get(alias) != value:
            result = sim._drive_articulation(alias, 'open' if value else 'close')
            if not result.success:
                raise ValueError('checkpoint articulation restore failed')
    for alias, rows in state['transforms'].items():
        for prim_path, matrix in rows:
            ops = UsdGeom.Xformable(sim._stage.GetPrimAtPath(prim_path)).GetOrderedXformOps()
            if not ops:
                raise ValueError('checkpoint prim transform missing')
            ops[0].Set(Gf.Matrix4d(*[q for row in matrix for q in row]))
    for k, v in state['backend'].items():
        setattr(sim, k, v)
    if state['world_model'] is not None:
        if sim._world_model is None:
            raise ValueError('checkpoint requires a world model')
        for k, v in state['world_model'].items():
            setattr(sim._world_model, k, v)
    held_box = state['held_visual_aabb']
    sim._held_visual_aabb = (held_box[0], np.asarray(held_box[1]), np.asarray(held_box[2])) if held_box else None
    if sim._held:
        sim._set_alias_visible(sim._held, False)
    set_pose(sim, state['pose'])
    for k, v in state['agent'].items():
        setattr(agent, k, v)
    for k, v in state['recorder'].items():
        setattr(rec, k, v)
    agent._vision_variants = {}
    agent.token_budget_exceeded = False
    return state
