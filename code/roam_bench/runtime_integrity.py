"""Runtime integrity utilities for ROAM-Bench."""
import types
from copy import deepcopy
import math

def require_baselines(art):
    names = [joint.joint_name for joint in art['plan'].joints]
    closed = art.get('closed_pose') or {}
    if not names or len(set(names)) != len(names) or any((name not in closed for name in names)):
        raise RuntimeError('missing or ambiguous actual closed joint baseline')
    for name in names:
        position, quaternion = closed[name]
        if len(position) != 3 or len(quaternion) != 4 or (not all((math.isfinite(float(value)) for value in (*position, *quaternion)))):
            raise RuntimeError('invalid actual closed joint baseline: ' + name)
    return set(names)

def reached(art, rows):
    names = require_baselines(art)
    return len(rows) == len(names) and {row.get('joint') for row in rows} == names and all((row.get('reached_target') is True and math.isfinite(float(row.get('measured', float('nan')))) and math.isfinite(float(row.get('target', float('nan')))) for row in rows))

def converge(backend, alias, action, set_action, plan_interact, pose_within, *, extra_steps=2000, chunk_steps=120):
    if action not in ('open', 'close') or extra_steps < 0 or chunk_steps <= 0:
        raise ValueError('invalid action or convergence budget')
    backend._ensure_sim_ctx()
    art = backend._articulations[alias]
    require_baselines(art)
    set_action(art['drives'], art['plan'], action)
    checkpoints = []
    count = 0
    budget = backend._interact_settle + extra_steps
    while count < budget:
        steps = backend._interact_settle if count == 0 else min(chunk_steps, budget - count)
        for _ in range(steps):
            backend._sim_ctx.step(render=False)
        count += steps
        rows = backend._read_articulation(art, action, plan_interact, pose_within)
        checkpoints.append(dict(steps=count, readouts=deepcopy(rows)))
        if reached(art, rows):
            break
    success = reached(art, rows)
    art['last_readouts'] = rows
    if success:
        art['last_action'] = action
        backend._open_state[alias] = action == 'open'
    return dict(success=success, checkpoints=checkpoints, total_steps=count, original_steps=backend._interact_settle, extra_steps=count - backend._interact_settle, tolerance_changed=False, target_changed=False, reset_used=False)

def restore(backend, set_action, plan_interact, pose_within, *, extra_steps=2000, chunk_steps=120):
    arts = backend._articulations or {}
    if not arts:
        return dict(success=True, no_articulations=True, checkpoints=[])
    backend._ensure_sim_ctx()
    before = {alias: backend._read_articulation(art, 'close', plan_interact, pose_within) for alias, art in arts.items()}
    dirty = any((not reached(art, before[alias]) or backend._open_state.get(alias, False) or art.get('last_action') == 'open' for alias, art in arts.items()))
    if not dirty:
        return dict(success=True, checkpoints=[], reset_used=False)
    if extra_steps < 0 or chunk_steps <= 0:
        raise ValueError('invalid restore budget')
    for art in arts.values():
        set_action(art['drives'], art['plan'], 'close')
    backend._sim_ctx.reset()
    checkpoints, count = ([], 0)
    budget = backend._interact_settle + extra_steps
    while count < budget:
        steps = backend._interact_settle if count == 0 else min(chunk_steps, budget - count)
        for _ in range(steps):
            backend._sim_ctx.step(render=False)
        count += steps
        rows = {alias: backend._read_articulation(art, 'close', plan_interact, pose_within) for alias, art in arts.items()}
        checkpoints.append(dict(steps=count, readouts=deepcopy(rows)))
        if all((reached(art, rows[alias]) for alias, art in arts.items())):
            break
    for alias, art in arts.items():
        art['last_readouts'] = rows[alias]
    failures = [alias for alias, art in arts.items() if not reached(art, rows[alias])]
    result = dict(success=not failures, failed_aliases=failures, checkpoints=checkpoints, reset_used=True, tolerance_changed=False, target_changed=False, total_steps=count)
    if failures:
        error = RuntimeError('actual articulation restore failed: ' + ', '.join(failures))
        error.diagnostic = result
        raise error
    for alias, art in arts.items():
        art['last_action'] = 'close'
        backend._open_state[alias] = False
    return result

def configure_sleep(backend, alias):
    from pxr import PhysxSchema
    art = backend._articulations[alias]
    plan = art['plan']
    rows = []
    targets = [(plan.instance_root, 'articulation', PhysxSchema.PhysxArticulationAPI)]
    targets += [(link.prim_path, 'rigid_body', PhysxSchema.PhysxRigidBodyAPI) for link in plan.links]
    for path, schema, api_class in targets:
        prim = backend._stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            raise RuntimeError('Missing articulation body: ' + path)
        api = api_class.Apply(prim)
        previous = api.GetSleepThresholdAttr().Get()
        attribute = api.CreateSleepThresholdAttr(0.0)
        if attribute.Get() != 0.0:
            raise RuntimeError('Sleep policy was not authored: ' + path)
        rows.append(dict(alias=alias, path=path, schema=schema, previous=previous, new=0.0))
    return rows

def _install_core(backend):
    if getattr(backend, '_sim_ctx', None) is not None or getattr(backend, '_articulations', {}) or getattr(backend, '_episode', None) is not None:
        raise RuntimeError('Configure scene integrity before the first episode')
    if getattr(backend, '_roam_candidate_version', None):
        raise RuntimeError('Candidate already installed')
    original_author = backend._author_articulation
    original_read = backend._read_articulation

    def author(self, inst, *args, **kwargs):
        original_author(inst, *args, **kwargs)
        self._roam_sleep_policy.extend(configure_sleep(self, inst.name))

    def read(self, art, action, plan_interact, pose_within):
        require_baselines(art)
        return original_read(art, action, plan_interact, pose_within)

    def reset_articulations(self):
        from roam_bench import artic_stage
        from roam_bench.joint_interact import plan_interact, pose_within
        result = restore(self, artic_stage.set_action, plan_interact, pose_within)
        self._roam_last_restore = result
    backend._author_articulation = types.MethodType(author, backend)
    backend._read_articulation = types.MethodType(read, backend)
    backend._restore_articulations_for_reset = types.MethodType(reset_articulations, backend)
    backend._roam_sleep_policy = []
    backend._roam_candidate_version = 'roam-scene-integrity-v1'
    backend._roam_physical_release_accepted = False
    return backend

def install_candidate(backend):
    _install_core(backend)
    original_drive = backend._drive_articulation

    def drive(self, alias, action):
        self._ensure_sim_ctx()
        context = self._sim_ctx
        was_playing = bool(context.is_playing())
        before_flags = dict(getattr(self, '_open_state', {}))
        before_time = float(context.current_time)
        if not was_playing:
            context.play()
        if not context.is_playing():
            raise RuntimeError('Physics timeline did not resume after rendering')
        if dict(getattr(self, '_open_state', {})) != before_flags:
            raise RuntimeError('Timeline handoff unexpectedly changed state flags')
        row = dict(alias=alias, action=action, was_playing=was_playing, resumed=not was_playing, reset_used=False, original_step_budget=self._interact_settle, before_time=before_time, after_play_time=float(context.current_time), completed=False)
        self._roam_handoffs.append(row)
        result = original_drive(alias, action)
        row.update(completed=True, success=bool(result.success), after_drive_time=float(context.current_time))
        return result
    backend._drive_articulation = types.MethodType(drive, backend)
    backend._roam_handoffs = []
    backend._roam_candidate_version = 'roam-scene-integrity-v1'
    return backend
