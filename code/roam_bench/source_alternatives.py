"""Keep an accepted source consistent across all relocation stages."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
TASK_ID = 'relocation__changing_room_left_shower_slide_sandal_0__to__reception_reception_service_counter_0'
SOURCES = ('changing_room_left_shower_slide_sandal_0', 'changing_room_right_spa_shower_sandal_0')
DESTINATION = 'reception_reception_service_counter_0'
POLICY_NAME = 'either-shower-sandal-same-object-v1'

def applies(task):
    return getattr(task, 'task_id', None) == TASK_ID and any((SOURCES[1] in goal.expr for goal in task.goals))

def candidate_task(task, source):
    from roam_bench.protocol.score import Goal
    return replace(task, goals=(Goal(expr=f"holding('{source}')", when='ever', label='picked up'), Goal(expr=f"on('{source}','{DESTINATION}')", when='final', label='placed')))

def candidate_spec(spec, source):
    truth = deepcopy(spec.truth)
    truth['object'] = source
    truth.pop('accepted_objects', None)
    bound = replace(spec.evidence_bound, must_see=frozenset((source, DESTINATION)))
    return replace(spec, task=candidate_task(spec.task, source), truth=truth, referents=(source, DESTINATION), evidence=(source, DESTINATION), evidence_bound=bound)

def alternative_family(original, policy, spec, trace):
    if not applies(getattr(spec, 'task', None)):
        return original(spec, trace)
    if tuple(spec.truth.get('accepted_objects', ())) != SOURCES or spec.truth.get('dest') != DESTINATION:
        raise ValueError('Relocation source alternatives do not match the task')
    branches = []
    for source in SOURCES:
        necessary, sufficient = original(candidate_spec(spec, source), trace)
        branches.append((source, necessary, sufficient))
    chosen = max(branches, key=lambda item: (item[1]['status'] == policy.PASS, item[2]['status'] == policy.PASS))
    source, necessary, sufficient = chosen
    necessary = dict(necessary, source_policy=POLICY_NAME, accepted_source=source, candidate_statuses={alias: dict(necessary=need['status'], sufficient=enough['status']) for alias, need, enough in branches})
    sufficient = dict(sufficient, source_policy=POLICY_NAME, accepted_source=source)
    return (necessary, sufficient)

def install():
    from roam_bench import enhanced_evidence as policy
    from roam_bench.protocol import score
    if getattr(score.Recorder, '_either_shower_sandal_installed_v1', False):
        raise ValueError('Source alternative scoring is already configured')
    original_poll, original_score = (score.Recorder.poll, score.Recorder.score)

    def poll(recorder):
        original_poll(recorder)
        if applies(recorder.task):
            for source in SOURCES:
                expression = f"holding('{source}')"
                if recorder._safe(expression):
                    recorder._ever.add(expression)

    def scored(recorder):
        task = recorder.task
        if not applies(task):
            return original_score(recorder)
        results = []
        try:
            for source in SOURCES:
                recorder.task = candidate_task(task, source)
                results.append(original_score(recorder))
        finally:
            recorder.task = task
        return max(results, key=lambda value: (value.success, value.progress))
    score.Recorder.poll, score.Recorder.score = (poll, scored)
    original_family = policy._family
    policy._family = lambda spec, trace: alternative_family(original_family, policy, spec, trace)
    policy.POLICY_VERSION += '+' + POLICY_NAME
    score.Recorder._either_shower_sandal_installed_v1 = True
    return dict(policy=POLICY_NAME, task_id=TASK_ID, accepted_sources=list(SOURCES), different_objects_cannot_pool_pick_place_or_stability=True, destination_unchanged=True)
