"""Check ordered relocation, stable support, and dynamic egocentric answers."""
from copy import deepcopy
from types import SimpleNamespace
from roam_bench.enhanced_evidence import judge
from roam_bench.protocol.score import Recorder, Task
from roam_bench.protocol.types import Pose


def relocation():
    return SimpleNamespace(family='relocation', truth={'object': 'jar', 'dest': 'bench'}, evidence=('jar', 'bench'), answer_instances=(), evidence_bound=None)

def trace(stable=True):
    return {'observations': [], 'placement_relations': {'jar': {'relation': 'on', 'host': 'bench'}}, 'interaction_events': [{'verb': 'pick', 'target': 'jar', 'success': True}, {'verb': 'place', 'source': 'jar', 'destination': 'bench', 'support': 'bench', 'success': True, 'stable_after_settle': stable}]}

def test_relocation_order_and_stability():
    spec = relocation()
    good = judge(spec, trace(), True)
    assert good['necessary_evidence']['status'] == good['sufficient_evidence']['status'] == 'pass'
    assert good['strict_success']
    reversed_events = trace()
    reversed_events['interaction_events'].reverse()
    wrong_order = judge(spec, reversed_events, True)
    assert wrong_order['necessary_evidence']['status'] == 'fail'
    assert not wrong_order['strict_success']
    unstable = judge(spec, trace(False), True)
    assert unstable['necessary_evidence']['status'] == 'pass'
    assert unstable['sufficient_evidence']['status'] == 'fail'
    assert unstable['strict_success']
    wrong_source = trace()
    wrong_source['interaction_events'][1]['source'] = 'other_jar'
    assert not judge(spec, wrong_source, True)['strict_success']
    assert not judge(spec, trace(), False)['strict_success']

def test_missing_telemetry_fails_verification():
    result = judge(relocation(), None, True)
    assert result['necessary_evidence']['status'] == 'fail'
    assert result['sufficient_evidence']['status'] == 'fail'
    assert not result['strict_success']

def test_path_integration_uses_final_pose():
    sim = SimpleNamespace(pose=Pose(0, 0, 0, 0))
    sim.agent_pose = lambda: sim.pose
    task = Task('path', 'return direction', answer_match='direction', answer_from_pose=('start',), answer_tolerance=5)
    recorder = Recorder(sim, task)
    sim.pose = Pose(0, 0, 2, 0)
    recorder.answer('behind')
    assert recorder.score().success
    sim.pose = Pose(0, 0, 2, 180)
    recorder.answer('ahead')
    assert recorder.score().success
    recorder.answer('behind')
    assert not recorder.score().success


def test_oracle_verdict_checks_relocation_events_even_with_correct_outcome(monkeypatch):
    from roam_bench.protocol import baselines, taskset
    from roam_bench.verdicts import verdict_for

    spec = relocation()
    spec.task = Task('relocation__jar__bench', 'Move the jar onto the bench.')
    spec.plan = ()
    evidence_trace = trace()
    # A correct final state alone does not establish a pick-before-place sequence.
    evidence_trace['interaction_events'].reverse()
    outcome = SimpleNamespace(success=1.0, answer_given=None, goals=(), to_dict=lambda: {'success': 1.0})
    rollout = SimpleNamespace(steps=(), score=lambda: outcome)
    sim = SimpleNamespace(reset=lambda episode: None, agent_pose=lambda: Pose(0, 0, 0, 0),
                          visible_instances=lambda fraction: [], exploration_evidence=lambda: evidence_trace)
    monkeypatch.setattr(taskset, 'oracle_answer_for', lambda task: None)
    monkeypatch.setattr(baselines, 'oracle_rollout', lambda *args, **kwargs: rollout)
    result = verdict_for(sim, spec, scene='offline', episode_factory=lambda *args: None, run_null=False)
    assert result['oracle_success'] == 1.0
    assert result['evidence']['necessary_evidence']['status'] == 'fail'
    assert result['evidence']['sufficient_evidence']['status'] == 'fail'
    assert result['strict_success'] is False


def test_all_released_tasks_use_the_paper_action_groups():
    import json
    from pathlib import Path
    from roam_bench.protocol.taskset import CAPABILITY_AXIS, TIER_REQUIRED
    from roam_bench.protocol.tiers import ActionSpace

    dataset = Path(__file__).resolve().parents[2] / 'dataset'
    manifest = json.loads((dataset / 'manifest.json').read_text())
    for row in manifest['tasks']:
        assert CAPABILITY_AXIS[row['family']] == row['group']
        skills = set(ActionSpace(None, TIER_REQUIRED[row['family']], refer='pixel').skills)
        assert {'move', 'strafe', 'turn', 'look', 'pick', 'place', 'open', 'close'} <= skills
        assert 'toggle' not in skills
        point_navigation = {'go_to_object_at_pixel', 'go_to_ground_at_pixel'}
        assert (point_navigation <= skills) == (row['group'] in {'G1', 'G2', 'G5'})


def test_summary_uses_all_episodes_for_gap_and_actions_and_only_S_for_error():
    from roam_bench.summary import summarize

    rows = [dict(A=True, N=False, S=False, T=False, score={'steps': 2}),
            dict(A=False, N=True, S=True, T=False, score={'steps': 5}),
            dict(A=True, N=True, S=True, T=True, score={'steps': 0})]
    result = summarize(rows)
    assert result['correctness_task_completion_gap'] == {'count': 1, 'rate': 1 / 3}
    assert result['error_after_S'] == {'numerator': 1, 'denominator': 2, 'rate': 0.5}
    assert result['average_actions'] == 7 / 3
    assert result['episodes_with_action_counts'] == 3
