"""Run a complete offline decision and validate initial-state schema."""
from pathlib import Path
import pytest
from roam_bench.cli import validate_initial_states
from roam_bench.fake_sim import FakeSim
from roam_bench.scene_facts import SceneFacts
from roam_bench.protocol.score import Task
from roam_bench.protocol.taskset import TaskSpec
from roam_bench.protocol.types import Pose
from roam_bench import harness

class OfflineClient:
    model = 'offline'
    prompt_tokens = 0
    completion_tokens = 0
    MAX_REQUEST_BYTES = 48000000
    def request_size(self, messages, token_scale=1):
        return 100
    def complete(self, messages):
        self.prompt_tokens += 10
        self.completion_tokens += 4
        return 'ACTION: answer("B")'

def test_harness_accepts_one_final_answer_without_charged_action():
    facts = SceneFacts('offline', room_types={'room': 'Room'}, room_names={'room': 'room'}, room_bounds={'room': (-2, -2, 2, 2)}, room_graph={'room': []})
    sim = FakeSim('offline', facts, visibility_model='room')
    sim.width = sim.height = 2
    task = Task('room_localization__offline', 'Choose B.', answer_keywords=(('B',),), answer_choices=('A', 'B'), max_steps=2)
    score, film, agent, meta = harness.run_model_episode(sim, TaskSpec(task, 'room_localization'), OfflineClient(), 'plan', 2, 8, Pose(0, 0, 0, 0), verbose=False)
    assert score.success
    assert score.steps == 0
    assert meta['rounds'] == 1
    assert agent.rounds[0]['raw'] == 'ACTION: answer("B")'

def test_initial_states_rejects_truthy_nonboolean_values():
    assert validate_initial_states([['drawer', 'open', False]]) == (('drawer', 'open', False),)
    with pytest.raises(ValueError):
        validate_initial_states([['drawer', 'open', 'false']])
    with pytest.raises(ValueError):
        validate_initial_states([['drawer', 'unknown', True]])
