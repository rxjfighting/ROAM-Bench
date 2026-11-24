"""Check odometry rendering and the episode HTML artifact."""
from pathlib import Path
from roam_bench.self_trajectory import render_self_trajectory
from roam_bench.protocol.episode_report import _PAGE
from roam_bench.backend import ROAMBackend
from types import SimpleNamespace
import numpy as np
import pytest
from roam_bench.protocol.types import Pose

def test_reference_view_prefers_visible_face_and_can_keep_a_narrow_object():
    from roam_bench.goal_image import best_shot, natural_projection_is_clear

    edge_view = dict(frac=.12, bbox_w=62, bbox_h=607, clipped=False)
    front_view = dict(frac=.08, bbox_w=200, bbox_h=240, clipped=False)
    assert natural_projection_is_clear(edge_view) is False
    assert natural_projection_is_clear(front_view) is True
    assert best_shot([edge_view, front_view]) is front_view
    assert best_shot([edge_view]) is edge_view

def test_odometry_has_expected_shape_and_no_scene_geometry():
    rgb, meta = render_self_trajectory([{'x': 0, 'z': 0, 'yaw_deg': 0}, {'x': 1, 'z': 2, 'yaw_deg': 90}])
    assert rgb.shape == (480, 640, 3)
    assert meta['pose_count'] == 2
    assert not meta['contains_scene_geometry']

def test_html_template_retains_record_fields():
    html = _PAGE.format(title='Episode', back='', intro='Task', head='Metrics', rows='Recorded row', video='', n=1, cols='Columns')
    assert '<!doctype html>' in html
    assert all(text in html for text in ('Episode', 'Task', 'Metrics', 'Recorded row', 'Columns'))

def test_motion_video_waits_for_lighting_without_advancing_pose():
    calls = []
    frame = np.full((2, 3, 4), 120, dtype=np.uint8)
    backend = object.__new__(ROAMBackend)
    backend._pose = object()
    original_pose = backend._pose
    backend._ensure_render_product = lambda: None
    backend._pose_to_camera = lambda pose: ('eye', 'target')
    backend._sync_embodiment_pose = lambda: None
    backend.set_camera_lookat = lambda *args: None
    backend._rep = SimpleNamespace(orchestrator=SimpleNamespace(step=lambda **kwargs: calls.append(kwargs)))
    backend._rgb_annot = SimpleNamespace(get_data=lambda: frame)
    assert backend._film_render_rgb().shape == (2, 3, 3)
    assert calls[-1] == {'rt_subframes': 16}
    backend.set_film_quality(8)
    backend._film_render_rgb(4)
    assert calls[-1] == {'rt_subframes': 8}
    backend._film_render_rgb(32)
    assert calls[-1] == {'rt_subframes': 32}
    assert backend._pose is original_pose

@pytest.mark.parametrize('fps', [12, 24])
def test_motion_sampling_preserves_speed_and_exact_end_pose(fps):
    backend = object.__new__(ROAMBackend)
    backend.set_film_rate(fps)
    poses = []
    backend.on_frame = lambda frame: poses.append(backend._pose)
    backend._film_render_rgb = lambda: np.zeros((2, 3, 3), dtype=np.uint8)
    start = Pose(x=0, y=0, z=0, yaw_deg=358)
    end = Pose(x=0.95, y=0, z=0, yaw_deg=358)
    backend._film_translation(start, end)
    assert len(poses) / fps == pytest.approx(10 / 12)
    assert poses[-1] == end
    assert max(b.x - a.x for a, b in zip([start] + poses, poses)) <= 1.2 / fps
    poses.clear()
    backend._film_rotation(end, yaw_delta=12, pitch_delta=6)
    assert len(poses) / fps == pytest.approx(3 / 12)
    assert poses[-1].yaw_deg == 10
    assert poses[-1].horizon_deg == 6
    delta = [((b.yaw_deg - a.yaw_deg + 180) % 360 - 180) for a, b in zip([end] + poses, poses)]
    assert max(delta) <= 60 / fps

def test_video_holds_scale_without_duplicating_motion_samples():
    backend = object.__new__(ROAMBackend)
    backend.set_film_rate(24)
    emitted = []
    backend.on_frame = emitted.append
    backend._film_render_rgb = lambda: np.zeros((2, 3, 3), dtype=np.uint8)
    backend._film_frame(repeat=8)
    assert len(emitted) / 24 == 8 / 12
    emitted.clear()
    backend._film_frame(sampled=True)
    assert len(emitted) == 1

def test_video_rate_does_not_change_model_motion_trace():
    backend = object.__new__(ROAMBackend)
    backend.set_film_rate(24)
    backend.on_frame = None
    backend.begin_model_motion_trace()
    start = Pose(x=0, y=0, z=0, yaw_deg=0)
    end = Pose(x=1, y=0, z=2, yaw_deg=0)
    backend._film_translation(start, end)
    backend._film_rotation(end, yaw_delta=10, pitch_delta=5)
    assert backend._pose == Pose(x=1, y=0, z=2, yaw_deg=10, horizon_deg=5)
    assert backend.finish_model_motion_trace() == [('translation', start, end), ('rotation', end, (10, 5))]
