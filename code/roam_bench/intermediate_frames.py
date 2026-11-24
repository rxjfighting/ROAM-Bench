"""Intermediate frames utilities for ROAM-Bench."""
from dataclasses import replace
import math
MAX_FRAME_URL_BYTES = 500000

def sample_motion(trace, *, spacing_m=1.0, rotation_deg=30.0, maximum=8):
    segments, total = ([], 0.0)
    for kind, start, end in trace:
        weight = math.hypot(end.x - start.x, end.z - start.z) / spacing_m if kind == 'translation' else max(abs(end[0]), abs(end[1])) / rotation_deg
        if weight > 1e-09:
            segments.append((total, weight, kind, start, end))
            total += weight
    count = min(max(0, int(maximum)), max(0, int(math.ceil(total - 1e-09)) - 1))
    poses = []
    for index in range(1, count + 1):
        target = total * index / (count + 1)
        for offset, weight, kind, start, end in segments:
            if target <= offset + weight + 1e-09:
                t = max(0.0, min(1.0, (target - offset) / weight))
                if kind == 'translation':
                    pose = replace(start, x=start.x + (end.x - start.x) * t, z=start.z + (end.z - start.z) * t, yaw_deg=end.yaw_deg, horizon_deg=end.horizon_deg)
                else:
                    pose = replace(start, yaw_deg=(start.yaw_deg + end[0] * t) % 360.0, horizon_deg=start.horizon_deg + end[1] * t)
                poses.append(pose)
                break
    return poses

def capture_intermediates(backend, poses, round_i, encode):
    final = backend._pose
    observations = backend._evidence_observations
    world_model = backend._world_model
    frames = []
    try:
        backend._world_model = None
        for i, pose in enumerate(poses, 1):
            backend._pose = pose
            backend._evidence_observations = []
            rgb = backend.observe().rgb
            url = encode(rgb)
            if len(url) > MAX_FRAME_URL_BYTES:
                raise ValueError('intermediate encoded image exceeds reserved capacity')
            evidence = [dict(row, frame_id=f'R{round_i + 1}.I{i}', source_action_round=round_i, model_visible=False) for row in backend._evidence_observations]
            frames.append({'id': f'R{round_i + 1}.I{i}', 'image': url, 'source_action_round': round_i, 'evidence': evidence})
    finally:
        backend._pose = final
        backend._world_model = world_model
        backend._evidence_observations = observations
    return frames

def publish_evidence(backend, frames, round_i):
    for frame in frames:
        for evidence in frame.get('evidence', []):
            backend._evidence_observations.append(dict(evidence, observation_index=len(backend._evidence_observations), model_visible=True, delivered_round=round_i))
