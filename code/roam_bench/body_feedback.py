"""Report body calibration and measured contact along attempted movement."""
from contextvars import ContextVar
from copy import deepcopy
import math
MIN = (-0.18158560618758202, -0.15257411628961565, -0.002272858321666682)
MAX = (0.18157560750842094, 0.3790984490476395, 1.3205725932121277)
POLICY_NAME = 'roam-body-feedback-v1'
TRACE = ContextVar('roam_bench_local_contact', default=None)
BODY = dict(version=POLICY_NAME, model='Unitree G1 / Inspire five-finger / fixed evaluation pose', length_front_back_m=MAX[1] - MIN[1], width_left_right_m=MAX[0] - MIN[0], height_m=MAX[2] - MIN[2], forward_extent_from_base_m=MAX[1], backward_extent_from_base_m=-MIN[1], left_extent_from_base_m=-MIN[0], right_extent_from_base_m=MAX[0], note='Visible fixed-pose bounds. Collision uses mesh boxes plus safety margins, not half the total length.')
BODY_TEXT = 'Robot body calibration: Unitree G1 with Inspire five-finger hands, fixed evaluation pose. Visible body length (front/back) 0.532 m, width (left/right) 0.363 m, height 1.323 m. Relative to the base reference: front 0.379 m, back 0.153 m, left/right about 0.182 m. These are visible-body bounds, not an exact collision radius; mesh collision boxes and safety margins apply. When a translation is blocked, Local contact range may report the attempted direction, the remaining collision-envelope travel to the blocking surface, and the base-to-contact projected range. The latter is along that direction to the actual blocking contact, not a nearest-wall reading or a room map. Unavailable contact range is unknown, never zero. No contact readings are provided for unattempted directions.'

def vector(value):
    try:
        result = tuple((float(value[i]) for i in range(3)))
        return result if all((math.isfinite(v) for v in result)) else None
    except (TypeError, ValueError, KeyError, IndexError, OverflowError):
        return None

def hit_value(hit, key, default=None):
    return hit.get(key, default) if isinstance(hit, dict) else getattr(hit, key, default)

class QueryObserver:

    def __init__(self, real, trace):
        self.real, self.trace = (real, trace)

    def __getattr__(self, name):
        method = getattr(self.real, name)
        if name not in ('sweep_box_closest', 'sweep_sphere_closest'):
            return method

        def call(*args, **kwargs):
            hit = method(*args, **kwargs)
            if hit_value(hit, 'hit', False):
                distance = hit_value(hit, 'distance')
                try:
                    distance = float(distance)
                    if math.isfinite(distance) and distance >= 0:
                        self.trace.append(dict(distance_m=distance, position=vector(hit_value(hit, 'position'))))
                except (TypeError, ValueError, OverflowError):
                    pass
            return hit
        return call

def observe_queries():
    import omni.physx
    if getattr(omni.physx, '_roam_bench_contact_observer_v1', False):
        return
    original = omni.physx.get_physx_scene_query_interface

    def get_query():
        query = original()
        trace = TRACE.get()
        return query if trace is None else QueryObserver(query, trace)
    omni.physx.get_physx_scene_query_interface = get_query
    omni.physx._roam_bench_contact_observer_v1 = True

def contact_result(verb, requested, result, pose, trace):
    if verb not in ('move', 'strafe') or not getattr(result, 'collided', False):
        return None
    if not trace:
        return dict(version=POLICY_NAME, direction=('front' if requested >= 0 else 'back') if verb == 'move' else 'right' if requested >= 0 else 'left', available=False, reason='No usable contact telemetry; distance is unknown.')
    hit = min(trace, key=lambda row: row['distance_m'])
    moved = abs(float(getattr(result, 'moved_m', 0.0)))
    remaining = max(0.0, hit['distance_m'] - moved)
    yaw = math.radians(float(pose.yaw_deg))
    sign = 1 if requested >= 0 else -1
    direction = (sign * math.sin(yaw), sign * math.cos(yaw), 0.0) if verb == 'move' else (sign * math.cos(yaw), -sign * math.sin(yaw), 0.0)
    point = hit['position']
    base_to_contact = None
    if point is not None:
        projected = (point[0] - float(pose.x)) * direction[0] + (point[1] - float(pose.z)) * direction[1]
        if math.isfinite(projected) and projected >= 0:
            base_to_contact = projected
    return dict(version=POLICY_NAME, direction=('front' if requested >= 0 else 'back') if verb == 'move' else 'right' if requested >= 0 else 'left', available=True, collision_envelope_remaining_travel_m=remaining, base_to_contact_projected_m=base_to_contact, reference='Final base reference; range along attempted movement direction to the blocking contact.', geometry='Existing mesh collision boxes and safety skin; not visible body-surface gap.', obstacle_identity_not_reported=True, unattempted_directions_not_probed=True)

def contact_text(data):
    if not data:
        return ''
    side = data['direction']
    if not data.get('available'):
        return f'Local contact range ({side}): unknown; contact telemetry unavailable.'
    line = f"Local contact range ({side}): collision-envelope remaining travel to the blocking surface {data['collision_envelope_remaining_travel_m']:.3f} m"
    range_m = data.get('base_to_contact_projected_m')
    line += f'; base-reference to blocking contact, projected along {side}, {range_m:.3f} m.' if range_m is not None else '; base-to-contact range unknown.'
    return line + ' This is local collision feedback, not visible-skin clearance or the nearest wall in the room.'

def install_backend(module):
    cls = module.ROAMBackend
    if getattr(cls, '_body_contact_feedback_v1', False):
        return
    original_sweep = cls._capsule_sweep_distance

    def sweep(self, *args, **kwargs):
        observe_queries()
        return original_sweep(self, *args, **kwargs)
    cls._capsule_sweep_distance = sweep
    for verb in ('move', 'strafe'):
        original = getattr(cls, verb)

        def motion(self, value, _original=original, _verb=verb):
            records = []
            token = TRACE.set(records)
            try:
                result = _original(self, value)
                result.local_contact = contact_result(_verb, float(value), result, self.agent_pose(), records)
                return result
            finally:
                TRACE.reset(token)
        setattr(cls, verb, motion)
    cls._body_contact_feedback_v1 = True

def install_feedback(module):
    cls = module.Feedback
    if getattr(cls, '_body_contact_feedback_v1', False):
        return
    original = cls.for_action

    def for_action(self, result, verb=None):
        line = original(self, result, verb)
        if self.mode == 'none':
            return line
        note = contact_text(getattr(result, 'local_contact', None))
        return line + (' ' + note if note else '')
    cls.for_action = for_action
    cls._body_contact_feedback_v1 = True

def install_prompt(module):
    if not getattr(module, '_body_contact_feedback_v1', False):
        module._SYSTEM += '\n\n' + BODY_TEXT + '\n'
        module._body_contact_feedback_v1 = True
