"""Legal oracle utilities for ROAM-Bench."""
from __future__ import annotations
import math
from typing import Iterable
from roam_bench.protocol.types import ActionResult, NameRef, PixelRef
VERSION = 'tier-legal-oracle-v3-rebuilt-tour-opt17'
INTERACTION_STANCE_RADIUS_M = 1.4
INTERACTION_STANCE_EXCLUSION_M = 0.45
INTERACTION_VERIFY_RADIUS_M = 1.55
GROUND_WAYPOINT_REACHED_M = 0.35
CRUISE_HORIZON_DEG = 10.0
CRUISE_MAX_LEG_M = 3.0
LEGAL_VERBS = {('plan', 'name'): frozenset({'go_to_visible_object', 'navigate_to_relative', 'move', 'turn', 'look', 'pick', 'place', 'open', 'close', 'toggle', 'propose'}), ('plan', 'pixel'): frozenset({'go_to_object_at_pixel', 'go_to_ground_at_pixel', 'move', 'turn', 'look', 'pick', 'place', 'open', 'close', 'toggle', 'propose'}), ('spatial', 'name'): frozenset({'move', 'turn', 'look', 'pick', 'place', 'open', 'close', 'toggle', 'propose'}), ('spatial', 'pixel'): frozenset({'move', 'turn', 'look', 'pick', 'place', 'open', 'close', 'toggle', 'propose'})}

class OracleBudgetExceeded(RuntimeError):
    oracle_budget_exceeded = True

    def __init__(self, recorder, budget: int):
        super().__init__(f'public action budget exceeded ({len(recorder.steps)} > {budget})')
        self.recorder = recorder
        self.budget = int(budget)

def required_tier(family: str) -> str:
    from roam_bench.protocol.taskset import TIER_REQUIRED
    try:
        return str(TIER_REQUIRED[family])
    except KeyError as exc:
        raise ValueError(f'no required oracle tier declared for family {family!r}') from exc

def _alias(ref) -> str:
    return str(ref.name if isinstance(ref, NameRef) else ref)

def _wrap180(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0

def _turn_to(start, end, yaw_deg: float) -> float:
    dx, dz = (float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    target_yaw = math.degrees(math.atan2(dx, dz)) % 360.0
    return _wrap180(float(yaw_deg) - target_yaw)

def _segment_yaw(start, end) -> float:
    return math.degrees(math.atan2(float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))) % 360.0

def _compress(polyline: Iterable, *, max_segment_m: float | None=None) -> list[tuple]:
    pts = [(float(p[0]), float(p[1])) for p in polyline]
    if len(pts) < 2:
        return pts
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = (out[-1], pts[i], pts[i + 1])
        ab, bc = ((b[0] - a[0], b[1] - a[1]), (c[0] - b[0], c[1] - b[1]))
        cross = ab[0] * bc[1] - ab[1] * bc[0]
        dot = ab[0] * bc[0] + ab[1] * bc[1]
        if abs(cross) > 1e-05 or dot < 0.0:
            out.append(b)
    out.append(pts[-1])
    if not max_segment_m:
        return out
    split = [out[0]]
    for end in out[1:]:
        start = split[-1]
        dist = math.dist(start, end)
        count = max(1, int(math.ceil(dist / max_segment_m)))
        for j in range(1, count + 1):
            t = j / count
            split.append((start[0] + (end[0] - start[0]) * t, start[1] + (end[1] - start[1]) * t))
    return split

def _waypoint_needs_turn_clearance(points, index: int, threshold_deg: float=8.0) -> bool:
    if index <= 0 or index >= len(points) - 1:
        return True
    incoming = math.atan2(points[index][0] - points[index - 1][0], points[index][1] - points[index - 1][1])
    outgoing = math.atan2(points[index + 1][0] - points[index][0], points[index + 1][1] - points[index][1])
    return abs(_wrap180(math.degrees(incoming - outgoing))) > threshold_deg

def prepare_plan(plan, *, must_open=(), reader_managed_open: bool=False):
    rows = list(plan or ())
    if reader_managed_open:
        hosts = set(map(str, must_open or ()))
        return [(verb, ref) for verb, ref in rows if not (str(verb).strip().lower() in {'navigate', 'open'} and _alias(ref) in hosts)]
    already_opened = {_alias(ref) for verb, ref in rows if str(verb).strip().lower() == 'open'}
    prefix = []
    for alias in sorted(map(str, must_open or ())):
        if alias in already_opened:
            continue
        prefix.extend((('navigate', NameRef(alias)), ('open', NameRef(alias))))
    return prefix + rows

class _RecorderFacade:

    def __init__(self, recorder, sim_proxy):
        object.__setattr__(self, '_rec', recorder)
        object.__setattr__(self, '_sim', sim_proxy)

    def step(self, kind, arg, result):
        pending = self._sim._pending.pop(id(result), None)
        event = pending[1] if pending is not None and pending[0] is result else None
        if event in ('navigation', 'skipped_interaction'):
            return result
        if isinstance(event, PixelRef):
            arg = event
        value = self._rec.step(kind, arg, result)
        self._sim._check_budget()
        return value

    def __getattr__(self, name):
        return getattr(self._rec, name)

    def __setattr__(self, name, value):
        setattr(self._rec, name, value)

class LegalOracleSim:

    def __init__(self, sim, recorder, *, tier: str, refer: str, max_steps: int | None=None):
        if (tier, refer) not in LEGAL_VERBS:
            raise ValueError(f'unsupported legal oracle combination: {tier}/{refer}')
        self._sim = sim
        self._rec = recorder
        self.tier = tier
        self.refer = refer
        self.max_steps = None if max_steps is None else int(max_steps)
        self._pending: dict[int, tuple[object, object]] = {}
        self.recorder = _RecorderFacade(recorder, self)

    def _check_budget(self):
        if self.max_steps is not None and len(self._rec.steps) > self.max_steps:
            raise OracleBudgetExceeded(self._rec, self.max_steps)

    def _record(self, kind, arg, result):
        value = self._rec.step(kind, arg, result)
        self._check_budget()
        return value

    def __getattr__(self, name):
        return getattr(self._sim, name)

    def _step(self, kind, arg, fn):
        aim = getattr(self._sim, 'aim_at_pixel', None)
        if callable(aim):
            if kind in ('go_to_object_at_pixel', 'go_to_ground_at_pixel'):
                aim(int(arg[0]), int(arg[1]))
            else:
                aim()
        result = fn()
        self._record(kind, arg, result)
        return result

    def _turn(self, degrees: float):
        if abs(degrees) <= 0.75:
            return None
        return self._step('turn', float(degrees), lambda: self._sim.turn(float(degrees)))

    def _turn_toward(self, start, end) -> ActionResult | None:
        pose = self._sim.agent_pose()
        requested = _turn_to(start, end, float(pose.yaw_deg))
        first = self._turn(requested)
        if first is None:
            return None
        applied = float(getattr(first, 'turned_deg', 0.0))
        if first.success and (not first.collided) and (abs(applied - requested) <= 2.0):
            return first
        pose = self._sim.agent_pose()
        remaining = _turn_to((float(pose.x), float(pose.z)), end, float(pose.yaw_deg))
        if abs(remaining) <= 0.75:
            return first
        alternate = remaining - math.copysign(360.0, remaining)
        second = self._turn(alternate)
        if second is None:
            return first
        second_applied = float(getattr(second, 'turned_deg', 0.0))
        if second.success and (not second.collided) and (abs(second_applied - alternate) <= 2.0):
            return second
        return second

    def turn(self, degrees: float):
        requested = float(degrees)
        result = self._sim.turn(requested)
        applied = float(getattr(result, 'turned_deg', 0.0))
        complete = bool(result.success) and (not bool(result.collided)) and (abs(applied - requested) <= 2.0)
        if complete:
            return result
        return ActionResult(False, result.reason, moved_m=result.moved_m, turned_deg=applied, collided=bool(result.collided), keyframe_idx=result.keyframe_idx, requested_m=result.requested_m)

    def _look_to(self, horizon: float):
        pose = self._sim.agent_pose()
        delta = float(pose.horizon_deg) - float(horizon)
        if abs(delta) <= 0.75:
            return None
        return self._step('look', delta, lambda: self._sim.look(delta))

    def _route(self, ref, *, max_goal_distance_m=None, excluded_goal_xz=(), preferred_goal_xz=None):
        query = getattr(self._sim, 'oracle_path_to_object', None)
        if not callable(query):
            raise RuntimeError('legal oracle requires backend.oracle_path_to_object; refusing to fall back to privileged navigate_to_object')
        if max_goal_distance_m is None and (not excluded_goal_xz):
            return query(ref)
        try:
            return query(ref, max_goal_distance_m=max_goal_distance_m, excluded_goal_xz=tuple(excluded_goal_xz), preferred_goal_xz=preferred_goal_xz)
        except TypeError:
            try:
                return query(ref, max_goal_distance_m=max_goal_distance_m, excluded_goal_xz=tuple(excluded_goal_xz))
            except TypeError:
                try:
                    return query(ref, max_goal_distance_m=max_goal_distance_m)
                except TypeError:
                    return query(ref)

    def _aim(self, alias: str, *, need_pixel: bool):
        if need_pixel:
            pixel = self._sim.instance_centroid(alias)
            if pixel is not None:
                return tuple(map(int, pixel))
        bearing = self._sim.bearing_to(alias)
        if bearing is not None:
            self._turn(float(bearing[0]))
        if not need_pixel:
            return None
        pixel = self._sim.instance_centroid(alias)
        if pixel is not None:
            return tuple(map(int, pixel))
        for horizon in (0.0, 30.0, 60.0, -30.0):
            self._look_to(horizon)
            pixel = self._sim.instance_centroid(alias)
            if pixel is not None:
                return tuple(map(int, pixel))
        return None

    def _interaction_pixel(self, alias: str, verb: str):
        normalized = str(verb).strip().lower()
        ordinary = self._aim(alias, need_pixel=True)
        if ordinary is None:
            return None
        selector = getattr(self._sim, 'oracle_interaction_pixel', None)
        if callable(selector):
            selected = selector(alias, normalized)
            if selected is not None:
                return tuple(map(int, selected))
            if normalized in ('place', 'put', 'drop'):
                for horizon in (30.0, 60.0, 0.0, -30.0):
                    self._look_to(horizon)
                    if self._sim.instance_centroid(alias) is None:
                        continue
                    selected = selector(alias, normalized)
                    if selected is not None:
                        return tuple(map(int, selected))
                return None
        return ordinary

    def _ground_pixel(self, point, *, tolerance_m=None, require_turn_clearance=False, turn_yaws=None, excluded_xz=()):
        query = self._sim.oracle_ground_pixel
        if tolerance_m is None and (not require_turn_clearance) and (turn_yaws is None):
            return query(*point)
        try:
            kwargs = {'require_turn_clearance': bool(require_turn_clearance)}
            if turn_yaws is not None:
                kwargs.update(turn_from_yaw_deg=float(turn_yaws[0]), turn_to_yaw_deg=float(turn_yaws[1]))
            if tolerance_m is not None:
                kwargs['tolerance_m'] = tolerance_m
            if excluded_xz:
                kwargs['excluded_xz'] = tuple(excluded_xz)
            return query(*point, **kwargs)
        except TypeError:
            if 'excluded_xz' in kwargs:
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop('excluded_xz', None)
                try:
                    return query(*point, **retry_kwargs)
                except TypeError:
                    pass
            try:
                return query(*point, tolerance_m=tolerance_m)
            except TypeError:
                return query(*point)

    def _interaction_distance(self, alias: str, pixel=None):
        query = getattr(self._sim, 'oracle_interaction_distance', None)
        if callable(query):
            return query(alias, pixel)
        bearing = self._sim.bearing_to(alias)
        return None if bearing is None else float(bearing[1])

    def _in_alias_room(self, alias: str) -> bool:
        room_query = getattr(self._sim, 'agent_room', None)
        inventory_query = getattr(self._sim, 'inventory', None)
        if not callable(room_query) or not callable(inventory_query):
            return True
        fact = next((row for row in inventory_query() if row.alias == alias), None)
        if fact is None:
            return True
        expected = {value for value in (fact.room_id, fact.room) if value}
        current = room_query()
        if current is None:
            return not expected
        return not expected or current in expected

    def _failed_probe(self, alias: str) -> ActionResult:
        bearing = self._sim.bearing_to(alias)
        if bearing is None:
            return ActionResult(False, 'The target has no resolvable location.')
        deg, distance = map(float, bearing)
        if self.tier == 'spatial':
            self._turn(deg)
            result = self._step('move', max(0.25, min(distance, 1.5)), lambda: self._sim.move(max(0.25, min(distance, 1.5))))
        elif self.refer == 'name':
            result = self._step('navigate_to_relative', (deg, max(0.25, min(distance, 1.5))), lambda: self._sim.navigate_to_relative(deg, max(0.25, min(distance, 1.5))))
        else:
            self._turn(deg)
            self._look_to(55.0)
            pose = self._sim.agent_pose()
            d = max(0.25, min(distance, 1.25))
            yaw = math.radians(float(pose.yaw_deg))
            point = (pose.x + d * math.sin(yaw), pose.z + d * math.cos(yaw))
            pixel = self._sim.oracle_ground_pixel(*point)
            if pixel is None:
                return ActionResult(False, 'No visible floor pixel leads toward the target.')
            result = self._step('go_to_ground_at_pixel', pixel, lambda: self._sim.go_to_ground_at_pixel(*pixel))
        return ActionResult(False, 'No connected legal route reaches the target.', moved_m=float(getattr(result, 'moved_m', 0.0)), collided=bool(getattr(result, 'collided', False)))

    def _plan_pixel_route(self, alias: str, *, max_goal_distance_m=None, excluded_goal_xz=(), prepared_route=None, interaction_verb='') -> ActionResult:
        moved = 0.0
        first_route = prepared_route
        active_goal = None if prepared_route is None else tuple(map(float, prepared_route['goal_xz']))
        rejected_goals = [tuple(map(float, q)) for q in excluded_goal_xz]
        interaction_route = max_goal_distance_m is not None
        max_leg_m = 0.65
        last_failure = None
        ever_collided = False
        escaped_turn_poses = set()
        failed_turn_routes = {}
        initial_pose = self._sim.agent_pose()
        visited_xz = [(float(initial_pose.x), float(initial_pose.z))]
        for _attempt in range(64):
            route = first_route
            first_route = None
            if route is None:
                route = self._route(NameRef(alias), max_goal_distance_m=max_goal_distance_m, excluded_goal_xz=rejected_goals, preferred_goal_xz=active_goal)
            if route is None:
                if last_failure is not None:
                    return ActionResult(False, last_failure.reason, moved_m=moved, collided=ever_collided)
                failed = self._failed_probe(alias)
                failed.moved_m += moved
                return failed
            pose = self._sim.agent_pose()
            goal = tuple(map(float, route['goal_xz']))
            if active_goal is None or math.dist(active_goal, goal) > 0.05:
                active_goal = goal
            if any((math.dist(goal, old) <= 0.05 for old in rejected_goals)):
                return ActionResult(False, 'No new visible stance remains for the target.', moved_m=moved)
            if math.dist((float(pose.x), float(pose.z)), goal) <= GROUND_WAYPOINT_REACHED_M and self._in_alias_room(alias):
                if interaction_route:
                    current = self._interaction_pixel(alias, interaction_verb)
                    distance = self._interaction_distance(alias, current)
                    if current is not None and distance is not None and (distance <= INTERACTION_VERIFY_RADIUS_M):
                        return ActionResult(True, 'Reached a visible interaction stance through legal ground-pixel actions.', moved_m=moved, collided=False)
                else:
                    current = self._aim(alias, need_pixel=True)
                    if current is not None:
                        visible_query = getattr(self._sim, 'visible_instances', None)
                        if callable(visible_query):
                            from roam_bench.protocol import baselines
                            visible = set(visible_query(baselines.COUNT_PIXEL_FRAC))
                            if alias in visible:
                                return ActionResult(True, 'Reached a visible observation stance through legal ground-pixel actions.', moved_m=moved, collided=False)
                        result = self._step('go_to_object_at_pixel', current, lambda p=current: self._sim.go_to_object_at_pixel(*p))
                        moved += abs(float(getattr(result, 'moved_m', 0.0)))
                        if result.success and (not result.collided):
                            return ActionResult(True, result.reason, moved_m=moved, collided=False)
                rejected_goals.append(goal)
                active_goal = None
                continue
            raw_points = _compress(route['polyline'], max_segment_m=max_leg_m)
            candidates = []
            for point in raw_points[1:]:
                waypoint_distance = math.dist((float(pose.x), float(pose.z)), point)
                if waypoint_distance <= 0.08:
                    continue
                if waypoint_distance > max_leg_m + 0.05:
                    break
                candidates.append(point)
            if not candidates:
                rejected_goals.append(goal)
                active_goal = None
                continue
            start, desired = ((float(pose.x), float(pose.z)), candidates[-1])
            deg = _turn_to(start, desired, float(pose.yaw_deg))
            turn = self._turn_toward(start, desired)
            if turn is not None and (not turn.success or turn.collided or abs(_wrap180(float(self._sim.agent_pose().yaw_deg) - math.degrees(math.atan2(desired[0] - start[0], desired[1] - start[1])))) > 2.0):
                last_failure = turn
                ever_collided = ever_collided or bool(turn.collided)
                route_key = (round(start[0], 2), round(start[1], 2), round(desired[0], 2), round(desired[1], 2))
                failed_turn_routes[route_key] = failed_turn_routes.get(route_key, 0) + 1
                repeated_turn_route = failed_turn_routes[route_key] >= 2
                if abs(float(getattr(turn, 'turned_deg', 0.0))) <= 1.0 or repeated_turn_route:
                    stuck = self._sim.agent_pose()
                    key = (round(float(stuck.x), 1), round(float(stuck.z), 1), round(float(stuck.yaw_deg) / 5.0) * 5.0)
                    if key not in escaped_turn_poses:
                        escaped_turn_poses.add(key)
                        self._look_to(55.0)
                        yaw = math.radians(float(stuck.yaw_deg))
                        escape_point = (float(stuck.x) + 0.4 * math.sin(yaw), float(stuck.z) + 0.4 * math.cos(yaw))
                        escape_pixel = self._ground_pixel(escape_point, tolerance_m=0.3, require_turn_clearance=False, excluded_xz=visited_xz)
                        if escape_pixel is not None:
                            before_escape = self._sim.agent_pose()
                            escaped = self._step('go_to_ground_at_pixel', escape_pixel, lambda p=escape_pixel: self._sim.go_to_ground_at_pixel(*p))
                            moved += abs(float(getattr(escaped, 'moved_m', 0.0)))
                            after_escape = self._sim.agent_pose()
                            escaped_distance = math.dist((float(before_escape.x), float(before_escape.z)), (float(after_escape.x), float(after_escape.z)))
                            ever_collided = ever_collided or bool(getattr(escaped, 'collided', False))
                            if escaped_distance >= 0.05:
                                visited_xz.append((float(after_escape.x), float(after_escape.z)))
                                first_route = None
                                continue
                    if repeated_turn_route:
                        rejected_goals.append(goal)
                        active_goal = None
                        continue
                    return ActionResult(False, turn.reason, moved_m=moved, collided=ever_collided)
                continue
            ground = None
            selected_waypoint = None
            if max_leg_m >= 0.64:
                cruise_points = _compress(route['polyline'], max_segment_m=CRUISE_MAX_LEG_M)
                cruise_candidates = []
                for point in cruise_points[1:]:
                    distance = math.dist((float(pose.x), float(pose.z)), point)
                    if distance < 1.8:
                        continue
                    if distance > CRUISE_MAX_LEG_M + 0.05:
                        break
                    cruise_candidates.append(point)
                if cruise_candidates:
                    self._look_to(CRUISE_HORIZON_DEG)
                    for candidate in reversed(cruise_candidates):
                        candidate_index = cruise_points.index(candidate)
                        needs_turn = _waypoint_needs_turn_clearance(cruise_points, candidate_index)
                        turn_yaws = None
                        if needs_turn and 0 < candidate_index < len(cruise_points) - 1:
                            turn_yaws = (_segment_yaw(cruise_points[candidate_index - 1], candidate), _segment_yaw(candidate, cruise_points[candidate_index + 1]))
                        ground = self._ground_pixel(candidate, tolerance_m=0.3 if interaction_route else None, require_turn_clearance=needs_turn, turn_yaws=turn_yaws, excluded_xz=visited_xz)
                        if ground is not None:
                            raw_points = cruise_points
                            candidates = cruise_candidates
                            selected_waypoint = candidate
                            break
            if ground is None:
                self._look_to(55.0)
            for candidate in () if ground is not None else reversed(candidates):
                candidate_index = raw_points.index(candidate)
                needs_turn = _waypoint_needs_turn_clearance(raw_points, candidate_index)
                turn_yaws = None
                if needs_turn and 0 < candidate_index < len(raw_points) - 1:
                    turn_yaws = (_segment_yaw(raw_points[candidate_index - 1], candidate), _segment_yaw(candidate, raw_points[candidate_index + 1]))
                ground = self._ground_pixel(candidate, tolerance_m=0.3 if interaction_route else None, require_turn_clearance=needs_turn, turn_yaws=turn_yaws, excluded_xz=visited_xz)
                if ground is not None:
                    selected_waypoint = candidate
                    break
            if ground is None and candidates and (max_leg_m <= 0.301):
                nearest_distance = min((math.dist((float(self._sim.agent_pose().x), float(self._sim.agent_pose().z)), candidate) for candidate in candidates))
                adaptive_pitch = max(55.0, min(60.0, math.degrees(math.atan2(1.55, nearest_distance))))
                for pitch in (adaptive_pitch,):
                    self._look_to(pitch)
                    for candidate in reversed(candidates):
                        candidate_index = raw_points.index(candidate)
                        needs_turn = _waypoint_needs_turn_clearance(raw_points, candidate_index)
                        turn_yaws = None
                        if needs_turn and 0 < candidate_index < len(raw_points) - 1:
                            turn_yaws = (_segment_yaw(raw_points[candidate_index - 1], candidate), _segment_yaw(candidate, raw_points[candidate_index + 1]))
                        ground = self._ground_pixel(candidate, tolerance_m=0.3 if interaction_route else None, require_turn_clearance=needs_turn, turn_yaws=turn_yaws, excluded_xz=visited_xz)
                        if ground is not None:
                            selected_waypoint = candidate
                            break
                    if ground is not None:
                        break
            if ground is None and candidates and (max_leg_m <= 0.301):
                lookahead_points = _compress(route['polyline'], max_segment_m=1.2)
                lookahead_candidates = []
                here = self._sim.agent_pose()
                here_xz = (float(here.x), float(here.z))
                for point in lookahead_points[1:]:
                    distance = math.dist(here_xz, point)
                    if distance <= GROUND_WAYPOINT_REACHED_M:
                        continue
                    if distance > 1.25:
                        break
                    lookahead_candidates.append(point)
                if lookahead_candidates:
                    for candidate in reversed(lookahead_candidates):
                        candidate_index = lookahead_points.index(candidate)
                        needs_turn = _waypoint_needs_turn_clearance(lookahead_points, candidate_index)
                        turn_yaws = None
                        if needs_turn and 0 < candidate_index < len(lookahead_points) - 1:
                            turn_yaws = (_segment_yaw(lookahead_points[candidate_index - 1], candidate), _segment_yaw(candidate, lookahead_points[candidate_index + 1]))
                        ground = self._ground_pixel(candidate, tolerance_m=0.35 if interaction_route else None, require_turn_clearance=needs_turn, turn_yaws=turn_yaws, excluded_xz=visited_xz)
                        if ground is not None:
                            raw_points = lookahead_points
                            candidates = lookahead_candidates
                            selected_waypoint = candidate
                            break
            if ground is None:
                if max_leg_m > 0.31:
                    max_leg_m = max(0.3, max_leg_m / 2.0)
                    first_route = route
                else:
                    rejected_goals.append(goal)
                    active_goal = None
                    max_leg_m = 0.65
                last_failure = ActionResult(False, 'No currently visible floor pixel resolves to the next collision-safe route waypoint.')
                continue
            before = self._sim.agent_pose()
            result = self._step('go_to_ground_at_pixel', ground, lambda p=ground: self._sim.go_to_ground_at_pixel(*p))
            moved += abs(float(getattr(result, 'moved_m', 0.0)))
            after = self._sim.agent_pose()
            progress = math.dist((float(before.x), float(before.z)), (float(after.x), float(after.z)))
            if progress >= 0.05:
                visited_xz.append((float(after.x), float(after.z)))
            continued_route = None
            if selected_waypoint is not None and progress >= 0.05:
                before_error = math.dist((float(before.x), float(before.z)), selected_waypoint)
                after_xz = (float(after.x), float(after.z))
                after_error = math.dist(after_xz, selected_waypoint)
                if after_error + 0.02 < before_error:
                    selected_index = raw_points.index(selected_waypoint)
                    tail = list(raw_points[selected_index + 1:])
                    if after_error > GROUND_WAYPOINT_REACHED_M:
                        tail.insert(0, selected_waypoint)
                    if not tail and (math.dist(after_xz, goal) > 0.15 or not self._in_alias_room(alias)):
                        tail = [goal]
                    chain = [after_xz, *tail]
                    continued_route = {'alias': alias, 'goal_xz': goal, 'polyline': tuple(chain), 'length_m': sum((math.dist(a, b) for a, b in zip(chain[:-1], chain[1:])))}
            if not result.success or result.collided or progress < 0.05:
                last_failure = result
                ever_collided = ever_collided or bool(result.collided)
                if progress >= 0.05:
                    max_leg_m = 0.65
                    first_route = continued_route
                elif max_leg_m > 0.31:
                    max_leg_m = max(0.3, max_leg_m / 2.0)
                else:
                    rejected_goals.append(goal)
                    active_goal = None
                    max_leg_m = 0.65
                continue
            max_leg_m = 0.65
            first_route = continued_route
        if interaction_route:
            if last_failure is not None:
                return ActionResult(False, last_failure.reason, moved_m=moved, collided=ever_collided)
            return ActionResult(False, 'No visible and reachable interaction stance could be reached through legal ground-pixel actions.', moved_m=moved)
        pixel = self._aim(alias, need_pixel=True)
        if pixel is None:
            return ActionResult(False, 'Reached the stance but the target is not visible.', moved_m=moved)
        result = self._step('go_to_object_at_pixel', pixel, lambda p=pixel: self._sim.go_to_object_at_pixel(*p))
        moved += abs(float(getattr(result, 'moved_m', 0.0)))
        return ActionResult(bool(result.success and (not result.collided)), result.reason, moved_m=moved, collided=bool(result.collided))

    def navigate_to_object(self, ref, *, max_goal_distance_m=None, excluded_goal_xz=(), _prepared_route=None, _interaction_verb='', _observation_retry=0):
        alias = _alias(ref)
        if self.tier == 'plan' and self.refer == 'pixel':
            aggregate = self._plan_pixel_route(alias, max_goal_distance_m=max_goal_distance_m, excluded_goal_xz=excluded_goal_xz, prepared_route=_prepared_route, interaction_verb=_interaction_verb)
            audit = getattr(self._sim, 'record_oracle_approach', None)
            if callable(audit):
                phase = str(_interaction_verb).strip().lower() or 'navigate'
                audit(alias, aggregate, f'plan_pixel_route:{phase}')
            self._pending[id(aggregate)] = (aggregate, 'navigation')
            return aggregate
        route = _prepared_route if _prepared_route is not None else self._route(NameRef(alias), max_goal_distance_m=max_goal_distance_m, excluded_goal_xz=excluded_goal_xz)
        if route is None:
            aggregate = self._failed_probe(alias)
            audit = getattr(self._sim, 'record_oracle_approach', None)
            if callable(audit):
                audit(alias, aggregate, f'{self.tier}_{self.refer}_route')
            self._pending[id(aggregate)] = (aggregate, 'navigation')
            return aggregate
        goal = tuple(map(float, route['goal_xz']))
        ok, collided, moved = (True, False, 0.0)
        active_route = route
        for _route_attempt in range(64):
            pose = self._sim.agent_pose()
            if math.dist((float(pose.x), float(pose.z)), goal) <= GROUND_WAYPOINT_REACHED_M and self._in_alias_room(alias):
                break
            points = _compress(active_route['polyline'], max_segment_m=0.65)
            end = next((point for point in points[1:] if math.dist((float(pose.x), float(pose.z)), point) > 0.05), None)
            if end is None:
                ok = False
                break
            start = (float(pose.x), float(pose.z))
            distance = math.dist(start, end)
            deg = _turn_to(start, end, float(pose.yaw_deg))
            if self.tier == 'spatial':
                turn = self._turn_toward(start, end)
                desired_yaw = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0
                if turn is not None and (not turn.success or turn.collided or abs(_wrap180(float(self._sim.agent_pose().yaw_deg) - desired_yaw)) > 2.0):
                    ok, collided = (False, bool(turn.collided))
                    break
                result = self._step('move', distance, lambda d=distance: self._sim.move(d))
            elif self.refer == 'name':
                result = self._step('navigate_to_relative', (deg, distance), lambda d=deg, m=distance: self._sim.navigate_to_relative(d, m))
            else:
                turn = self._turn_toward(start, end)
                desired_yaw = math.degrees(math.atan2(end[0] - start[0], end[1] - start[1])) % 360.0
                if turn is not None and (not turn.success or turn.collided or abs(_wrap180(float(self._sim.agent_pose().yaw_deg) - desired_yaw)) > 2.0):
                    ok, collided = (False, bool(turn.collided))
                    break
                self._look_to(55.0)
                pixel = self._sim.oracle_ground_pixel(*end)
                if pixel is None:
                    ok = False
                    break
                result = self._step('go_to_ground_at_pixel', pixel, lambda p=pixel: self._sim.go_to_ground_at_pixel(*p))
            actual = abs(float(getattr(result, 'moved_m', 0.0)))
            moved += actual
            now = self._sim.agent_pose()
            remaining = math.dist((float(now.x), float(now.z)), end)
            arrived = remaining <= GROUND_WAYPOINT_REACHED_M
            short = actual + 0.08 < distance
            needs_replan = False
            if not result.success or ((result.collided or short) and (not arrived)):
                collided = collided or bool(result.collided)
                if actual < 0.05:
                    ok = False
                    break
                needs_replan = True
            if needs_replan:
                active_route = self._route(NameRef(alias), max_goal_distance_m=max_goal_distance_m, excluded_goal_xz=excluded_goal_xz, preferred_goal_xz=goal)
                if active_route is None:
                    ok = False
                    break
                new_goal = tuple(map(float, active_route['goal_xz']))
                if math.dist(new_goal, goal) > 0.05:
                    goal = new_goal
            else:
                after_xz = (float(now.x), float(now.z))
                tail = list(points[2:])
                if not tail and (not self._in_alias_room(alias)):
                    tail = [goal]
                active_route = {'alias': alias, 'goal_xz': goal, 'polyline': tuple([after_xz, *tail]), 'length_m': sum((math.dist(a, b) for a, b in zip([after_xz, *tail][:-1], [after_xz, *tail][1:])))}
        else:
            ok = False
        if ok and self.tier == 'plan':
            if self.refer == 'name':
                self._aim(alias, need_pixel=False)
                result = self._step('go_to_visible_object', NameRef(alias), lambda: self._sim.go_to_visible_object(NameRef(alias)))
            else:
                pixel = self._aim(alias, need_pixel=True)
                if pixel is None:
                    result = ActionResult(False, 'The target is not visible for a pixel click.')
                else:
                    result = self._step('go_to_object_at_pixel', pixel, lambda p=pixel: self._sim.go_to_object_at_pixel(*p))
            ok = bool(result.success and (not result.collided))
            collided = collided or bool(result.collided)
            moved += abs(float(getattr(result, 'moved_m', 0.0)))
            if ok and self.refer == 'pixel' and (not _interaction_verb):
                self._aim(alias, need_pixel=True)
        elif ok:
            visible_pixel = self._aim(alias, need_pixel=self.refer == 'pixel')
            if self.refer == 'pixel' and visible_pixel is None and (_prepared_route is None) and (_observation_retry < 11):
                retry = self.navigate_to_object(ref, max_goal_distance_m=1.5, excluded_goal_xz=tuple(excluded_goal_xz) + (goal,), _observation_retry=_observation_retry + 1)
                aggregate = ActionResult(bool(retry.success), retry.reason, moved_m=moved + abs(float(getattr(retry, 'moved_m', 0.0))), collided=collided or bool(getattr(retry, 'collided', False)))
                audit = getattr(self._sim, 'record_oracle_approach', None)
                if callable(audit):
                    audit(alias, aggregate, f'{self.tier}_{self.refer}_route')
                self._pending[id(aggregate)] = (aggregate, 'navigation')
                return aggregate
            if self.refer == 'pixel' and visible_pixel is None:
                ok = False
        aggregate = ActionResult(ok, 'Reached through tier-legal public actions.' if ok else 'A tier-legal public action could not reach a visible observation stance.', moved_m=moved, collided=collided)
        audit = getattr(self._sim, 'record_oracle_approach', None)
        if callable(audit):
            audit(alias, aggregate, f'{self.tier}_{self.refer}_route')
        self._pending[id(aggregate)] = (aggregate, 'navigation')
        return aggregate

    def approach_for_interaction(self, ref, verb=''):
        alias = _alias(ref)
        if self.refer == 'pixel':
            current_pixel = self._interaction_pixel(alias, verb)
            current_distance = self._interaction_distance(alias, current_pixel)
            if current_pixel is not None and current_distance is not None and (current_distance <= INTERACTION_VERIFY_RADIUS_M):
                return ActionResult(True, 'The current stance is visible and within reach.')
        excluded = []
        last = ActionResult(False, 'No visible and reachable interaction stance was found.')
        if self.tier == 'plan' and self.refer == 'pixel':
            return self.navigate_to_object(ref, max_goal_distance_m=INTERACTION_STANCE_RADIUS_M, _interaction_verb=str(verb))
        for _attempt in range(12):
            route = self._route(NameRef(alias), max_goal_distance_m=INTERACTION_STANCE_RADIUS_M, excluded_goal_xz=excluded)
            if route is None:
                break
            goal = tuple(map(float, route['goal_xz']))
            if any((math.dist(goal, old) <= INTERACTION_STANCE_EXCLUSION_M for old in excluded)):
                break
            last = self.navigate_to_object(ref, max_goal_distance_m=INTERACTION_STANCE_RADIUS_M, excluded_goal_xz=excluded, _prepared_route=route, _interaction_verb=str(verb))
            if not last.success or last.collided:
                excluded.append(goal)
                continue
            pixel = self._interaction_pixel(alias, verb) if self.refer == 'pixel' else None
            distance = self._interaction_distance(alias, pixel)
            reachable = distance is not None and distance <= INTERACTION_VERIFY_RADIUS_M
            if self.refer != 'pixel':
                if reachable:
                    return last
            elif reachable and pixel is not None:
                return last
            excluded.append(goal)
            last = ActionResult(False, 'Reached a candidate stance, but the target was not both visible and within interaction reach.', moved_m=float(getattr(last, 'moved_m', 0.0)), collided=bool(getattr(last, 'collided', False)))
        return last

    def navigate_to_relative(self, deg: float, dist_m: float):
        deg, dist_m = (float(deg), float(dist_m))
        if self.tier == 'plan' and self.refer == 'name':
            result = self._step('navigate_to_relative', (deg, dist_m), lambda: self._sim.navigate_to_relative(deg, dist_m))
        elif self.tier == 'spatial':
            turn = self._turn(deg)
            if turn is not None and (not turn.success or turn.collided or abs(float(turn.turned_deg) - deg) > 2.0):
                result = turn
            else:
                result = self._step('move', dist_m, lambda: self._sim.move(dist_m))
        else:
            turn = self._turn(deg)
            if turn is not None and (not turn.success or turn.collided or abs(float(turn.turned_deg) - deg) > 2.0):
                self._pending[id(turn)] = (turn, 'navigation')
                return turn
            moved, collided, remaining = (0.0, False, max(0.0, dist_m))
            result = ActionResult(True, 'No movement was requested.')
            while remaining > 0.1:
                leg = min(0.65, remaining)
                self._look_to(55.0)
                pose = self._sim.agent_pose()
                yaw = math.radians(float(pose.yaw_deg))
                point = (pose.x + leg * math.sin(yaw), pose.z + leg * math.cos(yaw))
                pixel = self._ground_pixel(point, tolerance_m=0.3)
                if pixel is None:
                    result = ActionResult(False, 'The requested ground point is not visible.', moved_m=moved, collided=collided)
                    break
                step = self._step('go_to_ground_at_pixel', pixel, lambda p=pixel: self._sim.go_to_ground_at_pixel(*p))
                actual = abs(float(getattr(step, 'moved_m', 0.0)))
                moved += actual
                remaining = max(0.0, remaining - actual)
                collided = collided or bool(step.collided)
                if not step.success or step.collided or actual < 0.05:
                    result = ActionResult(False, step.reason, moved_m=moved, collided=collided)
                    break
                result = ActionResult(True, 'Reached through legal ground-pixel actions.', moved_m=moved, collided=False)
        self._pending[id(result)] = (result, 'navigation')
        return result

    def interact(self, verb, ref, into=None):
        target = into if into is not None else ref
        alias = _alias(target)
        pixel = None if self.refer == 'name' else self._interaction_pixel(alias, verb)
        distance = self._interaction_distance(alias, pixel)
        reachable = distance is not None and distance <= INTERACTION_VERIFY_RADIUS_M
        if not reachable or (self.refer == 'pixel' and pixel is None):
            ready = self.approach_for_interaction(NameRef(alias), verb=verb)
            if not ready.success:
                result = ActionResult(False, ready.reason, moved_m=float(getattr(ready, 'moved_m', 0.0)), collided=bool(getattr(ready, 'collided', False)))
                self._pending[id(result)] = (result, 'skipped_interaction')
                return result
            if self.refer != 'name':
                pixel = self._interaction_pixel(alias, verb)
        if self.refer == 'name':
            return self._sim.interact(verb, ref, into)
        if pixel is None:
            result = ActionResult(False, 'The target left the current frame before interaction.')
            self._pending[id(result)] = (result, 'skipped_interaction')
            return result
        px = PixelRef(*pixel)
        aim = getattr(self._sim, 'aim_at_pixel', None)
        if callable(aim):
            aim(px.u, px.v)
        result = self._sim.interact(verb, px)
        self._pending[id(result)] = (result, px)
        return result

def rollout(sim, task, plan, answer=None, *, tier: str, refer: str, interact_horizons=None, on_step=None, max_steps: int | None=None):
    from roam_bench.protocol import baselines
    from roam_bench.protocol.score import Recorder
    rec = Recorder(sim, task, on_step=on_step)
    proxy = LegalOracleSim(sim, rec, tier=tier, refer=refer, max_steps=max_steps)
    shown = proxy.recorder
    horizons = baselines.INTERACT_HORIZONS if interact_horizons is None else interact_horizons
    plan_holding = None
    prepared_interactions = {}
    plan = list(plan)
    for index, (verb, ref) in enumerate(plan):
        if verb in ('navigate', 'go'):
            next_verb, next_ref = plan[index + 1] if index + 1 < len(plan) else (None, None)
            normalized_next = str(next_verb).strip().lower()
            if plan_holding is False and normalized_next in {'place', 'put', 'drop'}:
                continue
            same_target = next_ref is not None and _alias(next_ref) == _alias(ref)
            if same_target and normalized_next in {'pick', 'pickup', 'take', 'place', 'put', 'drop', 'open', 'close', 'toggle'}:
                result = proxy.approach_for_interaction(ref, verb=normalized_next)
                prepared_interactions[index + 1] = result
                proxy._pending[id(result)] = (result, 'navigation')
            else:
                result = proxy.navigate_to_object(ref)
            shown.step('navigate', ref, result)
        elif verb == 'move':
            proxy._record('move', ref, sim.move(float(ref)))
        elif verb == 'turn':
            proxy._record('turn', ref, sim.turn(float(ref)))
        elif verb == 'look':
            proxy._record('look', ref, sim.look(float(ref)))
        elif verb == 'propose':
            proposed = sim.propose(str(ref))
            proxy._record('propose', ref, ActionResult(proposed.accepted, proposed.reason))
        else:
            normalized = str(verb).strip().lower()
            if normalized in ('place', 'put', 'drop') and plan_holding is False:
                continue
            prepared = prepared_interactions.pop(index, None)
            if prepared is not None and (not prepared.success):
                if normalized in ('pick', 'pickup', 'take'):
                    plan_holding = False
                continue
            if prepared is None and normalized in ('pick', 'pickup', 'take', 'place', 'put', 'drop', 'open', 'close', 'toggle'):
                ready = proxy.approach_for_interaction(ref, verb=normalized)
                if not ready.success:
                    if normalized in ('pick', 'pickup', 'take'):
                        plan_holding = False
                    continue
            result = baselines._interact_sweeping(proxy, shown, verb, ref, horizons)
            if normalized in ('pick', 'pickup', 'take'):
                plan_holding = bool(result.success)
            elif normalized in ('place', 'put', 'drop') and result.success:
                plan_holding = False
    if answer is not None:
        rec.answer(answer(proxy, shown) if callable(answer) else answer)
    illegal = sorted({kind for kind, _arg, _success, _reason in rec.steps} - LEGAL_VERBS[tier, refer])
    if illegal:
        raise RuntimeError(f'legal oracle emitted forbidden {tier}/{refer} actions: {illegal}')
    return rec
