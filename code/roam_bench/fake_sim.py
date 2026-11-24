"""Provide a deterministic simulator for offline protocol checks."""
from __future__ import annotations
import logging
import math
from collections import deque
from typing import Optional
import numpy as np
from roam_bench.protocol.types import ActionResult, Obs, Pose, ProposeResult, WorldView, NameRef, PixelRef, DirectionRef, HELD_UNNAMED
from roam_bench.scene_facts import load_facts, SceneFacts
from roam_bench import scene_geom, navmesh
_AGENT_EYE_Y = 1.5
_log = logging.getLogger('roam_bench.fake_sim')

def _ref_alias(ref) -> Optional[str]:
    if isinstance(ref, NameRef):
        return ref.name
    if isinstance(ref, str):
        return ref
    return None

class FakeSim:

    def __init__(self, scene_root: str, facts: Optional[SceneFacts]=None, *, visibility_model: str):
        self.scene_root = scene_root
        self.facts = facts or load_facts(scene_root)
        self._nav = navmesh.NavMesh(self.facts)
        if visibility_model not in ('room', 'geom'):
            raise ValueError(f"visibility_model must be 'room' or 'geom', got {visibility_model!r}")
        self.visibility_model = visibility_model
        self._geom = scene_geom.load_geom(scene_root, self.facts, scene_geom.REAL_RENDER_CAM) if visibility_model == 'geom' else None
        self._step = 0
        self._held: Optional[str] = None
        self._open: dict = {}
        self._pos_override: dict = {}
        self._seen: set = set()
        self._room = self._default_room()
        self._pose = self._room_center_pose(self._room)

    def _default_room(self) -> Optional[str]:
        for rid, nbrs in self.facts.room_graph.items():
            if nbrs:
                return rid
        return next(iter(self.facts.room_types), None)

    def _room_center_pose(self, room_id) -> Pose:
        b = self.facts.room_bounds.get(room_id)
        if b:
            return Pose((b[0] + b[2]) / 2.0, 0.0, (b[1] + b[3]) / 2.0, 0.0, 0.0, True)
        return Pose(0.0, 0.0, 0.0, 0.0, 0.0, True)

    def _validate_spawn(self, room_id, pose) -> bool:
        rid_here = self._room_localization_point(pose.x, pose.z)
        if rid_here is None:
            _log.warning('spawn pose (x=%.2f, z=%.2f) falls outside every room polygon; agent_room() is None and every navigate will fail with the same receipt as a genuinely unreachable room. Give reset() a pose inside a room, not a room centroid that may sit in furniture.', pose.x, pose.z)
            return False
        if room_id is not None and rid_here != room_id:
            _log.warning('spawn pose (x=%.2f, z=%.2f) is tracked as room %r but its polygon is room %r -- room_bounds overlap or the tracked room is wrong; point-in-room queries will disagree with the tracked room.', pose.x, pose.z, room_id, rid_here)
            return False
        return True

    def reset(self, episode) -> Obs:
        self._step = 0
        self._held = None
        self._open.clear()
        self._pos_override.clear()
        self._seen.clear()
        if getattr(episode, 'agent_pose', None):
            self._pose = episode.agent_pose
            self._room = self._room_localization_point(self._pose.x, self._pose.z)
            self._validate_spawn(self._room, self._pose)
        else:
            self._room = self._default_room()
            self._pose = self._room_center_pose(self._room)
            self._validate_spawn(self._room, self._pose)
        for alias, kind, val in getattr(episode, 'initial_states', ()) or ():
            if kind == 'open':
                self._open[alias] = bool(val)
        self._observe_room()
        return self.observe()

    def close(self) -> None:
        pass

    def observe(self) -> Obs:
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        return Obs(rgb=rgb, depth=None, pose=self._pose, step_idx=self._step)

    def _observe_room(self):
        for i in self.facts.instances:
            if i.room_id == self._room:
                self._seen.add(i.alias)

    def _room_aliases(self, room_id):
        return [i.alias for i in self.facts.instances if i.room_id == room_id]

    def world(self) -> WorldView:
        ep_here = self.agent_room_name()
        holding = None
        if self._held is not None:
            holding = HELD_UNNAMED
        visible = tuple((a for a in self._room_aliases(self._room)))
        rooms = tuple((self.facts.room_names.get(r, r) for r in self.facts.room_types))
        furniture = tuple((i.alias for i in self.facts.instances if not i.pickupable and i.alias in self._seen))
        objects = tuple((i.alias for i in self.facts.instances if i.pickupable and i.alias in self._seen))
        return WorldView(rooms=rooms, furniture=furniture, objects=objects, relations=(), holding=holding, here=ep_here, visible=visible)

    def inventory(self) -> tuple:
        return tuple(self.facts.instances)

    def room_labels(self) -> tuple:
        return self.facts.room_labels()

    def room_sizes(self) -> dict:
        return dict(self.facts.room_area)

    def footprint(self, alias):
        return self.facts.footprints.get(alias)

    def agent_pose(self) -> Pose:
        return self._pose

    def _room_localization_point(self, x, z) -> Optional[str]:
        return self.facts.room_localization_point(x, z)

    def agent_room(self) -> Optional[str]:
        if self._room is not None:
            return self.facts.room_types.get(self._room, self._room)
        rid = self._room_localization_point(self._pose.x, self._pose.z)
        return self.facts.room_types.get(rid, rid) if rid else None

    def agent_room_name(self) -> Optional[str]:
        if self._room is not None:
            return self.facts.room_names.get(self._room, self._room)
        rid = self._room_localization_point(self._pose.x, self._pose.z)
        return self.facts.room_names.get(rid) if rid else None

    def instance_centroid(self, alias):
        if alias in self._pos_override:
            return self._pos_override[alias]
        return self.facts.centroids.get(alias)

    def bearing_to(self, alias):
        c = self.instance_centroid(alias)
        if c is None:
            return None
        dx, dz = (c[0] - self._pose.x, c[2] - self._pose.z)
        dist = math.hypot(dx, dz)
        world_bearing = math.degrees(math.atan2(dx, dz))
        rel = world_bearing - self._pose.yaw_deg
        rel = (rel + 180.0) % 360.0 - 180.0
        return (-rel, dist)

    def _reachable(self, from_room, to_room) -> bool:
        if from_room == to_room:
            return True
        seen = {from_room}
        q = deque([from_room])
        while q:
            r = q.popleft()
            for nbr in self.facts.room_graph.get(r, {}):
                if nbr == to_room:
                    return True
                if nbr not in seen:
                    seen.add(nbr)
                    q.append(nbr)
        return False

    def navigate_to_object(self, ref) -> ActionResult:
        self._step += 1
        alias = _ref_alias(ref)
        inst = self.facts.by_alias.get(alias) if alias else None
        if inst is None:
            return ActionResult(False, 'there is no such object here.')
        if not self._reachable(self._room, inst.room_id):
            return ActionResult(False, 'i could not get to that room.')
        from roam_bench.backend import VISIBILITY_DISTANCE
        got = self._nav.nearest_standable(alias, (self._pose.x, self._pose.z), prefer_within=VISIBILITY_DISTANCE)
        if got is None:
            return ActionResult(False, 'i could not find a way to stand next to it.')
        (cx, cz), d_obj = got
        moved = math.hypot(cx - self._pose.x, cz - self._pose.z)
        yaw = math.degrees(math.atan2(cx - self._pose.x, cz - self._pose.z))
        self._pose = Pose(cx, 0.0, cz, yaw, 0.0, True)
        self._room = inst.room_id
        self._observe_room()
        if d_obj <= VISIBILITY_DISTANCE + 0.05:
            return ActionResult(True, 'you are now next to it.', moved_m=moved)
        return ActionResult(False, 'i walked toward it but could not get close enough.', moved_m=moved)

    def go_to_visible_object(self, ref) -> ActionResult:
        self._step += 1
        alias = _ref_alias(ref)
        inst = self.facts.by_alias.get(alias) if alias else None
        if inst is None:
            return ActionResult(False, 'there is no such object here.')
        if inst.room_id != self._room:
            return ActionResult(False, 'you cannot see that from where you are standing.')
        c = self.instance_centroid(alias)
        moved = 0.0
        if c is not None:
            moved = math.hypot(c[0] - self._pose.x, c[2] - self._pose.z)
            yaw = math.degrees(math.atan2(c[0] - self._pose.x, c[2] - self._pose.z))
            self._pose = Pose(c[0], 0.0, c[2], yaw, 0.0, True)
        return ActionResult(True, 'you are now next to it.', moved_m=moved)

    def navigate_to_relative(self, deg: float, dist_m: float) -> ActionResult:
        self._step += 1
        yaw = self._pose.yaw_deg - deg
        rad = math.radians(yaw)
        x = self._pose.x + dist_m * math.sin(rad)
        z = self._pose.z + dist_m * math.cos(rad)
        self._pose = Pose(x, 0.0, z, yaw, self._pose.horizon_deg, True)
        rid = self._room_localization_point(x, z)
        if rid:
            self._room = rid
            self._observe_room()
        return ActionResult(True, 'moved.', moved_m=dist_m, turned_deg=-deg)

    def move(self, distance_m: float) -> ActionResult:
        self._step += 1
        rad = math.radians(self._pose.yaw_deg)
        x = self._pose.x + distance_m * math.sin(rad)
        z = self._pose.z + distance_m * math.cos(rad)
        self._pose = Pose(x, 0.0, z, self._pose.yaw_deg, self._pose.horizon_deg, True)
        return ActionResult(True, 'moved.', moved_m=distance_m, requested_m=distance_m)

    def strafe(self, distance_m: float) -> ActionResult:
        self._step += 1
        rad = math.radians(self._pose.yaw_deg)
        x = self._pose.x + distance_m * math.cos(rad)
        z = self._pose.z - distance_m * math.sin(rad)
        self._pose = Pose(x, 0.0, z, self._pose.yaw_deg, self._pose.horizon_deg, True)
        return ActionResult(True, 'moved sideways.', moved_m=distance_m, requested_m=distance_m)

    def turn(self, degrees: float) -> ActionResult:
        self._step += 1
        yaw = (self._pose.yaw_deg + degrees) % 360.0
        self._pose = Pose(self._pose.x, self._pose.y, self._pose.z, yaw, self._pose.horizon_deg, self._pose.standing)
        return ActionResult(True, 'turned.', turned_deg=degrees)

    def look(self, delta_pitch_deg: float) -> ActionResult:
        self._step += 1
        h = self._pose.horizon_deg - delta_pitch_deg
        h = max(-30.0, min(60.0, h))
        self._pose = Pose(self._pose.x, self._pose.y, self._pose.z, self._pose.yaw_deg, h, self._pose.standing)
        return ActionResult(True, 'looked.')

    def interact(self, verb: str, ref, into=None) -> ActionResult:
        self._step += 1
        alias = _ref_alias(ref)
        inst = self.facts.by_alias.get(alias) if alias else None
        if inst is None:
            return ActionResult(False, 'there is no such object here.')
        if verb in ('open', 'close', 'toggle', 'slice'):
            return ActionResult(False, f'you cannot {verb} that.')
        if verb in ('pick', 'pickup', 'take'):
            if not inst.pickupable:
                return ActionResult(False, 'that cannot be picked up.')
            self._held = alias
            return ActionResult(True, 'picked it up.')
        if verb in ('place', 'put', 'drop'):
            if self._held is None:
                return ActionResult(False, 'you are not holding anything.')
            self._held = None
            return ActionResult(True, 'placed it.')
        return ActionResult(False, f'unknown action {verb!r}.')

    def propose(self, text: str) -> ProposeResult:
        self._step += 1
        t = text.lower()
        best = None
        for i in self.facts.instances:
            if i.room_id != self._room:
                continue
            key = (i.kind or '').lower().replace('_', ' ')
            if key and key in t:
                best = i.alias
                break
        if best is None:
            return ProposeResult(accepted=False, granted=text, reason='nothing here matches that.', phantom=True)
        return ProposeResult(accepted=True, granted=best, reason='granted.')

    def mutate(self, op: str, alias: str, arg) -> bool:
        if alias not in self.facts.by_alias:
            return False
        if op == 'open':
            return False
        if op == 'move':
            self._pos_override[alias] = tuple(arg)
            return True
        return False

    def predicate(self, expr: str) -> bool:
        e = expr.strip()
        try:
            if e.startswith('holding(') and e.endswith(')'):
                return self._held == e[len('holding('):-1].strip().strip('\'"')
            if e.startswith('open(') and e.endswith(')'):
                return bool(self._open.get(e[len('open('):-1].strip().strip('\'"')))
            if e.startswith('in_room(') and e.endswith(')'):
                a, r = (s.strip().strip('\'"') for s in e[len('in_room('):-1].split(','))
                inst = self.facts.by_alias.get(a)
                return inst is not None and inst.room_id == r
        except Exception:
            return False
        return False

    def visible_instances(self, min_pixel_frac=None) -> set:
        if self.visibility_model == 'geom' and self._geom is not None:
            aliases = [i.alias for i in self.facts.instances]
            return scene_geom.visible_from_pose(self._geom, self._pose, aliases, min_pixel_frac)
        return set(self._room_aliases(self._room))

    def instance_at_pixel(self, u: int, v: int):
        return None

    def instances_in_patch(self, u: int, v: int, k: int=2) -> set:
        return set()
