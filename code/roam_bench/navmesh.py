"""Navmesh utilities for ROAM-Bench."""
from __future__ import annotations
import math
from collections import deque
from typing import Optional
GRID = 0.25
AGENT_RADIUS = 0.2
AGENT_HEIGHT = 1.8
try:
    from roam_bench.protocol.navmesh import FLAT as _FLAT
except Exception:
    _FLAT = 0.0
_STRUCTURAL = {'wall'}

class NavMesh:

    def __init__(self, facts, grid: float=GRID, agent_radius: float=AGENT_RADIUS, agent_height: float=AGENT_HEIGHT, flat: float=_FLAT):
        self.facts = facts
        self.grid = grid
        self.agent_radius = agent_radius
        self.agent_height = agent_height
        self.flat = flat
        bands = self._vertical_extents(facts.root)
        self._room_fps: dict = {}
        for alias, fp in facts.footprints.items():
            inst = facts.by_alias.get(alias)
            if inst is None or fp is None:
                continue
            if (getattr(inst, 'kind', '') or '').lower() in _STRUCTURAL:
                continue
            cx, cz, sx, sz = fp
            y0, y1 = bands.get(alias, (0.0, float('inf')))
            self._room_fps.setdefault(inst.room_id, []).append((cx, cz, sx, sz, y0, y1))
        self._walkable: dict = {}

    @staticmethod
    def _vertical_extents(root: str) -> dict:
        try:
            from roam_bench.scene_facts import _find_house_state, _room_offsets, _obj_world_matrix, _world_aabb_drake, _STRUCTURAL as _SF_STRUCTURAL
            import json as _json
            offsets = _room_offsets(root)
            hs = _json.load(open(_find_house_state(root)))
        except Exception:
            return {}
        ext: dict = {}
        for rid, rm in (hs.get('rooms') or {}).items():
            off = offsets.get(rid)
            if off is None:
                continue
            for name, o in (rm.get('objects') or {}).items():
                if o.get('object_type') in _SF_STRUCTURAL:
                    continue
                alias = f"{rid}_{o.get('object_id') or name}"
                aabb = _world_aabb_drake(o.get('bbox_min'), o.get('bbox_max'), _obj_world_matrix(o.get('transform'), off))
                if aabb is not None:
                    ext[alias] = (float(aabb[0][2]), float(aabb[1][2]))
        return ext

    @staticmethod
    def _vertical_bands(root: str) -> dict:
        return {a: lo for a, (lo, _hi) in NavMesh._vertical_extents(root).items()}

    def reachable(self, from_room: Optional[str], to_room: Optional[str]) -> bool:
        if from_room is None or to_room is None:
            return False
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

    def _cell_blocked(self, x: float, z: float, room_id: str) -> bool:
        r = self.agent_radius
        h = self.agent_height
        flat = self.flat
        for cx, cz, sx, sz, y0, y1 in self._room_fps.get(room_id, ()):
            if y0 >= h:
                continue
            if y1 < flat:
                continue
            if abs(x - cx) <= sx / 2.0 + r and abs(z - cz) <= sz / 2.0 + r:
                return True
        return False

    def walkable_cells(self, room_id: str) -> list:
        if room_id in self._walkable:
            return self._walkable[room_id]
        b = self.facts.room_bounds.get(room_id)
        if b is None:
            self._walkable[room_id] = []
            return []
        x0, z0, x1, z1 = b
        nx = max(0, int(round((x1 - x0) / self.grid)))
        nz = max(0, int(round((z1 - z0) / self.grid)))
        free = set()
        for i in range(nx + 1):
            for j in range(nz + 1):
                x, z = (x0 + i * self.grid, z0 + j * self.grid)
                if not self._cell_blocked(x, z, room_id):
                    free.add((i, j))
        best_comp: set = set()
        unseen = set(free)
        while unseen:
            seed = unseen.pop()
            comp = {seed}
            stack = [seed]
            while stack:
                ci, cj = stack.pop()
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        n = (ci + di, cj + dj)
                        if n in unseen:
                            unseen.discard(n)
                            comp.add(n)
                            stack.append(n)
            if len(comp) > len(best_comp):
                best_comp = comp
        cells = [(x0 + i * self.grid, z0 + j * self.grid) for i, j in best_comp]
        self._walkable[room_id] = cells
        return cells

    def nearest_standable(self, alias: str, from_xz, prefer_within: float):
        c = self.facts.centroids.get(alias)
        inst = self.facts.by_alias.get(alias)
        if c is None or inst is None:
            return None
        ox, oz = (c[0], c[2])
        cells = self.walkable_cells(inst.room_id)
        if not cells:
            return None
        near = [p for p in cells if math.dist(p, (ox, oz)) <= prefer_within]
        if near:
            best = min(near, key=lambda p: math.dist(p, from_xz))
            best = self._prefer_own_room(best, near, inst.room_id, from_xz)
            return (best, math.dist(best, (ox, oz)))
        best = min(cells, key=lambda p: math.dist(p, (ox, oz)))
        best = self._prefer_own_room(best, cells, inst.room_id, (ox, oz))
        return (best, math.dist(best, (ox, oz)))

    def _prefer_own_room(self, best, pool, room_id, toward):
        if self.facts.room_localization_point(*best) == room_id:
            return best
        own = [p for p in pool if self.facts.room_localization_point(*p) == room_id]
        if not own:
            return best
        return min(own, key=lambda p: math.dist(p, toward))
