"""Astar path utilities for ROAM-Bench."""
import heapq
import math
from typing import Optional
from .navmesh import GRID, AGENT_RADIUS
from . import seg_geom
_LINK = GRID * 1.5
_DIAG_BIN = 1
_BRIDGE = 2.6
_BRIDGE_K = 6
_SYNTH_FINE = 0.05
_SYNTH_LINK = _SYNTH_FINE * 1.5
_SYNTH_SEAM_MAX = 3.0
_SYNTH_MARGIN = 0.4
_SYNTH_ROOM = '__synth__'

def _wall_rects(wall_boxes, band_lo=0.1, band_hi=1.7):
    rects = []
    for world, size in wall_boxes or ():
        cz = float(world[2][3])
        hz = 0.5 * (abs(world[2][0] * size[0]) + abs(world[2][1] * size[1]) + abs(world[2][2] * size[2]))
        if cz + hz <= band_lo or cz - hz >= band_hi:
            continue
        cx, cy = (float(world[0][3]), float(world[1][3]))
        hx = 0.5 * (abs(world[0][0] * size[0]) + abs(world[0][1] * size[1]) + abs(world[0][2] * size[2]))
        hy = 0.5 * (abs(world[1][0] * size[0]) + abs(world[1][1] * size[1]) + abs(world[1][2] * size[2]))
        rects.append((cx - hx, cy - hy, cx + hx, cy + hy))
    return rects

def _seg_hits_rect(p, q, rect, r):
    x0, y0, x1, y1 = (rect[0] - r, rect[1] - r, rect[2] + r, rect[3] + r)
    dx, dy = (q[0] - p[0], q[1] - p[1])
    t0, t1 = (0.0, 1.0)
    for d, s, lo, hi in ((dx, p[0], x0, x1), (dy, p[1], y0, y1)):
        if abs(d) < 1e-12:
            if s < lo or s > hi:
                return False
        else:
            a, b = ((lo - s) / d, (hi - s) / d)
            if a > b:
                a, b = (b, a)
            t0, t1 = (max(t0, a), min(t1, b))
            if t0 > t1:
                return False
    return True

class PathPlanner:

    def __init__(self, nav, room_ids=None, wall_boxes=None, agent_radius=AGENT_RADIUS, synth_seam_nodes=True, furniture_radius=AGENT_RADIUS):
        self._nav = nav
        self._room_ids = room_ids
        self._wall_rects = _wall_rects(wall_boxes)
        self._r = agent_radius
        self._furn_r = float(furniture_radius)
        self._synth = bool(synth_seam_nodes)
        self._synth_stats = {'enabled': self._synth, 'fine_grid': None, 'wall_radius': agent_radius, 'furniture_radius': self._furn_r, 'seam_pairs_considered': 0, 'seam_pairs_reconnected': 0, 'n_synth_nodes': 0, 'n_synth_nodes_with_at_least_one_edge': 0}
        self._furn_boxes = self._collect_furniture(nav)
        self._furn_stats = {'edges_checked': 0, 'cross_room_edges_checked': 0, 'edges_blocked': 0, 'boxes_dist_calcd': 0}
        self._pts = None
        self._adj = None
        self._bins = None
        self._room_localization = None

    @staticmethod
    def _collect_furniture(nav):
        fps = getattr(nav, '_room_fps', None)
        if not fps:
            return []
        h = getattr(nav, 'agent_height', float('inf'))
        flat = getattr(nav, 'flat', 0.0)
        out = []
        for _rid, items in fps.items():
            for cx, cz, sx, sz, y0, y1 in items:
                if y0 >= h:
                    continue
                if y1 < flat:
                    continue
                out.append((cx, cz, sx / 2.0, sz / 2.0))
        return out

    def _bin_key(self, x, z):
        return (int(math.floor(x / GRID)), int(math.floor(z / GRID)))

    def _cell_clear(self, x, z):
        for x0, y0, x1, y1 in self._wall_rects:
            dx = max(x0 - x, 0.0, x - x1)
            dy = max(y0 - z, 0.0, z - y1)
            if dx * dx + dy * dy < self._r * self._r:
                return False
        return True

    def point_clear(self, point, radius: float) -> bool:
        x, z = map(float, point)
        radius = float(radius)
        for x0, z0, x1, z1 in self._wall_rects:
            dx = max(x0 - x, 0.0, x - x1)
            dz = max(z0 - z, 0.0, z - z1)
            if dx * dx + dz * dz < radius * radius:
                return False
        for cx, cz, hx, hz in self._furn_boxes:
            dx = max(abs(x - cx) - hx, 0.0)
            dz = max(abs(z - cz) - hz, 0.0)
            if dx * dx + dz * dz < radius * radius:
                return False
        return True

    def _build(self):
        if self._pts is not None:
            return
        rooms = self._room_ids
        if rooms is None:
            rooms = list(getattr(self._nav.facts, 'room_bounds', {}).keys())
        pts = []
        room_localization = []
        seen = {}
        for rid in rooms:
            for x, z in self._nav.walkable_cells(rid):
                key = (round(x / (GRID * 0.5)), round(z / (GRID * 0.5)))
                if key in seen:
                    continue
                if not self._cell_clear(x, z):
                    continue
                seen[key] = len(pts)
                pts.append((float(x), float(z)))
                room_localization.append(rid)
        self._pts = pts
        bins = {}
        for i, (x, z) in enumerate(pts):
            bins.setdefault(self._bin_key(x, z), []).append(i)
        self._bins = bins
        adj = [[] for _ in pts]
        for i, (x, z) in enumerate(pts):
            bi, bj = self._bin_key(x, z)
            for di in range(-_DIAG_BIN, _DIAG_BIN + 1):
                for dj in range(-_DIAG_BIN, _DIAG_BIN + 1):
                    for j in bins.get((bi + di, bj + dj), ()):
                        if j <= i:
                            continue
                        d = math.dist(pts[i], pts[j])
                        if not 0 < d <= _LINK:
                            continue
                        if room_localization[i] != room_localization[j] and self._edge_blocked(pts[i], pts[j]):
                            continue
                        if self._furn_blocked(pts[i], pts[j], cross_room=room_localization[i] != room_localization[j]):
                            continue
                        adj[i].append((j, d))
                        adj[j].append((i, d))
        self._bridge(pts, room_localization, bins, adj)
        if self._synth:
            self._synthesize_seam_nodes(pts, room_localization, bins, adj)
        self._adj = adj
        self._room_localization = room_localization

    def measured_room_graph(self):
        self._build()
        facts = getattr(self._nav, 'facts', None)
        bounds = getattr(facts, 'room_bounds', {}) or {}
        authored = getattr(facts, 'room_graph', {}) or {}
        rooms = set(bounds) | set(authored)
        for neighbours in authored.values():
            rooms.update(neighbours or {})
        out = {rid: set() for rid in rooms}
        if not self._pts:
            return {rid: () for rid in sorted(out)}

        def authored_reachable(start, goal):
            if start == goal:
                return True
            seen, stack = ({start}, [start])
            while stack:
                room = stack.pop()
                for neighbour in authored.get(room, {}) or {}:
                    if neighbour == goal:
                        return True
                    if neighbour not in seen:
                        seen.add(neighbour)
                        stack.append(neighbour)
            return False

        def add_measured_pair(ra, rb):
            if authored_reachable(ra, rb):
                out.setdefault(ra, set()).add(rb)
            if authored_reachable(rb, ra):
                out.setdefault(rb, set()).add(ra)
        labels = self._room_localization
        active = set()
        for rid in sorted(rooms):
            nodes = [i for i, label in enumerate(labels) if label == rid]
            unseen = set(nodes)
            components = []
            while unseen:
                start = min(unseen)
                unseen.remove(start)
                stack, component = ([start], [])
                while stack:
                    u = stack.pop()
                    component.append(u)
                    for v, _cost in self._adj[u]:
                        if labels[v] == rid and v in unseen:
                            unseen.remove(v)
                            stack.append(v)
                components.append(component)
            if components:
                main = min(components, key=lambda c: (-len(c), min(c)))
                active.update(main)
        for u in active:
            ru = labels[u]
            for v, _cost in self._adj[u]:
                if v not in active:
                    continue
                rv = labels[v]
                if ru != rv:
                    add_measured_pair(ru, rv)
        synth = {i for i, label in enumerate(labels) if label == _SYNTH_ROOM}
        while synth:
            start = min(synth)
            synth.remove(start)
            stack, touched = ([start], set())
            while stack:
                u = stack.pop()
                for v, _cost in self._adj[u]:
                    if labels[v] == _SYNTH_ROOM:
                        if v in synth:
                            synth.remove(v)
                            stack.append(v)
                    elif v in active:
                        touched.add(labels[v])
            for ra in sorted(touched):
                for rb in sorted(touched):
                    if ra >= rb:
                        continue
                    hinted = rb in (authored.get(ra) or {}) or ra in (authored.get(rb) or {})
                    if hinted:
                        add_measured_pair(ra, rb)
        return {rid: tuple(sorted(out.get(rid, ()))) for rid in sorted(out)}

    def _bridge(self, pts, room_localization, bins, adj):
        rg = getattr(self._nav.facts, 'room_graph', {}) or {}

        def adjacent(ra, rb):
            return rb in (rg.get(ra) or {}) or ra in (rg.get(rb) or {})
        reach = int(math.ceil(_BRIDGE / GRID))
        for i, (x, z) in enumerate(pts):
            bi, bj = self._bin_key(x, z)
            cands = []
            for di in range(-reach, reach + 1):
                for dj in range(-reach, reach + 1):
                    for j in bins.get((bi + di, bj + dj), ()):
                        if j == i or room_localization[j] == room_localization[i]:
                            continue
                        if not adjacent(room_localization[i], room_localization[j]):
                            continue
                        d = math.dist(pts[i], pts[j])
                        if _LINK < d <= _BRIDGE:
                            cands.append((d, j))
            cands.sort()
            added = 0
            for d, j in cands:
                if added >= _BRIDGE_K:
                    break
                if not self._edge_blocked(pts[i], pts[j]) and (not self._furn_blocked(pts[i], pts[j], cross_room=True)):
                    adj[i].append((j, d))
                    adj[j].append((i, d))
                    added += 1

    def _edge_blocked(self, p, q):
        for rect in self._wall_rects:
            if _seg_hits_rect(p, q, rect, self._r):
                return True
        return False

    def _furn_blocked(self, p, q, cross_room=False):
        st = self._furn_stats
        st['edges_checked'] += 1
        if cross_room:
            st['cross_room_edges_checked'] += 1
        blocked = seg_geom.seg_furniture_blocked(p, q, self._furn_boxes, self._furn_r, stats=st)
        if blocked:
            st['edges_blocked'] += 1
        return blocked

    def _point_furn_clear(self, x, z):
        r2 = self._furn_r * self._furn_r
        for cx, cz, hx, hz in self._furn_boxes:
            dx = max(cx - hx - x, 0.0, x - (cx + hx))
            dz = max(cz - hz - z, 0.0, z - (cz + hz))
            if dx * dx + dz * dz < r2:
                return False
        return True

    @staticmethod
    def _components(adj, n):
        comp = [-1] * n
        c = 0
        for i in range(n):
            if comp[i] != -1:
                continue
            stack = [i]
            comp[i] = c
            while stack:
                u = stack.pop()
                for v, _w in adj[u]:
                    if comp[v] == -1:
                        comp[v] = c
                        stack.append(v)
            c += 1
        return comp

    def _synthesize_seam_nodes(self, pts, room_localization, bins, adj):
        st = self._synth_stats
        st['fine_grid'] = _SYNTH_FINE
        base = len(pts)
        comp0 = self._components(adj, base)
        room_cells = {}
        for i, rid in enumerate(room_localization):
            room_cells.setdefault(rid, []).append(i)
        rg = getattr(self._nav.facts, 'room_graph', {}) or {}
        pairs = set()
        for ra, nbrs in rg.items():
            for rb in nbrs or {}:
                if ra != rb and ra in room_cells and (rb in room_cells):
                    pairs.add(tuple(sorted((ra, rb))))

        def fine_key(x, z):
            return (round(x / (_SYNTH_FINE * 0.5)), round(z / (_SYNTH_FINE * 0.5)))
        occupied = set()
        for x, z in pts:
            occupied.add(fine_key(x, z))
        new_pts = []
        for ra, rb in sorted(pairs):
            A = room_cells.get(ra)
            B = room_cells.get(rb)
            if not A or not B:
                continue
            if {comp0[i] for i in A} & {comp0[i] for i in B}:
                continue
            st['seam_pairs_considered'] += 1
            nearpts = []
            for i in A:
                xi, zi = pts[i]
                for j in B:
                    if math.hypot(xi - pts[j][0], zi - pts[j][1]) <= _SYNTH_SEAM_MAX:
                        nearpts.append((xi, zi))
                        break
            for j in B:
                xj, zj = pts[j]
                for i in A:
                    if math.hypot(xj - pts[i][0], zj - pts[i][1]) <= _SYNTH_SEAM_MAX:
                        nearpts.append((xj, zj))
                        break
            if not nearpts:
                continue
            x0 = min((p[0] for p in nearpts)) - _SYNTH_MARGIN
            x1 = max((p[0] for p in nearpts)) + _SYNTH_MARGIN
            z0 = min((p[1] for p in nearpts)) - _SYNTH_MARGIN
            z1 = max((p[1] for p in nearpts)) + _SYNTH_MARGIN
            gx0, gx1 = (int(math.floor(x0 / _SYNTH_FINE)), int(math.ceil(x1 / _SYNTH_FINE)))
            gz0, gz1 = (int(math.floor(z0 / _SYNTH_FINE)), int(math.ceil(z1 / _SYNTH_FINE)))
            bounds = getattr(self._nav.facts, 'room_bounds', {}) or {}
            ba, bb = (bounds.get(ra), bounds.get(rb))

            def in_either_room(px, pz):
                if ba is None or bb is None:
                    return True
                eps = 1e-09
                return any((b[0] - eps <= px <= b[2] + eps and b[1] - eps <= pz <= b[3] + eps for b in (ba, bb)))
            for gx in range(gx0, gx1 + 1):
                x = gx * _SYNTH_FINE
                for gz in range(gz0, gz1 + 1):
                    z = gz * _SYNTH_FINE
                    k = fine_key(x, z)
                    if k in occupied:
                        continue
                    if not in_either_room(x, z):
                        continue
                    if not self._cell_clear(x, z):
                        continue
                    if not self._point_furn_clear(x, z):
                        continue
                    occupied.add(k)
                    new_pts.append((float(x), float(z)))
        for x, z in new_pts:
            pts.append((x, z))
            room_localization.append(_SYNTH_ROOM)
            adj.append([])
        st['n_synth_nodes'] = len(new_pts)
        n = len(pts)
        sbin = {}
        for s in range(base, n):
            x, z = pts[s]
            sbin.setdefault((int(round(x / _SYNTH_FINE)), int(round(z / _SYNTH_FINE))), []).append(s)
        near2 = _SYNTH_FINE * math.sqrt(2.0) * 1.001
        for s in range(base, n):
            x, z = pts[s]
            bi, bj = (int(round(x / _SYNTH_FINE)), int(round(z / _SYNTH_FINE)))
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for t in sbin.get((bi + di, bj + dj), ()):
                        if t <= s:
                            continue
                        d = math.hypot(x - pts[t][0], z - pts[t][1])
                        if 0.0 < d <= near2:
                            adj[s].append((t, d))
                            adj[t].append((s, d))
        reach = 2
        for s in range(base, n):
            x, z = pts[s]
            ps = pts[s]
            bi, bj = self._bin_key(x, z)
            for di in range(-reach, reach + 1):
                for dj in range(-reach, reach + 1):
                    for t in bins.get((bi + di, bj + dj), ()):
                        if t >= base:
                            continue
                        d = math.hypot(x - pts[t][0], z - pts[t][1])
                        if not 0.0 < d <= _SYNTH_LINK:
                            continue
                        if self._edge_blocked(ps, pts[t]):
                            continue
                        if self._furn_blocked(ps, pts[t], cross_room=True):
                            continue
                        adj[s].append((t, d))
                        adj[t].append((s, d))
        for s in range(base, n):
            bins.setdefault(self._bin_key(pts[s][0], pts[s][1]), []).append(s)
        st['n_synth_nodes_with_at_least_one_edge'] = sum((1 for s in range(base, n) if adj[s]))
        comp1 = self._components(adj, n)
        recon = 0
        for ra, rb in sorted(pairs):
            A = room_cells.get(ra)
            B = room_cells.get(rb)
            if not A or not B:
                continue
            if not {comp0[i] for i in A} & {comp0[i] for i in B} and {comp1[i] for i in A} & {comp1[i] for i in B}:
                recon += 1
        st['seam_pairs_reconnected'] = recon

    def _nearest_node(self, xz, max_dist):
        bi, bj = self._bin_key(xz[0], xz[1])
        best, best_d = (None, max_dist)
        reach = max(1, int(math.ceil(max_dist / GRID)))
        for di in range(-reach, reach + 1):
            for dj in range(-reach, reach + 1):
                for j in self._bins.get((bi + di, bj + dj), ()):
                    d = math.dist(self._pts[j], xz)
                    if d <= best_d:
                        best, best_d = (j, d)
        return best

    def plan(self, start_xz, goal_xz, snap_radius: float=None):
        self._build()
        if not self._pts:
            return None
        if snap_radius is None:
            snap_radius = GRID * 2.0
        s = self._nearest_node(start_xz, snap_radius)
        g = self._nearest_node(goal_xz, snap_radius)
        if s is None or g is None:
            return None
        node_path = self._astar(s, g)
        if node_path is None:
            return None
        poly = [tuple(map(float, start_xz))]
        poly += [self._pts[i] for i in node_path]
        poly.append(tuple(map(float, goal_xz)))
        poly = self._string_pull(poly)
        length = sum((math.dist(poly[k], poly[k + 1]) for k in range(len(poly) - 1)))
        return (poly, length)

    def _shortcut_clear(self, p, q):
        if self._edge_blocked(p, q) or self._furn_blocked(p, q):
            return False
        distance = math.dist(p, q)
        count = max(1, int(math.ceil(distance / (GRID * 0.5))))
        coverage = GRID * 0.8
        for i in range(1, count):
            t = i / count
            sample = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t)
            if self._nearest_node(sample, coverage) is None:
                return False
        return True

    def _string_pull(self, poly):
        if len(poly) <= 2:
            return list(poly)
        out = [poly[0]]
        anchor = 0
        while anchor < len(poly) - 1:
            farthest = anchor + 1
            probe = farthest + 1
            while probe < len(poly) and self._shortcut_clear(poly[anchor], poly[probe]):
                farthest = probe
                probe += 1
            out.append(poly[farthest])
            anchor = farthest
        return out

    def _astar(self, s, g):
        adj, pts = (self._adj, self._pts)
        gx, gz = pts[g]
        openh = [(math.dist(pts[s], pts[g]), 0.0, s)]
        came = {s: None}
        gcost = {s: 0.0}
        while openh:
            _, gc, u = heapq.heappop(openh)
            if u == g:
                path = []
                while u is not None:
                    path.append(u)
                    u = came[u]
                return list(reversed(path))
            if gc > gcost.get(u, math.inf):
                continue
            for v, w in adj[u]:
                ng = gc + w
                if ng < gcost.get(v, math.inf):
                    gcost[v] = ng
                    came[v] = u
                    h = math.hypot(pts[v][0] - gx, pts[v][1] - gz)
                    heapq.heappush(openh, (ng + h, ng, v))
        return None
