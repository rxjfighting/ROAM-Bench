"""Navmesh utilities for ROAM-Bench."""
import copy
import math
MARGIN = 0.05
MIN_OPENING = 0.3
REACH = 0.4
AGENT_HEIGHT = 1.8
FLAT = 0.05

def _geo(wall):
    poly = wall['polygon']
    xs = [round(p['x'], 4) for p in poly]
    zs = [round(p['z'], 4) for p in poly]
    if max(xs) - min(xs) < 0.001:
        return ('x', min(xs), min(zs), max(zs))
    return ('z', min(zs), min(xs), max(xs))

def _run_key(axis):
    return 'z' if axis == 'x' else 'x'

def _respan(wall, s0, s1, new_id):
    axis, _, a0, a1 = _geo(wall)
    run = _run_key(axis)
    out = copy.deepcopy(wall)
    out['id'] = new_id
    for src, dst in zip(wall['polygon'], out['polygon']):
        t = src[run]
        dst[run] = s1 if abs(t - a1) < abs(t - a0) else s0
    return out

def _rooms_rect(house):
    rect = {}
    for room in house['rooms']:
        poly = room['floorPolygon']
        rect[room['id']] = (min((p['x'] for p in poly)), min((p['z'] for p in poly)), max((p['x'] for p in poly)), max((p['z'] for p in poly)))
    return rect

def _hole_span(door, wall):
    axis, _, a0, _ = _geo(wall)
    hp = door['holePolygon']
    lo, hi = sorted((hp[0]['x'], hp[1]['x']))
    return (a0 + lo, a0 + hi, max((p['y'] for p in hp)))

def _rotation(rot):
    if not isinstance(rot, dict) or 'axis' not in rot:
        return None
    axis = rot.get('axis') or {}
    ax, ay, az = (axis.get('x', 0.0), axis.get('y', 0.0), axis.get('z', 0.0))
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm < 1e-12 or abs(rot.get('degrees') or 0.0) < 1e-09:
        return None
    ax, ay, az = (ax / norm, ay / norm, az / norm)
    t = math.radians(rot['degrees'])
    c, s, u = (math.cos(t), math.sin(t), 1.0 - math.cos(t))
    return ((c + ax * ax * u, ax * ay * u - az * s, ax * az * u + ay * s), (ay * ax * u + az * s, c + ay * ay * u, ay * az * u - ax * s), (az * ax * u - ay * s, az * ay * u + ax * s, c + az * az * u))

def boxes(house, payloads):
    local = {}
    for asset, payload in (payloads or {}).items():
        verts = (payload or {}).get('vertices') or ()
        if not verts:
            continue
        local[asset] = (min((v['x'] for v in verts)), min((v['y'] for v in verts)), min((v['z'] for v in verts)), max((v['x'] for v in verts)), max((v['y'] for v in verts)), max((v['z'] for v in verts)))
    out = {}
    for obj in house.get('objects') or ():
        span = local.get(obj.get('assetId'))
        pos = obj.get('position')
        if span is None or not pos:
            continue
        corners = [(x, y, z) for x in (span[0], span[3]) for y in (span[1], span[4]) for z in (span[2], span[5])]
        m = _rotation(obj.get('rotation'))
        if m is not None:
            corners = [(m[0][0] * x + m[0][1] * y + m[0][2] * z, m[1][0] * x + m[1][1] * y + m[1][2] * z, m[2][0] * x + m[2][1] * y + m[2][2] * z) for x, y, z in corners]
        half = [(max((c[i] for c in corners)) - min((c[i] for c in corners))) / 2.0 for i in range(3)]
        mid = (pos.get('x', 0.0), pos.get('y', 0.0), pos.get('z', 0.0))
        out[obj['id']] = tuple((mid[i] - half[i] for i in range(3))) + tuple((mid[i] + half[i] for i in range(3)))
    return out

def from_corners(source_boxes):
    out = {}
    for oid, corners in (source_boxes or {}).items():
        if not corners:
            continue
        out[oid] = (min((c[0] for c in corners)), min((c[1] for c in corners)), min((c[2] for c in corners)), max((c[0] for c in corners)), max((c[1] for c in corners)), max((c[2] for c in corners)))
    return out

def _merge(spans):
    out = []
    for lo, hi in sorted(spans):
        if out and lo <= out[-1][1] + 1e-09:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(lo, hi) for lo, hi in out]

def _occupancy(house, box_by_id, axis, const, reach):
    run = 0 if axis != 'x' else 2
    perp = 0 if axis == 'x' else 2
    spans = []
    for box in (box_by_id or {}).values():
        if box[1] > AGENT_HEIGHT or box[4] < FLAT:
            continue
        if max(box[perp] - const, const - box[perp + 3]) > reach:
            continue
        spans.append((box[run], box[run + 3]))
    walls = {w['id']: w for w in house.get('walls') or ()}
    for win in house.get('windows') or ():
        wall = walls.get(win.get('wall0'))
        if wall is None:
            continue
        wax, wc, _, _ = _geo(wall)
        if wax != axis or abs(wc - const) > 0.2:
            continue
        lo, hi, _ = _hole_span(win, wall)
        spans.append((lo, hi))
    return _merge(spans)

def _widest_free(blocked, lo, hi):
    best, cursor = (0.0, lo)
    for a, b in blocked:
        if b <= lo or a >= hi:
            continue
        best = max(best, min(a, hi) - cursor)
        cursor = max(cursor, min(b, hi))
    return max(best, hi - cursor)

def _place(lo_bound, hi_bound, width, prefer, blocked):
    top = hi_bound - width
    if top < lo_bound:
        return (lo_bound, 0.0)
    candidates = {lo_bound, top, min(max(prefer - width / 2.0, lo_bound), top)}
    for a, b in blocked:
        for edge in (b, a - width):
            candidates.add(min(max(edge, lo_bound), top))
    best, score, offset = (lo_bound, -1.0, None)
    for low in sorted(candidates):
        free = _widest_free(blocked, low, low + width)
        drift = abs(low + width / 2.0 - prefer)
        if free > score + 1e-09 or (free > score - 1e-09 and (offset is None or drift < offset)):
            best, score, offset = (low, free, drift)
    return (best, score)

def clearances(house):
    walls = {w['id']: w for w in house.get('walls', [])}
    out = {}
    for door in house.get('doors', []):
        name = door['id']
        w0 = walls.get(door.get('wall0'))
        if w0 is None:
            out[name] = {'exterior': False, 'net': 0.0, 'why': 'wall0 missing'}
            continue
        lo, hi, _ = _hole_span(door, w0)
        width = hi - lo
        w1 = walls.get(door.get('wall1'))
        if w1 is None:
            out[name] = {'exterior': True, 'hole_width': round(width, 4), 'delta': None, 'net': None, 'why': 'no wall1 (exterior door)'}
            continue
        _, _, a0, a0e = _geo(w0)
        _, _, a1, a1e = _geo(w1)
        delta = a1 - a0
        local_lo, local_hi = (lo - a0, hi - a0)
        on0 = local_lo >= -1e-06 and local_hi <= a0e - a0 + 1e-06
        on1 = local_lo >= -1e-06 and local_hi <= a1e - a1 + 1e-06
        net = width - abs(delta) if on0 and on1 else 0.0
        out[name] = {'exterior': False, 'hole_width': round(width, 4), 'delta': round(delta, 4), 'net': round(max(net, 0.0), 4), 'wall0_cut': on0, 'wall1_cut': on1, 'why': '' if on0 and on1 else 'hole falls off ' + ('wall0' if not on0 else 'wall1')}
    return out

def _plan(house, margin, only, box_by_id=None, reach=REACH):
    walls = {w['id']: w for w in house['walls']}
    rect = _rooms_rect(house)
    plans, skipped = ([], [])
    taken = {}
    for door in sorted(house['doors'], key=lambda d: d['id']):
        name = door['id']
        if only is not None and name not in only and (name.replace('door|', '') not in only):
            continue
        w0, w1 = (walls.get(door.get('wall0')), walls.get(door.get('wall1')))
        if w1 is None:
            skipped.append((name, 'no wall1 (exterior door)'))
            continue
        if w0 is None:
            skipped.append((name, 'wall0 %r not in walls' % door.get('wall0')))
            continue
        ax0, ac0, a0, a0e = _geo(w0)
        ax1, ac1, a1, a1e = _geo(w1)
        if ax0 != ax1 or abs(ac0 - ac1) > 0.2:
            skipped.append((name, 'wall0 and wall1 are not collinear'))
            continue
        r0, r1 = (w0.get('roomId'), w1.get('roomId'))
        if r0 not in rect or r1 not in rect:
            skipped.append((name, 'a wall has no usable roomId'))
            continue
        bx0, bz0, bx1, bz1 = rect[r0]
        cx0, cz0, cx1, cz1 = rect[r1]
        if ax0 == 'x':
            shared = (max(bz0, cz0), min(bz1, cz1))
        else:
            shared = (max(bx0, cx0), min(bx1, cx1))
        t0, t1 = (max(shared[0], a0, a1), min(shared[1], a0e, a1e))
        usable = t1 - t0 - 2 * margin
        if usable < MIN_OPENING:
            skipped.append((name, 'shared boundary leaves only %.3f m' % max(usable, 0.0)))
            continue
        lo, hi, height = _hole_span(door, w0)
        width = min(hi - lo, usable)
        prefer = (lo + hi) / 2
        key = tuple(sorted((w0['id'], w1['id'])))
        if box_by_id is not None:
            blocked = _occupancy(house, box_by_id, ax0, (ac0 + ac1) / 2, reach)
            blocked = _merge(list(blocked) + taken.get(key, []))
            start, free = _place(t0 + margin, t1 - margin, width, prefer, blocked)
            centre = start + width / 2
        else:
            centre = min(max(prefer, t0 + margin + width / 2), t1 - margin - width / 2)
            free = None
        taken.setdefault(key, []).append((centre - width / 2, centre + width / 2))
        plans.append({'door': door, 'name': name, 'axis': ax0, 'wall0': w0['id'], 'wall1': w1['id'], 'room0': r0, 'room1': r1, 'lo': centre - width / 2, 'hi': centre + width / 2, 'width': width, 'height': height, 'margin': margin, 'free': free, 'moved': abs(centre - prefer) > 1e-06})
    return (plans, skipped)

def _split(house, cuts):
    used = {w['id'] for w in house['walls']}
    segments, out = ({}, [])
    for wall in house['walls']:
        points = cuts.get(wall['id'])
        if not points:
            out.append(wall)
            axis, _, a0, a1 = _geo(wall)
            segments[wall['id']] = [(a0, a1, wall['id'])]
            continue
        axis, _, a0, a1 = _geo(wall)
        edges = sorted({a0, a1} | {c for c in points if a0 + 1e-06 < c < a1 - 1e-06})
        if len(edges) == 2:
            out.append(wall)
            segments[wall['id']] = [(a0, a1, wall['id'])]
            continue
        parts = wall['id'].split('|')
        room, side = (parts + ['?', '?', '?'])[1:3]
        segs = []
        for s0, s1 in zip(edges, edges[1:]):
            new_id = 'wall|%s|%s|%.3f|%.3f' % (room, side, s0, s1)
            n = 0
            while new_id in used:
                n += 1
                new_id = 'wall|%s|%s|%.3f|%.3f_%d' % (room, side, s0, s1, n)
            used.add(new_id)
            out.append(_respan(wall, s0, s1, new_id))
            segs.append((s0, s1, new_id))
        used.discard(wall['id'])
        segments[wall['id']] = segs
    house['walls'] = out
    return segments

def _pick(segs, lo, hi):
    for s0, s1, wid in segs:
        if s0 <= lo + 1e-06 and s1 >= hi - 1e-06:
            return (s0, s1, wid)
    best, score = (None, 0.0)
    for s0, s1, wid in segs:
        overlap = min(hi, s1) - max(lo, s0)
        if overlap > score:
            best, score = ((s0, s1, wid), overlap)
    return best

def _recentre_asset(hole):
    ap = hole.get('assetPosition')
    if not ap:
        return
    hp = hole['holePolygon']
    lo, hi = sorted((hp[0]['x'], hp[1]['x']))
    ap['x'] = round((lo + hi) / 2, 4)
    ap['y'] = round(max((p['y'] for p in hp)) / 2.0, 4)

def _relink(house, segments, planned):
    stale = []
    for kind in ('doors', 'windows'):
        for hole in house.get(kind) or []:
            if hole['id'] in planned:
                continue
            for key in ('wall0', 'wall1'):
                old = hole.get(key)
                segs = segments.get(old)
                if not segs or (len(segs) == 1 and segs[0][2] == old):
                    continue
                anchor = segments.get(hole.get('wall0')) or segs
                base = anchor[0][0]
                hp = hole['holePolygon']
                lo, hi = sorted((hp[0]['x'], hp[1]['x']))
                chosen = _pick(segs, base + lo, base + hi)
                if chosen is None:
                    stale.append((hole['id'], key, old))
                    continue
                hole[key] = chosen[2]
                if key == 'wall0':
                    height = max((p['y'] for p in hp))
                    hole['holePolygon'] = [{'x': round(base + lo - chosen[0], 4), 'y': 0.0, 'z': 0.0}, {'x': round(base + hi - chosen[0], 4), 'y': round(height, 4), 'z': 0.0}]
                    _recentre_asset(hole)
    return stale

def apply(house, margin=MARGIN, only=None, obstacles=None, reach=REACH, verbose=True):
    before = len(house.get('walls') or [])
    plans, skipped = _plan(house, margin, only, obstacles, reach)
    cuts = {}
    for plan in plans:
        for wall_id in (plan['wall0'], plan['wall1']):
            cuts.setdefault(wall_id, set()).update((plan['lo'] - plan['margin'], plan['hi'] + plan['margin']))
    planned = {p['name'] for p in plans}
    segments = _split(house, cuts)
    stale = _relink(house, segments, planned)
    fixed, failed = ([], [])
    for plan in plans:
        door = plan['door']
        lo, hi = (plan['lo'] - plan['margin'], plan['hi'] + plan['margin'])
        s0 = _pick(segments[plan['wall0']], lo, hi)
        s1 = _pick(segments[plan['wall1']], lo, hi)
        if s0 is None or s1 is None:
            failed.append((plan['name'], 'no segment covers the opening after splitting'))
            continue
        off0, off1 = (plan['lo'] - s0[0], plan['lo'] - s1[0])
        if abs(off0 - off1) > 0.001:
            failed.append((plan['name'], 'segments still start %.3f m apart' % abs(off0 - off1)))
            continue
        door['wall0'], door['wall1'] = (s0[2], s1[2])
        door['holePolygon'] = [{'x': round(off0, 4), 'y': 0.0, 'z': 0.0}, {'x': round(off0 + plan['width'], 4), 'y': round(plan['height'], 4), 'z': 0.0}]
        door['room0'], door['room1'] = (plan['room0'], plan['room1'])
        _recentre_asset(door)
        fixed.append({'door': plan['name'], 'rooms': [plan['room0'], plan['room1']], 'opening': [round(plan['lo'], 4), round(plan['hi'], 4)], 'width': round(plan['width'], 4), 'recentred': plan['moved'], 'free': None if plan['free'] is None else round(plan['free'], 4)})
    report = {'fixed': fixed, 'skipped': skipped, 'failed': failed, 'stale': stale, 'walls_before': before, 'walls_after': len(house['walls']), 'furniture_aware': obstacles is not None, 'clearances': clearances(house)}
    if verbose:
        recentred = [f['door'].replace('door|', '') for f in fixed if f['recentred']]
        print(f"navmesh: aligned {len(fixed)} doorways, walls {before} -> {len(house['walls'])}" + (', furniture-aware' if obstacles is not None else '') + (f', moved along the shared boundary: {recentred}' if recentred else ''), flush=True)
        tight = [(f['door'].replace('door|', ''), f['free']) for f in fixed if f['free'] is not None and f['free'] < MIN_OPENING]
        for name, free in tight:
            print(f'  WARNING {name}: furniture leaves only {free:.2f} m of the opening clear', flush=True)
        for name, why in skipped:
            print(f"  skipped {name.replace('door|', '')}: {why}", flush=True)
        for name, why in failed:
            print(f"  FAILED {name.replace('door|', '')}: {why}", flush=True)
        for hole_id, key, old in stale:
            print(f'  WARNING {hole_id} .{key} still points at split wall {old}', flush=True)
    return report
