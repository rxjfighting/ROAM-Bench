"""Houses utilities for ROAM-Bench."""
import json
import math
import os
import re
LAYOUT = '\n    0 0 0 0 0 0\n    0 1 1 1 1 0\n    0 1 1 1 1 0\n    0 1 1 1 1 0\n    0 1 1 1 1 0\n    0 2 2 2 2 0\n    0 2 2 2 2 0\n    0 2 2 2 2 0\n    0 2 2 2 2 0\n    0 3 3 3 3 0\n    0 3 3 3 3 0\n    0 3 3 3 3 0\n    0 0 0 0 0 0\n'
FURNITURE = '\n    0 0 0 0 0 0\n    0 F 0 0 0 0\n    0 0 0 0 C 0\n    0 0 0 0 0 0\n    0 0 = 0 0 0\n    0 0 = 0 0 0\n    0 S 0 0 0 0\n    0 0 0 0 T 0\n    0 0 0 0 + 0\n    0 0 0 0 + 0\n    0 D 0 0 0 0\n    0 0 0 0 B 0\n    0 0 0 0 0 0\n'
ROOMS = {'1': {'wallTemplate': {'material': {'name': 'DrywallOrange'}}, 'floorTemplate': {'roomType': 'Kitchen', 'floorMaterial': {'name': 'DarkWoodFloors'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '2': {'wallTemplate': {'material': {'name': 'DrywallBeige'}}, 'floorTemplate': {'roomType': 'LivingRoom', 'floorMaterial': {'name': 'RedBrick'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '3': {'wallTemplate': {'material': {'name': 'DrywallGreen'}}, 'floorTemplate': {'roomType': 'Bedroom', 'floorMaterial': {'name': 'DarkWoodFloors'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}}
FURNISHINGS = {'F': {'kinematic': True, 'assetId': 'Fridge_16'}, 'C': {'kinematic': True, 'assetId': 'Countertop_C_8x4'}, 'S': {'kinematic': True, 'assetId': 'Sofa_206_1'}, 'T': {'kinematic': True, 'assetId': 'Coffee_Table_211_1'}, 'D': {'kinematic': True, 'assetId': 'Desk_204_1'}, 'B': {'kinematic': True, 'assetId': 'Bed_17'}}
TEMPLATE = {'id': 'm6_three_room', 'layout': LAYOUT, 'objectsLayouts': [FURNITURE], 'rooms': ROOMS, 'doors': {'=': {'room0': '1', 'room1': '2', 'openness': 1.0, 'assetId': 'Doorway_1'}, '+': {'room0': '2', 'room1': '3', 'openness': 1.0, 'assetId': 'Doorway_1'}}, 'objects': FURNISHINGS, 'proceduralParameters': {'floorColliderThickness': 1.0, 'receptacleHeight': 0.7, 'skyboxId': 'Sky1', 'ceilingMaterial': {'name': 'PureWhite'}}, 'metadata': {'schema': '1.0.0'}}
PLACEMENTS = [('Mug_1', 'D', 0.0, 0.0), ('Apple_1', 'T', -0.18, 0.0), ('Book_5', 'C', 0.3, 0.0), ('Bowl_2', 'B', 0.0, 0.0), ('Tomato_3', 'C', -0.3, -0.12), ('Remote_1', 'T', -0.32, 0.16), ('CreditCard_1', 'T', 0.2, 0.0)]

def _unpack(p):
    asset, host, dx, dz = p[:4]
    where = p[4] if len(p) > 4 else 'on'
    if where not in ('on', 'in'):
        raise ValueError(f"placement {p!r}: `where` must be 'on' or 'in', not {where!r}")
    return (asset, host, dx, dz, where)
EPS_LAYOUT = '\n    0 0 0 0 0 0\n    0 1 1 1 1 0\n    0 1 1 1 1 0\n    0 1 1 1 1 0\n    0 1 1 1 1 0\n    0 2 2 2 2 0\n    0 2 2 2 2 0\n    0 2 2 2 2 0\n    0 2 2 2 2 0\n    0 3 3 3 3 0\n    0 3 3 3 3 0\n    0 3 3 3 3 0\n    0 3 3 3 3 0\n    0 4 4 4 4 0\n    0 4 4 4 4 0\n    0 4 4 4 4 0\n    0 4 4 4 4 0\n    0 0 0 0 0 0\n'
EPS_FURNITURE = '\n    0 0 0 0 0 0\n    0 F 0 0 0 0\n    0 0 0 0 C 0\n    0 0 0 0 0 0\n    0 0 = 0 0 0\n    0 0 = 0 0 0\n    0 M 0 0 0 0\n    0 0 0 0 E 0\n    0 0 0 0 + 0\n    0 0 0 0 + 0\n    0 G 0 0 0 0\n    0 0 0 0 B 0\n    0 0 # 0 0 0\n    0 0 # 0 0 0\n    0 K 0 0 0 0\n    0 0 0 0 H 0\n    0 P 0 0 0 0\n    0 0 0 0 0 0\n'
EPS_ROOMS = {'1': {'wallTemplate': {'material': {'name': 'DrywallOrange'}}, 'floorTemplate': {'roomType': 'Kitchen', 'floorMaterial': {'name': 'DarkWoodFloors'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '2': {'wallTemplate': {'material': {'name': 'DrywallBeige'}}, 'floorTemplate': {'roomType': 'LivingRoom', 'floorMaterial': {'name': 'RedBrick'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '3': {'wallTemplate': {'material': {'name': 'DrywallGreen'}}, 'floorTemplate': {'roomType': 'Bedroom', 'floorMaterial': {'name': 'DarkWoodFloors'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '4': {'wallTemplate': {'material': {'name': 'BrownDrywall'}}, 'floorTemplate': {'roomType': 'Bathroom', 'floorMaterial': {'name': 'RedBrick'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}}
EPS_FURNISHINGS = {'F': {'kinematic': True, 'assetId': 'Fridge_16'}, 'C': {'kinematic': True, 'assetId': 'Countertop_C_8x4'}, 'M': {'kinematic': True, 'assetId': 'Microwave_1'}, 'E': {'kinematic': True, 'assetId': 'Dresser_205_1', 'rotateY': 180}, 'G': {'kinematic': True, 'assetId': 'Dresser_209_1'}, 'B': {'kinematic': True, 'assetId': 'Bed_17'}, 'K': {'kinematic': True, 'assetId': 'Safe_18_1'}, 'H': {'kinematic': True, 'assetId': 'Dresser_210_1_1', 'rotateY': 180}, 'P': {'kinematic': True, 'assetId': 'Toilet_1'}}
EPS_TEMPLATE = {'id': 'eps_four_room', 'layout': EPS_LAYOUT, 'objectsLayouts': [EPS_FURNITURE], 'rooms': EPS_ROOMS, 'doors': {'=': {'room0': '1', 'room1': '2', 'openness': 1.0, 'assetId': 'Doorway_1'}, '+': {'room0': '2', 'room1': '3', 'openness': 1.0, 'assetId': 'Doorway_1'}, '#': {'room0': '3', 'room1': '4', 'openness': 1.0, 'assetId': 'Doorway_1'}}, 'objects': EPS_FURNISHINGS, 'proceduralParameters': {'floorColliderThickness': 1.0, 'receptacleHeight': 0.7, 'skyboxId': 'Sky1', 'ceilingMaterial': {'name': 'PureWhite'}}, 'metadata': {'schema': '1.0.0'}}
EPS_PLACEMENTS = [('Apple_10', 'K', 0.0, 0.0, 'in'), ('Apple_1', 'C', 0.0, -0.33), ('Apple_11', 'E', 0.0, 0.0), ('Apple_12', 'G', 0.0, 0.0), ('Mug_1', 'F', 0.0, 0.0, 'in'), ('Mug_2', 'E', -0.25, 0.0), ('Mug_3', 'G', -0.25, 0.0), ('Bowl_10', 'F', 0.0, 0.3, 'in'), ('Bowl_1', 'B', 0.0, 0.0), ('Bowl_11', 'H', 0.0, 0.0)]
GAMMA_LAYOUT = '\n    0 0 0 0 0 0 0\n    0 1 1 1 1 1 0\n    0 1 1 1 1 1 0\n    0 1 1 1 1 1 0\n    0 1 1 1 1 1 0\n    0 1 1 1 1 1 0\n    0 2 2 2 2 2 0\n    0 2 2 2 2 2 0\n    0 2 2 2 2 2 0\n    0 2 2 2 2 2 0\n    0 2 2 2 2 2 0\n    0 3 3 3 3 3 0\n    0 3 3 3 3 3 0\n    0 3 3 3 3 3 0\n    0 3 3 3 3 3 0\n    0 3 3 3 3 3 0\n    0 4 4 4 4 4 0\n    0 4 4 4 4 4 0\n    0 4 4 4 4 4 0\n    0 4 4 4 4 4 0\n    0 4 4 4 4 4 0\n    0 0 0 0 0 0 0\n'
GAMMA_FURNITURE = '\n    0 0 0 0 0 0 0\n    0 F C 0 0 0 0\n    0 0 0 0 0 0 0\n    0 M 0 0 0 0 0\n    0 0 0 0 0 0 0\n    0 0 0 = 0 0 0\n    0 0 0 = 0 S 0\n    0 T 0 N 0 0 0\n    0 0 0 V 0 0 0\n    0 0 0 0 0 0 0\n    0 0 0 + 0 0 0\n    0 0 G + D 0 0\n    0 0 0 0 0 0 0\n    0 B 0 0 0 0 0\n    0 0 0 0 O 0 0\n    0 0 0 # 0 0 0\n    0 0 P # L 0 0\n    0 0 0 0 A 0 0\n    0 0 K 0 0 0 0\n    0 0 0 0 0 0 0\n    0 0 0 0 0 0 0\n    0 0 0 0 0 0 0\n'
GAMMA_ROOMS = {'1': {'wallTemplate': {'material': {'name': 'DrywallOrange'}}, 'floorTemplate': {'roomType': 'Kitchen', 'floorMaterial': {'name': 'DarkWoodFloors'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '2': {'wallTemplate': {'material': {'name': 'DrywallBeige'}}, 'floorTemplate': {'roomType': 'LivingRoom', 'floorMaterial': {'name': 'RedBrick'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '3': {'wallTemplate': {'material': {'name': 'DrywallGreen'}}, 'floorTemplate': {'roomType': 'Bedroom', 'floorMaterial': {'name': 'DarkWoodFloors'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}, '4': {'wallTemplate': {'material': {'name': 'BrownDrywall'}}, 'floorTemplate': {'roomType': 'Bathroom', 'floorMaterial': {'name': 'RedBrick'}}, 'floorYPosition': 0.0, 'wallHeight': 2.8}}
GAMMA_FURNISHINGS = {'F': {'kinematic': True, 'assetId': 'Fridge_16'}, 'C': {'kinematic': True, 'assetId': 'Countertop_C_8x4'}, 'M': {'kinematic': True, 'assetId': 'Microwave_1'}, 'S': {'kinematic': True, 'assetId': 'Sofa_206_1'}, 'T': {'kinematic': True, 'assetId': 'Coffee_Table_211_1', 'rotateY': 90}, 'N': {'kinematic': True, 'assetId': 'RoboTHOR_side_table_hol'}, 'V': {'kinematic': True, 'assetId': 'TV_Stand_201_1', 'rotateY': 90}, 'G': {'kinematic': True, 'assetId': 'Dresser_210_1_1', 'rotateY': 90}, 'D': {'kinematic': True, 'assetId': 'Desk_204_1'}, 'B': {'kinematic': True, 'assetId': 'Bed_17', 'rotateY': 90}, 'O': {'kinematic': True, 'assetId': 'Ottoman_203_1'}, 'P': {'kinematic': True, 'assetId': 'Toilet_1', 'rotateY': 90}, 'L': {'kinematic': True, 'assetId': 'Stool_1_1'}, 'A': {'kinematic': True, 'assetId': 'Dining_Table_16_1'}, 'K': {'kinematic': True, 'assetId': 'Safe_18_1'}}
GAMMA_TEMPLATE = {'id': 'gamma_clearance', 'layout': GAMMA_LAYOUT, 'objectsLayouts': [GAMMA_FURNITURE], 'rooms': GAMMA_ROOMS, 'doors': {'=': {'room0': '1', 'room1': '2', 'openness': 1.0, 'assetId': 'Doorway_1'}, '+': {'room0': '2', 'room1': '3', 'openness': 1.0, 'assetId': 'Doorway_1'}, '#': {'room0': '3', 'room1': '4', 'openness': 1.0, 'assetId': 'Doorway_1'}}, 'objects': GAMMA_FURNISHINGS, 'proceduralParameters': {'floorColliderThickness': 1.0, 'receptacleHeight': 0.7, 'skyboxId': 'Sky1', 'ceilingMaterial': {'name': 'PureWhite'}}, 'metadata': {'schema': '1.0.0'}}
GAMMA_PLACEMENTS = [('Apple_1', 'C', 0.0, -0.33), ('Mug_1', 'D', 0.0, 0.0), ('Bowl_1', 'A', 0.0, 0.0)]
TEMPLATES = {'m6_three_room': (TEMPLATE, PLACEMENTS), 'eps_four_room': (EPS_TEMPLATE, EPS_PLACEMENTS), 'gamma_clearance': (GAMMA_TEMPLATE, GAMMA_PLACEMENTS)}

def named(name):
    if name not in TEMPLATES:
        raise KeyError(f'unknown house template {name!r}; have {sorted(TEMPLATES)}')
    return TEMPLATES[name]
WALL_INSET = 0.05
OBJECT_INSET = 0.2
EDGE_INSET = 0.08
FLOOR_TOL = 0.01

def _span(box, axis):
    c, s = (box['center'], box['size'])
    return (c[axis] - s[axis] / 2.0, c[axis] + s[axis] / 2.0)

def _room_bounds(poly, inset=0.0):
    xs = [p[0] for p in poly]
    zs = [p[1] for p in poly]
    return (min(xs) + inset, max(xs) - inset, min(zs) + inset, max(zs) - inset)

def _shift_inside(lo, hi, rlo, rhi):
    if hi - lo > rhi - rlo:
        return (rlo + rhi - (lo + hi)) / 2.0
    if lo < rlo:
        return rlo - lo
    if hi > rhi:
        return rhi - hi
    return 0.0

def _overflow(box, poly, inset=0.0):
    rx0, rx1, rz0, rz1 = _room_bounds(poly, inset)
    x0, x1 = _span(box, 'x')
    z0, z1 = _span(box, 'z')
    return (max(0.0, rx0 - x0) + max(0.0, x1 - rx1), max(0.0, rz0 - z0) + max(0.0, z1 - rz1))

def _mesh_bottom(controller, oid):
    ev = controller.step(action='GetInSceneAssetGeometry', objectId=oid)
    if not ev.metadata['lastActionSuccess']:
        return None
    ys = [v['y'] for g in ev.metadata['actionReturn'] or [] for v in g.get('vertices') or []]
    return min(ys) if ys else None

def _floor_of(house):
    return {r['id']: r.get('floorYPosition', 0.0) for r in house.get('rooms', [])}

def _place_on(box, poly, dx=0.0, dz=0.0, clearance=0.06):
    c, s = (box['center'], box['size'])
    hx0, hx1 = _span(box, 'x')
    hz0, hz1 = _span(box, 'z')
    rx0, rx1, rz0, rz1 = _room_bounds(poly, OBJECT_INSET)
    ax0, ax1 = (max(hx0 + EDGE_INSET, rx0), min(hx1 - EDGE_INSET, rx1))
    az0, az1 = (max(hz0 + EDGE_INSET, rz0), min(hz1 - EDGE_INSET, rz1))
    if ax0 > ax1 or az0 > az1:
        raise RuntimeError(f'no point is both on this receptacle and inside its room: host footprint x {hx0:.2f}..{hx1:.2f} z {hz0:.2f}..{hz1:.2f}, room (inset {OBJECT_INSET}) x {rx0:.2f}..{rx1:.2f} z {rz0:.2f}..{rz1:.2f}')
    return {'x': min(max(c['x'] + dx, ax0), ax1), 'y': c['y'] + s['y'] / 2.0 + clearance, 'z': min(max(c['z'] + dz, az0), az1)}

def check_assets(controller, template=None, placements=None):
    template = template or TEMPLATE
    placements = placements if placements is not None else PLACEMENTS
    ev = controller.step(action='GetAssetDatabase')
    if not ev.metadata['lastActionSuccess']:
        raise RuntimeError('GetAssetDatabase failed: ' + str(ev.metadata.get('errorMessage'))[:200])
    known = set(ev.metadata['actionReturn'])
    wanted = [o['assetId'] for o in (template.get('objects') or {}).values()]
    wanted += [d['assetId'] for d in (template.get('doors') or {}).values()]
    wanted += [p[0] for p in placements]
    return (wanted, [a for a in wanted if a not in known])

def _create(controller, house, tag, refresh=True):
    ev = controller.step(action='CreateHouse', house=house)
    if not ev.metadata['lastActionSuccess']:
        raise RuntimeError(f'CreateHouse ({tag}) failed: ' + str(ev.metadata.get('errorMessage'))[:300])
    if refresh:
        ev = controller.step(action='Pass')
    return ({o['objectId']: o for o in ev.metadata['objects']}, ev)

def _spawn_points(controller, host, poly, inset=OBJECT_INSET):
    ev = controller.step(action='GetSpawnCoordinatesAboveReceptacle', objectId=host, anywhere=True)
    if not ev.metadata['lastActionSuccess']:
        return []
    rx0, rx1, rz0, rz1 = _room_bounds(poly, inset)
    return [p for p in ev.metadata.get('actionReturn') or [] if rx0 <= p['x'] <= rx1 and rz0 <= p['z'] <= rz1]

def _rank(points, want, taken, min_gap=0.18):
    ranked = sorted(points, key=lambda p: (round(math.dist((p['x'], p['z']), want), 4), round(p['x'], 4), round(p['z'], 4)))
    return [p for p in ranked if all((math.dist((p['x'], p['z']), (q['x'], q['z'])) >= min_gap for q in taken))]

def _spread(ranked, k):
    if len(ranked) <= k:
        return list(ranked)
    step = max(1, (len(ranked) - 1) // (k - 1))
    out = [ranked[min(i * step, len(ranked) - 1)] for i in range(k)]
    return out

def _settle(controller, steps):
    ev = controller.step(action='AdvancePhysicsStep', timeStep=0.02)
    if ev.metadata['lastActionSuccess']:
        for _ in range(steps - 1):
            controller.step(action='AdvancePhysicsStep', timeStep=0.02)
        return ('AdvancePhysicsStep', steps)
    for _ in range(steps):
        controller.step(action='Pass')
    return ('Pass', steps)

def _on_host(obj, host):
    return host in (obj.get('parentReceptacles') or [])

def _in_host(obj, host):
    return any((p == host or p.split('___')[0] == host for p in obj.get('parentReceptacles') or []))

def _set_open(controller, hosts, opened):
    for h in hosts:
        if opened:
            controller.step(action='OpenObject', objectId=h, openness=1.0, forceAction=True)
        else:
            controller.step(action='CloseObject', objectId=h, forceAction=True)
    live = {o['objectId']: o for o in controller.last_event.metadata['objects']}
    return {h: live[h].get('isOpen') if h in live else None for h in hosts}
DOOR_CENTRE_TOL = 0.02
RELOAD_TOL = 0.02
RELOAD_TOL_IN = 0.1
SETTLE_STEPS = 200

def _wall_length(wall):
    a, b = (wall['polygon'][0], wall['polygon'][1])
    return math.dist((a['x'], a['z']), (b['x'], b['z']))

def _centre_doors(house, tol=DOOR_CENTRE_TOL):
    walls = {w['id']: w for w in house.get('walls', [])}
    moved = {}
    for d in house.get('doors', []):
        wall = walls.get(d.get('wall0'))
        hole = d.get('holePolygon')
        if wall is None or not hole or len(hole) < 2:
            continue
        length = _wall_length(wall)
        lo, hi = (hole[0]['x'], hole[1]['x'])
        width = hi - lo
        new_lo = (length - width) / 2.0
        shift = new_lo - lo
        if abs(shift) <= tol:
            continue
        hole[0]['x'], hole[1]['x'] = (new_lo, new_lo + width)
        if d.get('assetPosition'):
            d['assetPosition']['x'] -= shift
        moved[d.get('id')] = {'wall': d.get('wall0'), 'wall_len': round(length, 3), 'width': round(width, 3), 'from': round(lo, 3), 'to': round(new_lo, 3)}
    return moved

def connectivity(sim, house, tries=2):
    ev = sim.c.step(action='GetReachablePositions')
    pts = ev.metadata.get('actionReturn') or []
    by_room = {}
    for p in pts:
        by_room.setdefault(room_localization((p['x'], p['z']), house), []).append(p)

    def anchor(rid):
        cells = by_room.get(rid) or []
        if not cells:
            return None
        poly = room_polygons(house).get(rid) or []
        cx = sum((x for x, _ in poly)) / len(poly)
        cz = sum((z for _, z in poly)) / len(poly)
        return min(cells, key=lambda p: (round(math.dist((p['x'], p['z']), (cx, cz)), 4), round(p['x'], 4), round(p['z'], 4)))
    results, failures = ([], [])
    pairs = []
    for d in house.get('doors', []):
        a, b = (d.get('room0'), d.get('room1'))
        pairs += [(a, b, d.get('id')), (b, a, d.get('id'))]
    for src, dst, door in pairs:
        start, target = (anchor(src), anchor(dst))
        if start is None or target is None:
            failures.append(f'door {door}: room {(src if start is None else dst)} has no reachable cell')
            continue
        ok, reason, moved = (False, '', 0.0)
        for _ in range(tries):
            sim.c.step(action='TeleportFull', position=dict(x=start['x'], y=start['y'], z=start['z']), rotation=dict(x=0, y=0, z=0), horizon=0, standing=True, forceAction=True)
            res = sim._walk_to(target)
            ok, reason, moved = (res.success, res.reason, res.moved_m)
            if ok:
                break
        end = sim.c.last_event.metadata['agent']['position']
        landed = room_localization((end['x'], end['z']), house)
        row = {'door': door, 'from': src, 'to': dst, 'walked_m': round(moved, 2), 'ok': bool(ok and landed == dst), 'landed_in': landed, 'reason': reason}
        results.append(row)
        if not row['ok']:
            failures.append(f'door {door}: cannot walk {src} -> {dst} (walked {moved:.2f} m, ended in room {landed}): {reason}')
    return {'walks': results, 'failures': failures}

def liftable(sim, house):
    ev = sim.c.step(action='GetReachablePositions')
    pts = ev.metadata.get('actionReturn') or []
    origin = sim.agent_pose()
    rows, failures = ([], [])
    for f in sorted(sim.inventory(), key=lambda x: x.alias):
        if not f.pickupable:
            continue
        oid = sim.table.object_id(f.alias)
        here = _pose_of(sim, oid)
        _go(sim, origin)
        at_walk, reason, _arrival = _try_lift(sim, f.alias, oid, walk=True)
        elsewhere = None
        if not at_walk:
            elsewhere = _lift_from_anywhere(sim, f.alias, oid, pts, here)
        rows.append({'alias': f.alias, 'on': list(f.on), 'at_walk_pose': at_walk, 'reason': reason, 'from_elsewhere': elsewhere, 'arrived_at': _arrival})
        if not at_walk:
            where = 'but it lifts from another cell -- the placement is wedged, move it' if elsewhere else 'and from no cell tried -- the asset cannot be lifted in this house'
            failures.append(f"{f.alias} (on {', '.join(f.on) or 'the floor'}) cannot be picked up where the walk stops ({_arrival}), {where}: {reason}")
    return {'origin': {'x': round(origin.x, 2), 'z': round(origin.z, 2), 'yaw': round(origin.yaw_deg, 1)}, 'objects': rows, 'failures': failures}

def _pose_of(sim, oid):
    o = sim._object(oid)
    return None if o is None else (dict(o['position']), dict(o['rotation']))

def _restore(sim, oid, pose):
    if sim._held_alias() is not None:
        sim.c.step(action='DropHandObject', forceAction=True)
    if pose is not None:
        sim.c.step(action='TeleportObject', objectId=oid, position=pose[0], rotation=pose[1], forceAction=True, forceKinematic=True)

def _go(sim, pose):
    sim.c.step(action='TeleportFull', position=dict(x=pose.x, y=pose.y, z=pose.z), rotation=dict(x=0, y=pose.yaw_deg, z=0), horizon=pose.horizon_deg, standing=True, forceAction=True)

def _try_lift(sim, alias, oid, walk=False):
    from .types import NameRef
    pose = _pose_of(sim, oid)
    arrival = None
    if walk:
        if not sim.navigate_to_object(NameRef(alias)).success:
            _restore(sim, oid, pose)
            return (False, 'the walk to it did not arrive', None)
        a = sim.agent_pose()
        arrival = {'x': round(a.x, 2), 'z': round(a.z, 2), 'yaw': round(a.yaw_deg, 1)}
    res = sim.interact('pick', NameRef(alias))
    _restore(sim, oid, pose)
    return (bool(res.success), res.reason, arrival)

def _lift_from_anywhere(sim, alias, oid, pts, pose, cells=10, yaws=(0.0, 90.0, 180.0, 270.0), horizons=(0.0, 30.0, 60.0)):
    if pose is None:
        return None
    p0 = pose[0]
    near = sorted(pts, key=lambda p: math.dist((p['x'], p['z']), (p0['x'], p0['z'])))[:cells]
    for p in near:
        for yaw in yaws:
            for hz in horizons:
                sim.c.step(action='TeleportFull', position=dict(x=p['x'], y=p['y'], z=p['z']), rotation=dict(x=0, y=yaw, z=0), horizon=hz, standing=True, forceAction=True)
                ok, _, _ = _try_lift(sim, alias, oid)
                if ok:
                    return {'x': round(p['x'], 2), 'z': round(p['z'], 2), 'yaw': yaw, 'horizon': hz, 'distance_m': round(math.dist((p['x'], p['z']), (p0['x'], p0['z'])), 2)}
    return None

def refreeze(controller, house, placements=None, settle_steps=SETTLE_STEPS):
    placements = placements if placements is not None else PLACEMENTS
    unpacked = [_unpack(p) for p in placements]
    hosts = {f'pk_{a}': h for a, h, _, _, _ in unpacked}
    inside = {f'pk_{a}' for a, _, _, _, w in unpacked if w == 'in'}
    ev = controller.step(action='CreateHouse', house=house)
    if not ev.metadata['lastActionSuccess']:
        raise RuntimeError('refreeze: CreateHouse failed: ' + str(ev.metadata.get('errorMessage'))[:300])
    _settle(controller, settle_steps)
    live = {o['objectId']: o for o in controller.last_event.metadata['objects']}
    drift, moving, off_host = ({}, [], [])
    for o in house['objects']:
        oid = o['id']
        if oid not in hosts or oid not in live:
            continue
        p, r = (live[oid]['position'], live[oid]['rotation'])
        drift[oid] = round(math.dist((o['position']['x'], o['position']['y'], o['position']['z']), (p['x'], p['y'], p['z'])), 3)
        if live[oid].get('isMoving'):
            moving.append(oid)
        held = _in_host(live[oid], hosts[oid]) if oid in inside else _on_host(live[oid], hosts[oid])
        if not held:
            off_host.append((oid, hosts[oid], live[oid].get('parentReceptacles') or []))
        o['position'] = {'x': p['x'], 'y': p['y'], 'z': p['z']}
        o['rotation'] = {'x': r['x'], 'y': r['y'], 'z': r['z']}
    floor = min((r.get('floorYPosition', 0.0) for r in house.get('rooms', [])), default=0.0)
    fell = sorted((o['id'] for o in house['objects'] if o['id'] in drift and o['position']['y'] < floor - 0.5))
    if fell:
        raise RuntimeError(f'placements fell out of the level on reload: {fell} -- y below the floor ({floor}) by more than 0.5 m')
    over_tol = sorted((oid for oid, d in drift.items() if d > (RELOAD_TOL_IN if oid in inside else RELOAD_TOL)))
    return {'drift_m': drift, 'moving': moving, 'off_host': off_host, 'worst_m': round(max(drift.values(), default=0.0), 3), 'over_tol': over_tol, 'settled': bool(not moving and (not off_host) and (not over_tol))}

def verify(controller, house, placements=None, settle_steps=SETTLE_STEPS):
    placements = placements if placements is not None else PLACEMENTS
    ev = controller.step(action='CreateHouse', house=house)
    if not ev.metadata['lastActionSuccess']:
        raise RuntimeError('verify: CreateHouse failed: ' + str(ev.metadata.get('errorMessage'))[:300])
    _settle(controller, settle_steps)
    live = {o['objectId']: o for o in controller.last_event.metadata['objects']}
    polys = room_polygons(house)
    room_localization_marker = {o['id']: o.get('room') for o in house['objects']}
    problems, overflow, floor_gap = ([], {}, {})
    floors = _floor_of(house)
    for marker, room in room_localization_marker.items():
        poly = polys.get(room)
        if poly is None or marker not in live or marker.startswith('pk_'):
            continue
        ov = tuple((round(v, 3) for v in _overflow(live[marker]['axisAlignedBoundingBox'], poly, WALL_INSET)))
        if any(ov):
            overflow[marker] = ov
            problems.append(f'{marker} sticks {ov} m outside room {room}')
        bottom = _mesh_bottom(controller, marker)
        if bottom is not None:
            gap = round(bottom - floors.get(room, 0.0), 3)
            if abs(gap) > FLOOR_TOL:
                floor_gap[marker] = gap
                problems.append(f"{marker} {('floats' if gap > 0 else 'is buried')} {abs(gap)} m {('above' if gap > 0 else 'below')} the floor of room {room}")
    parents, moving, shut = ({}, [], {})
    for asset, host, _, _, where in (_unpack(p) for p in placements):
        oid = f'pk_{asset}'
        o = live.get(oid)
        if o is None:
            problems.append(f'{oid} did not instantiate')
            continue
        parents[oid] = sorted(set(o.get('parentReceptacles') or []))
        if not (_in_host(o, host) if where == 'in' else _on_host(o, host)):
            problems.append(f'{oid} is on {parents[oid]}, not {where} {host}')
        if where == 'in':
            shut[host] = live[host].get('isOpen') if host in live else None
            if shut[host] is not False:
                problems.append(f'{host} is not shut on load (isOpen={shut[host]}), so {oid} is not hidden and no must_open bound will be generated for it')
        if o.get('isMoving'):
            moving.append(oid)
        rid = room_localization((o['position']['x'], o['position']['z']), house)
        if rid is None:
            problems.append(f'{oid} is outside every room polygon')
    if moving:
        problems.append(f'still moving after {settle_steps} settle steps: {moving}')
    return {'overflow': overflow, 'floor_gap': floor_gap, 'parents': parents, 'moving': moving, 'shut': shut, 'problems': problems}

def _strip_rotations(template):
    objs = template.get('objects') or {}
    rotations = {m: float(s['rotateY']) for m, s in objs.items() if s and s.get('rotateY')}
    if not rotations:
        return (template, {})
    clean = dict(template)
    clean['objects'] = {m: {k: v for k, v in (s or {}).items() if k != 'rotateY'} for m, s in objs.items()}
    return (clean, rotations)

def _apply_rotations(house, rotations):
    turned = {}
    for o in house['objects']:
        if o['id'] not in rotations:
            continue
        r = dict(o.get('rotation') or {})
        y = (r.get('y') or 0.0) + rotations[o['id']]
        o['rotation'] = {'x': r.get('x') or 0.0, 'y': y % 360.0, 'z': r.get('z') or 0.0}
        turned[o['id']] = round(o['rotation']['y'], 1)
    missing = sorted(set(rotations) - set(turned))
    if missing:
        raise RuntimeError(f"`rotateY` named furnishings {missing} that the template did not produce.  That is a typo in the marker name, not an asset problem -- the markers this house has are {sorted((o['id'] for o in house['objects']))}.")
    return turned

def build(controller, template=None, placements=None, settle_steps=SETTLE_STEPS, tries=6, fresh=None):
    template = template or TEMPLATE
    placements = placements if placements is not None else PLACEMENTS
    template, rotations = _strip_rotations(template)
    wanted, missing = check_assets(controller, template, placements)
    if missing:
        raise RuntimeError(f'{len(missing)} assetId(s) not in the database: {missing}')
    ev = controller.step(action='GetHouseFromTemplate', template=template)
    if not ev.metadata['lastActionSuccess']:
        raise RuntimeError('GetHouseFromTemplate failed: ' + str(ev.metadata.get('errorMessage'))[:300])
    house = json.loads(json.dumps(ev.metadata['actionReturn']))
    doors_moved = _centre_doors(house)
    turned = _apply_rotations(house, rotations)
    live, _ = _create(controller, house, 'pass 1')
    unturned = {m: (deg, round((live[m]['rotation'] or {}).get('y') or 0.0, 1)) for m, deg in turned.items() if m in live and min((((live[m]['rotation'] or {}).get('y') or 0.0) - deg) % 360.0, (deg - ((live[m]['rotation'] or {}).get('y') or 0.0)) % 360.0) > 1.0}
    if unturned:
        raise RuntimeError(f"`rotateY` did not reach the scene for {sorted(unturned)}: { {m: f'asked {a}, got {b}' for m, (a, b) in sorted(unturned.items())}}.  The usual cause is a `rotation` dict that still carries Unity's axis-angle pair (`axis` plus `degrees`), which the deserialiser prefers over the Euler `y` -- so `degrees: 0.0` silently means 'do not rotate'.")
    markers = list(template.get('objects') or {})
    absent = [m for m in markers if m not in live]
    if absent:
        gridless = [m for m in absent if not any((m in (l or '').split() for l in '\n'.join(template.get('objectsLayouts') or []).splitlines()))]
        raise RuntimeError(f'furniture markers {absent} did not instantiate.  ' + (f"{gridless} appear in the template's `objects` but in none of its `objectsLayouts` cells, so there is nowhere to put them -- that is the whole cause, and it is not an asset problem." if gridless else 'Every one of them is in the grid, so this is almost certainly an assetId that is not in GetAssetDatabase, which CreateHouse drops without an error -- run check_assets.'))
    polys = room_polygons(house)
    room_localization_marker = {o['id']: o.get('room') for o in house['objects']}
    shifts, lifts, before = ({}, {}, {})
    floors = _floor_of(house)
    for o in house['objects']:
        marker = o['id']
        poly = polys.get(room_localization_marker.get(marker))
        if marker not in live or poly is None:
            continue
        box = live[marker]['axisAlignedBoundingBox']
        before[marker] = _overflow(box, poly, WALL_INSET)
        rx0, rx1, rz0, rz1 = _room_bounds(poly, WALL_INSET)
        sx = _shift_inside(*_span(box, 'x'), rx0, rx1)
        sz = _shift_inside(*_span(box, 'z'), rz0, rz1)
        bottom = _mesh_bottom(controller, marker)
        floor_y = floors.get(room_localization_marker.get(marker), 0.0)
        sy = 0.0 if bottom is None or abs(bottom - floor_y) <= FLOOR_TOL else floor_y - bottom
        if abs(sx) > 0.0001 or abs(sz) > 0.0001 or abs(sy) > 0.0001:
            p = dict(o['position'])
            p['x'], p['z'] = (p['x'] + sx, p['z'] + sz)
            p['y'] = p['y'] + sy
            o['position'] = p
            if abs(sx) > 0.0001 or abs(sz) > 0.0001:
                shifts[marker] = (round(sx, 3), round(sz, 3))
            if abs(sy) > 0.0001:
                lifts[marker] = round(sy, 3)
    if fresh is not None:
        controller = fresh()
    live, _ = _create(controller, house, 'pass 2a')
    overflow = {m: tuple((round(v, 3) for v in _overflow(live[m]['axisAlignedBoundingBox'], polys[room_localization_marker[m]], WALL_INSET))) for m in before if m in live}
    gap_after = {}
    for marker in before:
        if marker not in live:
            continue
        bottom = _mesh_bottom(controller, marker)
        if bottom is None:
            continue
        g = round(bottom - floors.get(room_localization_marker.get(marker), 0.0), 3)
        if abs(g) > FLOOR_TOL:
            gap_after[marker] = g
    unpacked = [_unpack(p) for p in placements]
    in_hosts = sorted({h for _, h, _, _, w in unpacked if w == 'in'})
    opened = _set_open(controller, in_hosts, True) if in_hosts else {}
    refused = sorted((h for h, v in opened.items() if v is not True))
    if refused:
        raise RuntimeError(f'hosts {refused} would not open, so nothing can be put inside them.  An `in` placement whose host is not openable is a template error, and THOR openability is narrower than it looks: `Cabinet_Straight_01` is not openable, and a `Dresser` is not either -- only its four `Drawer` children are.')
    extra, cands, taken, fallbacks, relaxed = ([], {}, [], [], [])
    for asset, host, dx, dz, where in unpacked:
        oid = f'pk_{asset}'
        room = room_localization_marker.get(host)
        poly = polys[room]
        box = live[host]['axisAlignedBoundingBox']
        want = (box['center']['x'] + dx, box['center']['z'] + dz)
        ranked = _rank(_spawn_points(controller, host, poly), want, taken)
        if not ranked and where == 'in':
            ranked = _rank(_spawn_points(controller, host, poly, WALL_INSET), want, taken)
            if ranked:
                relaxed.append(oid)
        if not ranked:
            if where == 'in':
                raise RuntimeError(f'{oid}: no spawn point inside {host} that is also inside room {room}. Either the container is too small for this object or THOR will not enumerate its interior.  There is no safe fallback for an `in` placement -- the on-top one would hide nothing -- so this is fatal.')
            ranked = [_place_on(box, poly, dx, dz)]
            fallbacks.append(oid)
        cands[oid] = _spread(ranked, tries)
        taken.append(ranked[0])
        extra.append({'id': oid, 'assetId': asset, 'kinematic': False, 'room': room, 'position': {k: ranked[0][k] for k in ('x', 'y', 'z')}, 'rotation': {'x': 0, 'y': 0, 'z': 0}})
    house['objects'] = list(house['objects']) + extra
    live, _ = _create(controller, house, 'pass 2b', refresh=False)
    lost = [e['id'] for e in extra if e['id'] not in live]
    if lost:
        raise RuntimeError(f'placements {lost} did not instantiate')
    if in_hosts:
        _set_open(controller, in_hosts, True)
    how, n_settle = _settle(controller, settle_steps)
    hosts = {f'pk_{a}': h for a, h, _, _, _ in unpacked}
    inside = {f'pk_{a}' for a, _, _, _, w in unpacked if w == 'in'}

    def _held(obj, oid):
        if not (_in_host(obj, hosts[oid]) if oid in inside else _on_host(obj, hosts[oid])):
            return False
        parents = obj.get('parentReceptacles') or []
        return not any((p != oid and p in hosts for p in parents))
    attempts = {e['id']: 0 for e in extra}
    for e in extra:
        oid = e['id']
        for pt in cands[oid]:
            attempts[oid] += 1
            r = controller.step(action='PlaceObjectAtPoint', objectId=oid, position={k: pt[k] for k in ('x', 'y', 'z')})
            if not r.metadata['lastActionSuccess']:
                continue
            _settle(controller, settle_steps)
            live = {o['objectId']: o for o in controller.last_event.metadata['objects']}
            if _held(live[oid], oid):
                break
    shut_after = _set_open(controller, in_hosts, False) if in_hosts else {}
    if in_hosts:
        _settle(controller, settle_steps)
    still_open = sorted((h for h, v in shut_after.items() if v is not False))
    if still_open:
        raise RuntimeError(f'containers {still_open} would not shut, so their contents are not hidden')
    live = {o['objectId']: o for o in controller.last_event.metadata['objects']}
    drift, still_moving, off_host = ({}, [], [])
    for e in extra:
        o = live[e['id']]
        p, r = (o['position'], o['rotation'])
        drift[e['id']] = round(math.dist((e['position']['x'], e['position']['y'], e['position']['z']), (p['x'], p['y'], p['z'])), 3)
        if o.get('isMoving'):
            still_moving.append(e['id'])
        if not _held(o, e['id']):
            off_host.append((e['id'], hosts[e['id']], o.get('parentReceptacles') or []))
        e['position'] = {'x': p['x'], 'y': p['y'], 'z': p['z']}
        e['rotation'] = {'x': r['x'], 'y': r['y'], 'z': r['z']}
    types = room_types(house)
    homeless = [(e['id'], round(e['position']['x'], 2), round(e['position']['z'], 2)) for e in extra if room_localization((e['position']['x'], e['position']['z']), house) is None]
    if homeless:
        raise RuntimeError(f'placements outside every room polygon: {homeless}')
    live, _ = _create(controller, house, 'pass 3', refresh=False)
    _settle(controller, settle_steps)
    live = {o['objectId']: o for o in controller.last_event.metadata['objects']}
    reload_drift = {e['id']: round(math.dist((e['position']['x'], e['position']['y'], e['position']['z']), (live[e['id']]['position']['x'], live[e['id']]['position']['y'], live[e['id']]['position']['z'])), 3) for e in extra}
    reload_off_host = [e['id'] for e in extra if not _held(live[e['id']], e['id'])]
    reload_moving = [e['id'] for e in extra if live[e['id']].get('isMoving')]
    picks = [o for o in live.values() if o.get('pickupable')]
    report = {'rooms': [(r['id'], r.get('roomType')) for r in house['rooms']], 'walls': len(house.get('walls', [])), 'doors': [(d.get('id'), d.get('room0'), d.get('room1'), d.get('assetId')) for d in house.get('doors', [])], 'doors_recentred': doors_moved, 'rotated': turned, 'n_objects': len(house['objects']), 'n_live': len(live), 'pickupable': sorted((o['objectType'] for o in picks)), 'parented': {o['objectId']: o.get('parentReceptacles') or [] for o in picks}, 'shifts': shifts, 'overflow_before': {m: tuple((round(x, 3) for x in b)) for m, b in before.items() if any(b)}, 'overflow_after': {m: v for m, v in overflow.items() if any(v)}, 'lifts': lifts, 'floor_gap_after': gap_after, 'placement_tries': attempts, 'placement_fallbacks': fallbacks, 'hidden': sorted(inside), 'shut_after': shut_after, 'inset_relaxed': relaxed, 'off_host': off_host, 'settle': {'action': how, 'steps': n_settle, 'still_moving': still_moving, 'drift_m': drift}, 'reload': {'drift_m': reload_drift, 'off_host': reload_off_host, 'moving': reload_moving}, 'placement_rooms': {e['id']: types.get(room_localization((e['position']['x'], e['position']['z']), house)) for e in extra}}
    return (house, report)

def room_polygons(house):
    return {r['id']: [(p['x'], p['z']) for p in r['floorPolygon']] for r in house['rooms']}

def room_sizes(house):
    out = {}
    for r in house['rooms']:
        poly = [(p['x'], p['z']) for p in r['floorPolygon']]
        s = 0.0
        for i in range(len(poly)):
            x0, z0 = poly[i]
            x1, z1 = poly[(i + 1) % len(poly)]
            s += x0 * z1 - x1 * z0
        out[r['id']] = abs(s) / 2.0
    return out

def room_types(house):
    return {r['id']: r.get('roomType') for r in house['rooms']}
_GENERIC_ROOM_WORDS = frozenset({'room', 'area', 'space', 'zone', 'main', 'the'})
_ROOM_INDEX = re.compile('_\\d+$')
_PLACE_NOUN = re.compile('(room|way|hall|office|kitchen|lobby|closet|garage|corridor|pantry|foyer|studio|gym|porch|patio|deck|reception|basement|attic|cellar|nursery|library)$')

def room_name(room_id):
    if not room_id:
        return None
    raw = room_id.split('|')[-1]
    raw = _ROOM_INDEX.sub('', raw)
    toks = [t for t in re.split('[_\\s]+', raw.lower()) if t]
    if not any((re.search('[a-z]', t) for t in toks)):
        return None
    core = [t for t in toks if t not in _GENERIC_ROOM_WORDS and t != 'activity']
    if not core:
        core = toks
    name = ' '.join(core)
    if len(core) == 1 and (not _PLACE_NOUN.search(core[0])):
        name += ' room'
    return name

def room_names(house):
    return {r['id']: n for r in house['rooms'] if (n := room_name(r['id'])) is not None}

def point_in_polygon(pt, poly):
    x, z = pt
    n, hit = (len(poly), False)
    for i in range(n):
        x0, z0 = poly[i]
        x1, z1 = poly[(i + 1) % n]
        if (z0 > z) != (z1 > z):
            xi = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
            if x < xi:
                hit = not hit
    return hit

def room_localization(pt, house):
    for rid, poly in room_polygons(house).items():
        if point_in_polygon(pt, poly):
            return rid
    return None

def save(house, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(house, f, indent=1)
    return path

def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)
