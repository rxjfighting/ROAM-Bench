"""Goalimg utilities for ROAM-Bench."""
import os
from . import vision
CUTOUT_PX = 256
KIND_FAMILIES = ('object_count', 'count_in_room', 'room_count', 'room_comparison', 'hidden_count', 'state_count', 'both_in_room')
GLOBAL_EXEMPLAR_FAMILIES = ('count_absent', 'unanswerable')
NO_IMAGE_FAMILIES = ('room_area', 'topo_route')
DARK_MEAN = 25.0
DARK_LIT_FRAC = 0.15
INSTANCE_CAPTION = 'the {noun} this task refers to -- this exact object'
EXEMPLAR_CAPTION = "one visual example of the object category '{noun}' -- the task is about every {noun} in the house, and the others need not look like this one"
DIRNAME = 'goal_images'

def plan_for(spec, inv, exemplars=True):
    if spec.family in NO_IMAGE_FAMILIES:
        return ()
    by_alias = {str(f.alias): f for f in inv}
    if spec.referents:
        out = []
        for a in spec.referents:
            f = by_alias.get(str(a))
            if f is None:
                continue
            out.append((str(a), INSTANCE_CAPTION.format(noun=_noun_of(f.kind))))
        return tuple(out)
    if not exemplars or spec.family not in KIND_FAMILIES:
        return ()
    out = []
    for kind in spec.kinds or spec.receptacle_kinds:
        peers = sorted((str(f.alias) for f in inv if f.kind == kind))
        if not peers:
            continue
        out.append((peers[0], EXEMPLAR_CAPTION.format(noun=_noun_of(kind))))
    return tuple(out)

def global_exemplar_kinds(spec):
    if spec.referents or spec.family not in GLOBAL_EXEMPLAR_FAMILIES:
        return ()
    return tuple((str(kind) for kind in spec.kinds or spec.receptacle_kinds))

def graft_global_exemplars(specs, image_by_kind):
    output, missing = ([], {})
    for spec in specs:
        if spec.task.goal_images and (not spec.referents) and (spec.family in KIND_FAMILIES):
            kinds = tuple(spec.kinds or spec.receptacle_kinds)
            if len(kinds) == len(spec.task.goal_images):
                spec = _with_images(spec, tuple(((EXEMPLAR_CAPTION.format(noun=_noun_of(kind)), old[1]) for kind, old in zip(kinds, spec.task.goal_images))))
        kinds = global_exemplar_kinds(spec)
        if not kinds or spec.task.goal_images:
            output.append(spec)
            continue
        absent = tuple((kind for kind in kinds if kind not in image_by_kind))
        if absent:
            missing[spec.task.task_id] = absent
            output.append(spec)
            continue
        images = tuple(((EXEMPLAR_CAPTION.format(noun=_noun_of(kind)), str(image_by_kind[kind])) for kind in kinds))
        output.append(_with_images(spec, images))
    return (output, missing)

def _noun_of(kind):
    from .taskset import _noun
    return _noun(kind)

def render(b, specs, inv, out_dir, ep, exemplars=True, max_side=None, image_subdir=DIRNAME):
    max_side = CUTOUT_PX if max_side is None else max_side
    img_dir = os.path.join(out_dir, image_subdir)
    os.makedirs(img_dir, exist_ok=True)
    cache, misses = ({}, {})
    orbited = []
    dark = {}

    def png_for(alias):
        if alias in cache:
            return cache[alias]
        rgb, info = b.cutout(alias)
        b.reset(ep)
        if rgb is None:
            misses[str(info)] = misses.get(str(info), 0) + 1
            cache[alias] = None
            return None
        lum, lit = vision.legibility(rgb)
        if lum is not None and (lum < DARK_MEAN or lit < DARK_LIT_FRAC):
            dark[str(alias)] = (round(lum, 1), round(lit, 3))
        if isinstance(info, dict) and info.get('pose') == 'orbit':
            orbited.append(str(alias))
        rel = os.path.join(image_subdir, f'{alias}.png')
        with open(os.path.join(out_dir, rel), 'wb') as fh:
            fh.write(vision.png_bytes(rgb, max_side=max_side))
        cache[alias] = rel
        return rel
    out = []
    stats = {'with_images': 0, 'no_images': 0, 'incomplete': 0, 'images': 0, 'objects': 0, 'misses': misses}
    for s in specs:
        plan = plan_for(s, inv, exemplars=exemplars)
        if not plan:
            stats['no_images'] += 1
            out.append(s)
            continue
        pairs = [(caption, png_for(alias)) for alias, caption in plan]
        if any((p is None for _c, p in pairs)):
            stats['incomplete'] += 1
            out.append(s)
            continue
        stats['with_images'] += 1
        stats['images'] += len(pairs)
        out.append(_with_images(s, tuple(((c, p) for c, p in pairs))))
    stats['objects'] = sum((1 for v in cache.values() if v))
    stats['unphotographable'] = sorted((a for a, v in cache.items() if v is None))
    stats['orbit_posed'] = sorted(orbited)
    stats['dark'] = dict(sorted(dark.items()))
    return (out, stats)

def encode(images, base_dir):
    from PIL import Image
    import numpy as np
    out = []
    for caption, rel in images or ():
        path = rel if os.path.isabs(rel) else os.path.join(base_dir, rel)
        if not os.path.exists(path):
            raise FileNotFoundError(f'goal image {rel!r} is missing from {base_dir!r}. The instruction names its target by noun and relies on this picture to say which one, so running without it measures a different (unanswerable) task.')
        with Image.open(path) as im:
            rgb = np.asarray(im.convert('RGB'))
        out.append((caption, vision.data_url(rgb)))
    return tuple(out)

def regraft(specs, ts):
    by_id = {t.get('task_id'): tuple((tuple(g) for g in t.get('goal_images') or ())) for t in (ts or {}).get('tasks') or ()}
    n = 0
    for tid, sp in list(specs.items()):
        images = by_id.get(tid)
        if images:
            specs[tid] = _with_images(sp, images)
            n += 1
    return n

def _with_images(spec, images):
    import dataclasses
    return dataclasses.replace(spec, task=dataclasses.replace(spec.task, goal_images=images))

def note(stats):
    if not stats:
        return '目标图：没跑'
    miss = stats.get('misses') or {}
    tail = f"，{stats['incomplete']} 道因为有referent拍不到而没给图" if stats.get('incomplete') else ''
    why = f'；拍不到的原因 {dict(miss)}' if miss else ''
    orb = stats.get('orbit_posed') or ()
    ring = f'；{len(orb)} 张只有自由相机拍到（agent 视角看不见）{sorted(orb)}' if orb else ''
    dk = stats.get('dark') or {}
    blk = f'；{len(dk)} 张暗到可能读不出来（均值/亮像素占比）{dict(dk)}' if dk else ''
    return f"目标图：{stats.get('with_images', 0)} 道带图（共 {stats.get('images', 0)} 张，来自 {stats.get('objects', 0)} 个物体），{stats.get('no_images', 0)} 道按设计不带图{tail}{why}{ring}{blk}"
