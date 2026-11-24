"""Goal image utilities for ROAM-Bench."""
from __future__ import annotations
import dataclasses
import os
import numpy as np
from . import seg_alias
CUTOUT_MIN_FRAC = 0.01
CUTOUT_PAD = 0.04
CUTOUT_ORBIT_ENOUGH = 0.25
CUTOUT_ORBIT_MIN_RADIUS = 0.3
CUTOUT_NATURAL_MIN_BBOX_RATIO = 0.25
DIRNAME = 'goal_images'

def alias_mask_from_seg(seg_array, id_to_labels, safe2alias, alias: str) -> np.ndarray:
    seg = np.asarray(seg_array)
    mask = np.zeros(seg.shape[:2], dtype=bool)
    for seg_id in np.unique(seg).tolist():
        label = seg_alias._lookup_label(id_to_labels, seg_id)
        if seg_alias.resolve_label(label, safe2alias) == alias:
            mask |= seg == seg_id
    return mask

def _shift(array: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.zeros_like(array)
    h, w = array.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = array[max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out

def fill_over(rgb, keep, cover, rounds: int=64) -> np.ndarray:
    out = np.asarray(rgb).copy()
    known = np.asarray(keep, dtype=bool).copy()
    todo = np.asarray(cover, dtype=bool) & ~known
    for _ in range(rounds):
        if not todo.any():
            return out
        acc = np.zeros(out.shape, dtype=np.uint32)
        count = np.zeros(known.shape, dtype=np.uint16)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            near = _shift(known, dy, dx)
            acc[near] += _shift(out, dy, dx)[near]
            count[near] += 1
        grow = todo & (count > 0)
        if not grow.any():
            break
        out[grow] = (acc[grow] // count[grow][:, None]).astype(np.uint8)
        known |= grow
        todo &= ~grow
    if todo.any() and np.asarray(keep).any():
        out[todo] = np.median(np.asarray(rgb)[np.asarray(keep, dtype=bool)], axis=0)
    return out

def compose_cutout(rgb, target_mask, cover_mask=None, pad_frac: float=CUTOUT_PAD):
    rgb = np.asarray(rgb)
    keep = np.asarray(target_mask, dtype=bool)
    if rgb.ndim != 3 or rgb.shape[2] < 3 or keep.shape != rgb.shape[:2]:
        return (None, 'rgb/mask shape mismatch')
    if not keep.any():
        return (None, 'no target pixels')
    rgb = rgb[:, :, :3].astype(np.uint8, copy=False)
    cover = np.zeros_like(keep) if cover_mask is None else np.asarray(cover_mask, dtype=bool) & ~keep
    ys, xs = np.nonzero(keep)
    h, w = keep.shape
    x0, x1 = (int(xs.min()), int(xs.max()))
    y0, y1 = (int(ys.min()), int(ys.max()))
    clipped = bool(x0 == 0 or y0 == 0 or x1 == w - 1 or (y1 == h - 1))
    pad = max(2, int(float(pad_frac) * max(x1 - x0, y1 - y0)))
    x0, x1 = (max(0, x0 - pad), min(w - 1, x1 + pad))
    y0, y1 = (max(0, y0 - pad), min(h - 1, y1 + pad))
    crop = rgb[y0:y1 + 1, x0:x1 + 1].copy()
    crop_keep = keep[y0:y1 + 1, x0:x1 + 1]
    crop_cover = cover[y0:y1 + 1, x0:x1 + 1]
    if crop_cover.any():
        crop = fill_over(crop, crop_keep, crop_cover)
    crop[~(crop_keep | crop_cover)] = 255
    ch, cw = crop.shape[:2]
    side = max(ch, cw)
    canvas = np.full((side, side, 3), 255, dtype=np.uint8)
    oy, ox = ((side - ch) // 2, (side - cw) // 2)
    canvas[oy:oy + ch, ox:ox + cw] = crop
    return (canvas, {'crop': (cw, ch), 'clipped': clipped, 'covered': int(crop_cover.sum())})

def best_shot(shots, min_frac: float=CUTOUT_MIN_FRAC):
    """Prefer complete projections with visible width and height, then mask area."""
    usable = [shot for shot in shots if shot and float(shot.get('frac', 0.0)) >= min_frac and (not bool(shot.get('clipped', False)))]
    if not usable:
        return None
    return max(enumerate(usable), key=lambda pair: (
        min(int(pair[1].get('bbox_w', 0)), int(pair[1].get('bbox_h', 0))),
        float(pair[1]['frac']), -pair[0]))[1]

def projected_bbox_ratio(shot) -> float:
    """Return the ratio of the shorter to the longer projected dimension."""
    width = int((shot or {}).get('bbox_w', 0))
    height = int((shot or {}).get('bbox_h', 0))
    if width <= 0 or height <= 0:
        return 1.0
    return min(width, height) / float(max(width, height))

def natural_projection_is_clear(shot) -> bool:
    """Check whether a reachable camera view can skip additional orbit views."""
    return bool(shot and float(shot.get('frac', 0.0)) >= CUTOUT_MIN_FRAC
                and not bool(shot.get('clipped', False))
                and projected_bbox_ratio(shot) >= CUTOUT_NATURAL_MIN_BBOX_RATIO)

def orbit_radii(extent: float) -> tuple[float, ...]:
    return tuple((max(CUTOUT_ORBIT_MIN_RADIUS, float(extent) * scale) for scale in (0.8, 1.3, 2.2, 3.5)))

def isolation_hide_paths(object_roots, target_roots, extra_roots=()):
    targets = {str(path).rstrip('/') for path in target_roots}
    hidden = []
    for raw in tuple(object_roots) + tuple(extra_roots):
        path = str(raw).rstrip('/')
        keeps_target = any((target == path or target.startswith(path + '/') for target in targets))
        if path and (not keeps_target):
            hidden.append(path)
    return tuple(sorted(set(hidden)))

def render_specs(backend, specs, inventory, out_dir, episode, *, max_side: int=256, exemplars: bool=True, image_subdir: str=DIRNAME):
    from roam_bench.protocol import goalimg, vision
    image_dir = os.path.join(out_dir, image_subdir)
    os.makedirs(image_dir, exist_ok=True)
    cache = {}
    misses = {}
    orbited = []
    dark = {}
    captures = {}

    def png_for(alias):
        if alias in cache:
            return cache[alias]
        rgb, info = backend.cutout(alias)
        if rgb is None:
            misses[str(info)] = misses.get(str(info), 0) + 1
            cache[alias] = None
            return None
        mean, lit = vision.legibility(rgb)
        if mean is not None and (mean < 25.0 or lit < 0.15):
            dark[str(alias)] = (round(float(mean), 1), round(float(lit), 3))
        if isinstance(info, dict) and info.get('pose') == 'orbit':
            orbited.append(str(alias))
        if isinstance(info, dict):
            captures[str(alias)] = dict(info)
        rel = os.path.join(image_subdir, f'{alias}.png')
        with open(os.path.join(out_dir, rel), 'wb') as handle:
            handle.write(vision.png_bytes(rgb, max_side=max_side))
        cache[alias] = rel
        return rel
    output = []
    stats = {'with_images': 0, 'no_images': 0, 'incomplete': 0, 'images': 0, 'objects': 0, 'misses': misses}
    for spec in specs:
        plan = goalimg.plan_for(spec, inventory, exemplars=exemplars)
        if not plan:
            stats['no_images'] += 1
            output.append(spec)
            continue
        pairs = [(caption, png_for(alias)) for alias, caption in plan]
        if any((path is None for _caption, path in pairs)):
            stats['incomplete'] += 1
            output.append(spec)
            continue
        images = tuple(((caption, path) for caption, path in pairs))
        task = dataclasses.replace(spec.task, goal_images=images)
        output.append(dataclasses.replace(spec, task=task))
        stats['with_images'] += 1
        stats['images'] += len(images)
    stats['objects'] = sum((path is not None for path in cache.values()))
    stats['unphotographable'] = sorted((alias for alias, path in cache.items() if path is None))
    stats['orbit_posed'] = sorted(orbited)
    stats['dark'] = dict(sorted(dark.items()))
    stats['captures'] = dict(sorted(captures.items()))
    return (output, stats)
