"""Seg alias utilities for ROAM-Bench."""
import re
import numpy as np
_OBJS_PREFIX = '/World/objs/'
_TRAILING_IDX = re.compile('^(.*)_(\\d+)$')
_BACKGROUND_LABELS = {'', 'BACKGROUND', 'UNLABELLED', 'unlabelled', 'unlabeled'}

def _safe(name: str) -> str:
    return re.sub('[^A-Za-z0-9_]', '_', name)

def build_safe2alias(known_aliases):
    out = {}
    for a in known_aliases:
        s = _safe(a)
        if s in out and out[s] != a:
            raise ValueError(f'_safe collision: {a!r} and {out[s]!r} both map to {s!r}')
        out[s] = a
    return out

def instance_root_of(prim_path):
    if not prim_path or not prim_path.startswith(_OBJS_PREFIX):
        return None
    parts = prim_path.split('/')
    if len(parts) < 5 or parts[1] != 'World' or parts[2] != 'objs':
        return None
    return '/'.join(parts[:5])

def alias_of_root(root_path, safe2alias):
    if not root_path:
        return None
    name = root_path.rsplit('/', 1)[-1]
    m = _TRAILING_IDX.match(name)
    if not m:
        return None
    return safe2alias.get(m.group(1))

def resolve_label(label, safe2alias):
    if label is None:
        return None
    label = str(label)
    if label in _BACKGROUND_LABELS:
        return None
    if label.startswith(_OBJS_PREFIX):
        return alias_of_root(instance_root_of(label), safe2alias)
    if label in safe2alias:
        return safe2alias[label]
    return safe2alias.get(_safe(label))

def _lookup_label(id_to_labels, seg_id):
    if seg_id in id_to_labels:
        return id_to_labels[seg_id]
    return id_to_labels.get(str(seg_id))

def alias_pixel_counts(seg_array, id_to_labels, safe2alias):
    seg = np.asarray(seg_array)
    ids, counts = np.unique(seg, return_counts=True)
    per_alias = {}
    for seg_id, c in zip(ids.tolist(), counts.tolist()):
        alias = resolve_label(_lookup_label(id_to_labels, seg_id), safe2alias)
        if alias is None:
            continue
        per_alias[alias] = per_alias.get(alias, 0) + int(c)
    return (per_alias, int(seg.size))

def alias_mask_from_seg(seg_array, id_to_labels, safe2alias, alias):
    seg = np.asarray(seg_array)
    ids = [sid for sid in np.unique(seg).tolist() if resolve_label(_lookup_label(id_to_labels, sid), safe2alias) == alias]
    return np.isin(seg, ids) if ids else np.zeros(seg.shape[:2], dtype=bool)

def alias_masks_from_seg(seg_array, id_to_labels, safe2alias, only=None):
    seg = np.asarray(seg_array)
    wanted = set(only) if only is not None else None
    ids_by_alias = {}
    for sid in np.unique(seg).tolist():
        alias = resolve_label(_lookup_label(id_to_labels, sid), safe2alias)
        if alias and (wanted is None or alias in wanted):
            ids_by_alias.setdefault(alias, []).append(sid)
    return {alias: np.isin(seg, ids) for alias, ids in ids_by_alias.items()}

def visible_instances_from_seg(seg_array, id_to_labels, safe2alias, min_pixel_frac):
    per_alias, total = alias_pixel_counts(seg_array, id_to_labels, safe2alias)
    if total == 0:
        return set()
    thr = 0.0 if min_pixel_frac is None else float(min_pixel_frac)
    return {a for a, c in per_alias.items() if c / total >= thr}

def instance_at_pixel_from_seg(seg_array, id_to_labels, safe2alias, u, v):
    seg = np.asarray(seg_array)
    h, w = seg.shape[:2]
    if not (0 <= v < h and 0 <= u < w):
        return None
    return resolve_label(_lookup_label(id_to_labels, int(seg[v, u])), safe2alias)

def instances_in_patch_from_seg(seg_array, id_to_labels, safe2alias, u, v, k=2):
    seg = np.asarray(seg_array)
    h, w = seg.shape[:2]
    v0, v1 = (max(0, v - k), min(h, v + k + 1))
    u0, u1 = (max(0, u - k), min(w, u + k + 1))
    out = set()
    for seg_id in set(seg[v0:v1, u0:u1].flatten().tolist()):
        alias = resolve_label(_lookup_label(id_to_labels, seg_id), safe2alias)
        if alias is not None:
            out.add(alias)
    return out

def instance_centroid_from_seg(seg_array, id_to_labels, safe2alias, alias):
    seg = np.asarray(seg_array)
    ids = [sid for sid in np.unique(seg).tolist() if resolve_label(_lookup_label(id_to_labels, sid), safe2alias) == alias]
    if not ids:
        return None
    mask = np.isin(seg, ids)
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return None
    cy, cx = (ys.mean(), xs.mean())
    h, w = mask.shape
    pad = 2
    for i in np.argsort((ys - cy) ** 2 + (xs - cx) ** 2):
        y, x = (int(ys[i]), int(xs[i]))
        if pad <= y < h - pad and pad <= x < w - pad and mask[y - pad:y + pad + 1, x - pad:x + pad + 1].all():
            return (int(x), int(y))
    return (int(xs[0]), int(ys[0]))
