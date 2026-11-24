"""Imgcheck utilities for ROAM-Bench."""
from __future__ import annotations
import logging
import math
import os
import numpy as np
from roam_bench.protocol import vision
if not hasattr(vision, 'legibility'):
    raise ImportError(f"roam_bench.protocol.vision has no legibility(): the bundled protocol ({getattr(vision, '__file__', '?')}) is incomplete. Do not add a second brightness implementation; repair the bundled protocol instead.")
_log = logging.getLogger('roam_bench.imgcheck')
_DEFAULT_MEAN_THRESHOLD = 25.0
_DEFAULT_NONBLACK_FRAC = 0.15

def _env_float(name: str, default: float) -> float:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return float(v)
    except ValueError:
        _log.warning('%s=%r is not a float; using %.3f', name, v, default)
        return default
DARK_MEAN_THRESHOLD = _env_float('ISAAC_DARK_MEAN_THRESHOLD', _DEFAULT_MEAN_THRESHOLD)
DARK_NONBLACK_FRAC = _env_float('ISAAC_DARK_NONBLACK_FRAC', _DEFAULT_NONBLACK_FRAC)

def _measure(rgb) -> tuple[float, float]:
    if rgb is None or np.asarray(rgb).size == 0:
        return (0.0, 0.0)
    mean, lit = vision.legibility(rgb)
    if mean is None or lit is None or math.isnan(mean) or math.isnan(lit):
        return (0.0, 0.0)
    return (float(mean), float(lit))

def frame_mean_brightness(rgb) -> float:
    return _measure(rgb)[0]

def frame_nonblack_fraction(rgb) -> float:
    return _measure(rgb)[1]

def frame_stats(rgb) -> dict:
    mean, frac = _measure(rgb)
    return {'mean': mean, 'nonblack_frac': frac}

def warn_if_dark(rgb, *, context: str='', threshold=None, frac_threshold=None, logger=None):
    thr = DARK_MEAN_THRESHOLD if threshold is None else float(threshold)
    fthr = DARK_NONBLACK_FRAC if frac_threshold is None else float(frac_threshold)
    lg = logger or _log
    mean, frac = _measure(rgb)
    ok = mean >= thr and frac >= fthr
    if not ok:
        reason = 'mean=%.1f < %.1f' % (mean, thr) if mean < thr else 'clearly-lit fraction=%.2f < %.2f' % (frac, fthr)
        lg.warning('target image too dark: %s (mean=%.1f, lit-fraction=%.2f)%s -- every task photographed here scores against pixels the model cannot see; do NOT ship silently (re-light the room or drop the frame).', reason, mean, frac, f' [{context}]' if context else '')
    return (ok, mean)
