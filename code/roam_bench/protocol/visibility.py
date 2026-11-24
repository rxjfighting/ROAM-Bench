"""Visibility utilities for ROAM-Bench."""
from __future__ import annotations
from typing import Any, Optional
MIN_VISIBLE_PIXELS = 16
DEFAULT_FRAME_WIDTH = 1280
DEFAULT_FRAME_HEIGHT = 720

def pixel_fraction(width: int, height: int, min_pixels: int=MIN_VISIBLE_PIXELS) -> float:
    pixels = int(width) * int(height)
    if pixels <= 0:
        raise ValueError(f'frame dimensions must be positive, got {width}x{height}')
    if int(min_pixels) <= 0:
        raise ValueError(f'min_pixels must be positive, got {min_pixels}')
    return float(min_pixels) / float(pixels)

def for_frame(width: int, height: int, *, override: Optional[float]=None, min_pixels: int=MIN_VISIBLE_PIXELS) -> float:
    if override is not None:
        value = float(override)
        if value < 0.0:
            raise ValueError(f'visibility fraction must be non-negative, got {value}')
        return value
    return pixel_fraction(width, height, min_pixels)

def for_sim(sim: Any, *, min_pixels: Optional[int]=None) -> float:
    episode = getattr(sim, '_episode', None)
    override = getattr(episode, 'view_min_pixel_fraction', None)
    requested_pixels = min_pixels if min_pixels is not None else getattr(episode, 'view_min_pixels', MIN_VISIBLE_PIXELS)
    width = getattr(sim, 'width', None)
    height = getattr(sim, 'height', None)
    if width is None or height is None:
        cam = getattr(getattr(sim, '_geom', None), 'cam', None)
        width = getattr(cam, 'width', DEFAULT_FRAME_WIDTH)
        height = getattr(cam, 'height', DEFAULT_FRAME_HEIGHT)
    return for_frame(width, height, override=override, min_pixels=requested_pixels)
