"""Render cumulative robot odometry without scene semantics."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def _finite_pose(value):
    try:
        row = {key: float(value[key]) for key in ('x', 'z', 'yaw_deg')}
    except (KeyError, TypeError, ValueError):
        return None
    return row if all((math.isfinite(number) for number in row.values())) else None

def _grid_step(span: float) -> float:
    target = max(0.25, span / 6.0)
    power = 10 ** math.floor(math.log10(target))
    for factor in (1, 2, 5, 10):
        if factor * power >= target:
            return factor * power
    return 10 * power

def render_self_trajectory(poses, *, width: int=640, height: int=480):
    if width < 320 or height < 240:
        raise ValueError('self trajectory canvas must be at least 320x240')
    route = [row for value in poses if (row := _finite_pose(value)) is not None]
    if not route:
        raise ValueError('self trajectory requires at least one finite pose')
    xs, zs = ([p['x'] for p in route], [p['z'] for p in route])
    span = max(max(xs) - min(xs), max(zs) - min(zs), 4.0)
    pad_world = max(1.0, span * 0.12)
    cx = (min(xs) + max(xs)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    half = span / 2.0 + pad_world
    min_x, max_x, min_z, max_z = (cx - half, cx + half, cz - half, cz + half)
    left, right, top, bottom = (66, 20, 70, 54)
    plot_w, plot_h = (width - left - right, height - top - bottom)
    scale = min(plot_w / (max_x - min_x), plot_h / (max_z - min_z))
    sx = lambda x: left + (x - min_x) * scale
    sy = lambda z: height - bottom - (z - min_z) * scale
    image = Image.new('RGB', (width, height), '#fbfcfe')
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((18, 14), 'SELF TRAJECTORY (ODOMETRY ONLY)', fill='#14213d', font=font)
    draw.text((18, 32), 'No floor plan, walls, rooms, or task-object locations', fill='#52606d', font=font)
    current = route[-1]
    pose_text = f"current: x={current['x']:.2f} m, z={current['z']:.2f} m, yaw={current['yaw_deg'] % 360:.1f} deg | poses={len(route)}"
    draw.text((18, 49), pose_text, fill='#263238', font=font)
    step = _grid_step(max_x - min_x)
    start_x = math.ceil(min_x / step) * step
    start_z = math.ceil(min_z / step) * step
    grid_color = '#dce4ec'
    value = start_x
    while value <= max_x + 1e-09:
        px = sx(value)
        draw.line((px, top, px, height - bottom), fill=grid_color, width=1)
        draw.text((px + 2, height - bottom + 5), f'{value:g}', fill='#607080', font=font)
        value += step
    value = start_z
    while value <= max_z + 1e-09:
        py = sy(value)
        draw.line((left, py, width - right, py), fill=grid_color, width=1)
        draw.text((8, py - 6), f'{value:g}', fill='#607080', font=font)
        value += step
    draw.rectangle((left, top, width - right, height - bottom), outline='#91a4b7', width=2)
    draw.text((width - 55, height - 26), '+X m', fill='#34495e', font=font)
    draw.text((18, top - 14), '+Z m ↑', fill='#34495e', font=font)
    points = [(sx(p['x']), sy(p['z'])) for p in route]
    if len(points) > 1:
        draw.line(points, fill='#2563eb', width=4, joint='curve')
    for index, (px, py) in enumerate(points):
        if index == 0 or index == len(points) - 1 or index % 5 == 0:
            radius = 6 if index in (0, len(points) - 1) else 4
            color = '#15803d' if index == 0 else '#dc2626' if index == len(points) - 1 else '#2563eb'
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline='white', width=2)
            draw.text((px + 7, py - 12), str(index + 1), fill='#1f2937', font=font)
    px, py = points[-1]
    yaw = math.radians(current['yaw_deg'])
    length = 24
    tip = (px + length * math.sin(yaw), py - length * math.cos(yaw))
    draw.line((px, py, tip[0], tip[1]), fill='#b91c1c', width=4)
    angle = math.atan2(tip[1] - py, tip[0] - px)
    for delta in (-2.55, 2.55):
        wing = (tip[0] + 10 * math.cos(angle + delta), tip[1] + 10 * math.sin(angle + delta))
        draw.line((tip[0], tip[1], wing[0], wing[1]), fill='#b91c1c', width=3)
    meta = {'pose_count': len(route), 'grid_step_m': step, 'bounds': {'min_x': min_x, 'max_x': max_x, 'min_z': min_z, 'max_z': max_z}, 'current': dict(current), 'contains_scene_geometry': False}
    return (np.asarray(image), meta)
