"""Mark utilities for ROAM-Bench."""
from PIL import ImageDraw
OFF, VIDEO, BOTH = ('off', 'video', 'both')
MODES = (OFF, VIDEO, BOTH)
INK = (255, 0, 255)
EDGE = (0, 0, 0)

def valid(mode):
    return mode in MODES

def marks_video(mode):
    return mode in (VIDEO, BOTH)

def marks_frames(mode):
    return mode == BOTH

def radius(w, h):
    return max(5, int(round(min(int(w), int(h)) / 90.0)))

def stamp(im, u, v, scale=1.0):
    x = float(u) * float(scale)
    y = float(v) * float(scale)
    r = radius(im.width, im.height)
    w = max(2, r // 5)
    draw = ImageDraw.Draw(im, 'RGBA')
    for colour, grow in ((EDGE, 1), (INK, 0)):
        lw = w + 2 * grow
        draw.ellipse([x - r - grow, y - r - grow, x + r + grow, y + r + grow], outline=colour, width=lw)
    return im
