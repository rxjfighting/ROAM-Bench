"""Vision utilities for ROAM-Bench."""
import base64
import hashlib
import io
import numpy as np
from PIL import Image
from . import mark
MIME = 'image/png'
MAX_SIDE = None
LIT_FLOOR = 32.0

def png_bytes(rgb, max_side=MAX_SIDE, point=None):
    im = Image.fromarray(rgb).convert('RGB')
    scale = 1.0
    if max_side and max(im.size) != max_side:
        scale = max_side / float(max(im.size))
        im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
    if point is not None:
        mark.stamp(im, point[0], point[1], scale=scale)
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return buf.getvalue()

def data_url(rgb, max_side=MAX_SIDE, point=None, mime=MIME, quality=90):
    if mime == MIME:
        encoded = png_bytes(rgb, max_side, point=point)
    elif mime == 'image/jpeg':
        im = Image.fromarray(rgb).convert('RGB')
        scale = 1.0
        if max_side and max(im.size) != max_side:
            scale = max_side / float(max(im.size))
            im = im.resize((max(1, round(im.width * scale)), max(1, round(im.height * scale))), Image.BICUBIC)
        if point is not None:
            mark.stamp(im, point[0], point[1], scale=scale)
        buf = io.BytesIO()
        im.save(buf, format='JPEG', quality=max(1, min(100, int(quality))), optimize=False, subsampling=0)
        encoded = buf.getvalue()
    else:
        raise ValueError(f'unsupported vision MIME type: {mime!r}')
    return 'data:' + mime + ';base64,' + base64.b64encode(encoded).decode('ascii')

def scaled_size(size, max_side):
    width, height = (int(size[0]), int(size[1]))
    side = max(1, min(int(max_side), max(width, height)))
    scale = side / float(max(width, height))
    return (max(1, round(width * scale)), max(1, round(height * scale)))

def data_url_ladder(rgb, max_sides, point=None, mime=MIME, quality=90, preserve_canvas=True):
    source = Image.fromarray(rgb).convert('RGB')
    canvas = source.size
    out = []
    seen = set()
    for requested in max_sides or (max(canvas),):
        side = max(1, min(int(requested), max(canvas)))
        if side in seen:
            continue
        seen.add(side)
        output_size = scaled_size(canvas, side)
        if side == max(canvas):
            im = source.copy()
        else:
            reduced = source.resize(output_size, Image.BICUBIC)
            im = reduced.resize(canvas, Image.BICUBIC) if preserve_canvas else reduced
        if point is not None:
            if preserve_canvas:
                marked = point
            else:
                marked = (point[0] * im.width / source.width, point[1] * im.height / source.height)
            mark.stamp(im, marked[0], marked[1], scale=1.0)
        buf = io.BytesIO()
        if mime == MIME:
            im.save(buf, format='PNG', optimize=True)
        elif mime == 'image/jpeg':
            im.save(buf, format='JPEG', quality=max(1, min(100, int(quality))), optimize=False, subsampling=0)
        else:
            raise ValueError(f'unsupported vision MIME type: {mime!r}')
        url = 'data:' + mime + ';base64,' + base64.b64encode(buf.getvalue()).decode('ascii')
        out.append((side, url))
    return tuple(out)

def pack_data_urls(urls, columns=2, mime='image/jpeg', quality=90):
    urls = tuple(urls or ())
    if not urls:
        raise ValueError('pack_data_urls needs at least one image')
    images = []
    for url in urls:
        head, sep, payload = str(url).partition(',')
        if not sep or not head.startswith('data:image/'):
            raise ValueError('pack_data_urls accepts image data URLs only')
        images.append(Image.open(io.BytesIO(base64.b64decode(payload))).convert('RGB'))
    columns = max(1, min(int(columns), len(images)))
    rows = (len(images) + columns - 1) // columns
    tile_w = max((im.width for im in images))
    tile_h = max((im.height for im in images))
    sheet = Image.new('RGB', (columns * tile_w, rows * tile_h), (0, 0, 0))
    for i, im in enumerate(images):
        x = i % columns * tile_w
        y = i // columns * tile_h
        sheet.paste(im, (x, y))
    buf = io.BytesIO()
    if mime == 'image/jpeg':
        sheet.save(buf, format='JPEG', quality=max(1, min(100, int(quality))), optimize=False, subsampling=0)
    elif mime == MIME:
        sheet.save(buf, format='PNG', optimize=True)
    else:
        raise ValueError(f'unsupported packed vision MIME type: {mime!r}')
    return 'data:' + mime + ';base64,' + base64.b64encode(buf.getvalue()).decode('ascii')

def legibility(rgb):
    a = np.asarray(rgb)
    if a.ndim != 3 or a.shape[-1] < 3:
        return (None, None)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    return (float(lum.mean()), float((lum > LIT_FLOOR).mean()))

def digest(url_or_bytes, n=12):
    raw = url_or_bytes.encode('ascii') if isinstance(url_or_bytes, str) else url_or_bytes
    return hashlib.sha256(raw).hexdigest()[:n]

def frame_note(url):
    if not url:
        return None
    head, _, b64 = url.partition(',')
    return {'mime': MIME if not head else head[5:].split(';')[0], 'bytes': len(base64.b64decode(b64)) if b64 else 0, 'sha256': digest(url)}
