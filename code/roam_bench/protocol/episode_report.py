"""Episode report utilities for ROAM-Bench."""
import base64
import html
import io
import os
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont
from . import mark
CRF = 18
FPS = 8
THUMB = 160
FILM_FPS = 24
FILM_SCALE = 1
FILM_HOLD_S = 0.5
MIN_ACTION_FRAMES = 8
_FONT_PATHS = ('/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf')
_font_cache = {}

def _film_font(px):
    if px in _font_cache:
        return _font_cache[px]
    font = None
    for path in _FONT_PATHS:
        try:
            font = ImageFont.truetype(path, px)
            break
        except Exception:
            continue
    if font is None:
        try:
            font = ImageFont.load_default(size=px)
        except Exception:
            font = ImageFont.load_default()
    _font_cache[px] = font
    return font

class FilmStrip:

    def __init__(self, path, fps=FILM_FPS, scale=FILM_SCALE, crf=CRF, min_action_frames=MIN_ACTION_FRAMES):
        self.path = path
        self.fps = int(fps)
        self.scale = max(1, int(scale))
        self.crf = int(crf)
        self.min_action_frames = max(1, int(min_action_frames))
        self.label = ''
        self.n_frames = 0
        self.n_actions = 0
        self._proc = None
        self._dead = False
        self._last = None
        self._action_start = 0
        self._aim = None

    def aim(self, u=None, v=None):
        self._floor()
        self._aim = None if u is None else (float(u), float(v))

    def track(self, uv):
        self._aim = None if uv is None else (float(uv[0]), float(uv[1]))

    def begin(self, label):
        self._floor()
        self.label = _ascii(label)
        self.n_actions += 1
        self._action_start = self.n_frames

    def add(self, frame):
        self._last = frame
        self._write(frame, self.label)

    def mark(self):
        return self.n_frames

    def _floor(self):
        if self._last is None:
            return
        for _ in range(self.min_action_frames - (self.n_frames - self._action_start)):
            self._write(self._last, self.label)

    def hold(self, label=None, seconds=FILM_HOLD_S):
        if self._last is None:
            return
        if label is not None:
            self.begin(label)
        for _ in range(max(1, int(round(seconds * self.fps)))):
            self._write(self._last, self.label)

    def finish(self):
        if self._proc is None:
            return None
        self._floor()
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        try:
            self._proc.wait(timeout=300)
        except Exception:
            self._proc.kill()
            return None
        ok = self._proc.returncode == 0 and os.path.exists(self.path)
        self._proc = None
        return self.path if ok else None

    def abort(self):
        self._dead = True
        if self._proc is not None:
            try:
                self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def _write(self, frame, label):
        if self._dead:
            return
        try:
            im = self._annotate(frame, label)
            if self._proc is None and (not self._spawn(im.size)):
                return
            im.save(self._proc.stdin, format='PNG', compress_level=1)
            self.n_frames += 1
        except Exception:
            self.abort()

    def _spawn(self, size):
        if shutil.which('ffmpeg') is None:
            self._dead = True
            return False
        cmd = ['ffmpeg', '-y', '-f', 'image2pipe', '-vcodec', 'png', '-framerate', str(self.fps), '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(self.crf), '-preset', 'medium', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', self.path]
        try:
            self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            self._dead = True
            return False
        return True

    def _annotate(self, frame, label):
        im = Image.fromarray(frame).convert('RGB')
        if self.scale > 1:
            im = im.resize((im.width * self.scale, im.height * self.scale), Image.BICUBIC)
        if self._aim is not None:
            mark.stamp(im, self._aim[0], self._aim[1], scale=self.scale)
        if not label:
            return im
        px = max(10, im.height // 22)
        font = _film_font(px)
        draw = ImageDraw.Draw(im, 'RGBA')
        pad = px // 2
        bar = px + 2 * pad
        draw.rectangle([0, im.height - bar, im.width, im.height], fill=(0, 0, 0, 150))
        draw.text((pad, im.height - bar + pad), label, font=font, fill=(255, 255, 255, 255))
        return im

def _ascii(text):
    s = '' if text is None else str(text)
    return s.encode('ascii', 'replace').decode('ascii')

class _NoStep:
    __slots__ = ('success', 'reason', 'moved_m', 'collided')

    def __init__(self, success, reason):
        self.success = bool(success)
        self.reason = reason
        self.moved_m = 0.0
        self.collided = False

class EpisodeFilm:

    def __init__(self, sim, title=''):
        self.sim = sim
        self.title = title
        self.rows = []

    def capture(self, kind, arg, result, note='', thought='', prompt='', raw=''):
        frame = None
        try:
            frame = self.sim.observe().rgb
        except Exception:
            pass
        self.rows.append({'i': len(self.rows) + 1, 'kind': kind, 'arg': _short(arg), 'ok': bool(result.success), 'reason': getattr(result, 'reason', ''), 'note': note, 'thought': thought, 'prompt': prompt, 'raw': raw, 'moved': round(getattr(result, 'moved_m', 0.0) or 0.0, 3), 'collided': bool(getattr(result, 'collided', False)), 'frame': frame})

    def note_round(self, kind, arg, ok, reason, note='', thought='', prompt='', raw=''):
        self.capture(kind, arg, _NoStep(bool(ok), reason), note=note, thought=thought, prompt=prompt, raw=raw)

    def as_callback(self):

        def cb(_pol, kind, arg, result, note):
            self.capture(kind, arg, result, note)
        return cb

    def as_recorder_callback(self):

        def cb(kind, arg, result):
            self.capture(kind, arg, result)
        return cb

    def write_video(self, path, fps=FPS, hold=1):
        frames = [r['frame'] for r in self.rows if r['frame'] is not None]
        if not frames or shutil.which('ffmpeg') is None:
            return None
        tmp = path + '.frames'
        os.makedirs(tmp, exist_ok=True)
        try:
            n = 0
            for f in frames:
                im = Image.fromarray(f)
                for _ in range(max(1, hold)):
                    im.save(os.path.join(tmp, f'f_{n:05d}.png'))
                    n += 1
            cmd = ['ffmpeg', '-y', '-framerate', str(fps), '-i', os.path.join(tmp, 'f_%05d.png'), '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(CRF), '-preset', 'medium', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', path]
            subprocess.run(cmd, check=True, capture_output=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return path

    def write_html(self, path, score=None, meta=None, video=None, intro='', back='', withheld=False):
        has_model = any((r.get('thought') or r.get('prompt') for r in self.rows))
        rows = []
        for r in self.rows:
            thumb = _b64_thumb(r['frame'])
            img = f'<img src="data:image/png;base64,{thumb}" width="{THUMB}">' if thumb else '<div class="noimg">no frame</div>'
            extra = []
            if r['moved']:
                extra.append(f"位移 {r['moved']:.2f} m")
            if r['collided']:
                extra.append('撞停')
            model = ''
            if has_model:
                bits = [f"""<div class="thought">{html.escape(r.get('thought') or '')}</div>"""]
                if r.get('raw'):
                    bits.append(f"<details><summary>原始回复</summary><pre>{html.escape(r['raw'])}</pre></details>")
                if r.get('prompt'):
                    bits.append(f"<details><summary>这一轮模型收到的 prompt</summary><pre>{html.escape(r['prompt'])}</pre></details>")
                model = f"""<td class="model">{''.join(bits)}</td>"""
            rows.append(f'''<tr class="{('ok' if r['ok'] else 'bad')}"><td>{r['i']}</td><td>{img}</td>{model}<td><code>{html.escape(r['kind'])}({html.escape(r['arg'])})</code><div class="note">{html.escape(r['note'])}</div></td><td>{('成功' if r['ok'] else '失败')}<div class="note">{html.escape(' · '.join(extra))}</div></td><td class="reason">{html.escape(r['reason'])}</td></tr>''')
        head = []
        for k, v in (meta or {}).items():
            head.append(f'<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>')
        if score is not None:
            head.append(f'<tr><th>判分</th><td>success=<b>{score.success}</b>，progress=<b>{score.progress:.2f}</b>，步数 {score.steps}，无效动作 {score.invalid_actions}，碰撞 {score.collisions}</td></tr>')
            for label, expr, when, ok in score.goals:
                head.append(f"""<tr><th>目标</th><td>{('✅' if ok else '❌')} {html.escape(label)} <code>{html.escape(expr)}</code> <span class="note">when={when}</span></td></tr>""")
        vid = ''
        if video:
            vid = f'<video controls width="360" src="{html.escape(os.path.basename(video))}"></video><div class="note">第一人称，与判分用的是同一台相机（the evaluation protocol）；录像不进入判分链路</div>'
        cols = ['#', '画面'] + (['模型（推理 / 原文 / prompt）'] if has_model else []) + ['动作', '结果', '仿真器真实回执（<b>模型没收到</b>，只有你能看）' if withheld else '回执（模型收到的原文）']
        with open(path, 'w', encoding='utf-8') as f:
            f.write(_PAGE.format(title=html.escape(self.title), head=''.join(head), rows=''.join(rows), video=vid, n=len(self.rows), intro=intro, back=f'<p><a href="{html.escape(back)}">← 返回索引</a></p>' if back else '', cols=''.join((f'<th>{c}</th>' for c in cols))))
        return path

def _short(arg):
    for attr in ('name',):
        if hasattr(arg, attr):
            return str(getattr(arg, attr))
    if hasattr(arg, 'u'):
        return f'{arg.u},{arg.v}'
    if hasattr(arg, 'deg'):
        return f'{arg.deg:+.0f}°/{arg.dist_m:.2f}m'
    if isinstance(arg, float):
        return f'{arg:.2f}'
    return '' if arg is None else str(arg)

def _b64_thumb(frame):
    if frame is None:
        return None
    im = Image.fromarray(frame)
    im.thumbnail((THUMB, THUMB))
    buf = io.BytesIO()
    im.save(buf, format='PNG', optimize=True)
    return base64.b64encode(buf.getvalue()).decode('ascii')
_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px system-ui,sans-serif;margin:24px;color:#172235}}table{{border-collapse:collapse;width:100%;margin:16px 0}}th,td{{border:1px solid #cbd5e1;padding:8px;text-align:left;vertical-align:top}}img{{max-width:320px;height:auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}.note{{font-size:13px;color:#475569}}.bad{{background:#fff1f2}}.ok{{background:#f0fdf4}}</style></head>
<body>{back}<h1>{title}</h1>{intro}<table>{head}</table>{video}<p>{n} recorded actions</p><table><thead><tr>{cols}</tr></thead><tbody>{rows}</tbody></table></body></html>"""
