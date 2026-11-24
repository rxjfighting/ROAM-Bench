"""Llm utilities for ROAM-Bench."""
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from . import feedback
from .tiers import MANIPULATION
from .types import DirectionRef, NameRef, PixelRef
DEFAULT_BASE = 'https://api.openai.com/v1'
DEFAULT_MODEL = 'model'
DEFAULT_OUTPUT_TOKENS = 32768
MAX_RETRY_OUTPUT_TOKENS = 65536
KEY_VARS = ('ROAM_BENCH_API_KEY', 'OPENAI_API_KEY')
_SECRET_SHAPE = re.compile('(^|[^A-Za-z0-9])(sk-[A-Za-z0-9]{16,})')

def _redact_secrets(text):
    return _SECRET_SHAPE.sub(lambda m: m.group(1) + 'sk-<redacted>', text)

class ChatError(RuntimeError):
    pass

class RequestBodyTooLarge(ChatError):
    pass

class EpisodeTokenBudgetExceeded(ChatError):
    pass

def bounded_image_messages(messages, maximum=0, client=None):
    request = [dict(m, content=[dict(p) for p in m['content']]) if isinstance(m.get('content'), list) else dict(m) for m in messages]
    original = _count_image_parts(request)
    removed = []

    def remove_oldest():
        for mi, msg in enumerate(request):
            content = msg.get('content')
            if not isinstance(content, list):
                continue
            for pi, part in enumerate(content):
                if part.get('type') == 'image_url':
                    removed.append({'message': mi, 'part': pi})
                    del content[pi]
                    return True
        return False
    while maximum and _count_image_parts(request) > maximum:
        remove_oldest()
    byte_limit = getattr(client, 'MAX_REQUEST_BYTES', 48000000)
    while client and client.request_size(request) > byte_limit:
        if _count_image_parts(request) <= 1 or not remove_oldest():
            raise RequestBodyTooLarge('text plus the current image exceeds request body limit')
    event = {'policy': 'recent-images-v1', 'limit': maximum, 'original_parts': original, 'retained_parts': _count_image_parts(request), 'dropped_images': len(removed), 'removed': removed, 'request_bytes': client.request_size(request) if client else None} if removed else None
    return (request, event)

class ChatClient:

    def __init__(self, base_url=None, api_key=None, model=DEFAULT_MODEL, temperature=None, max_tokens=DEFAULT_OUTPUT_TOKENS, timeout=120, retries=10):
        self.base = (base_url or os.environ.get('ROAM_BENCH_BASE_URL') or DEFAULT_BASE).rstrip('/')
        self.key = api_key or next((os.environ[k] for k in KEY_VARS if os.environ.get(k)), None)
        if not self.key:
            raise ChatError('no API key: set ' + ' or '.join(KEY_VARS))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failures = []
        self.throttled = 0
        self.dropped = []
        self.image_parts = 0
        self.last_request_bytes = 0
        self.largest_request_bytes = 0
        self.request_bytes = []
        self._policy_debugged = False

    def _dump_policy_images(self, messages):
        root_value = os.environ.get('ROAM_BENCH_POLICY_DEBUG_DIR')
        if not root_value or self._policy_debugged:
            return
        self._policy_debugged = True
        stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
        target = Path(root_value) / f'{stamp}_pid{os.getpid()}'
        target.mkdir(parents=True, exist_ok=False)
        records = []
        image_index = 0
        for message_index, message in enumerate(messages):
            content = message.get('content') if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for part_index, part in enumerate(content):
                if not isinstance(part, dict) or part.get('type') != 'image_url':
                    continue
                image_value = part.get('image_url') or {}
                url = image_value.get('url') if isinstance(image_value, dict) else None
                if not isinstance(url, str) or not url.startswith('data:image/'):
                    continue
                try:
                    header, encoded = url.split(',', 1)
                    raw = base64.b64decode(encoded, validate=True)
                except (ValueError, TypeError):
                    continue
                mime = header[5:].split(';', 1)[0]
                extension = 'png' if mime == 'image/png' else 'jpg'
                name = f'image_{image_index:03d}_m{message_index}_p{part_index}.{extension}'
                (target / name).write_bytes(raw)
                records.append({'index': image_index, 'message_index': message_index, 'part_index': part_index, 'mime': mime, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'file': name})
                image_index += 1
        (target / 'manifest.json').write_text(json.dumps({'schema': 'ROAM-Bench-content-policy-image-debug-v1', 'schema_version': 1, 'image_count': len(records), 'images': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    def _body(self, token_scale=1):
        body = {'model': self.model, 'messages': None}
        if self.temperature is not None:
            body['temperature'] = self.temperature
        if self.max_tokens is not None:
            ceiling = max(self.max_tokens, MAX_RETRY_OUTPUT_TOKENS)
            body['max_tokens'] = min(self.max_tokens * token_scale, ceiling)
        for k in self.dropped:
            body.pop(k, None)
        return body

    def request_size(self, messages, token_scale=1):
        body = self._body(token_scale=token_scale)
        body['messages'] = messages
        return len(json.dumps(body).encode('utf-8'))
    DROPPABLE = ('temperature', 'max_tokens', 'top_p')
    MAX_TOKEN_SCALE = 16
    MAX_BACKOFF = 60.0
    MAX_REQUEST_BYTES = 48000000

    def _droppable_from(self, detail):
        low = (detail or '').lower()
        for k in self.DROPPABLE:
            if k in low and k not in self.dropped:
                return k
        return None

    def complete(self, messages):
        last = ''
        scale = 1
        self.image_parts += _count_image_parts(messages)
        for attempt in range(self.retries):
            body = self._body(token_scale=scale)
            body['messages'] = messages
            guard = getattr(self, 'episode_token_guard', None)
            if guard:
                guard(messages, body)
            payload_bytes = json.dumps(body).encode('utf-8')
            self.last_request_bytes = len(payload_bytes)
            self.largest_request_bytes = max(self.largest_request_bytes, self.last_request_bytes)
            self.request_bytes.append(self.last_request_bytes)
            if self.last_request_bytes > self.MAX_REQUEST_BYTES:
                raise RequestBodyTooLarge(f'request body is {self.last_request_bytes} bytes; local safety limit is {self.MAX_REQUEST_BYTES} bytes (gateway limit 50000000)')
            req = urllib.request.Request(self.base + '/v1/chat/completions', data=payload_bytes, headers={'Authorization': f'Bearer {self.key}', 'Content-Type': 'application/json'})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode('utf-8'))
                self.calls += 1
                usage = payload.get('usage') or {}
                self.prompt_tokens += int(usage.get('prompt_tokens') or 0)
                self.completion_tokens += int(usage.get('completion_tokens') or 0)
                choices = payload.get('choices') or []
                if not choices:
                    last = 'empty choices'
                    continue
                msg = choices[0].get('message') or {}
                text = (msg.get('content') or '').strip()
                if text:
                    return text
                fin = choices[0].get('finish_reason') or '?'
                last = f"empty content (finish_reason={fin}, max_tokens={body.get('max_tokens')})"
                scale = min(scale * 2, self.MAX_TOKEN_SCALE)
                time.sleep(0.5)
                continue
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError) as exc:
                last = f'{type(exc).__name__}: {exc}'
                detail = getattr(exc, 'read', None)
                body_text = ''
                if detail is not None:
                    try:
                        body_text = detail()[:400].decode('utf-8', 'replace')
                        last += ' :: ' + body_text
                    except Exception:
                        pass
                code = getattr(exc, 'code', None)
                if code == 400:
                    if 'contentpolicyviolation' in body_text.lower():
                        self._dump_policy_images(messages)
                    bad = self._droppable_from(body_text)
                    if bad:
                        self.dropped.append(bad)
                        continue
                if code == 429:
                    self.throttled += 1
                time.sleep(self._backoff(attempt, code, exc))
        redacted = _redact_secrets(last)
        self.failures.append(redacted)
        print(f'[chat] exhausted {self.retries} attempts: {redacted}', file=sys.stderr, flush=True)
        return ''

    def _backoff(self, attempt, code, exc=None):
        after = None
        headers = getattr(exc, 'headers', None)
        if headers is not None:
            try:
                after = float(headers.get('Retry-After'))
            except (TypeError, ValueError):
                after = None
        if after is not None:
            return max(0.5, min(after, self.MAX_BACKOFF))
        base = 4.0 if code == 429 else 1.5
        return min(base * 2 ** attempt, self.MAX_BACKOFF)

    def stats(self):
        return {'model': self.model, 'base_url': self.base, 'calls': self.calls, 'temperature': '(端点默认)' if self.temperature is None else self.temperature, 'dropped_params': list(self.dropped), 'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens, 'image_parts': self.image_parts, 'last_request_bytes': self.last_request_bytes, 'largest_request_bytes': self.largest_request_bytes, 'failed_calls': len(self.failures), 'throttled_429': self.throttled, 'failures': self.failures[:5]}

def _count_image_parts(messages):
    n = 0
    for m in messages or ():
        content = m.get('content')
        if isinstance(content, list):
            n += sum((1 for p in content if isinstance(p, dict) and p.get('type') == 'image_url'))
    return n
_ACTION_RE = re.compile('(?:^|\\n)\\s*(?:ACTION|动作)\\s*[:：]\\s*(.+)', re.I)
_THOUGHT_RE = re.compile('(?:^|\\n)\\s*(?:THOUGHT|REASON|推理|思考)\\s*[:：]\\s*(.+)', re.I)
_CALL_RE = re.compile('([A-Za-z_][A-Za-z0-9_]*)\\s*\\(\\s*(.*?)\\s*\\)\\s*$', re.S)
TALK_VERBS = ('answer', 'done')

def split_args(inner):
    args, kwargs, buf, quote = ([], {}, '', '')
    for ch in inner:
        if quote:
            if ch == quote:
                quote = ''
            else:
                buf += ch
        elif ch in '"\'':
            quote = ch
        elif ch == ',':
            args.append(buf.strip())
            buf = ''
        else:
            buf += ch
    if buf.strip() or args:
        args.append(buf.strip())
    out = []
    for a in args:
        if not a:
            continue
        if '=' in a and re.match('^[A-Za-z_][A-Za-z0-9_]*\\s*=', a):
            k, v = a.split('=', 1)
            kwargs[k.strip()] = v.strip().strip('"\'')
        else:
            out.append(a.strip().strip('"\''))
    return (out, kwargs)

def parse_action(text):
    if not text:
        return (None, (), {}, '模型没有输出（调用失败或空回复）')
    m = _ACTION_RE.search(text)
    if not m:
        return (None, (), {}, '没有 ACTION: 行')
    line = re.sub('^[\\s`*]+|[\\s`*.。]+$', '', m.group(1))
    call = _CALL_RE.search(line)
    if not call:
        bare = line.split()[0].strip('`') if line.split() else ''
        if bare in TALK_VERBS:
            return (bare, (line[len(bare):].strip(' :：'),), {}, '')
        return (None, (), {}, f'看不懂的动作：{line[:80]}')
    verb = call.group(1)
    args, kwargs = split_args(call.group(2))
    return (verb, tuple(args), kwargs, '')

def parse_thought(text):
    m = _THOUGHT_RE.search(text or '')
    if m:
        return m.group(1).strip()
    head = (text or '').split('ACTION')[0].strip()
    return head[:400]
_SYSTEM = 'You are the decision-making brain of a household robot in a simulated house.\n\n{observation_note}\n\n{memory_note}\n\nYour task:\n{instruction}\n{goal_note}\nSkills available to you this run (and only these):\n{skills}\n  answer("...")   -- give your final answer to a question, in English\n  done()          -- declare the task finished\n\nRules that matter:\n{refer_rule}\n{tier_rule}\n- You have a budget of {budget} actions. Rooms are not all visible from where you\n  stand; turning and walking to look around is expected and costs budget.\n{feedback_rule}{mark_note}\n\nReply in exactly this shape, nothing else:\n\nTHOUGHT: <one short line: what you believe and why this action>\nACTION: <one call from the list above>\n'

def _goal_note(images):
    if not images:
        return ''
    lines = '\n'.join((f'  Image {i}: {caption}' for i, (caption, _url) in enumerate(images, 1)))
    n = 'One image is' if len(images) == 1 else f'{len(images)} images are'
    return f'\n{n} attached above, showing what the task is about:\n{lines}\nEach is rendered on a plain white background -- the object has been cut out of its surroundings, so the white says nothing about where it is. These are reference pictures, not what you see now.\n'
_MEMORY_NOTE = {'text': 'The observation separates what is true *now* from what you have *learned*, and\nthe difference decides which actions are legal:\n- `you are standing in:` and `in view right now:` are this instant. They change\n  with every step you take.\n- `rooms in this house`, `furniture you know of`, `objects you have seen` and\n  `placement` are your accumulated memory. Something listed there is a name you\n  may use, but it is not necessarily in front of you.', 'blind': 'You are given no text description of this house. Not the rooms it has, not the\nfurniture in it, not the name of a single object, and no list that fills in as\nyou explore. Nothing will ever tell you what you are looking at.\n\nYour only sense is the photograph. Your only memory is this conversation: every\nTHOUGHT line you write is kept and shown back to you on every later round, while\nthe older photographs are dropped. So the notes you take are the only durable\nrecord you will have -- when you see that the kitchen is through the left-hand\ndoorway, write that down, because the picture proving it will be gone.', 'hand': 'You are given no text description of this house. Not the rooms it has, not the\nfurniture in it, not the name of a single object, and no list that fills in as\nyou explore. The one line of text you get each round is `holding:`, and it says\nonly whether your hand is full or empty -- never what is in it.\n\nYour senses are the photograph and what the robot can feel in its own body. Your\nonly memory is this conversation: every THOUGHT line you write is kept and shown\nback to you on every later round, while the older photographs are dropped. So the\nnotes you take are the only durable record you will have -- when you see that the\nkitchen is through the left-hand doorway, write that down, because the picture\nproving it will be gone. If you pick something up and it matters later what it\nwas, write that down too: the `holding:` line will not remind you.'}
_FEEDBACK_RULE = {'verdict': '- If an action is refused, the refusal text tells you why. Read it and change\n  something -- repeating the identical action usually fails identically.', 'none': '- **You are not told whether an action worked.** Nothing will say "that\n  failed", or "you cannot reach that from here", or "you are not holding\n  anything". After each action you get a new photograph and that is the entire\n  report. Compare it with the previous one and decide for yourself.\n- An action that achieved nothing leaves the view **exactly** as it was. So two\n  identical photographs in a row mean your last action did nothing at all, and\n  the thing to change is the action, not the number of times you try it.\n- Two things you are still told, because they are about your own message rather\n  than about the house: a reply this harness could not parse, and a call that is\n  not on the skill list above. Those come back as text; read them.', 'sensor': '- **After each action you are told what the robot\'s own body measured, and\n  nothing else.** Each line opens with the instrument it came from, and these are\n  the ones you will see: `Odometry:` -- the wheels counting how far the base\n  rolled and how far it turned; `Head encoder:` -- whether the camera\'s pitch\n  joint moved or hit the end of its travel; `Gripper:` -- whether anything is\n  between the fingers; `Wrist force sensor:` -- whether the thing the hand pushed\n  on actually moved under it; `Depth camera:` and `Local map:` -- what the range\n  image and the floor plan made of the point you pointed at; `Arm:` -- whether a\n  reachable arm pose exists from where the base is standing. The label tells you\n  what kind of claim you are reading.\n- Those lines are measurements, not judgements, and they are complete. "The\n  fingers closed and came back empty" is not a refusal with the reason left out\n  -- it is everything the robot felt. Nothing will ever tell you "that is not\n  something you can pick up", or "you cannot reach that from here", or what any\n  object is called, because no sensor on the robot measures those things.\n- Two outcomes end with "read the new photograph", and they mean it literally.\n  When you put something down, the hand knows it let go and nothing on the robot\n  knows where the object came to rest. When you switch something, the finger feels\n  the click and nothing feels whether the device is now on. Those two you have to\n  see for yourself.\n- An action that achieved nothing leaves the view **exactly** as it was, so two\n  identical photographs in a row mean your last action did nothing at all -- and\n  the thing to change is the action, not the number of times you try it.\n- You are still told when your own message could not be used: a reply this harness\n  could not parse, a call that is not on the skill list, a coordinate off the edge\n  of the frame. Those are about what you wrote rather than about the house, and\n  they arrive before the robot moves at all.'}
_MARK_NOTE = {False: '', True: "\n- The photographs from earlier rounds carry a hollow magenta ring around the\n  pixel that round's action was aimed at, so you can see which spot each of your\n  past choices picked. It is your own aim drawn back to you, not a hint: it marks\n  what you pointed at, not what was really there. The newest photograph has no\n  ring, because you have not chosen a pixel for this round yet. The ring is drawn\n  on top of the picture and is not an object in the house."}

def _frame_geometry(frame_size):
    w, h = (int(frame_size[0]), int(frame_size[1]))
    return {'w': w, 'h': h, 'umax': w - 1, 'vmax': h - 1, 'ucen': w // 2, 'vcen': h // 2, 'vfloor': int(h * 0.93), 'vceil': int(h * 0.07)}
_REFER_RULE = {'name': '- Names like `mug_1`, `shelf_2`, `counter_top_1` are the only way to refer to\n  things. You may only pass a name that has appeared in an observation.\n- To put something somewhere: pick(X) first, then walk to the destination, then\n  place(DEST) -- `place` takes the *destination*, the held object is implied.', 'pixel': "- You refer to things by **pointing at them in the photograph**, never by name.\n  Every skill below takes `u, v`: `u` is the column counting from the **left**\n  edge, `v` is the row counting from the **top** edge, both in pixels, so `u` is\n  0-{umax} and `v` is 0-{vmax} on this {w}x{h} frame. So ({ucen}, {vcen}) is the\n  centre of the view, ({ucen}, {vfloor}) is the floor just in front of your feet,\n  and ({ucen}, {vceil}) is up near the ceiling. `open({ucen}, {vcen})` opens\n  whatever is at that point.\n- Aim for the **middle** of the thing you mean, not its edge. An edge pixel\n  usually belongs to whatever is behind it, and you will act on that instead.\n- Objects and ground are **separate skills**, and choosing the wrong one fails\n  rather than guessing: `go_to_object_at_pixel(u, v)` walks up to a *thing* and\n  stops within reach of it, while `go_to_ground_at_pixel(u, v)` walks onto a\n  patch of *floor* and stands there. Point the object skill at bare floor, or\n  the ground skill at a chair, and you are told which one you should have used.\n  When that happens, change the skill -- not the pixel.\n- The text observation still tells you what exists and is still your memory of\n  the house, and it is still better evidence than the photograph about *what* a\n  thing is. But pointing needs no vocabulary: if you can see something in the\n  photograph that the text has not named, you may point at it anyway. That is\n  this channel's advantage, and equally its risk -- you may be pointing at\n  something other than what you think it is.\n- To put something somewhere: pick(u, v) on the thing, then walk to the\n  destination, then place(u, v) pointing at the destination -- `place` points at\n  where it goes, the held object is implied.", 'pixel_blind': '- You refer to things by **pointing at them in the photograph**, and there is no\n  other way to refer to anything. Every skill below takes `u, v`: `u` is the\n  column counting from the **left** edge, `v` is the row counting from the **top**\n  edge, both in pixels, so `u` is 0-{umax} and `v` is 0-{vmax} on this {w}x{h}\n  frame. So ({ucen}, {vcen}) is the centre of the view, ({ucen}, {vfloor}) is the\n  floor just in front of your feet, and ({ucen}, {vceil}) is up near the ceiling.\n  `open({ucen}, {vcen})` opens whatever is at that point.\n- Aim for the **middle** of the thing you mean, not its edge. An edge pixel\n  usually belongs to whatever is behind it, and you will act on that instead.\n- Objects and ground are **separate skills**, and choosing the wrong one does\n  nothing rather than guessing: `go_to_object_at_pixel(u, v)` walks up to a\n  *thing* and stops within reach of it, while `go_to_ground_at_pixel(u, v)` walks\n  onto a patch of *floor* and stands there. Point the object skill at bare floor,\n  or the ground skill at a chair, and you simply will not move -- and nobody will\n  tell you which mistake you made. So when a pixel you are confident about\n  produces an unchanged photograph, try the **other skill** on the same pixel\n  before you go looking for a different pixel.\n- Nothing names anything for you. What you are pointing at is entirely your own\n  reading of the image, with no second opinion to check it against, so a\n  coordinate that looks like a mug handle may be a spoon or a shadow. When it\n  matters, walk closer and look again before you act.\n- To put something somewhere: pick(u, v) on the thing, then walk to the\n  destination, then place(u, v) pointing at the destination -- `place` points at\n  where it goes, the held object is implied. You can see whether your hand is\n  full: a held object is carried in front of the camera, in the lower middle of\n  the frame.', 'pixel_hand': "- You refer to things by **pointing at them in the photograph**, and there is no\n  other way to refer to anything. Every skill below takes `u, v`: `u` is the\n  column counting from the **left** edge, `v` is the row counting from the **top**\n  edge, both in pixels, so `u` is 0-{umax} and `v` is 0-{vmax} on this {w}x{h}\n  frame. So ({ucen}, {vcen}) is the centre of the view, ({ucen}, {vfloor}) is the\n  floor just in front of your feet, and ({ucen}, {vceil}) is up near the ceiling.\n  `open({ucen}, {vcen})` opens whatever is at that point.\n- Aim for the **middle** of the thing you mean, not its edge. An edge pixel\n  usually belongs to whatever is behind it, and you will act on that instead.\n- Objects and ground are **separate skills**, and choosing the wrong one fails\n  rather than guessing: `go_to_object_at_pixel(u, v)` walks up to a *thing* and\n  stops within reach of it, while `go_to_ground_at_pixel(u, v)` walks onto a patch\n  of *floor* and stands there. The robot's depth camera can work out whether the\n  point you gave lies on the floor plane or stands above it, so when you pick the\n  wrong skill it says which -- and then the thing to change is the skill, not the\n  pixel.\n- Nothing names anything for you. What you are pointing at is entirely your own\n  reading of the image, with no second opinion to check it against, so a\n  coordinate that looks like a mug handle may be a spoon or a shadow. When it\n  matters, walk closer and look again before you act.\n- To put something somewhere: pick(u, v) on the thing, then walk to the\n  destination, then place(u, v) pointing at the destination -- `place` points at\n  where it goes, the held object is implied. The `holding:` line tells you whether\n  your hand is full but never what is in it, and a held object is also carried in\n  front of the camera, in the lower middle of the frame."}
_TIER_RULE = {'high': '- `navigate_to_object(name)` reaches anything by name, even something you\n  have not seen yet. Searching is done for you.', 'plan': "- `go_to_visible_object(name)` only works on a name that is listed under\n  `in view right now:`. Being in `objects you have seen` is not enough --\n  that is memory, and you may have walked away from it. If the name is not\n  in view, turn/look/move until it is; that search is your job, not the\n  skill's. `navigate_to_relative(deg, m)` needs no visibility at all and is\n  the right tool for crossing a room you cannot yet see into.\n- For measured motion, positive move is forward, positive strafe is right,\n  positive turn is left and positive look is up. Negative values reverse\n  each direction.", 'spatial': '- You have no name-based navigation at all. `move(m)`, `strafe(m)`,\n  `turn(deg)` and `look(deg)` are the only ways to change where you are.\n  Positive move is forward; positive strafe is right; positive turn is\n  left; positive look is up. Negative values reverse each direction.'}
_TIER_RULE_PIXEL = {'plan': "- `go_to_object_at_pixel(u, v)` only reaches what is on screen at this\n  instant -- that is what pointing means. Something you saw two rooms ago is\n  memory, not a target: there is no pixel you can give for it. If what you\n  want is not in the frame, turn, look or walk until it is; that search is\n  your job, not the skill's.\n- `go_to_ground_at_pixel(u, v)` is how you cross a room. Point it at floor\n  as far away as you can still see clearly -- through a doorway, at the far\n  wall's base -- and you will walk there. It needs no object in view, so it\n  is the right tool for getting into a room you cannot see into yet.\n- For measured motion, positive move is forward, positive strafe is right,\n  positive turn is left and positive look is up. Negative values reverse\n  each direction."}
_OBS_NOTE = {'text': 'You do not receive images. You receive a text observation each round, and you\nissue exactly one action per round. Everything you know about the house comes\nfrom those observations: an object that has not appeared in an observation may\nstill exist -- go and look.', 'vision': "Each round you receive two things about the same instant: a photograph from the\nrobot's own camera, and a text observation. You issue exactly one action per\nround. Everything you know about the house comes from these; an object that has\nappeared in neither may still exist -- go and look.\n\nThe two are not interchangeable, and getting this backwards will cost you the\nepisode:\n- The **text** is the only source of *names*. You may only pass a name that has\n  appeared in an observation, so if you can see something in the photograph but\n  it is not named in the text, you cannot act on it yet -- get closer or look\n  straight at it until the text lists it.\n- The **photograph** is the only source of *layout*: which way a doorway leads,\n  whether the thing you want is left or right, how far a wall is, whether the\n  room continues past what is named. Use it to decide where to move and which\n  way to turn.\n- It is {w}x{h} pixels with a wide lens, taken from the robot's eye height. Small\n  or distant things are a few pixels and easy to mistake for something else. It\n  is evidence, not proof: when the photograph and the text disagree about what is\n  in front of you, the text is what the harness will accept.", 'vision_pixel': "Each round you receive two things about the same instant: a photograph from the\nrobot's own camera, and a text observation. You issue exactly one action per\nround. Everything you know about the house comes from these; an object that has\nappeared in neither may still exist -- go and look.\n\nThe two do different jobs, and in this run the photograph does the load-bearing\none:\n- The **photograph** is where you *act*. Every skill takes a pixel coordinate\n  read off this image, so where a thing sits in the frame is not a hint about\n  your action, it *is* your action. Look at it before every move.\n- The **text** is your *memory and your interpreter*: which room you are in,\n  what you have already seen, what the harness believes each object is. Use it\n  to decide what is worth pointing at and to check that the thing you think you\n  see is the thing that is there.\n- It is {w}x{h} pixels with a wide lens, taken from the robot's eye height. Small\n  or distant things are a few pixels wide, so a coordinate that misses by a\n  little can land on the wall behind your target. When something is too small to\n  point at confidently, walk closer first and point then.", 'blind_pixel': "Each round you receive exactly one thing: a photograph from the robot's own\ncamera. You issue exactly one action per round. There is no text description of\nthe house, no list of what is in it, and no report of what your last action did.\nThe picture is your only sensor, and everything you ever know about this house\nyou will have worked out from pictures.\n\n- It is {w}x{h} pixels with a wide lens, taken from the robot's eye height. Small\n  or distant things are only a few pixels wide and are easy to mistake for\n  something else, so treat a distant identification as a guess until you have\n  walked closer.\n- Where a thing sits in the frame is not a hint about your action, it **is** your\n  action: every skill takes a pixel coordinate read off this image. Look at the\n  picture before every single move.\n- You are shown the {keep} most recent photographs, not just the newest one. That\n  is deliberate and it is your only way of telling what just happened: the\n  *difference* between consecutive frames is the whole of your feedback. Two\n  identical frames mean nothing moved.", 'hand_pixel': "Each round you receive three things: a photograph from the robot's own camera, one\nline of text saying whether your hand is full, and a report of what the robot's\nown body measured while carrying out your last action. You issue exactly one\naction per round. There is no description of this house, no list of what is in it,\nand nothing anywhere that names anything -- everything you come to know about the\nplace you will have worked out from pictures.\n\n- It is {w}x{h} pixels with a wide lens, taken from the robot's eye height. Small\n  or distant things are only a few pixels wide and are easy to mistake for\n  something else, so treat a distant identification as a guess until you have\n  walked closer.\n- Where a thing sits in the frame is not a hint about your action, it **is** your\n  action: every skill takes a pixel coordinate read off this image. Look at the\n  picture before every single move.\n- You are shown the {keep} most recent photographs, not just the newest one. Use\n  them together with the body report: the report says what the robot *did*, and\n  the difference between consecutive frames says what that achieved. Where the\n  report stops short -- and it says so when it does -- the pictures are all you\n  have.\n- `holding:` says `(something)` or `(nothing)`, and that is the only fact about\n  the world's state you are ever given in words. It will never say what you are\n  carrying: a gripper measures force and finger position, not names."}

def _user_content(text, image=None):
    if not image:
        return text
    return [{'type': 'text', 'text': text}, {'type': 'image_url', 'image_url': {'url': image}}]

def text_of(content):
    if isinstance(content, str):
        return content
    return '\n'.join((p.get('text') or '' for p in content or [] if isinstance(p, dict) and p.get('type') == 'text'))

def _image_note(image):
    if not image:
        return None
    return {'bytes': len(image), 'sha256': hashlib.sha256(image.encode('ascii')).hexdigest()[:12]}

def conversation_size(messages):
    chars = 0
    parts = 0
    image_bytes = 0
    for m in messages or ():
        content = m.get('content')
        chars += len(text_of(content))
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get('type') != 'image_url':
                    continue
                parts += 1
                image_bytes += len((part.get('image_url') or {}).get('url') or '')
    return {'chars': chars, 'messages': len(messages or ()), 'image_parts': parts, 'image_bytes': image_bytes}
MEMORY_HEADER = '[Earlier rounds, condensed into your own notes]'
_SUMMARY_SYSTEM = "You compress one robot's episode log so it can keep acting with a shorter context.\nRewrite the rounds below as a compact first-person note the robot will read as its own memory.\n\nKeep, in this order of priority:\n  1. Where it has been and what it saw there. This matters more than anything else: the robot cannot re-read these rounds, and one that forgets which places it already searched will search them again until its budget runs out.\n  2. What it is holding, and anything it picked up, put down, opened or closed.\n  3. What it tried that did not work, and the reason given. Repeating a refused action is the second most common way an episode is wasted.\n  4. Any conclusion it reached about the task, and how sure it was.\n\nRules:\n  - The input intentionally ends in the middle of an episode. Do not continue the episode, ask for later results, or comment that the log is incomplete. Your only job is to write the memory note for the rounds supplied.\n  - Write only what these rounds say. Do not add, guess, or resolve anything left open. If the robot was unsure which object it had found, say that it was unsure.\n  - Keep it under 400 words.\n  - No preamble. Start with the note itself."
_SUMMARY_META_MARKERS = ('not able to keep playing', "can't keep playing", 'cannot keep playing', "log you've given me ends", 'log you have given me ends', 'what i can do is give you', 'results that would tell me what happened next')

def clean_history_summary(note):
    note = (note or '').strip()
    prefix = note[:1000].lower()
    if not any((marker in prefix for marker in _SUMMARY_META_MARKERS)):
        return note
    for divider in ('\n---\n', '\n***\n'):
        if divider in note[:1200]:
            memory = note.split(divider, 1)[1].strip()
            return memory if memory else ''
    return ''

def summarize_history(client, segment):
    body = []
    for m in segment:
        text = text_of(m.get('content')).strip()
        if not text:
            continue
        who = 'ROBOT' if m.get('role') == 'assistant' else 'HARNESS'
        body.append(f'{who}: {text}')
    if not body:
        return ''
    return (client.complete([{'role': 'system', 'content': _SUMMARY_SYSTEM}, {'role': 'user', 'content': '\n\n'.join(body)}]) or '').strip()

class LLMAgent:

    def __init__(self, client, space, instruction, budget, tier_key=None, vision=False, vision_keep=None, frame_size=None, world_text=True, action_feedback=feedback.VERDICT, mark_frames=False, hand_only=False, compact_every=0, compact_min_chars=None, compact_keep=None, compact_after_prompt_tokens=0, summarizer=None, goal_images=(), vision_max_bytes=None, vision_resolution_ladder=(), vision_request_target_bytes=None):
        self.client = client
        self.space = space
        self.instruction = instruction
        self.goal_images = tuple(goal_images or ())
        self.budget = budget
        self.tier_key = tier_key or space.tier.key
        self.vision = bool(vision)
        self.refer = getattr(space, 'refer', 'name')
        self.world_text = bool(world_text)
        self.action_feedback = action_feedback
        self.hand_only = bool(hand_only)
        self.mark_frames = bool(mark_frames)
        if action_feedback not in feedback.MODES:
            raise ValueError('action_feedback must be one of ' + ', '.join((repr(m) for m in feedback.MODES)) + f', got {action_feedback!r}')
        if self.hand_only and (not self.world_text):
            raise ValueError('hand_only=True needs world_text=True: it narrows the text observation to one line, so it cannot also be the switch that removes it. For no text at all, pass world_text=False alone.')
        if self.hand_only and self.refer != 'pixel':
            raise ValueError("hand_only=True needs refer='pixel': it names nothing, so no name is ever granted and a name-channel brain can only guess handles. Pass refer='pixel'.")
        if self.hand_only and (not self.vision):
            raise ValueError("hand_only=True needs vision=True: one line about the robot's own hand is not an observation of a house, so without the frame the model is asked to act on no percept at all.")
        if not self.world_text and self.refer != 'pixel':
            raise ValueError("world_text=False needs refer='pixel': with no text observation no name is ever granted, so a name-channel brain can only guess handles and every action it takes is refused. Pass refer='pixel'.")
        if not self.world_text and (not self.vision):
            raise ValueError('world_text=False needs vision=True: removing the text observation leaves the frame as the only observation, and without it the model is asked to act on no percept at all.')
        if self.refer == 'pixel' and (not self.vision):
            raise ValueError('the pixel channel needs vision=True: the model is being asked for coordinates in an image, so it has to be sent the image.')
        if self.vision and frame_size is None:
            raise ValueError("vision=True needs frame_size=(width, height): the prompt quotes the frame's pixel dimensions and its coordinate range, so it has to be told what they are. Pass the render size, e.g. (b.width, b.height).")
        if self.mark_frames and (not (self.vision and self.refer == 'pixel')):
            raise ValueError("mark_frames=True needs vision=True and refer='pixel': the mark is a crosshair drawn on a frame at the coordinate an action was aimed at, so without the frame there is nothing to draw on and without the pixel channel there is no coordinate to draw.")
        self.frame_size = None if frame_size is None else (int(frame_size[0]), int(frame_size[1]))
        self.vision_keep = self.VISION_KEEP if vision_keep is None else max(1, int(vision_keep))
        self.vision_max_bytes = None if vision_max_bytes is None else max(1, int(vision_max_bytes))
        self.vision_resolution_ladder = tuple((max(1, int(side)) for side in vision_resolution_ladder or ()))
        self.vision_request_target_bytes = None if vision_request_target_bytes is None else max(1, int(vision_request_target_bytes))
        self.vision_max_parts = None
        self.vision_packer = None
        self.vision_packing_events = []
        self.model_image_limit = 0
        self.image_limit_events = []
        self.vision_resolution_index = 0
        self.vision_resolution_events = []
        self._vision_variants = {}
        self.compact_every = max(0, int(compact_every or 0))
        self.compact_min_chars = self.COMPACT_MIN_CHARS if compact_min_chars is None else max(0, int(compact_min_chars))
        self.compact_keep = self.COMPACT_KEEP if compact_keep is None else max(1, int(compact_keep))
        self.compact_after_prompt_tokens = max(0, int(compact_after_prompt_tokens or 0))
        self._compact_prompt_token_base = int(getattr(self.client, 'prompt_tokens', 0) or 0)
        self._next_prompt_token_compaction = self.compact_after_prompt_tokens
        self.summarizer = summarizer or summarize_history
        self._since_compact = 0
        self.compactions = []
        skills = '\n'.join((f'  {s}' for s in space.prompt_lines()))
        tier_rule = (_TIER_RULE_PIXEL.get(self.tier_key) if self.refer == 'pixel' else None) or _TIER_RULE[self.tier_key]
        refer_key = 'pixel_blind' if self.refer == 'pixel' and (not self.world_text) else 'pixel_hand' if self.refer == 'pixel' and self.hand_only else self.refer
        obs_key = 'blind_pixel' if self.vision and (not self.world_text) else 'hand_pixel' if self.vision and self.hand_only else 'vision_pixel' if self.vision and self.refer == 'pixel' else 'vision' if self.vision else 'text'
        memory_key = 'blind' if not self.world_text else 'hand' if self.hand_only else 'text'
        geom = _frame_geometry(self.frame_size) if self.frame_size else {}
        if geom:
            geom['keep'] = self.vision_keep
        self.system = _SYSTEM.format(instruction=instruction, skills=skills, budget=budget, goal_note=_goal_note(self.goal_images), tier_rule=tier_rule, refer_rule=_REFER_RULE[refer_key].format(**geom), memory_note=_MEMORY_NOTE[memory_key], feedback_rule=_FEEDBACK_RULE[self.action_feedback], mark_note=_MARK_NOTE[self.mark_frames], observation_note=_OBS_NOTE[obs_key].format(**geom))
        self.messages = [{'role': 'system', 'content': [{'type': 'image_url', 'image_url': {'url': url}} for _caption, url in self.goal_images] + [{'type': 'text', 'text': self.system}] if self.goal_images else self.system}]
        self.rounds = []
        self.images_sent = 0
        self.image_bytes = 0
    LAST_CALL = 3
    VISION_KEEP = 3
    COMPACT_EVERY = 50
    COMPACT_MIN_CHARS = 60000
    COMPACT_KEEP = 10

    def turn_prompt(self, view_text, receipt, round_i, spent):
        parts = [f'Round {round_i}. Budget: {spent}/{self.budget} actions used.']
        left = self.budget - spent
        if left <= self.LAST_CALL:
            parts.append(f"""You have {left} action{('s' if left != 1 else '')} left, and after that the episode ends. If the task asked you a question, an unanswered question scores zero -- answer("...") is itself an action and needs one of those remaining rounds.""")
        if receipt:
            parts.append('Result of your last action:\n' + receipt)
        if view_text:
            parts.append('Observation:\n' + view_text)
        parts.append('What is your next action?')
        return '\n\n'.join(parts)

    def ask(self, view_text, receipt, round_i, spent, image=None, image_variants=(), trajectory_image=None, trajectory_meta=None):
        prompt = self.turn_prompt(view_text, receipt, round_i, spent)
        variants = tuple(image_variants or ())
        if variants:
            chosen = variants[min(self.vision_resolution_index, len(variants) - 1)][1]
            image = chosen
        marker = 'SELF TRAJECTORY (ODOMETRY ONLY).'
        for old in self.messages:
            content = old.get('content')
            if not isinstance(content, list):
                continue
            cleaned, skip_image = ([], False)
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'text' and str(part.get('text') or '').startswith(marker):
                    skip_image = True
                    continue
                if skip_image and isinstance(part, dict) and (part.get('type') == 'image_url'):
                    skip_image = False
                    continue
                cleaned.append(part)
            old['content'] = cleaned
        content = _user_content(prompt, image)
        if trajectory_image:
            if not isinstance(content, list):
                raise ValueError('self trajectory requires a current visual observation')
            detail = json.dumps(trajectory_meta or {}, ensure_ascii=False, sort_keys=True)
            content[-1:-1] = [{'type': 'text', 'text': marker + ' Full route through the current pose; contains no map or task-object truth. Metadata: ' + detail}, {'type': 'image_url', 'image_url': {'url': trajectory_image}}, {'type': 'text', 'text': 'CURRENT FINAL camera observation follows; use only that final image for action pixels.'}]
        message = {'role': 'user', 'content': content}
        self.messages.append(message)
        if variants:
            self._vision_variants[id(message)] = variants
        self._trim_images()
        self._since_compact += 1
        compacted = self._compact_text(round_i)
        resolution_event = self._fit_request_resolution(round_i)
        sent_image = self._message_image(message)
        self._current_trajectory_image = trajectory_image
        if sent_image:
            self.images_sent += 1
            self.image_bytes += len(sent_image)
        if trajectory_image:
            self.images_sent += 1
            self.image_bytes += len(trajectory_image)
        request_messages, packing_event = self._request_messages()
        sent = conversation_size(request_messages)
        if packing_event:
            sent['image_packing'] = packing_event
            self.vision_packing_events.append(packing_event)
        raw = self._complete_action_request(request_messages, round_i)
        retrieval = getattr(self, 'last_retrieval', None)
        if retrieval and retrieval.get('action_context'):
            sent = retrieval['action_context']
        sent['request_bytes'] = getattr(self.client, 'last_request_bytes', None)
        self.messages.append({'role': 'assistant', 'content': raw or '(no reply)'})
        verb, args, kwargs, why = parse_action(raw)
        row = {'round': round_i, 'prompt': prompt, 'raw': raw, 'thought': parse_thought(raw), 'verb': verb, 'args': list(args), 'kwargs': dict(kwargs), 'parse_error': why, 'image': _image_note(sent_image), 'self_trajectory': {'image': _image_note(trajectory_image), 'meta': trajectory_meta or {}} if trajectory_image else None, 'context': sent, 'compaction': compacted, 'vision_resolution': {'effective_max_side': self._effective_resolution(), 'request_target_bytes': self.vision_request_target_bytes, 'event': resolution_event}}
        row['memory_retrieval'] = getattr(self, 'last_retrieval', None)
        self.rounds.append(row)
        return row

    def _complete_action_request(self, messages, round_i):
        return self._memory_call(messages)

    def _memory_call(self, messages):
        limit = getattr(self, 'episode_token_limit', 0)

        def guard(request, body=None):
            used = int(getattr(self.client, 'prompt_tokens', 0)) + int(getattr(self.client, 'completion_tokens', 0)) - getattr(self, '_episode_token_base', 0)
            size = conversation_size(request)
            text_bytes = sum((len(text_of(m.get('content')).encode('utf-8')) for m in request))
            reserve = text_bytes + 2048 * size['image_parts'] + int((body or {}).get('max_tokens') or getattr(self.client, 'max_tokens', DEFAULT_OUTPUT_TOKENS) or DEFAULT_OUTPUT_TOKENS)
            if used + reserve > limit:
                self.token_budget_exceeded = True
                raise EpisodeTokenBudgetExceeded('episode token budget would be exceeded before next request')
        if not limit:
            return self.client.complete(messages)
        guard(messages)
        previous = getattr(self.client, 'episode_token_guard', None)
        self.client.episode_token_guard = guard
        try:
            return self.client.complete(messages)
        finally:
            self.client.episode_token_guard = previous

    def _trim_images(self):
        used = 0
        for msg in self.messages:
            if msg.get('role') == 'user':
                continue
            content = msg.get('content')
            if not isinstance(content, list):
                continue
            used += sum((len((part.get('image_url') or {}).get('url') or '') for part in content if isinstance(part, dict) and part.get('type') == 'image_url'))
        seen = 0
        for msg in reversed(self.messages):
            if msg.get('role') != 'user' or not isinstance(msg.get('content'), list):
                continue
            seen += 1
            frame_bytes = sum((len((part.get('image_url') or {}).get('url') or '') for part in msg['content'] if isinstance(part, dict) and part.get('type') == 'image_url'))
            within_bytes = getattr(self, 'vision_request_target_bytes', None) is not None or self.vision_max_bytes is None or used + frame_bytes <= self.vision_max_bytes
            if seen > self.vision_keep or not within_bytes:
                msg['content'] = text_of(msg['content'])
            else:
                used += frame_bytes

    def configure_image_part_limit(self, maximum, packer):
        self.vision_max_parts = None if not maximum else max(1, int(maximum))
        self.vision_packer = packer

    def _request_messages(self):
        if getattr(self, 'model_image_limit', 0):
            request, event = bounded_image_messages(self.messages, self.model_image_limit, self.client)
            if event:
                self.image_limit_events.append(event)
            return (request, None)
        maximum = self.vision_max_parts
        total = _count_image_parts(self.messages)
        if not maximum or total <= maximum:
            return (self.messages, None)
        pinned = _count_image_parts([msg for msg in self.messages if msg.get('role') != 'user'])
        available = maximum - pinned
        if available < 1:
            raise ChatError(f'{pinned} pinned goal images leave no room under the {maximum}-image limit')
        observations = []
        for index, msg in enumerate(self.messages):
            if msg.get('role') != 'user' or not isinstance(msg.get('content'), list):
                continue
            url = self._message_image(msg)
            if url:
                observations.append((index, url))
        if len(observations) <= available:
            return (self.messages, None)
        if self.vision_packer is None:
            raise ChatError('image-part limit requires a vision-side contact-sheet packer')
        recent = min(16, available, len(observations))
        while recent >= 0:
            old_count = len(observations) - recent
            archive_slots = available - recent
            if old_count == 0 or (archive_slots > 0 and (old_count + archive_slots - 1) // archive_slots <= 4):
                break
            recent -= 1
        if recent < 0 or (len(observations) - recent and available - recent < 1):
            raise ChatError(f'cannot preserve {len(observations)} observation frames within {available} image parts using four-frame contact sheets')
        old = observations[:len(observations) - recent]
        archive_slots = available - recent
        group_count = min(archive_slots, len(old))
        base, extra = divmod(len(old), group_count)
        groups = []
        cursor = 0
        for group_i in range(group_count):
            size = base + (1 if group_i < extra else 0)
            group = old[cursor:cursor + size]
            cursor += size
            if len(group) > 4:
                raise ChatError('contact-sheet group unexpectedly exceeds four frames')
            groups.append(group)
        request = []
        old_indices = {index for group in groups for index, _url in group}
        for index, msg in enumerate(self.messages):
            copied = dict(msg)
            if index in old_indices:
                copied['content'] = text_of(msg.get('content'))
            request.append(copied)
        archive = [{'type': 'text', 'text': 'Archived observation frames follow. Contact-sheet tiles are chronological in row-major order (top-left to top-right, then the next row). Original round text remains below in its normal position.'}]
        first_frame = 1
        for group in groups:
            last_frame = first_frame + len(group) - 1
            archive.append({'type': 'text', 'text': f'Observation frames {first_frame}-{last_frame}:'})
            packed = group[0][1] if len(group) == 1 else self.vision_packer([url for _index, url in group])
            archive.append({'type': 'image_url', 'image_url': {'url': packed}})
            first_frame = last_frame + 1
        request.insert(1, {'role': 'user', 'content': archive})
        actual = _count_image_parts(request)
        if actual > maximum:
            raise ChatError(f'packed request still has {actual}>{maximum} image parts')
        return (request, {'limit': maximum, 'original_parts': total, 'packed_parts': actual, 'pinned_goal_parts': pinned, 'archived_observation_frames': len(old), 'contact_sheets': len(groups), 'recent_independent_frames': recent, 'max_tiles_per_sheet': max((len(group) for group in groups))})

    def _effective_resolution(self):
        if not self.vision_resolution_ladder:
            return None
        return self.vision_resolution_ladder[min(self.vision_resolution_index, len(self.vision_resolution_ladder) - 1)]

    @staticmethod
    def _replace_message_image(message, url):
        content = message.get('content')
        if not isinstance(content, list):
            return False
        for part in reversed(content):
            if isinstance(part, dict) and part.get('type') == 'image_url':
                part['image_url'] = {'url': url}
                return True
        return False

    @staticmethod
    def _message_image(message):
        content = message.get('content')
        if not isinstance(content, list):
            return None
        for part in reversed(content):
            if isinstance(part, dict) and part.get('type') == 'image_url':
                return (part.get('image_url') or {}).get('url') or None
        return None

    def _apply_resolution_index(self):
        live = {id(message): message for message in self.messages}
        for message_id, variants in tuple(self._vision_variants.items()):
            message = live.get(message_id)
            if message is None or not variants:
                continue
            url = variants[min(self.vision_resolution_index, len(variants) - 1)][1]
            self._replace_message_image(message, url)

    def _drop_oldest_observation_image(self):
        for message in self.messages:
            if message.get('role') != 'user':
                continue
            content = message.get('content')
            if not isinstance(content, list):
                continue
            if any((isinstance(part, dict) and part.get('type') == 'image_url' for part in content)):
                message['content'] = text_of(content)
                return True
        return False

    def _fit_request_resolution(self, round_i):
        target = self.vision_request_target_bytes
        if not target:
            return None
        before = self.client.request_size(self.messages)
        measured = before
        old_index = self.vision_resolution_index
        while measured > target and self.vision_resolution_ladder and (self.vision_resolution_index < len(self.vision_resolution_ladder) - 1):
            self.vision_resolution_index += 1
            self._apply_resolution_index()
            measured = self.client.request_size(self.messages)
        dropped = 0
        while measured > target and self._drop_oldest_observation_image():
            dropped += 1
            measured = self.client.request_size(self.messages)
        if self.vision_resolution_index == old_index and (not dropped):
            return None
        event = {'round': round_i, 'request_bytes_before': before, 'request_bytes_after': measured, 'from_max_side': self.vision_resolution_ladder[old_index] if self.vision_resolution_ladder else None, 'to_max_side': self._effective_resolution(), 'dropped_oldest_frames': dropped}
        self.vision_resolution_events.append(event)
        return event

    def _compact_text(self, round_i):
        prompt_tokens = max(0, int(getattr(self.client, 'prompt_tokens', 0) or 0) - int(getattr(self, '_compact_prompt_token_base', 0) or 0))
        next_token_compaction = int(getattr(self, '_next_prompt_token_compaction', 0) or 0)
        token_due = bool(next_token_compaction and prompt_tokens >= next_token_compaction)
        round_due = bool(self.compact_every and self._since_compact >= self.compact_every)
        if not token_due and (not round_due):
            return None
        trigger = 'prompt_tokens' if token_due else 'rounds'
        threshold = next_token_compaction if token_due else None
        if token_due:
            interval = int(getattr(self, 'compact_after_prompt_tokens', 0) or 0)
            while self._next_prompt_token_compaction <= prompt_tokens:
                self._next_prompt_token_compaction += interval
        before = conversation_size(self.messages)
        if not token_due and before['chars'] < self.compact_min_chars:
            return None
        seen, cut = (0, None)
        for i in range(len(self.messages) - 1, 0, -1):
            if self.messages[i].get('role') == 'user':
                seen += 1
                if seen >= self.compact_keep:
                    cut = i
                    break
        if cut is None or cut <= 1:
            return None
        span = self.messages[1:cut]
        span_user_turns = sum((msg.get('role') == 'user' for msg in span))
        min_batch = 1 if token_due else self.compact_every
        if span_user_turns < min_batch:
            return None
        note = clean_history_summary(self.summarizer(self.client, span))
        if not note:
            self._since_compact = 0
            if token_due:
                self._next_prompt_token_compaction = threshold
            return None
        source_chars = sum((len(text_of(msg.get('content'))) for msg in span))
        replacement = f'{MEMORY_HEADER}\n{note}'
        if len(replacement) >= source_chars:
            self._since_compact = 0
            if token_due:
                self._next_prompt_token_compaction = threshold
            return None
        self.messages[1:cut] = [{'role': 'user', 'content': replacement}]
        self._since_compact = 0
        rec = {'round': round_i, 'replaced_messages': len(span), 'chars_before': before['chars'], 'chars_after': conversation_size(self.messages)['chars'], 'trigger': trigger, 'prompt_tokens_before': prompt_tokens, 'prompt_token_threshold': threshold, 'note': note}
        self.compactions.append(rec)
        return rec

    def remark_last_frame(self, image, image_variants=()):
        if not image:
            return False
        for msg in reversed(self.messages):
            if msg.get('role') != 'user':
                continue
            content = msg.get('content')
            if not isinstance(content, list):
                return False
            for part in content:
                if isinstance(part, dict) and part.get('type') == 'image_url':
                    variants = tuple(image_variants or ())
                    if variants:
                        self._vision_variants[id(msg)] = variants
                        image = variants[min(self.vision_resolution_index, len(variants) - 1)][1]
                    part['image_url'] = {'url': image}
                    if self.rounds:
                        self.rounds[-1]['marked_image'] = _image_note(image)
                    return True
            return False
        return False

    def tell(self, text):
        self.messages.append({'role': 'user', 'content': text})
NAME_VERBS = ('navigate_to_object', 'go_to_visible_object', 'pick', 'place', 'open', 'close', 'toggle')
SCALAR_VERBS = {'move': 'dist_m', 'strafe': 'dist_m', 'turn': 'deg', 'look': 'deg'}
PIXEL_NAV_VERBS = ('go_to_object_at_pixel', 'go_to_ground_at_pixel')

def to_float(s):
    m = re.search('[-+]?\\d*\\.?\\d+', str(s))
    return float(m.group(0)) if m else None

def to_pixel(args):
    nums = [to_float(a) for a in args]
    nums = [n for n in nums if n is not None]
    if len(nums) < 2:
        return None
    return (int(round(nums[0])), int(round(nums[1])))

def bind(space, verb, args, kwargs, pixel_scale=None):
    if verb in SCALAR_VERBS:
        if not space.has(verb):
            return (None, None, f"这一档没有 {verb}，可用：{', '.join(sorted(dir(space)))}")
        v = to_float(args[0]) if args else None
        if v is None:
            return (None, None, f'{verb} 需要一个数字参数，收到 {args!r}')
        fn = getattr(space, verb)
        return (verb, v, lambda: fn(v))
    if verb == 'navigate_to_relative':
        if not space.has(verb):
            return (None, None, f"这一档没有 {verb}，可用：{', '.join(sorted(dir(space)))}")
        nums = [to_float(a) for a in args]
        nums = [n for n in nums if n is not None]
        if len(nums) < 2:
            return (None, None, 'navigate_to_relative 需要 (deg, dist_m) 两个数字')
        deg, dist = (nums[0], nums[1])
        ref = DirectionRef(deg=deg, dist_m=dist)
        fn = getattr(space, verb)
        return (verb, ref, lambda: fn(deg, dist))
    pixel_mode = getattr(space, 'refer', 'name') == 'pixel'
    if verb in PIXEL_NAV_VERBS or (pixel_mode and verb in MANIPULATION):
        if not space.has(verb):
            return (None, None, f"这一档没有 {verb}，可用：{', '.join(sorted(dir(space)))}")
        uv = to_pixel(args)
        if uv is None:
            return (None, None, f'{verb} 需要两个像素坐标 (u, v)，收到 {args!r}')
        u, v = uv
        if pixel_scale is not None:
            u = int(round(u * float(pixel_scale[0])))
            v = int(round(v * float(pixel_scale[1])))
        ref = PixelRef(u, v)
        fn = getattr(space, verb)
        return (verb, ref, lambda: fn(u, v))
    if verb in NAME_VERBS:
        if not space.has(verb):
            return (None, None, f"这一档没有 {verb}，可用：{', '.join(sorted(dir(space)))}")
        name = args[0] if args else kwargs.get('into') or kwargs.get('name')
        if verb == 'place' and kwargs.get('into'):
            name = kwargs['into']
        if not name:
            return (None, None, f'{verb} 需要一个物体名字')
        ref = NameRef(str(name))
        fn = getattr(space, verb)
        return (verb, ref, lambda: fn(ref))
    return (None, None, f"没有 {verb} 这个技能，可用：{', '.join(sorted(dir(space)))}, answer(...), done()")
