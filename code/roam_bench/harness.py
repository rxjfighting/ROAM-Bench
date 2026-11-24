"""Run observation, retrieval, action, and answer rounds with independent scoring."""
import argparse
import concurrent.futures as futures
import contextlib
import hashlib
import html as html_mod
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
import traceback
import unicodedata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from roam_bench.protocol import baselines, feedback, goalimg, llm, mark, policies, taskset, vision
from roam_bench.protocol.episode_report import EpisodeFilm, FilmStrip
from roam_bench.protocol.score import Recorder
from roam_bench.protocol.tiers import ActionSpace
from roam_bench.protocol.types import Episode, NameRef, PixelRef, Pose
OUT = os.path.join(ROOT, 'output')
HOUSE = os.path.join(OUT, 'scene.json')
TASKSET = os.path.join(OUT, 'tasks.json')

def episode_for(task_id, spawn, partial_obs=True, blind=False, mark_pixels=mark.OFF, sensor=False):
    if blind and sensor:
        raise ValueError("blind and sensor are two different observation modes, not a scale: blind removes the text observation and every outcome report, sensor keeps one line of text and reports what the robot's own body measured. Pick one.")
    unnamed = blind or sensor
    return Episode(episode_id=task_id, scene=HOUSE, agent_pose=spawn, partial_obs=partial_obs, room_prior=not unnamed, furniture_prior=not unnamed, strict_object_discovery=False, world_text=not blind, hand_only=sensor, action_feedback=feedback.NONE if blind else feedback.SENSOR if sensor else feedback.VERDICT, mark_pixels=mark_pixels)

class Stages:

    def __init__(self):
        self.sec = {}
        self.n = {}

    @contextlib.contextmanager
    def time(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t0)

    def add(self, name, seconds, count=1):
        self.sec[name] = self.sec.get(name, 0.0) + seconds
        self.n[name] = self.n.get(name, 0) + count

    def merge(self, other, prefix=''):
        for k, v in (other or {}).get('sec', {}).items():
            self.add(prefix + k, v, count=(other.get('n') or {}).get(k, 0))

    def to_dict(self, total=None):
        out = {'sec': {k: round(v, 3) for k, v in self.sec.items()}, 'n': dict(self.n)}
        if total is not None:
            out['total_sec'] = round(total, 3)
            out['unaccounted_sec'] = round(total - sum(self.sec.values()), 3)
        return out

def _short_arg(arg):
    if hasattr(arg, 'name'):
        return arg.name
    if hasattr(arg, 'deg'):
        return f'{arg.deg:+.0f}°/{arg.dist_m:.2f}m'
    return f'{arg:.2f}' if isinstance(arg, float) else str(arg)

def _record_reason(agent, res, receipt):
    if not agent.rounds:
        return
    row = agent.rounds[-1]
    row['raw_reason'] = getattr(res, 'reason', '') or ''
    row['sense'] = receipt
CONTINUOUS = True

@contextlib.contextmanager
def filming(b, case_dir):
    os.makedirs(case_dir, exist_ok=True)
    strip = FilmStrip(os.path.join(case_dir, 'episode.mp4'))
    if CONTINUOUS:
        set_rate = getattr(b, 'set_film_rate', None)
        if set_rate is not None:
            set_rate(strip.fps)
        b.on_frame, b.on_label = (strip.add, strip.begin)
        b.on_aim = strip.track
    try:
        yield strip
    finally:
        b.on_frame = b.on_label = b.on_aim = None
_LEGAL_ROUND_SECONDS = 64.0

def episode_wallclock_deadline_s(budget, extra_rounds, *, legal_round_s=_LEGAL_ROUND_SECONDS, margin=1.5):
    return margin * (budget + extra_rounds) * legal_round_s

def wallclock_exhausted(elapsed_s, deadline_s):
    return bool(deadline_s) and elapsed_s > deadline_s

def run_model_episode(b, spec, client, tier, budget, extra_rounds, spawn, verbose=True, strip=None, vision_on=False, vision_keep=None, stages=None, refer='name', blind=False, mark_pixels=mark.OFF, sensor=False, compact_every=0, compact_min_chars=None, compact_keep=None, compact_after_prompt_tokens=0, summarizer=None, wallclock_deadline_s=None, vision_mime=vision.MIME, vision_quality=90, vision_max_bytes=None, vision_resolution_ladder=(), vision_request_target_bytes=None, vision_preserve_canvas=True, vision_max_parts=None, model_image_limit=0, memory_harness='standard', retrieval_max_passes=2, retrieval_max_images=12, episode_token_limit=0, memory_archive_dir=None, intermediate_spacing_m=1.0, intermediate_rotation_deg=30.0, intermediate_max_frames=8, episode_resume=None, episode_checkpoint_path=None):
    st = stages if stages is not None else Stages()
    task = spec.task
    with st.time('episode_reset'):
        ep = episode_for(f'demo-{task.task_id}', spawn, blind=blind, sensor=sensor, mark_pixels=mark_pixels)
        b.reset(ep)
    fb = feedback.Feedback.from_episode(ep)
    space = ActionSpace(b, tier, refer=refer)
    film = EpisodeFilm(b, f'{task.task_id} · {client.model}')
    rec = Recorder(b, task)
    if vision_on and vision_resolution_ladder and (not vision_preserve_canvas):
        model_frame_size = vision.scaled_size((b.width, b.height), vision_resolution_ladder[0])
    else:
        model_frame_size = (b.width, b.height) if vision_on else None
    pixel_scale = None
    if model_frame_size and model_frame_size != (b.width, b.height):
        pixel_scale = ((b.width - 1) / max(1, model_frame_size[0] - 1), (b.height - 1) / max(1, model_frame_size[1] - 1))
    agent_class = llm.LLMAgent
    extra_agent_kwargs = {}
    if memory_harness in {'retrieve', 'intermediate', 'trajectory'}:
        from roam_bench.protocol.unified_memory import UnifiedRetrievalAgent as RetrievalAgent
        agent_class = RetrievalAgent
        if memory_harness in {'intermediate', 'trajectory'}:
            from roam_bench.protocol.unified_memory import UnifiedIntermediateAgent as IntermediateAgent
            agent_class = IntermediateAgent
        extra_agent_kwargs = {'retrieval_max_passes': retrieval_max_passes, 'retrieval_max_images': retrieval_max_images, 'archive_dir': memory_archive_dir, 'task_family': task.task_id.split('__', 1)[0]}
    agent = agent_class(client, space, task.instruction, budget, tier, vision=vision_on, vision_keep=vision_keep, frame_size=model_frame_size, world_text=ep.world_text, hand_only=ep.hand_only, action_feedback=ep.action_feedback, mark_frames=mark.marks_frames(ep.mark_pixels), compact_every=compact_every, compact_min_chars=compact_min_chars, compact_keep=compact_keep, summarizer=summarizer, compact_after_prompt_tokens=compact_after_prompt_tokens, vision_max_bytes=vision_max_bytes, vision_resolution_ladder=vision_resolution_ladder, vision_request_target_bytes=vision_request_target_bytes, goal_images=goalimg.encode(task.goal_images, os.path.dirname(TASKSET)), **extra_agent_kwargs)
    agent.model_image_limit = int(model_image_limit or 0)
    agent.episode_token_limit = int(episode_token_limit or 0)
    agent._episode_token_base = int(getattr(client, 'prompt_tokens', 0)) + int(getattr(client, 'completion_tokens', 0))
    if memory_harness in {'intermediate', 'trajectory'}:
        from roam_bench.intermediate_frames import publish_evidence
        agent.on_candidates_delivered = lambda frames, round_i: publish_evidence(b, frames, round_i)
    if memory_harness == 'trajectory':
        note = "\nSelf-trajectory sensor: every decision includes one cumulative top-down odometry plot through the current pose. It contains only the robot's own x/z/yaw history, metric grid and step order; it contains no floor plan, room boundary, room label, object location or task truth. Use it to reason about where you have moved and turned. The CURRENT FINAL camera image remains the last image and is the only source for action pixel coordinates.\n"
        agent.system += note
        first = agent.messages[0]
        if isinstance(first['content'], list):
            first['content'][-1]['text'] += note
        else:
            first['content'] += note
        agent.trajectory_points = []
    if vision_max_parts:
        agent.configure_image_part_limit(vision_max_parts, lambda urls: vision.pack_data_urls(urls, columns=2, mime=vision_mime, quality=vision_quality))
    if wallclock_deadline_s is None:
        wallclock_deadline_s = episode_wallclock_deadline_s(budget, extra_rounds)
    receipt, ended, malformed, refused = ('', '预算用完（没有宣布完成）', 0, 0)
    timed_out = False
    pixel_clicks = []
    t0 = time.time()
    from roam_bench import episode_checkpoint as checkpoint
    checkpoint_identity = checkpoint.identity(task, {'task_id': task.task_id, 'instruction': task.instruction, 'budget': budget, 'extra_rounds': extra_rounds, 'model': client.model, 'harness': memory_harness, 'image_limit': agent.model_image_limit, 'vision_keep': vision_keep, 'frame_size': list(model_frame_size or ()), 'tier': tier, 'refer': refer, 'sensor': sensor, 'episode_token_limit': agent.episode_token_limit})
    start_round = 1
    resumed = None
    inherited_usage = {'prompt_tokens': 0, 'completion_tokens': 0}
    terminal = None
    if episode_checkpoint_path and os.path.exists(episode_checkpoint_path):
        state = checkpoint.load(episode_checkpoint_path, b, rec, agent, checkpoint_identity)
        start_round, receipt = (state['next_round'], state['receipt'])
        malformed, refused = (state['malformed'], state['refused'])
        inherited_usage = state['usage']
        terminal = state.get('terminal')
        pixel_clicks = state.get('pixel_clicks', [])
        resumed = {'mode': 'exact-episode-checkpoint', 'next_round': start_round, 'source_steps': len(rec.steps), 'model_calls_for_prefix': 0, 'navigation_steps_replayed': 0}
    elif episode_resume:
        source_log = episode_resume['log']
        source_row = episode_resume['row']
        resumed = checkpoint.replay_restore(b, rec, agent, source_log, source_row, episode_resume['case'])
        start_round = len(agent.rounds) + 1
        last = agent.rounds[-1]
        if last.get('sense'):
            receipt = last['sense']
        elif last.get('parse_error'):
            receipt = fb.for_parse_error(last['parse_error'])
        else:
            receipt = ''
        malformed, refused = (source_log['meta']['malformed_rounds'], source_log['meta']['tier_refusals'])
        pixel_clicks = list(source_log['meta'].get('pixel_clicks', []))
        inherited_usage = source_row['gateway_usage']
    if resumed:
        print(f"[episode-resume] task={task.task_id} next_round={start_round} steps={len(rec.steps)}/{budget} mode={resumed['mode']}", flush=True)

    def save_checkpoint(next_round, terminal_reason=None):
        if not episode_checkpoint_path:
            return
        used = {'prompt_tokens': inherited_usage['prompt_tokens'] + client.prompt_tokens - episode_resume_client_base[0], 'completion_tokens': inherited_usage['completion_tokens'] + client.completion_tokens - episode_resume_client_base[1]}
        checkpoint.save(episode_checkpoint_path, b, rec, agent, next_round, receipt, malformed, refused, checkpoint_identity, used, terminal=terminal_reason, pixel_clicks=pixel_clicks)
    episode_resume_client_base = (client.prompt_tokens, client.completion_tokens)
    save_checkpoint(start_round, terminal)
    if terminal:
        ended = terminal

    def pointed_at(arg):
        if not isinstance(arg, PixelRef):
            return None
        try:
            return b.instance_at_pixel(arg.u, arg.v)
        except Exception:
            return None

    def hold(label):
        if strip is not None:
            strip.hold(label)

    def say(round_i, what, ok, detail=''):
        if verbose:
            tag = f'{verbose} ' if isinstance(verbose, str) else ''
            print(f"    {tag}r{round_i:<3} {what} -> {('ok' if ok else 'no')} :: {(detail or '')[:70]}")
    for round_i in range(start_round, budget + extra_rounds + 1):
        if terminal:
            break
        if len(rec.steps) >= budget:
            break
        if wallclock_exhausted(time.time() - t0, wallclock_deadline_s):
            timed_out = True
            ended = f'挂钟超时（{wallclock_deadline_s:.0f} s / {wallclock_deadline_s / 3600:.1f} h，第 {round_i} 轮，{budget} 步预算未走完）'
            say(round_i, '(挂钟超时)', False, ended)
            hold(f'r{round_i}: wall-clock timeout')
            break
        with st.time('observe_text'):
            view = b.world().text()
        img, rgb, image_variants = (None, None, ())
        if vision_on:
            with st.time('frame_encode'):
                rgb = b.observe().rgb
                if vision_resolution_ladder:
                    image_variants = vision.data_url_ladder(rgb, vision_resolution_ladder, mime=vision_mime, quality=vision_quality, preserve_canvas=vision_preserve_canvas)
                    img = image_variants[0][1]
                else:
                    img = vision.data_url(rgb, mime=vision_mime, quality=vision_quality)
        if strip is not None and mark.marks_video(ep.mark_pixels):
            strip.aim()
            b.aim_at_pixel()
        trajectory_image, trajectory_meta = (None, None)
        if memory_harness == 'trajectory':
            from roam_bench.self_trajectory import render_self_trajectory
            pose = b.agent_pose()
            agent.trajectory_points.append({'round': round_i, 'action_steps': len(rec.steps), 'x': float(pose.x), 'z': float(pose.z), 'yaw_deg': float(pose.yaw_deg), 'horizon_deg': float(pose.horizon_deg)})
            trajectory_rgb, trajectory_meta = render_self_trajectory(agent.trajectory_points)
            trajectory_meta.update(round=round_i, action_steps=len(rec.steps))
            trajectory_image = vision.data_url(trajectory_rgb, max_side=640, mime='image/jpeg', quality=86)
        try:
            with st.time('model_wait'):
                row = agent.ask(view, receipt, round_i, len(rec.steps), image=img, image_variants=image_variants, trajectory_image=trajectory_image, trajectory_meta=trajectory_meta)
        except llm.EpisodeTokenBudgetExceeded:
            ended = '每题Token成本上限触发，停止继续请求'
            say(round_i, '(Token成本上限)', False, ended)
            break
        common = {'thought': row['thought'], 'prompt': row['prompt'], 'raw': row['raw']}
        if rgb is not None:
            _lit_mean, _lit_frac = vision.legibility(rgb)
            agent.rounds[-1]['lit_frac'] = _lit_frac
            agent.rounds[-1]['lit_mean'] = _lit_mean
        if row['verb'] is None:
            malformed += 1
            with st.time('report_capture'):
                film.note_round('(无法解析)', '', False, row['parse_error'], note='没有产生动作，只消耗一轮', **common)
            hold(f'r{round_i}: unparsable reply')
            say(round_i, '(无法解析)', False, row['parse_error'])
            receipt = fb.for_parse_error(row['parse_error'])
            if malformed >= 8:
                ended = f'连续无法解析（{malformed} 轮）'
                break
            save_checkpoint(round_i + 1)
            continue
        if row['verb'] == 'answer':
            text = row['args'][0] if row['args'] else ''
            rec.answer(text)
            with st.time('report_capture'):
                film.note_round('answer', text, True, f'答案已记录：{text!r}', note='回答问题，不消耗动作预算', **common)
            hold(f'answer({text})')
            hold(f'answer({text})')
            say(round_i, f'answer({text!r})', True, '本集结束')
            ended = '模型给出答案'
            save_checkpoint(round_i + 1, ended)
            break
        if row['verb'] == 'done':
            with st.time('report_capture'):
                film.note_round('done', '', True, '模型宣布任务完成', note='终止本集', **common)
            hold('done() -- self-reported, never scored')
            hold('done() -- self-reported, never scored')
            say(round_i, 'done()', True, '模型宣布完成')
            ended = '模型宣布完成'
            save_checkpoint(round_i + 1, ended)
            break
        kind, arg, thunk = llm.bind(space, row['verb'], tuple(row['args']), row['kwargs'], pixel_scale=pixel_scale)
        if kind is None:
            refused += 1
            with st.time('report_capture'):
                film.note_round(row['verb'], ', '.join(row['args']), False, thunk, note='这一档不提供 / 参数不对，没有到达仿真器', **common)
            hold(f"{row['verb']}(...) REFUSED: not in this tier")
            say(round_i, f"{row['verb']}(…) [越档]", False, thunk)
            receipt = fb.for_refusal(thunk)
            save_checkpoint(round_i + 1)
            continue
        frames_before = strip.mark() if strip is not None else 0
        if strip is not None and mark.marks_video(ep.mark_pixels):
            if isinstance(arg, PixelRef):
                strip.aim(arg.u, arg.v)
                b.aim_at_pixel(arg.u, arg.v)
            else:
                strip.aim()
                b.aim_at_pixel()
        with st.time('report_capture'):
            aimed = pointed_at(arg)
        with st.time('sim_step'):
            interaction_start = len(getattr(b, '_evidence_interactions', ()))
            if memory_harness in {'intermediate', 'trajectory'}:
                b._model_motion_trace = []
                try:
                    res = thunk()
                    trace = b._model_motion_trace
                finally:
                    b._model_motion_trace = None
            else:
                res = thunk()
        rec.step(kind, arg, res)
        from roam_bench.interaction_events import stamp_interaction_events
        stamp_interaction_events(getattr(b, '_evidence_interactions', []),
                                 interaction_start, round_i, len(rec.steps))
        if memory_harness in {'intermediate', 'trajectory'}:
            from roam_bench.intermediate_frames import sample_motion, capture_intermediates
            capacity = agent.candidate_capacity(intermediate_max_frames)
            poses = sample_motion(trace, spacing_m=intermediate_spacing_m, rotation_deg=intermediate_rotation_deg, maximum=capacity)
            with st.time('intermediate_capture'):
                agent.pending = capture_intermediates(b, poses, round_i, lambda frame: vision.data_url(frame, max_side=model_frame_size[0], mime=vision_mime, quality=vision_quality))
                agent.save_archive()
        if rgb is not None and mark.marks_frames(ep.mark_pixels) and isinstance(arg, PixelRef):
            with st.time('frame_encode'):
                if vision_resolution_ladder:
                    marked_variants = vision.data_url_ladder(rgb, vision_resolution_ladder, point=(arg.u, arg.v), mime=vision_mime, quality=vision_quality, preserve_canvas=vision_preserve_canvas)
                    agent.remark_last_frame(marked_variants[0][1], image_variants=marked_variants)
                else:
                    agent.remark_last_frame(vision.data_url(rgb, point=(arg.u, arg.v), mime=vision_mime, quality=vision_quality))
        if isinstance(arg, PixelRef):
            pixel_clicks.append({'round': round_i, 'verb': kind, 'u': arg.u, 'v': arg.v, 'at_pixel': aimed, 'ok': bool(res.success)})
        with st.time('report_capture'):
            film.capture(kind, arg, res, note=f'第 {len(rec.steps)}/{budget} 个动作' + ('' if aimed is None else f'；这个像素上渲染的是 `{aimed}`'), **common)
        if strip is not None and strip.n_frames == frames_before:
            hold(f'{kind}({_short_arg(arg)}) -- nothing moved')
        receipt = fb.for_action(res, kind)
        _record_reason(agent, res, receipt)
        say(round_i, f'{kind}({_short_arg(arg)})', res.success, res.reason)
        save_checkpoint(round_i + 1)
    save_checkpoint(len(agent.rounds) + 1, ended)
    score = rec.score()
    meta = {'rounds': len(agent.rounds), 'steps': len(rec.steps), 'malformed_rounds': malformed, 'tier_refusals': refused, 'ended': ended, 'seconds': round(time.time() - t0, 1), 'timed_out': timed_out, 'wallclock_deadline_s': wallclock_deadline_s, 'vision': bool(vision_on), 'frames_shown': agent.images_sent, 'frame_bytes': agent.image_bytes, 'refer': refer, 'vision_mime': vision_mime, 'vision_quality': vision_quality, 'vision_max_bytes': agent.vision_max_bytes, 'vision_resolution_ladder': list(agent.vision_resolution_ladder), 'vision_frame_size': list(model_frame_size) if model_frame_size else None, 'vision_preserve_canvas': bool(vision_preserve_canvas), 'vision_request_target_bytes': agent.vision_request_target_bytes, 'vision_resolution_events': list(agent.vision_resolution_events), 'vision_max_parts': agent.vision_max_parts, 'vision_packing_events': list(agent.vision_packing_events), 'image_limit_events': list(agent.image_limit_events), 'retrieval_events': list(getattr(agent, 'retrieval_events', [])), 'intermediate_events': list(getattr(agent, 'intermediate_events', [])), 'self_trajectory': {'enabled': memory_harness == 'trajectory', 'pose_count': len(getattr(agent, 'trajectory_points', [])), 'poses': list(getattr(agent, 'trajectory_points', [])), 'contains_scene_geometry': False}, 'token_budget_exceeded': bool(getattr(agent, 'token_budget_exceeded', False)), 'blind': bool(blind), 'world_text': ep.world_text, 'sensor': bool(sensor), 'hand_only': ep.hand_only, 'action_feedback': ep.action_feedback, 'mark_pixels': ep.mark_pixels, 'compact_every': int(compact_every or 0), 'compact_after_prompt_tokens': agent.compact_after_prompt_tokens, 'compact_min_chars': agent.compact_min_chars, 'compact_keep': agent.compact_keep, 'compactions': agent.compactions, 'context_chars_final': agent.rounds[-1]['context']['chars'] if agent.rounds else 0, 'pixel_clicks': pixel_clicks, 'clicks_on_nothing': sum((1 for c in pixel_clicks if c['at_pixel'] is None))}
    if resumed:
        meta['episode_resume'] = resumed
    meta['inherited_gateway_usage'] = inherited_usage
    return (score, film, agent, meta)
