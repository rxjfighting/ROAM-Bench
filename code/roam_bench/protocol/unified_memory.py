"""Manage recent visual history, selected intermediate views, and image retrieval."""
import hashlib
import json
import re
from .llm import conversation_size, text_of, RequestBodyTooLarge
from .retrieval_memory import RetrievalAgent
from .intermediate_memory import IntermediateAgent, image_part

def header(raw, name, default=None):
    found = re.search('(?m)^\\s*' + name + '\\s*:\\s*([^\\n]*)', raw or '')
    if not found:
        return default
    try:
        return json.loads(found.group(1))
    except (TypeError, ValueError):
        return default

def request_record(messages):
    result = []
    for msg in messages:
        content = msg.get('content', '')
        if isinstance(content, list):
            parts = []
            for part in content:
                if part.get('type') == 'image_url':
                    url = part['image_url']['url']
                    parts.append({'type': 'image_reference', 'url_sha256': hashlib.sha256(url.encode()).hexdigest(), 'url_bytes': len(url.encode())})
                else:
                    parts.append(dict(part))
            content = parts
        result.append({'role': msg['role'], 'content': content})
    return result

class UnifiedDecision:
    intermediate = False

    def __init__(self, *args, task_family=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.task_family = str(task_family or '')
        self.system = self.system.split('Reply in exactly this shape, nothing else:')[0]
        self.system += '\nUnified memory and action decision. Recent observation images remain visible; older rounds have an indexed summary, chosen action and original execution receipt. All previous original observation/reply text is archived and retrievable. Summaries are uncertain notes, not ground truth. Choose EITHER retrieve information OR one normal action in each reply. After retrieval you see its results and make the SAME decision again. Retrieval does not spend physical action steps; at most two retrieval requests are allowed per action round, including empty or invalid requests. No more flag or separate selector is needed. If the information is sufficient, select the action directly. Avoid repeatedly retrieving the same landmarks without a concrete unresolved question.\nFor retrieval reply:\nTHOUGHT: <brief information gap>\nRETRIEVE: {"retrieve": [image_ids], "retrieve_text": [integer_rounds]}\nFor an action reply:\nMEMORY: <at most 100 words of observed facts, distinctive objects/rooms, orientation, counting/de-duplication evidence and unresolved uncertainty>\nTHOUGHT: <brief reason>\nACTION: <one call from the skills above>\nNever output RETRIEVE and ACTION in the same reply. The current action has NOT executed; only summarize its outcome after the next execution receipt. All historical images have historical coordinates. Use ONLY the CURRENT FINAL observation for action pixels.\n'
        if self.intermediate:
            self.system += 'C intermediate memory: the first decision previews ALL candidate intermediate frames from the previous action. On that first reply optionally add KEEP: [candidate_frame_ids] before other headers, whether retrieving or acting. Default KEEP is []. Keep only key views; unselected candidates are discarded from later requests and cannot be retrieved. Kept views enter recent history and the archive alongside the final observation. Keep selection is final for this round. Image IDs use R#.F and R#.I#; text IDs use integer round numbers.\n'
        else:
            self.system += 'B image IDs and text IDs both use integer round numbers.\n'
        if self.task_family == 'surface_contents':
            self.system += 'CONTENTS CHECK: inspect the entire named surface or container from enough views to cover it. Include only objects directly on or in that target, keep distinct instances, and exclude nearby objects on another surface. Before answer(), check this evidence once; retrieve a relevant old view only when a specific part of the target remains unresolved.\n'
        msg = self.messages[0]
        if isinstance(msg['content'], list):
            msg['content'][-1]['text'] = self.system
        else:
            msg['content'] = self.system
        if not 0 <= self.retrieval_max_passes <= 2:
            raise ValueError('unified retrieval allows zero, one or two requests per round')

    def _indexed_messages(self, messages):
        request = RetrievalAgent._indexed_messages(self, messages)
        old = 'retrieve=[round_numbers] for images.'
        ids = 'retrieve=["R#.F", "R#.I#"] for images.' if self.intermediate else old
        return [dict(m, content=m['content'].replace(old, ids).replace('Request retrieve_text=', 'In a RETRIEVE JSON request, use retrieve_text=')) if m.get('role') == 'user' and isinstance(m.get('content'), str) and m['content'].startswith('INDEXED HISTORY.') else m for m in request]
    _bounded = IntermediateAgent._bounded

    def _request_messages(self):
        final = self._message_image(self.messages[-1])
        trajectory = getattr(self, '_current_trajectory_image', None)
        request, gate = self._bounded(self._indexed_messages(self.messages), [image for image in (trajectory, final) if image])
        self.image_limit_events.append(gate)
        return (request, None)

    def _catalogue(self, round_i):
        available = {}
        for entry in self.archive:
            if entry['round'] > round_i - self.vision_keep:
                continue
            key = f"R{entry['round']}.F" if self.intermediate else entry['round']
            if entry.get('image'):
                available[key] = dict(image=entry['image'], summary=entry['summary'])
            if self.intermediate:
                for frame in entry.get('intermediates', []):
                    available[frame['id']] = dict(frame, summary=entry['summary'])
        return available

    @staticmethod
    def _retrieved_message(by_id, selected):
        if not selected:
            return []
        content = [{'type': 'text', 'text': 'Retrieved historical images; historical coordinates only.'}]
        for key in selected:
            content.extend([{'type': 'text', 'text': f"{key}: {by_id[key]['summary']}"}, image_part(by_id[key]['image'])])
        return [{'role': 'user', 'content': content}]

    def _complete_action_request(self, messages, round_i):
        final = self._message_image(self.messages[-1])
        trajectory = getattr(self, '_current_trajectory_image', None)
        current = self.messages[-1]
        candidates = list(self.pending) if self.intermediate else []
        candidate_by_id = {f['id']: f for f in candidates}
        by_id = self._catalogue(round_i)
        selected, selected_text, text_results, dialogue = ([], [], [], [])
        retained = []
        if self.intermediate:
            self._retained = []
            preview = dict(round=round_i, generated_ids=list(candidate_by_id), delivered_ids=[], retained_ids=[], selector=None, selection_policy='unified-first-decision-v3')
            self.intermediate_events.append(preview)
        event = self.last_retrieval = dict(round=round_i, policy='unified-retrieve-or-action-v3', passes=[], calls=[], retrieved_rounds=[], retrieved_frames=[], retrieved_text_rounds=[])
        self.retrieval_events.append(event)
        for attempt in range(self.retrieval_max_passes + 1):
            first = attempt == 0
            preview_parts = []
            if first:
                for frame in candidates:
                    preview_parts.extend([{'type': 'text', 'text': f"Candidate {frame['id']} from previous action; historical coordinates only."}, image_part(frame['image'])])
            extras = self._retrieved_message(by_id, selected) + text_results + dialogue
            if preview_parts:
                extras += [{'role': 'user', 'content': preview_parts}]
            bundle = list(messages[:-1]) + extras + list(messages[-1:])
            protected = [image for image in (trajectory, final) if image]
            protected += [by_id[i]['image'] for i in selected]
            protected += [f['image'] for f in candidates] if first else [f['image'] for f in retained]
            bundle, _ = self._bounded(bundle, protected)
            used = conversation_size(bundle)['image_parts']
            remaining = max(0, min(self.retrieval_max_images - len(selected), self.model_image_limit - used))
            left = self.retrieval_max_passes - attempt
            instruction = f'UNIFIED DECISION, round {round_i}. Retrieval requests remaining: {left}. Choose RETRIEVE JSON OR one ACTION using the same system format. Retrieval results will be shown before you decide again. Already fetched images: {selected}; text rounds: {selected_text}. This input has {used} images, including all references, history, candidates, retrieval results and CURRENT FINAL. Total image limit {self.model_image_limit}; request at most {remaining} NEW images, and at most {self.retrieval_max_images} images total this round. Original text: at most {max(0, 12 - len(selected_text))} NEW rounds (12 total); available for ALL previous integer rounds. Image IDs are available only in this catalogue: ' + json.dumps(list(by_id)) + '. No more flag. No separate selector. CURRENT FINAL is the last image of the current observation; only it provides current action pixels. '
            if not left:
                instruction += 'RETRIEVE IS DISABLED: select one normal ACTION now. '
            if self.intermediate:
                instruction += f'Candidate IDs shown: {list(candidate_by_id)}. On the FIRST decision only, optionally output KEEP: [IDs]; default []. ' if first else f"Keep selection is fixed: {[f['id'] for f in retained]}. "
            request, gate = self._bounded(bundle + [{'role': 'user', 'content': instruction}], protected)
            before = (self.client.prompt_tokens, self.client.completion_tokens)
            raw = self._memory_call(request)
            call = dict(call=attempt + 1, raw=raw, kind='unclassified', image_parts=conversation_size(request)['image_parts'], request_bytes=self.client.last_request_bytes, image_limit=gate, prompt_tokens=self.client.prompt_tokens - before[0], completion_tokens=self.client.completion_tokens - before[1], tokens=self.client.prompt_tokens + self.client.completion_tokens - sum(before), context_images=used, remaining_retrieval_images=remaining, retrieval_requests_remaining=left, total_image_limit=self.model_image_limit, input=request_record(request))
            event['calls'].append(call)
            if not raw:
                call['kind'] = 'failed'
                event.update(action_context=conversation_size(request), action_image_parts=call['image_parts'], action_request_bytes=call['request_bytes'])
                self.save_archive()
                return ''
            if self.intermediate and first:
                preview['delivered_ids'] = list(candidate_by_id)
                if candidates:
                    self.images_sent += len(candidates)
                    self.image_bytes += sum((len(f['image']) for f in candidates))
                    if self.on_candidates_delivered:
                        self.on_candidates_delivered(candidates, round_i)
                keep = header(raw, 'KEEP', [])
                ids = list(dict.fromkeys((i for i in keep if isinstance(i, str) and i in candidate_by_id))) if isinstance(keep, list) else []
                retained = self._retained = [candidate_by_id[i] for i in ids]
                preview.update(retained_ids=ids, selector=call)
                prefix = [{'type': 'text', 'text': f'Intermediate selection: shown={list(candidate_by_id)}, retained={ids}. CURRENT FINAL is the last image in this observation; use it for action pixels.'}]
                for frame in retained:
                    prefix.extend([{'type': 'text', 'text': frame['id']}, image_part(frame['image'])])
                current['content'] = prefix + current['content']
                messages = list(messages[:-1]) + [current]
            retrieve = bool(re.search('(?m)^\\s*RETRIEVE\\s*:', raw))
            action = bool(re.search('(?m)^\\s*(?:ACTION|动作)\\s*[:：]', raw))
            if retrieve and (action or not left):
                call.update(kind='invalid', error='ambiguous reply' if action else 'retrieval cap reached')
                raw = 'THOUGHT: Invalid unified decision: ' + call['error']
                retrieve = False
            if not retrieve:
                if call['kind'] != 'invalid':
                    call['kind'] = 'action' if action else 'invalid'
                call['returned_raw'] = raw
                event.update(retrieved_frames=selected if self.intermediate else [], retrieved_rounds=sorted({int(i.split('.')[0][1:]) for i in selected}) if self.intermediate else list(selected), retrieved_text_rounds=selected_text, action_context=conversation_size(request), action_image_parts=call['image_parts'], action_image_limit=gate, action_request_bytes=call['request_bytes'])
                if self.intermediate:
                    self.pending = []
                self.save_archive()
                return raw
            call.update(kind='retrieve', **{'pass': attempt + 1})
            event['passes'].append(call)
            choice = header(raw, 'RETRIEVE', {})
            if not isinstance(choice, dict):
                choice = {}
            requested = choice.get('retrieve', [])
            keys = [i for i in requested if (isinstance(i, str) if self.intermediate else isinstance(i, int) and (not isinstance(i, bool))) and i in by_id and (i not in selected)] if isinstance(requested, list) else []
            keys = list(dict.fromkeys(keys))[:remaining]
            accepted = []
            next_dialogue = dialogue + [{'role': 'assistant', 'content': raw}]
            for key in keys:
                probe = list(messages[:-1]) + self._retrieved_message(by_id, selected + accepted + [key]) + text_results + next_dialogue + list(messages[-1:]) + [{'role': 'user', 'content': instruction}]
                try:
                    self._bounded(probe, [image for image in (trajectory, final) if image] + [f['image'] for f in retained] + [by_id[i]['image'] for i in selected + accepted + [key]])
                    accepted.append(key)
                except RequestBodyTooLarge:
                    break
            selected.extend(accepted)
            requested_text, accepted_text, parts = self._text_retrieval(choice, round_i, selected_text)
            text_capacity_rejected = []
            fitted_parts, fitted_rounds = ([], [])
            for number, part in zip(accepted_text, parts):
                probe = list(messages[:-1]) + self._retrieved_message(by_id, selected) + text_results + [{'role': 'user', 'content': fitted_parts + [part]}] + next_dialogue + list(messages[-1:]) + [{'role': 'user', 'content': instruction}]
                try:
                    self._bounded(probe, [image for image in (trajectory, final) if image] + [f['image'] for f in retained] + [by_id[i]['image'] for i in selected])
                    fitted_parts.append(part)
                    fitted_rounds.append(number)
                except RequestBodyTooLarge:
                    text_capacity_rejected.append(number)
            selected_text.extend(fitted_rounds)
            if fitted_parts:
                text_results.append({'role': 'user', 'content': fitted_parts})
            call.update(requested=requested, accepted=accepted, requested_text=requested_text, accepted_text=fitted_rounds, text_capacity_rejected=text_capacity_rejected)
            dialogue = next_dialogue + [{'role': 'user', 'content': f'Retrieval receipt: accepted images={accepted}, accepted text={fitted_rounds}. Unknown, duplicate, recent image IDs and over-capacity entries were not fetched. Retrieval requests remaining={left - 1}. Inspect results and decide again.'}]
            event.update(retrieved_frames=list(selected) if self.intermediate else [], retrieved_rounds=sorted({int(i.split('.')[0][1:]) for i in selected}) if self.intermediate else list(selected), retrieved_text_rounds=list(selected_text))
            self.save_archive()

class UnifiedRetrievalAgent(UnifiedDecision, RetrievalAgent):
    pass

class UnifiedIntermediateAgent(UnifiedDecision, IntermediateAgent):
    intermediate = True
