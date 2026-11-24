"""Intermediate memory utilities for ROAM-Bench."""
import base64
import hashlib
import json
import os
import re
from collections import Counter
from .llm import conversation_size, text_of, RequestBodyTooLarge
from .retrieval_memory import RetrievalAgent
from roam_bench.intermediate_frames import MAX_FRAME_URL_BYTES

def image_part(url):
    return {'type': 'image_url', 'image_url': {'url': url}}

class IntermediateAgent(RetrievalAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending = []
        self.intermediate_events = []
        self.on_candidates_delivered = None
        self._retained = []
        note = '\nIntermediate-frame memory: the pre-action CURRENT FINAL image is the only image for action coordinates. Before actions, a JSON memory selector can preview all intermediate frames from the previous action, retain useful frame IDs, and retrieve archived frame IDs. Default retention is none; the final image is always archived. Unselected intermediates cannot be retrieved. MEMORY must summarize the observed outcome and retained frames, with uncertainty. The current action has NOT run yet. Retrieve images using frame IDs (R#.F / R#.I#), and original text using integer round numbers.'
        self.system += note
        if isinstance(self.messages[0]['content'], list):
            self.messages[0]['content'][-1]['text'] += note
        else:
            self.messages[0]['content'] += note

    def candidate_capacity(self, maximum=8):
        pinned = [m for m in self.messages if m.get('role') != 'user']
        references = conversation_size(pinned)['image_parts']
        text_request = [dict(m, content=text_of(m['content'])) if m.get('role') == 'user' else m for m in self.messages]
        current_images = 1 + int(bool(getattr(self, '_current_trajectory_image', None)))
        reserve = self.client.request_size(text_request) + current_images * MAX_FRAME_URL_BYTES + 1000000
        slots = max(0, (self.client.MAX_REQUEST_BYTES - reserve) // (MAX_FRAME_URL_BYTES + 256))
        return max(0, min(int(maximum), self.model_image_limit - references - current_images, slots))

    def _bounded(self, messages, protected=()):
        request = [dict(m, content=[dict(p) for p in m['content']]) if isinstance(m.get('content'), list) else dict(m) for m in messages]
        protected = Counter(protected)
        before = conversation_size(request)['image_parts']
        while conversation_size(request)['image_parts'] > self.model_image_limit or self.client.request_size(request) > self.client.MAX_REQUEST_BYTES:
            removed = False
            counts = Counter((p['image_url']['url'] for m in request if m.get('role') == 'user' and isinstance(m.get('content'), list) for p in m['content'] if p.get('type') == 'image_url'))
            for msg in request:
                if msg.get('role') != 'user' or not isinstance(msg.get('content'), list):
                    continue
                for i, p in enumerate(msg['content']):
                    if p.get('type') == 'image_url' and counts[p['image_url']['url']] > protected[p['image_url']['url']]:
                        del msg['content'][i]
                        removed = True
                        break
                if removed:
                    break
            if not removed:
                raise RequestBodyTooLarge('references, current bundle and all text exceed request capacity')
        return (request, {'policy': 'protected-current-bundle-v1', 'limit': self.model_image_limit, 'dropped_images': before - conversation_size(request)['image_parts'], 'retained_parts': conversation_size(request)['image_parts'], 'request_bytes': self.client.request_size(request)})

    def _indexed_messages(self, messages):
        request = super()._indexed_messages(messages)
        return [dict(m, content=m['content'].replace('retrieve=[round_numbers] for images.', 'retrieve=["R#.F", "R#.I#"] for archived images; only IDs in the selector catalogue are valid.')) if m.get('role') == 'user' and isinstance(m.get('content'), str) and m['content'].startswith('INDEXED HISTORY.') else m for m in request]

    def _request_messages(self):
        current = self._message_image(self.messages[-1])
        request, event = self._bounded(self._indexed_messages(self.messages), [current] if current else [])
        self.image_limit_events.append(event)
        return (request, None)

    def _complete_action_request(self, messages, round_i):
        current_message = self.messages[-1]
        final = self._message_image(current_message)
        candidates = list(self.pending)
        candidate_by_id = {f['id']: f for f in candidates}
        older = [a for a in self.archive if a['round'] <= round_i - self.vision_keep]
        by_id = {}
        for a in older:
            if a.get('image'):
                by_id[f"R{a['round']}.F"] = {'id': f"R{a['round']}.F", 'image': a['image'], 'summary': a['summary']}
            for f in a.get('intermediates', []):
                by_id[f['id']] = dict(f, summary=a['summary'])
        catalogue = ', '.join(by_id)
        selected, retained = ([], [])
        selected_text = []
        text_results = []
        self._retained = []
        self.last_retrieval = {'round': round_i, 'passes': [], 'retrieved_rounds': [], 'retrieved_frames': []}
        preview_event = {'round': round_i, 'generated_ids': list(candidate_by_id), 'delivered_ids': [], 'retained_ids': [], 'selector': None}
        self.intermediate_events.append(preview_event)
        passes = self.retrieval_max_passes if candidates or older else 0
        for attempt in range(passes):
            first_preview = bool(candidates) and attempt == 0
            used_images = conversation_size(self._with_retrieved(messages, by_id, selected))['image_parts']
            preview_images = len(candidates) if first_preview else 0
            remaining = max(0, min(self.retrieval_max_images - len(selected), self.model_image_limit - used_images - preview_images))
            content = [{'type': 'text', 'text': 'MEMORY SELECTION ONLY. Return JSON {"keep": [candidate_frame_ids], "retrieve": [archived_frame_ids], "retrieve_text": [round_numbers], "more": false}. Full archived text can be fetched by round number; old messages are indexed in the context. Set more=true only if another selection is needed after seeing these results. Original text is available for ALL previous rounds, even when their images are still in the recent window. Request at most 12 original-text rounds total. Already fetched text rounds: ' + str(selected_text) + f". Do not select an action yet. Retain only key intermediate views; keep=[] is the default. Retrieve only needed archived views. At most {self.retrieval_max_images} retrieved images total; already retrieved: {selected}. Context contains {used_images} images plus {preview_images} preview candidates; all references, finals, retained intermediates, previews and retrievals share the {self.model_image_limit}-image limit. Request at most {remaining} NEW archive images this pass. keep selects already shown candidates and adds no new image slots. When capacity is zero use retrieve=[]. Only IDs listed in the archive catalogue are valid. Pass {attempt + 1}/{passes}.\nArchived frame catalogue:\n{catalogue or '(empty)'}"}]
            if first_preview:
                for f in candidates:
                    content.extend([{'type': 'text', 'text': f"Candidate {f['id']} from previous action; historical coordinates only."}, image_part(f['image'])])
            request = self._with_retrieved(messages, by_id, selected) + text_results + [{'role': 'user', 'content': content}]
            protected = [final] + ([f['image'] for f in candidates] if first_preview else [])
            request, gate = self._bounded(request, protected + [by_id[i]['image'] for i in selected] + ([] if first_preview else [f['image'] for f in self._retained]))
            before = self.client.prompt_tokens + self.client.completion_tokens
            raw = self._memory_call(request)
            event = {'pass': attempt + 1, 'raw': raw, 'requested': [], 'accepted': [], 'image_parts': conversation_size(request)['image_parts'], 'request_bytes': self.client.last_request_bytes, 'image_limit': gate, 'tokens': self.client.prompt_tokens + self.client.completion_tokens - before}
            self.last_retrieval['passes'].append(event)
            if not raw:
                self.save_archive()
                return ''
            if first_preview:
                preview_event['delivered_ids'] = list(candidate_by_id)
                self.images_sent += len(candidates)
                self.image_bytes += sum((len(f['image']) for f in candidates))
                if self.on_candidates_delivered:
                    self.on_candidates_delivered(candidates, round_i)
            try:
                match = re.search('\\{.*\\}', raw, re.S)
                choice = json.loads(match.group(0) if match else raw)
                if not isinstance(choice, dict):
                    choice = {}
            except (ValueError, TypeError):
                choice = {}
            if first_preview:
                keep = choice.get('keep', [])
                retained = list(dict.fromkeys((i for i in keep if isinstance(i, str) and i in candidate_by_id))) if isinstance(keep, list) else []
                self._retained = [candidate_by_id[i] for i in retained]
                preview_event.update(retained_ids=retained, selector=event)
                if retained:
                    prefix = [{'type': 'text', 'text': 'Retained intermediate memory:'}]
                    for i in retained:
                        prefix.extend([{'type': 'text', 'text': i}, image_part(candidate_by_id[i]['image'])])
                    current_message['content'] = prefix + current_message['content']
                messages = list(messages[:-1]) + [current_message]
            requested = choice.get('retrieve', [])
            new = list(dict.fromkeys((i for i in requested if isinstance(i, str) and i in by_id and (i not in selected)))) if isinstance(requested, list) else []
            new = new[:remaining]
            accepted = []
            for i in new:
                probe = self._with_retrieved(messages, by_id, selected + accepted + [i])
                try:
                    self._bounded(probe, [final] + [f['image'] for f in self._retained] + [by_id[j]['image'] for j in selected + accepted + [i]])
                    accepted.append(i)
                except RequestBodyTooLarge:
                    break
            selected.extend(accepted)
            event.update(requested=requested, accepted=accepted)
            text_requested, text_accepted, text_parts = self._text_retrieval(choice, round_i, selected_text)
            selected_text.extend(text_accepted)
            if text_parts:
                text_results.append({'role': 'user', 'content': text_parts})
            event.update(requested_text=text_requested, accepted_text=text_accepted, more=choice.get('more') is True)
            event.update(context_images=used_images, preview_images=preview_images, remaining_retrieval_images=remaining, total_image_limit=self.model_image_limit)
            self.save_archive()
            if not accepted and (not text_accepted) or choice.get('more') is not True:
                break
        current_message['content'].insert(0, {'type': 'text', 'text': f'Intermediate selection: shown={list(candidate_by_id)}, retained={retained}. CURRENT FINAL is the last image in this observation; use it for action pixels.'})
        messages = list(messages[:-1]) + [current_message]
        action_request = self._with_retrieved(messages, by_id, selected)
        action_request = list(action_request[:-1]) + text_results + list(action_request[-1:])
        action_request.append({'role': 'user', 'content': 'Memory selection is complete. Choose one action; include MEMORY, THOUGHT, ACTION. Use only the current final image for action coordinates.'})
        action_request, gate = self._bounded(action_request, [final] + [f['image'] for f in self._retained] + [by_id[i]['image'] for i in selected])
        self.last_retrieval.update(retrieved_frames=selected, retrieved_text_rounds=selected_text, retrieved_rounds=sorted(set((int(i.split('.')[0][1:]) for i in selected))), action_image_limit=gate, action_image_parts=conversation_size(action_request)['image_parts'], action_context=conversation_size(action_request))
        self.retrieval_events.append(self.last_retrieval)
        self.pending = []
        raw = self._memory_call(action_request)
        self.last_retrieval['action_request_bytes'] = self.client.last_request_bytes
        return raw

    @staticmethod
    def _with_retrieved(messages, by_id, selected):
        if not selected:
            return list(messages)
        content = [{'type': 'text', 'text': 'Retrieved archive: historical coordinates only.'}]
        for i in selected:
            content.extend([{'type': 'text', 'text': f"{i}: {by_id[i]['summary']}"}, image_part(by_id[i]['image'])])
        return list(messages[:-1]) + [{'role': 'user', 'content': content}] + list(messages[-1:])

    @staticmethod
    def _message_image(message):
        content = message.get('content')
        if isinstance(content, list):
            for p in reversed(content):
                if p.get('type') == 'image_url':
                    return p['image_url']['url']
        return None

    def ask(self, *args, **kwargs):
        row = super().ask(*args, **kwargs)
        self.archive[-1]['intermediates'] = list(self._retained)
        row['intermediate_selection'] = self.intermediate_events[-1]
        self.save_archive()
        return row

    def save_archive(self):
        if not self.archive_dir:
            return
        frames_by_round = {a['round']: a.pop('intermediates', []) for a in self.archive}
        try:
            super().save_archive()
        finally:
            for a in self.archive:
                a['intermediates'] = frames_by_round[a['round']]
        path = self.archive_dir / 'manifest.json'
        doc = json.loads(path.read_text())
        for entry in doc['rounds']:
            entry['intermediates'] = []
            original = next((a for a in self.archive if a['round'] == entry['round']))
            for f in original.get('intermediates', []):
                meta = self._persist_frame(f, self.archive_dir)
                entry['intermediates'].append(meta)
        doc.update(schema='episode-intermediate-memory-v1', selections=self.intermediate_events)
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps(doc, ensure_ascii=False, indent=2))
        os.replace(temp, path)
        audit = self.archive_dir / 'candidate_audit'
        audit.mkdir(exist_ok=True)
        for f in self.pending:
            meta = self._persist_frame(f, audit)
            target = audit / (f['id'] + '.json')
            target.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    @staticmethod
    def _persist_frame(frame, directory):
        meta = {k: v for k, v in frame.items() if k != 'image'}
        raw = base64.b64decode(frame['image'].split(',', 1)[1], validate=True)
        name = frame['id'] + '.jpg'
        temp = directory / (name + '.tmp')
        temp.write_bytes(raw)
        os.replace(temp, directory / name)
        meta.update(image_file=name, image_sha256=hashlib.sha256(raw).hexdigest())
        return meta
