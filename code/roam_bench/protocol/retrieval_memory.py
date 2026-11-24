"""Retrieval memory utilities for ROAM-Bench."""
import json
import re
import base64
import hashlib
import os
from pathlib import Path
from .llm import LLMAgent, bounded_image_messages, text_of, conversation_size

class RetrievalAgent(LLMAgent):

    def __init__(self, *args, retrieval_max_passes=2, retrieval_max_images=12, archive_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.archive = []
        self.archive_dir = Path(archive_dir) if archive_dir else None
        self.retrieval_events = []
        self.last_retrieval = None
        self.retrieval_max_passes = int(retrieval_max_passes)
        self.retrieval_max_images = int(retrieval_max_images)
        note = "\nVisual memory: only recent observation images stay in context. Before actions, you can retrieve earlier round images and verbatim text by round number. Older context contains an indexed summary of every round; full original messages remain archived and retrievable. Every action reply must include MEMORY: followed by at most 100 words: visible objects and distinctive features, room/door landmarks, orientation or distances supported by observations, counting/de-duplication evidence and unresolved uncertainty. Distinguish observed facts from hypotheses. The action below has NOT run yet; summarize its outcome only after receiving the next round's execution receipt. Then use the normal THOUGHT/ACTION format. Historical images are memory; choose action pixels only from the latest current image."
        self.system += note
        msg = self.messages[0]
        if isinstance(msg['content'], list):
            msg['content'][-1]['text'] += note
        else:
            msg['content'] += note

    def ask(self, view_text, receipt, round_i, spent, image=None, image_variants=(), trajectory_image=None, trajectory_meta=None):
        row = super().ask(view_text, receipt, round_i, spent, image, image_variants, trajectory_image=trajectory_image, trajectory_meta=trajectory_meta)
        image = self._message_image(self.messages[-2])
        raw = row.get('raw') or ''
        match = re.search('(?ims)^MEMORY\\s*:\\s*(.*?)(?=^THOUGHT\\s*:|^ACTION\\s*:|\\Z)', raw)
        summary = match.group(1).strip() if match else row.get('thought') or raw
        summary = ' '.join(summary.split()[:100])[:800]
        self.archive.append({'round': round_i, 'summary': summary, 'image': image, 'prompt': row['prompt'], 'reply': raw, 'context': row['context'], 'retrieval': row.get('memory_retrieval'), 'episode_tokens_so_far': self.client.prompt_tokens + self.client.completion_tokens - getattr(self, '_episode_token_base', 0)})
        self.save_archive()
        return row

    def _indexed_messages(self, messages):
        if not self.archive:
            return list(messages)
        latest = max((a['round'] for a in self.archive)) + 1
        older = {a['round']: a for a in self.archive if a['round'] <= latest - self.vision_keep}
        if not older:
            return list(messages)
        index = []
        for number, entry in sorted(older.items()):
            raw = entry.get('reply', '')
            action = re.search('(?im)^ACTION\\s*:\\s*(.*)', raw)
            following = next((a for a in self.archive if a['round'] == number + 1), None)
            receipt = ''
            if following:
                found = re.search('Result of your last action:\\n(.*?)(?=\\n\\nObservation:|\\n\\nWhat is your next action\\?|\\Z)', following.get('prompt', ''), re.S)
                receipt = found.group(1).strip() if found else ''
            frame_ids = ', '.join((f['id'] for f in entry.get('intermediates', [])))
            index.append(f"R{number}: {entry['summary']}\nRetained intermediate IDs: {frame_ids or '(none)'}\nChosen action: {(action.group(1) if action else '(no parsed action)')}\nExecution receipt: {receipt or '(not yet available in archive)'}")
        output = [messages[0], {'role': 'user', 'content': 'INDEXED HISTORY. Every older round is listed below. Summaries are uncertain model notes; execution receipts are original public feedback. Request retrieve_text=[round_numbers] for exact original observation/reply, and retrieve=[round_numbers] for images. Do not treat repeated sightings as new objects without distinguishing evidence.\n' + '\n'.join(index)}]
        previous_round = None
        for msg in messages[1:]:
            match = re.search('(?m)^Round (\\d+)\\.', text_of(msg.get('content', '')))
            if match:
                previous_round = int(match.group(1))
            if previous_round not in older:
                output.append(msg)
        return output

    def _request_messages(self):
        request, event = bounded_image_messages(self._indexed_messages(self.messages), self.model_image_limit, self.client)
        if event:
            self.image_limit_events.append(event)
        return (request, None)

    def _text_retrieval(self, choice, round_i, selected):
        requested = choice.get('retrieve_text', [])
        available = {a['round']: a for a in self.archive if a['round'] < round_i}
        accepted = list(dict.fromkeys((n for n in requested if isinstance(n, int) and (not isinstance(n, bool)) and (n in available) and (n not in selected)))) if isinstance(requested, list) else []
        accepted = accepted[:max(0, 12 - len(selected))]
        parts = [{'type': 'text', 'text': f"Original archived text R{n} (historical; not current):\nUSER:\n{available[n]['prompt']}\nASSISTANT:\n{available[n]['reply']}"} for n in accepted]
        return (requested, accepted, parts)

    def save_archive(self):
        if not self.archive_dir:
            return
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        entries = []
        for original in self.archive:
            entry = dict(original)
            url = entry.pop('image', None)
            if url:
                head, data = url.split(',', 1)
                image_bytes = base64.b64decode(data, validate=True)
                name = f"round_{entry['round']:03d}." + ('jpg' if 'jpeg' in head else 'png')
                target = self.archive_dir / name
                if not target.exists() or target.read_bytes() != image_bytes:
                    temp = target.with_suffix(target.suffix + '.tmp')
                    temp.write_bytes(image_bytes)
                    os.replace(temp, target)
                entry.update(image_file=name, image_sha256=hashlib.sha256(image_bytes).hexdigest())
            entries.append(entry)
        path = self.archive_dir / 'manifest.json'
        temp = path.with_suffix('.json.tmp')
        temp.write_text(json.dumps({'schema': 'episode-visual-memory-v1', 'rounds': entries}, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(temp, path)

    def _complete_action_request(self, messages, round_i):
        older = [a for a in self.archive if a['round'] <= round_i - self.vision_keep]
        self.last_retrieval = {'round': round_i, 'passes': [], 'retrieved_rounds': []}
        if not older:
            return self._memory_call(messages)
        catalogue = ', '.join((f"R{a['round']}" for a in older))
        by_round = {a['round']: a for a in older}
        selected = []
        results = []
        selected_text = []
        for attempt in range(self.retrieval_max_passes):
            used_images = conversation_size(list(messages[:-1]) + results + list(messages[-1:]))['image_parts']
            remaining = max(0, self.retrieval_max_images - len(selected))
            if self.model_image_limit:
                remaining = min(remaining, max(0, self.model_image_limit - used_images))
            request = list(messages[:-1]) + results + list(messages[-1:]) + [{'role': 'user', 'content': f"""MEMORY RETRIEVAL ONLY. Review the current task, all round text and recent images. Choose archived rounds needed before deciding the next action. Return only JSON {{"retrieve": [round_numbers], "retrieve_text": [round_numbers], "more": false}}. Original text is indexed for ALL previous rounds and can be fetched without images. Inspect the index for evidence needed for the task: distinctive objects/rooms, orientation, scale, and possible repeat sightings. Set more=true ONLY when you need a second selection after seeing these retrieval results; otherwise the next call selects an action directly. At most {self.retrieval_max_images} images total. Already retrieved: {selected}. This call already has {used_images} context images (references, recent and current observations, and earlier retrievals). The total image limit is {self.model_image_limit or 'unlimited'}; request at most {remaining} NEW archived images in this pass. If remaining capacity is zero return retrieve=[]. Recent images are already visible; only catalogue rounds <= {round_i - self.vision_keep} are retrievable. This is retrieval pass {attempt + 1}/{self.retrieval_max_passes}.\nValid archived image IDs (summaries appear in indexed history):\n""" + catalogue}]
            request, limit_event = bounded_image_messages(request, self.model_image_limit, self.client)
            before = (self.client.prompt_tokens, self.client.completion_tokens)
            raw = self._memory_call(request)
            if not raw:
                return ''
            try:
                match = re.search('\\{.*\\}', raw, re.S)
                choice = json.loads(match.group(0) if match else raw)
                choice = choice if isinstance(choice, dict) else {}
                requested = choice.get('retrieve', [])
            except (ValueError, AttributeError):
                choice = {}
                requested = []
            new = []
            for number in requested if isinstance(requested, list) else []:
                if isinstance(number, int) and (not isinstance(number, bool)) and (number in by_round):
                    if number not in selected and number not in new:
                        new.append(number)
            new = new[:remaining]
            event = {'pass': attempt + 1, 'requested': requested, 'accepted': new, 'raw': raw, 'image_parts': conversation_size(request)['image_parts'], 'request_bytes': self.client.last_request_bytes, 'prompt_tokens': self.client.prompt_tokens - before[0], 'completion_tokens': self.client.completion_tokens - before[1], 'image_limit': limit_event}
            event.update(context_images=used_images, remaining_retrieval_images=remaining, total_image_limit=self.model_image_limit)
            self.last_retrieval['passes'].append(event)
            text_requested, text_accepted, text_parts = self._text_retrieval(choice, round_i, selected_text)
            event.update(requested_text=text_requested, accepted_text=text_accepted, more=choice.get('more') is True)
            selected_text.extend(text_accepted)
            if not new and (not text_accepted):
                break
            selected.extend(new)
            content = [{'type': 'text', 'text': 'Retrieved historical memory; use latest image for action coordinates.'}]
            for number in new:
                a = by_round[number]
                content.append({'type': 'text', 'text': f"Archived round {number}: {a['summary']}"})
                if a.get('image'):
                    content.append({'type': 'image_url', 'image_url': {'url': a['image']}})
            content.extend(text_parts)
            results.append({'role': 'user', 'content': content})
            if choice.get('more') is not True:
                break
        action_request = list(messages[:-1]) + results + list(messages[-1:])
        action_request.append({'role': 'user', 'content': 'Memory retrieval is complete for this round. Now select exactly one action using the latest current observation; include your MEMORY summary.'})
        action_request, event = bounded_image_messages(action_request, self.model_image_limit, self.client)
        self.last_retrieval.update({'retrieved_rounds': selected, 'retrieved_text_rounds': selected_text, 'action_image_limit': event, 'action_image_parts': conversation_size(action_request)['image_parts'], 'action_context': conversation_size(action_request)})
        self.retrieval_events.append(self.last_retrieval)
        raw = self._memory_call(action_request)
        self.last_retrieval['action_request_bytes'] = self.client.last_request_bytes
        return raw
