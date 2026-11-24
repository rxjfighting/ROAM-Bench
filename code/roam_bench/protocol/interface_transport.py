"""Preserve image content and order when an endpoint requires user-role images."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
import uuid
POLICY = 'native-first-interface-transport-v1'
IMAGE_ROLES = ('system', 'assistant', 'developer')

def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def image_part(part):
    return isinstance(part, dict) and part.get('type') == 'image_url'

def flattened(messages):
    return [(message['role'], part) for message in messages for part in (message['content'] if isinstance(message.get('content'), list) else [message.get('content', '')])]

def images_to_user(messages):
    output = []
    mapping = []
    for index, message in enumerate(messages):
        content = message.get('content')
        if message.get('role') not in IMAGE_ROLES or not isinstance(content, list) or (not any((image_part(part) for part in content))):
            output.append(deepcopy(message))
            continue
        if any((message.get(field) for field in ('tool_calls', 'tool_call_id', 'function_call', 'refusal'))):
            raise ValueError('Cannot split an image message containing tool-call/refusal semantics')
        start = 0
        while start < len(content):
            is_image = image_part(content[start])
            end = start + 1
            while end < len(content) and image_part(content[end]) == is_image:
                end += 1
            chunk = deepcopy(message)
            chunk['content'] = deepcopy(content[start:end])
            if is_image:
                chunk['role'] = 'user'
                mapping.extend((dict(source_message=index, source_part=part, source_role=message['role'], output_message=len(output), output_part=part - start, destination_role='user') for part in range(start, end)))
            output.append(chunk)
            start = end
    before = flattened(messages)
    after = flattened(output)
    assert [part for _, part in before] == [part for _, part in after], 'Content/order changed'
    assert len(before) == len(after)
    for (role, part), (wire_role, _) in zip(before, after):
        assert wire_role == ('user' if image_part(part) and role in IMAGE_ROLES else role)
    if not mapping:
        return (messages, None)
    return (output, dict(policy=POLICY, source_messages_sha256=digest(messages), wire_messages_sha256=digest(output), ordered_content_sha256=digest([part for _, part in before]), content_order_preserved=True, text_roles_preserved=True, original_messages_mutated=False, image_parts=sum((image_part(part) for _, part in before)), moved_image_parts=len(mapping), mapping=mapping))

def role_refusal(code, detail):
    if code not in (400, 422):
        return False
    low = detail.lower()
    image = any((word in low for word in ('image', 'picture', '图片', '图像')))
    role = any((word in low for word in IMAGE_ROLES))
    refusal = any((word in low for word in ('not allowed', 'cannot', 'not support', 'unsupported', 'only', 'must', '不支持', '不能', '不允许', '只能')))
    return image and role and refusal

def unsupported_vision(code, detail):
    if code not in (400, 422):
        return False
    low = detail.lower()
    return any((word in low for word in ('image_url', 'image input', 'vision', '图片', '图像'))) and any((word in low for word in ('not supported', 'does not support', 'unsupported', '不支持'))) and (not role_refusal(code, detail))

def sampling_refusal(detail):
    low = detail.lower()
    return any((word in low for word in ('not supported', 'does not support', 'unsupported', 'not allowed', 'only the default', 'invalid parameter', 'unknown parameter', '不支持')))

def assistant_text(message):
    content = message.get('content')
    if content is None:
        return ''
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        if not all((isinstance(part, dict) and part.get('type') in ('text', 'output_text') and isinstance(part.get('text'), str) for part in content)):
            raise ValueError('Unsupported assistant content parts; no tool/reasoning coercion')
        return ''.join((part['text'] for part in content)).strip()
    raise ValueError('Unsupported assistant content schema')

def endpoint(base):
    return base + '/chat/completions' if base.rstrip('/').endswith('/v1') else base + '/v1/chat/completions'

def client_class(original, llm, capabilities=None, event_sink=None):
    capabilities = capabilities or {}

    class CompatibleChatClient(original):

        def __init__(self, *args, interface_policy=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.interface_policy = interface_policy or os.environ.get('ROAM_BENCH_INTERFACE_POLICY', 'auto')
            if self.interface_policy not in ('auto', 'native', 'user_images'):
                raise llm.ChatError('Interface policy must be auto, native or user_images')
            self.wire_image_policy = 'native'
            self.token_parameter = 'max_tokens'
            self.capacity_rejections = 0
            self.interface_adaptations = []
            self.last_interface_call = None
            self._transport_call = 0

        def _body(self, token_scale=1):
            body = super()._body(token_scale)
            if self.token_parameter != 'max_tokens' and 'max_tokens' in body:
                body[self.token_parameter] = body.pop('max_tokens')
            return body

        def _wire(self, messages):
            contains_images = llm._count_image_parts(messages) > 0
            profile = capabilities.get((self.base.rstrip('/'), self.model)) if self.interface_policy == 'auto' else None
            if contains_images and profile and (profile['vision'] == 'unsupported'):
                raise llm.ChatError('Vision unsupported by independently probed endpoint/model; no text-only fallback')
            reason = None
            if self.interface_policy == 'user_images':
                reason = 'explicit_user_images_policy'
            elif self.wire_image_policy == 'user_images':
                reason = 'prior_explicit_image_role_refusal_in_this_client'
            elif profile and profile.get('non_user_images') == 'failed':
                reason = 'independently_verified_same_endpoint_native_role_probe_failed'
            if reason:
                wire, event = images_to_user(messages)
                if event:
                    event.update(reason=reason, capability_proof=(profile or {}).get('proof'))
                return (wire, event)
            return (messages, None)

        def request_size(self, messages, token_scale=1):
            wire, _ = self._wire(messages)
            body = self._body(token_scale)
            body['messages'] = wire
            return len(json.dumps(body).encode('utf-8'))

        def _emit(self, event):
            self.last_interface_call = event
            if event_sink:
                event_sink(event)

        def complete(self, messages):
            self._transport_call += 1
            record = dict(policy=POLICY, call=self._transport_call, model=self.model, endpoint_sha256=hashlib.sha256(endpoint(self.base).encode()).hexdigest(), interface_policy=self.interface_policy, source_messages_sha256=digest(messages), attempts=[], images_never_deleted=True, model_or_endpoint_substituted=False)
            last = ''
            scale = 1
            transient = 0
            format_changes = 0
            self.image_parts += llm._count_image_parts(messages)
            try:
                while transient < self.retries and len(record['attempts']) < self.retries + 4:
                    wire, mapping = self._wire(messages)
                    body = self._body(scale)
                    body['messages'] = wire
                    guard = getattr(self, 'episode_token_guard', None)
                    if guard:
                        guard(wire, body)
                    payload_bytes = json.dumps(body).encode('utf-8')
                    self.last_request_bytes = len(payload_bytes)
                    self.largest_request_bytes = max(self.largest_request_bytes, len(payload_bytes))
                    self.request_bytes.append(len(payload_bytes))
                    if len(payload_bytes) > self.MAX_REQUEST_BYTES:
                        raise llm.RequestBodyTooLarge('Adapted request exceeds unchanged local body-byte limit')
                    attempt = dict(index=len(record['attempts']), request_sha256=hashlib.sha256(payload_bytes).hexdigest(), request_bytes=len(payload_bytes), image_parts=llm._count_image_parts(wire), image_role_mapping=mapping, output_token_parameter=self.token_parameter, dropped_sampling_params=list(self.dropped))
                    record['attempts'].append(attempt)
                    started = time.monotonic()
                    request = urllib.request.Request(endpoint(self.base), data=payload_bytes, headers={'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
                    try:
                        with urllib.request.urlopen(request, timeout=self.timeout) as response:
                            payload = json.loads(response.read().decode('utf-8'))
                        attempt.update(http_status=200, seconds=time.monotonic() - started)
                        self.calls += 1
                        usage = payload.get('usage') or {}
                        self.prompt_tokens += int(usage.get('prompt_tokens') or 0)
                        self.completion_tokens += int(usage.get('completion_tokens') or 0)
                        attempt['usage'] = {key: usage.get(key) for key in ('prompt_tokens', 'completion_tokens', 'total_tokens')}
                        choices = payload.get('choices') or []
                        if not choices:
                            last = 'empty choices'
                            transient += 1
                            continue
                        text = assistant_text(choices[0].get('message') or {})
                        if text:
                            record['accepted_attempt'] = attempt['index']
                            record['success'] = True
                            return text
                        last = 'empty assistant content'
                        scale = min(scale * 2, self.MAX_TOKEN_SCALE)
                        transient += 1
                        time.sleep(0.5)
                    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
                        code = getattr(error, 'code', None)
                        detail = ''
                        if isinstance(error, urllib.error.HTTPError):
                            detail = error.read(4096).decode('utf-8', 'replace')
                        attempt.update(http_status=code, seconds=time.monotonic() - started)
                        last = 'HTTP ' + str(code) if code else type(error).__name__
                        if code in (429, 503):
                            self.capacity_rejections += 1
                            if code == 429:
                                self.throttled += 1
                            attempt['classification'] = 'capacity_rejection_no_format_fallback'
                        elif role_refusal(code, detail):
                            attempt['classification'] = 'explicit_image_role_refusal'
                            if self.interface_policy == 'native' or self.wire_image_policy == 'user_images':
                                raise llm.ChatError('Provider rejects image role; native policy or adaptation already exhausted')
                            _, change = images_to_user(messages)
                            if not change:
                                raise llm.ChatError('Image role refusal cannot be repaired without changing non-image semantics')
                            self.wire_image_policy = 'user_images'
                            format_changes += 1
                            self.interface_adaptations.append(dict(kind='images_to_user', trigger='explicit_HTTP_role_refusal'))
                            continue
                        elif unsupported_vision(code, detail):
                            attempt['classification'] = 'vision_unsupported_no_text_only_fallback'
                            raise llm.ChatError('Provider explicitly rejects image input; no text-only fallback')
                        elif code in (400, 422):
                            low = detail.lower()
                            if 'contentpolicyviolation' in low:
                                self._dump_policy_images(wire)
                                raise llm.ChatError('Provider content-policy rejection; no format fallback')
                            if 'max_tokens' in low and 'max_completion_tokens' in low and sampling_refusal(detail) and (self.token_parameter == 'max_tokens') and (self.interface_policy != 'native'):
                                self.token_parameter = 'max_completion_tokens'
                                format_changes += 1
                                attempt['classification'] = 'output_token_parameter_alias'
                                self.interface_adaptations.append(dict(kind='max_completion_tokens', trigger='explicit_HTTP_parameter_alias'))
                                continue
                            bad = self._droppable_from(detail)
                            if bad and sampling_refusal(detail) and (self.interface_policy != 'native') and (format_changes < 4):
                                self.dropped.append(bad)
                                format_changes += 1
                                attempt['classification'] = 'refused_sampling_parameter'
                                continue
                            raise llm.ChatError('Permanent provider format/policy rejection; no speculative fallback')
                        elif code in (401, 403, 404):
                            attempt['classification'] = 'auth_or_route_rejection_no_fallback'
                            raise llm.ChatError('Provider authentication/route rejection; no credential, endpoint or model substitution')
                        transient += 1
                        if transient < self.retries:
                            time.sleep(self._backoff(transient - 1, code, error))
                self.failures.append(last)
                record.update(success=False, terminal_classification=last)
                return ''
            except Exception as error:
                record.update(success=False, terminal_exception=type(error).__name__)
                raise
            finally:
                self._emit(record)

        def stats(self):
            return dict(super().stats(), interface_policy=self.interface_policy, interface_adaptations=list(self.interface_adaptations), capacity_rejections_429_503=self.capacity_rejections, wire_image_policy=self.wire_image_policy, output_token_parameter=self.token_parameter)
    return CompatibleChatClient

def journal_sink(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def save(event):
        path = directory / ('transport_' + str(os.getpid()) + '_' + str(event['call']).zfill(6) + '_' + uuid.uuid4().hex + '.json')
        with path.open('x') as handle:
            json.dump(event, handle, ensure_ascii=False, indent=2)
            handle.write('\n')
    return save

def install(llm, capabilities=None, journal_directory=None):
    if getattr(llm, '_native_first_interface_transport_installed', False):
        raise ValueError('Interface transport already installed; prevent stacked wrappers')
    llm.ChatClient = client_class(llm.ChatClient, llm, capabilities, journal_sink(journal_directory) if journal_directory else None)
    original_ask = llm.LLMAgent.ask

    def ask(self, *args, **kwargs):
        row = original_ask(self, *args, **kwargs)
        row['interface_transport'] = deepcopy(getattr(self.client, 'last_interface_call', None))
        return row
    llm.LLMAgent.ask = ask
    llm._native_first_interface_transport_installed = True
    return dict(policy=POLICY, frozen_source_changed=False, native_first=True, image_deletion_or_text_only_fallback=False, capability_profiles=len(capabilities or {}))
