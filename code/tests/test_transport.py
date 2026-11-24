"""Exercise request adaptation using local fake HTTP responses."""
from copy import deepcopy
import io
import json
import urllib.error
import pytest
from roam_bench.client import EvaluationClient
from roam_bench.protocol.interface_transport import endpoint, flattened, images_to_user

MESSAGES = [{'role': 'assistant', 'content': [{'type': 'text', 'text': 'prior'}, {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AAAA'}}, {'type': 'text', 'text': 'after'}]}, {'role': 'user', 'content': 'decide'}]

class Response:
    def __init__(self, text='ACTION: answer("B")'):
        self.text = text
    def __enter__(self):
        return self
    def __exit__(self, *values):
        return False
    def read(self):
        return json.dumps({'choices': [{'message': {'content': self.text}}], 'usage': {'prompt_tokens': 10, 'completion_tokens': 2}}).encode()

def test_image_role_mapping_preserves_order_and_text():
    source = deepcopy(MESSAGES)
    wire, event = images_to_user(source)
    assert [p for _, p in flattened(wire)] == [p for _, p in flattened(source)]
    assert source == MESSAGES
    assert event['moved_image_parts'] == 1
    assert [role for role, _ in flattened(wire)] == ['assistant', 'user', 'assistant', 'user']
    assert endpoint('https://example.invalid/v1') == 'https://example.invalid/v1/chat/completions'
    assert endpoint('https://example.invalid') == 'https://example.invalid/v1/chat/completions'

def test_native_first_and_explicit_role_refusal(monkeypatch):
    requests = []
    def open_request(request, **kwargs):
        body = json.loads(request.data)
        requests.append(body)
        if len(requests) == 1:
            raise urllib.error.HTTPError(request.full_url, 400, 'bad role', {}, io.BytesIO(b'Image input in assistant role is not allowed; only user role supports images.'))
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', open_request)
    client = EvaluationClient(api_key='offline-test', base_url='https://example.invalid/v1', model='offline', retries=2)
    assert client.complete(MESSAGES).endswith('("B")')
    assert requests[0]['messages'] == MESSAGES
    for body in requests:
        assert body['temperature'] == 0
        assert body['top_p'] == 1
        assert body['reasoning_effort'] == 'max'
    assert [p for _, p in flattened(requests[1]['messages'])] == [p for _, p in flattened(MESSAGES)]
    assert client.stats()['wire_image_policy'] == 'user_images'
    assert client.calls == 1

def test_refused_top_p_is_removed_and_completed_answer_not_resampled(monkeypatch):
    bodies = []
    def open_request(request, **kwargs):
        bodies.append(json.loads(request.data))
        if len(bodies) == 1:
            raise urllib.error.HTTPError(request.full_url, 400, 'bad field', {}, io.BytesIO(b'top_p is not supported; only the default is allowed'))
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', open_request)
    client = EvaluationClient(api_key='offline-test', model='offline', top_p=.7)
    assert client.complete(MESSAGES)
    assert len(bodies) == 2
    assert bodies[0]['top_p'] == .7
    assert 'top_p' not in bodies[1]

def test_empty_response_scales_token_cap(monkeypatch):
    bodies = []
    def open_request(request, **kwargs):
        bodies.append(json.loads(request.data))
        return Response('' if len(bodies) < 4 else 'ACTION: answer("B")')
    monkeypatch.setattr('urllib.request.urlopen', open_request)
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    client = EvaluationClient(api_key='offline-test', model='offline', retries=4)
    assert client.complete(MESSAGES)
    assert [body['max_tokens'] for body in bodies] == [32768, 65536, 65536, 65536]

def test_custom_output_budget_doubles_with_bounded_retries(monkeypatch):
    bodies = []
    def open_request(request, **kwargs):
        bodies.append(json.loads(request.data))
        return Response('' if len(bodies) < 5 else 'ACTION: answer("B")')
    monkeypatch.setattr('urllib.request.urlopen', open_request)
    monkeypatch.setattr('time.sleep', lambda seconds: None)
    client = EvaluationClient(api_key='offline-test', model='offline', max_tokens=8192, retries=5)
    assert client.complete(MESSAGES)
    assert [body['max_tokens'] for body in bodies] == [8192, 16384, 32768, 65536, 65536]

def test_explicit_larger_output_budget_is_preserved():
    client = EvaluationClient(api_key='offline-test', model='offline', max_tokens=131072)
    assert client._body()['max_tokens'] == 131072
    assert client._body(16)['max_tokens'] == 131072

def test_auth_failure_does_not_adapt(monkeypatch):
    def open_request(request, **kwargs):
        raise urllib.error.HTTPError(request.full_url, 401, 'unauthorized', {}, io.BytesIO(b'Invalid authentication'))
    monkeypatch.setattr('urllib.request.urlopen', open_request)
    client = EvaluationClient(api_key='offline-test', model='offline')
    with pytest.raises(RuntimeError, match='authentication'):
        client.complete(MESSAGES)
    assert not client.interface_adaptations

def test_endpoint_specific_maximum_reasoning_is_preserved(monkeypatch):
    bodies = []
    def open_request(request, **kwargs):
        bodies.append(json.loads(request.data))
        return Response()
    monkeypatch.setattr('urllib.request.urlopen', open_request)
    client = EvaluationClient(api_key='offline-test', model='offline', reasoning_effort='xhigh')
    assert client.complete(MESSAGES)
    assert len(bodies) == 1
    assert bodies[0]['reasoning_effort'] == 'xhigh'

def test_rejected_reasoning_does_not_fall_back_to_provider_default(monkeypatch):
    bodies = []
    def open_request(request, **kwargs):
        bodies.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, 400, 'bad field', {},
                                     io.BytesIO(b'reasoning_effort max is not supported'))
    monkeypatch.setattr('urllib.request.urlopen', open_request)
    client = EvaluationClient(api_key='offline-test', model='offline')
    with pytest.raises(RuntimeError, match='Permanent provider format'):
        client.complete(MESSAGES)
    assert len(bodies) == 1
    assert not client.dropped
