"""The openai_compatible chat provider for OpenAI-compatible endpoints
(e.g. vLLM), with chat_extra_body merged into every request (Qwen: thinking off). Request fields only: the messages and
every other argument of 2.0.0's call sites reach the endpoint unchanged. Never sends OPENAI_KEY. No network: the
OpenAI client is a fake."""
import pytest

import utils.utils as uu
from utils import llm_calls


class FakeOpenAI:
    made = []

    def __init__(self, **kwargs):
        self.init = kwargs
        self.sent = []
        outer = self

        class Completions:
            def create(self, *args, **kw):
                outer.sent.append(kw)
                msg = type('m', (), {'content': None})()
                return type('r', (), {'choices': [type('c', (), {'message': msg})()]})()
        self.chat = type('chat', (), {'completions': Completions()})()
        FakeOpenAI.made.append(self)


@pytest.fixture
def fake(monkeypatch):
    FakeOpenAI.made.clear()
    monkeypatch.setattr(uu, 'OpenAI', FakeOpenAI)
    monkeypatch.setenv('OPENAI_KEY', 'sk-must-not-be-sent')
    return FakeOpenAI


def cfg(**extra):
    base = {'openai_provider': 'openai_compatible', 'chat_endpoint': 'https://example.org/qwen/v1/',
            'chat_model': 'qwen', 'embeddings_provider': 'sentence-transformers',
            'chat_rate_limit': None}          # no rate gate file in these tests (tests/test_rate_limit.py)
    base.update(extra)
    return base


def test_endpoint_key_and_extra_body(fake):
    body = {'chat_template_kwargs': {'enable_thinking': False}}
    client = uu._build_chat_client('openai_compatible', cfg(chat_extra_body=body), 'sk-must-not-be-sent')
    made = fake.made[-1]
    assert made.init == {'base_url': 'https://example.org/qwen/v1/', 'api_key': 'EMPTY', 'max_retries': 0}
    msgs = [{'role': 'user', 'content': 'Hello!'}]
    resp = llm_calls.chat(client, 'startup_check', model='qwen', messages=msgs, temperature=0.7, max_tokens=80)
    sent = made.sent[-1]
    assert sent['messages'] is msgs and sent['temperature'] == 0.7 and sent['max_tokens'] == 80
    assert sent['extra_body'] == body and sent['timeout'] == 60.0
    assert resp.choices[0].message.content == ''           # 2.0.0's None -> '' still applies


def test_without_extra_body_the_request_is_2_0_0s(fake):
    client = uu._build_chat_client('openai_compatible', cfg(), None)
    client.chat.completions.create(model='gemma', messages=[])
    assert 'extra_body' not in fake.made[-1].sent[-1]


def test_key_from_chat_api_key_env(fake, monkeypatch):
    monkeypatch.setenv('MY_ENDPOINT_KEY', 'k-123')
    uu._build_chat_client('openai_compatible', cfg(chat_api_key_env='MY_ENDPOINT_KEY'), None)
    assert fake.made[-1].init['api_key'] == 'k-123'
    monkeypatch.delenv('MY_ENDPOINT_KEY')
    with pytest.raises(ValueError):
        uu._build_chat_client('openai_compatible', cfg(chat_api_key_env='MY_ENDPOINT_KEY'), None)


def test_extra_body_must_be_a_mapping(fake):
    with pytest.raises(ValueError):
        uu._build_chat_client('openai_compatible', cfg(chat_extra_body='thinking off'), None)


def test_get_openai_clients_never_passes_openai_key(fake, monkeypatch):
    monkeypatch.setattr(uu, '_build_embeddings_client', lambda provider, config, key: None)
    uu.get_openai_clients(cfg())
    assert fake.made[-1].init['api_key'] == 'EMPTY'
