"""Waiting out an endpoint's rate limits for openai_compatible (utils/rate_limit.py, wired in
utils.utils._wrap_chat_client / _build_chat_client). A 429, 5xx, timeout or connection error is retried with
exponential backoff (Retry-After followed, per-wait and total caps); the jitter never touches the global `random`
state; a call that succeeds after retries returns the same text; retries are counted; the rate gate holds across
processes. No network: the OpenAI client is a fake, waits are recorded instead of slept (except the gate test)."""
import json
import multiprocessing
import os
import random
import time
import types

import openai
import pytest

import utils.rate_limit as rl
import utils.utils as uu
from utils import llm_calls

REQ = types.SimpleNamespace(method='POST', url='https://example.org/v1/chat/completions')


def response(status, headers=None):
    """What openai's APIStatusError reads from a response (request, status_code, headers): no HTTP library needed."""
    return types.SimpleNamespace(request=REQ, status_code=status, headers=dict(headers or {}))


def rate_limited(retry_after=None):
    headers = {'retry-after': str(retry_after)} if retry_after is not None else {}
    return openai.RateLimitError('Error code: 429', response=response(429, headers), body=None)


def server_error(code=503):
    return openai.InternalServerError(f'Error code: {code}', response=response(code), body=None)


def bad_request():
    return openai.BadRequestError('Error code: 400', response=response(400), body=None)


def ok(content):
    msg = types.SimpleNamespace(content=content, reasoning_content=None, reasoning=None)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=None)


def scripted(outcomes):
    """A fake client whose create() raises or returns the given outcomes in turn."""
    it = iter(outcomes)
    sent = []

    def create(*args, **kwargs):
        sent.append(kwargs)
        x = next(it)
        if isinstance(x, Exception):
            raise x
        return ok(x)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))), sent


def policy(waits, **kw):
    return rl.RetryPolicy(sleep=waits.append, rng=random.Random(0), **kw)


def test_429_5xx_timeout_and_connection_errors_are_retried_then_the_same_text_returned():
    errors = [rate_limited(), server_error(503), openai.APITimeoutError(request=REQ),
              openai.APIConnectionError(request=REQ)]
    waits, notes = [], []
    client, sent = scripted(errors + ['<think>x</think>How about "Heat"?'])
    pol = rl.RetryPolicy(sleep=waits.append, rng=random.Random(0), note=notes.append)
    wrapped = uu._wrap_chat_client(client, strip_thinking=True, retry=pol)
    reply = wrapped.chat.completions.create(model='m', messages=[{'role': 'user', 'content': 'hi'}], temperature=0.7)
    assert reply.choices[0].message.content == 'How about "Heat"?'
    plain, _ = scripted(['<think>x</think>How about "Heat"?'])
    assert uu._wrap_chat_client(plain, strip_thinking=True).chat.completions.create(
        model='m', messages=[]).choices[0].message.content == reply.choices[0].message.content
    assert len(sent) == 5 and all(s == sent[0] for s in sent)            # the same request every time
    assert notes == [429, 503, 0, 0]
    for w, b in zip(waits, [2, 4, 8, 16]):
        assert 0.9 * b <= w <= 1.1 * b                                   # doubling from 2 s, +-10% jitter


def test_retry_after_is_followed_within_the_per_wait_cap():
    waits = []
    pol = policy(waits)
    client, _ = scripted([rate_limited(7), rate_limited(900), 'fine'])
    uu._wrap_chat_client(client, retry=pol).chat.completions.create(model='m', messages=[])
    assert waits == [7.0, 300.0]


def test_per_wait_cap_and_total_cap():
    waits = []
    pol = policy(waits, base=100, max_wait=300, total=1000)
    client, sent = scripted([rate_limited()] * 10)
    with pytest.raises(openai.RateLimitError):
        uu._wrap_chat_client(client, retry=pol).chat.completions.create(model='m', messages=[])
    assert max(waits) <= 300 and sum(waits) <= 1000
    assert len(sent) == len(waits) + 1                                    # the last error is raised, no extra wait
    assert waits[0] == pytest.approx(100 * random.Random(0).uniform(0.9, 1.1))


def test_other_errors_are_not_retried():
    waits = []
    client, sent = scripted([bad_request(), 'x'])
    with pytest.raises(openai.BadRequestError):
        uu._wrap_chat_client(client, retry=policy(waits)).chat.completions.create(model='m', messages=[])
    assert waits == [] and len(sent) == 1


def test_backoff_never_touches_the_global_random_state():
    random.seed(2025)
    before = random.getstate()
    client, _ = scripted([rate_limited(), server_error(), rate_limited(), 'ok'])
    uu._wrap_chat_client(client, retry=rl.RetryPolicy(sleep=lambda s: None)).chat.completions.create(model='m',
                                                                                                    messages=[])
    assert random.getstate() == before


def test_retries_are_counted_in_the_run_summary_record():
    rec = llm_calls.Record()
    llm_calls._local.record = rec
    try:
        client, _ = scripted([rate_limited(), server_error(502), 'ok'])
        pol = rl.RetryPolicy(sleep=lambda s: None, note=llm_calls.note_retry)
        llm_calls.chat(uu._wrap_chat_client(client, retry=pol), 'generate_chit_chat', model='m', messages=[])
    finally:
        llm_calls._local.record = None
    assert rec.retries == {('generate_chit_chat', 429): 1, ('generate_chit_chat', 502): 1}


def test_build_chat_client_wires_retry_and_gate(monkeypatch, tmp_path):
    made = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            made.update(kw)
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **k: ok('x')))
    monkeypatch.setattr(uu, 'OpenAI', FakeOpenAI)
    window = tmp_path / 'w.json'
    cfg = {'openai_provider': 'openai_compatible', 'chat_endpoint': 'https://example.org/v1/', 'chat_model': 'm',
           'chat_rate_limit': '3/60', 'chat_rate_window_file': str(window)}
    client = uu._build_chat_client('openai_compatible', cfg, None)
    assert made['max_retries'] == 0
    client.chat.completions.create(model='m', messages=[])
    assert len(json.load(open(window))['stamps']) == 1
    assert rl.from_config({})[1].max_req == 70 and rl.from_config({})[1].window == 60.0        # the default gate
    assert rl.from_config({'chat_rate_limit': None})[1] is None                                 # turned off
    pol = rl.from_config({})[0]
    assert (pol.base, pol.max_wait, pol.total) == (2.0, 300.0, 1800.0)


def _gate_worker(path, n, out):
    gate = rl.SharedWindow(path, 5, 1.0)
    stamps = []
    for _ in range(n):
        gate.acquire()
        stamps.append(time.time())
    with open(out, 'w') as f:
        json.dump(stamps, f)


def test_rate_gate_holds_across_processes(tmp_path):
    path = str(tmp_path / 'window.json')
    outs = [str(tmp_path / f'out{k}.json') for k in range(3)]
    ctx = multiprocessing.get_context('fork')
    procs = [ctx.Process(target=_gate_worker, args=(path, 4, o)) for o in outs]
    for p in procs:
        p.start()
    for p in procs:
        p.join(30)
        assert p.exitcode == 0
    stamps = sorted(t for o in outs for t in json.load(open(o)))
    assert len(stamps) == 12
    # never more than 5 grants within any 1-second window (small slack for clock reads outside the lock)
    for i, t in enumerate(stamps):
        assert sum(1 for u in stamps[i:] if u - t < 0.95) <= 5
    assert stamps[-1] - stamps[0] >= 1.9                                  # 12 grants at 5/s need at least two waits


def test_parse_rate():
    assert rl.parse_rate('70/60') == (70, 60.0)
    assert rl.parse_rate('0') is None and rl.parse_rate(None) is None and rl.parse_rate('none') is None
    with pytest.raises(ValueError):
        rl.parse_rate('0/60x')
