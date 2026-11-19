"""The passive call log of the chat client (utils/llm_log.py, wired in utils.utils._wrap_chat_client
/ _build_chat_client by llm_log_file). With the log on or off, the requests that reach the service are identical (same
kwargs, same order), the replies the call sites get are identical, the global random state is untouched; one line per
returned request, each line parses, and carries the site, the conversation's user / item and the request as passed.
No network: the OpenAI client is a fake."""
import copy
import json
import random
import types

import pytest

import utils.llm_log as ll
import utils.utils as uu
from utils import llm_calls


def fake_client(replies, sent):
    """A fake client whose create() records a deep copy of what it is sent and returns the replies in turn."""
    it = iter(replies)

    def create(*args, **kwargs):
        sent.append(copy.deepcopy(kwargs))
        text, reason = next(it)
        msg = types.SimpleNamespace(content=text, reasoning_content=None, reasoning=None)
        usage = types.SimpleNamespace(prompt_tokens=7, completion_tokens=3, total_tokens=10)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=reason)], usage=usage)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create)))


REPLIES = [('Hello there.', 'stop'), ('<think>plan</think>A dark thriller.', 'stop'), ('{"value": "1994"}', 'stop'),
           ('Sure, try it!', 'length')]


def calls(client, history):
    """A fixed sequence of call sites, inside and outside a conversation; history is mutated after the calls, as the
    simulator appends to chat_history."""
    out = [llm_calls.chat(client, 'startup_check', model='m', messages=[{'role': 'user', 'content': 'Hello!'}],
                          max_tokens=50).choices[0].message.content]

    @llm_calls.conversation
    def conversation(config, user_id, target_item_id):
        r = []
        r.append(llm_calls.chat(client, 'generate_chit_chat', model='m', messages=history, temperature=1.2,
                                max_tokens=60).choices[0].message.content)
        r.append(llm_calls.chat(client, 'extract_attribute_value', model='m',
                                messages=[{'role': 'user', 'content': 'year?'}], temperature=0,
                                timeout=60).choices[0].message.content)
        r.append(llm_calls.chat(client, 'rephrase_phrase', model='m', messages=history, temperature=0.7,
                                max_tokens=30).choices[0].message.content)
        history.append({'role': 'user', 'content': 'appended after the calls'})
        return r
    out += conversation({'generation_strategy': 'user_item_pairs'}, 'u1', 'tt0000001')
    return out


def run(tmp_path, log_on):
    sent = []
    log = ll.CallLog(str(tmp_path / 'calls.jsonl'), 'adaptive') if log_on else None
    client = uu._wrap_chat_client(fake_client(REPLIES, sent), extra_body={'chat_template_kwargs': {'x': False}},
                                  strip_thinking=True, call_log=log)
    random.seed(5)
    before = random.getstate()
    got = calls(client, [{'role': 'user', 'content': 'I want a thriller.'}])
    assert random.getstate() == before
    return sent, got


def test_requests_and_replies_are_identical_with_and_without_the_log(tmp_path):
    sent_off, got_off = run(tmp_path / 'off', False)
    sent_on, got_on = run(tmp_path / 'on', True)
    assert sent_on == sent_off
    assert got_on == got_off
    assert not (tmp_path / 'off' / 'calls.jsonl').exists()


def test_one_parsable_line_per_returned_request(tmp_path):
    sent, got = run(tmp_path, True)
    lines = [json.loads(x) for x in open(tmp_path / 'calls.jsonl', encoding='utf-8')]
    assert len(lines) == len(sent) == 4
    assert [x['site'] for x in lines] == ['startup_check', 'generate_chit_chat', 'extract_attribute_value',
                                          'rephrase_phrase']
    assert [x['scope'] for x in lines] == ['run'] + ['user u1 item tt0000001 mode adaptive#0'] * 3
    assert lines[0]['user_id'] is None and lines[1]['user_id'] == 'u1' and lines[1]['item_id'] == 'tt0000001'
    # the request as the call site passed it (no timeout / extra_body added by the wrapper), messages at call time
    assert lines[1]['request'] == {'model': 'm', 'messages': [{'role': 'user', 'content': 'I want a thriller.'}],
                                   'temperature': 1.2, 'max_tokens': 60}
    assert 'max_tokens' not in lines[2]['request'] and lines[2]['request']['temperature'] == 0
    # the reply text the call site got (thinking block removed), finish reason and usage
    assert [x['response']['choices'][0]['message']['content'] for x in lines] == got
    assert lines[1]['response']['choices'][0]['message']['content'] == 'A dark thriller.'
    assert lines[3]['response']['choices'][0]['finish_reason'] == 'length'
    assert lines[0]['response']['usage']['total_tokens'] == 10
    assert all(isinstance(x['thread'], int) and x['ts'] > 0 and x['mode'] == 'adaptive' for x in lines)


def test_a_failed_request_is_not_logged_and_a_write_error_does_not_affect_the_call(tmp_path):
    def create(*args, **kwargs):
        raise RuntimeError('service down')
    client = uu._wrap_chat_client(types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create))), call_log=ll.CallLog(str(tmp_path / 'a.jsonl')))
    with pytest.raises(RuntimeError):
        client.chat.completions.create(model='m', messages=[])
    assert not (tmp_path / 'a.jsonl').exists()
    sent = []
    log = ll.CallLog(str(tmp_path / 'b.jsonl'))
    log.path = str(tmp_path)                      # a directory: the write fails
    client = uu._wrap_chat_client(fake_client(REPLIES, sent), call_log=log)
    assert client.chat.completions.create(model='m', messages=[]).choices[0].message.content == 'Hello there.'
    assert log.errors == 1 and log.lines == 0


def test_from_config_and_build_chat_client(tmp_path, monkeypatch):
    assert ll.from_config({}) is None
    assert ll.from_config({'llm_log_file': None}) is None
    c = ll.from_config({'llm_log_file': str(tmp_path / 'x' / 'c.jsonl'), 'mode_refinement': 'free'})
    assert c.mode == 'free' and (tmp_path / 'x').is_dir()
    seen = {}

    def fake_wrap(client, **kw):
        seen.update(kw)
        return client
    monkeypatch.setattr(uu, '_wrap_chat_client', fake_wrap)
    cfg = {'chat_endpoint': 'http://127.0.0.1:9/v1/', 'chat_rate_limit': None, 'mode_refinement': 'static'}
    uu._build_chat_client('openai_compatible', cfg, None)
    assert seen['call_log'] is None
    uu._build_chat_client('openai_compatible', {**cfg, 'llm_log_file': str(tmp_path / 'c.jsonl')}, None)
    assert isinstance(seen['call_log'], ll.CallLog) and seen['call_log'].mode == 'static'
