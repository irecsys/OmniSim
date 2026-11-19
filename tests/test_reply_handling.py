"""Reply handling of the openai_compatible provider: a leading <think>...</think> block is removed, a reasoning field
is ignored (the reply is message.content), an unclosed <think> or nothing after </think> is asked once more and then
fails, and thinking at the startup check is a configuration error. No network: the client is a fake."""
import types

import pytest

import utils.utils as uu
from utils import llm_calls


def response(content, reasoning_content=None, reasoning=None, reasoning_tokens=0):
    msg = types.SimpleNamespace(content=content, reasoning_content=reasoning_content, reasoning=reasoning)
    usage = types.SimpleNamespace(completion_tokens_details=types.SimpleNamespace(reasoning_tokens=reasoning_tokens))
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)], usage=usage)


def fake_client(replies):
    """A client whose create() returns the given replies in turn (a fresh object each time)."""
    it = iter(replies)
    calls = []

    def create(*args, **kwargs):
        calls.append(kwargs)
        return response(**next(it))
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))), calls


CASES = {
    'plain': [{'content': 'How about a crime film?'}],
    'think block': [{'content': '<think>The user wants crime.</think>\n\nHow about a crime film?'}],
    'empty think block': [{'content': '<think></think>How about a crime film?'}],
    'reasoning_content field': [{'content': 'How about a crime film?', 'reasoning_content': 'The user wants...'}],
    'reasoning field and tokens': [{'content': 'Sure.', 'reasoning': 'r', 'reasoning_tokens': 12}],
    'None content': [{'content': None}],
    'closing tag only': [{'content': 'planning the answer</think>\nHow about a crime film?'}],
    'unclosed, then fine': [{'content': '<think>the user wants'}, {'content': 'How about a crime film?'}],
    'unclosed twice': [{'content': '<think>the user'}, {'content': '<think>still thinking'}],
    'nothing after </think>, then fine': [{'content': 'plan</think>  '}, {'content': '<think>x</think>Okay.'}],
    'nothing after </think> twice': [{'content': 'plan</think>'}, {'content': 'more</think>\n'}],
}


def outcome(wrapped):
    try:
        return ('text', wrapped.chat.completions.create(model='m', messages=[]).choices[0].message.content)
    except Exception as e:
        return ('error', type(e).__name__)


# per case: the outcome the call site gets, and the number of requests sent (one retry at most)
EXPECTED = {
    'plain': (('text', 'How about a crime film?'), 1),
    'think block': (('text', 'How about a crime film?'), 1),
    'empty think block': (('text', 'How about a crime film?'), 1),
    'reasoning_content field': (('text', 'How about a crime film?'), 1),
    'reasoning field and tokens': (('text', 'Sure.'), 1),
    'None content': (('text', ''), 1),
    'closing tag only': (('text', 'How about a crime film?'), 1),
    'unclosed, then fine': (('text', 'How about a crime film?'), 2),
    'unclosed twice': (('error', 'ThinkingTruncated'), 2),
    'nothing after </think>, then fine': (('text', 'Okay.'), 2),
    'nothing after </think> twice': (('error', 'NoReplyAfterThinking'), 2),
}


@pytest.mark.parametrize('case', sorted(CASES))
def test_reply_text_and_number_of_requests(case):
    client, calls = fake_client(CASES[case])
    assert outcome(uu._wrap_chat_client(client, strip_thinking=True)) == EXPECTED[case][0]
    assert len(calls) == EXPECTED[case][1]
    # the requests carry the call site's arguments and the 60 s timeout, nothing else (no seed, no extra fields)
    assert all({k: v for k, v in c.items() if k not in ('model', 'messages')} == {'timeout': 60.0} for c in calls)


def test_expected_texts():
    assert outcome(uu._wrap_chat_client(fake_client(CASES['think block'])[0], strip_thinking=True)) == \
        ('text', 'How about a crime film?')
    assert outcome(uu._wrap_chat_client(fake_client(CASES['unclosed twice'])[0], strip_thinking=True)) == \
        ('error', 'ThinkingTruncated')
    assert outcome(uu._wrap_chat_client(fake_client(CASES['None content'])[0], strip_thinking=True)) == ('text', '')


@pytest.mark.parametrize('case', ['think block', 'reasoning_content field', 'reasoning field and tokens'])
def test_thinking_at_the_startup_check_stops_the_run(case):
    client = uu._wrap_chat_client(fake_client(CASES[case])[0], strip_thinking=True)
    with pytest.raises(uu.ThinkingNotOff):
        llm_calls.chat(client, 'startup_check', model='m', messages=[{'role': 'user', 'content': 'Hello!'}])


def test_other_providers_keep_2_0_0s_reply():
    client = uu._wrap_chat_client(fake_client(CASES['think block'])[0])          # azure / openai: no stripping
    assert outcome(client) == ('text', '<think>The user wants crime.</think>\n\nHow about a crime film?')
