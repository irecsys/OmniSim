"""
generate_chit_chat reads temp_chitchat (default 1.2, the value it used to hard-code), the same way
generate_bot_closing_failed does (default 1.0). Offline: a fake chat client records the call arguments.
"""
import types

import pytest

import utils.utils as uu
from utils.configurator import Config

CONTEXTS = ['greeting', 'category_react', 'pre_recommendation', 'follow_up', 'transition', 'closing']


class RecordingChat:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=self)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        msg = types.SimpleNamespace(content='Nice pick.')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def _no_key_config():
    """Config always loads system.yaml (which sets temp_chitchat), so the missing-key case uses a plain dict."""
    return {'role_bot': 'movie recommender', 'role_user': 'movie seeker', 'chat_model': 'fake-model',
            'prompts_file': 'configs/prompts/default.yaml'}


@pytest.mark.parametrize('context', CONTEXTS)
def test_chitchat_uses_configured_temperature(context):
    config = Config(['configs/imdb/imdb.yaml'])
    config['temp_chitchat'] = 0.5
    client = RecordingChat()
    assert uu.generate_chit_chat(config, client, 'bot', context=context, chat_history=[]) == 'Nice pick.'
    assert [c['temperature'] for c in client.calls] == [0.5]


@pytest.mark.parametrize('context', CONTEXTS)
def test_chitchat_default_temperature_without_key(context):
    config = _no_key_config()
    assert 'temp_chitchat' not in config
    client = RecordingChat()
    uu.generate_chit_chat(config, client, 'bot', context=context, chat_history=[])
    assert [c['temperature'] for c in client.calls] == [1.2]


def test_system_yaml_value_unchanged():
    """system.yaml sets 1.2, so default runs produce the same calls as before the change."""
    client = RecordingChat()
    uu.generate_chit_chat(Config(['configs/imdb/imdb.yaml']), client, 'bot', context='greeting')
    assert client.calls[0]['temperature'] == 1.2


def test_closing_failed_reads_the_same_key():
    """Unchanged: generate_bot_closing_failed reads temp_chitchat too, default 1.0."""
    config = Config(['configs/imdb/imdb.yaml'])
    config['temp_chitchat'] = 0.5
    client = RecordingChat()
    uu.generate_bot_closing_failed(config, client)
    uu.generate_bot_closing_failed(_no_key_config(), client)
    assert [c['temperature'] for c in client.calls] == [0.5, 1.0]
