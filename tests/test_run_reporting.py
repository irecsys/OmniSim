"""
Three small changes found by the smoke test. Offline, fake clients.

1. A dialogue stopped by max_total_rounds ends with its own <FAILED> System line; the line for running out of
   max_rec_attempts (the only one in v1) is unchanged.
2. The log file name starts with the chat model (it was 'None-…': config['model'] never existed).
3. Visibility of a missing target: a target that is not in df_items (H&M-shaped int ids after user_guest) logs one
   WARNING per conversation and is counted as target_not_found; the conversation files are unchanged.
"""
import glob
import logging
import os

import numpy as np
import pandas as pd
import pytest

import utils.quick_start as qs
import utils.simulator as sim
import utils.user_profile_builder as upb
from utils.configurator import Config
from utils.logger import init_logger
from test_round_limit import make_config, run
from test_llm_fallback import FakeChat, FakeEmbeddings, finals, run_toy, summary_file, toy

ROUND_LIMIT_END = ("System: <FAILED> Session ended. The target item 'Target Movie' (tt1) was not found before "
                   "the dialogue round limit (max_total_rounds) was reached.")
REC_ATTEMPTS_END = ("System: <FAILED> Session ended. The target item 'Target Movie' (tt1) was not found within "
                    "the maximum recommendation attempts.")


# ── 1. System line of the two failure exits ───────────────────────────────────
@pytest.fixture
def rejecting(monkeypatch):
    """_do_recommendation that shows an item every round and has it rejected (show=False: shows nothing)."""
    state = {'calls': 0, 'show': True}

    def fake(config, client_chat, backend, emb, target_title, target_item_id, item_row, profile, chat_history,
             rejected_item_ids, previous_rejections, disliked_context, *args, **kwargs):
        state['calls'] += 1
        return False, 0, disliked_context, state['show'], state['show']

    monkeypatch.setattr(sim, '_do_recommendation', fake)
    monkeypatch.setattr(sim, '_build_rec_constraints', lambda user_context, backend, config: ({}, None))
    return state


@pytest.mark.parametrize('mode', ['free', 'static'])
def test_round_limit_has_its_own_end_line(mode, tmp_path, rejecting):
    rejecting['show'] = False                                   # the endless loop: nothing shown, stopped at round 4
    out = run(mode, make_config(tmp_path, max_total_rounds=3))
    assert rejecting['calls'] == 3 and out['succeed'] == 0
    assert out['text'].splitlines()[-1] == ROUND_LIMIT_END


@pytest.mark.parametrize('mode', ['free', 'static'])
def test_rec_attempts_end_line_is_unchanged(mode, tmp_path, rejecting):
    out = run(mode, make_config(tmp_path, max_rec_attempts=2))  # two items shown and rejected
    assert rejecting['calls'] == 2 and (out['shown'], out['succeed']) == (2, 0)
    assert out['text'].splitlines()[-1] == REC_ATTEMPTS_END


def test_both_end_lines_keep_the_parsed_prefix():
    """Downstream tools that parse the end line read '<FAILED>', the title and the id; the corpus comparison
    tells the two reasons apart."""
    import legacy_es_equivalence.compare_dialogue_corpora as cdc
    for line in (ROUND_LIMIT_END, REC_ATTEMPTS_END):
        assert line.startswith("System: <FAILED> Session ended. The target item 'Target Movie' (tt1) was not found ")
    text = 'Bot: Hi\nUser: A horror film.\nBot: Sorry.\n'
    assert cdc.classify(text + REC_ATTEMPTS_END + '\n') == ('max_rec_attempts:target_not_shown', 'Target Movie')
    assert cdc.classify(text + ROUND_LIMIT_END + '\n') == ('max_total_rounds', 'Target Movie')


# ── 2. log file name ──────────────────────────────────────────────────────────
def logger_config(tmp_path, **values):
    config = Config.__new__(Config)
    config.final_config_dict = {'log_path_outputs': str(tmp_path / 'logs') + '/', 'state': 'INFO', **values}
    return config


@pytest.mark.parametrize('values, prefix', [({'chat_model': 'qwen3_8_flash'}, 'qwen3_8_flash-'),
                                            ({'chat_model': 'meta-llama/Llama-3.1 8B'}, 'meta-llama_Llama-3.1_8B-'),
                                            ({'chat_model': ''}, 'omnisim-'),
                                            ({}, 'omnisim-')])                    # no chat_model key at all
def test_log_file_name_starts_with_the_chat_model(tmp_path, values, prefix):
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        path = init_logger(logger_config(tmp_path, **values))
    finally:
        for h in root.handlers[:]:
            if h not in before:
                root.removeHandler(h)
                h.close()
    name = os.path.basename(path)
    assert name.startswith(prefix) and name.endswith('.log') and not name.startswith('None')
    assert os.path.dirname(path) == str(tmp_path / 'logs')


# ── 3. target not found ────────────────────────────────────
WARN = 'not found in the items of dataset'


def hm_shaped(tmp_path, monkeypatch, notify=True):
    """H&M-shaped ids (all digits → int in the index rows): user_guest replaces df_items with index rows, so its
    target and the target of the items strategy after it are not found."""
    (tmp_path / 'users_in.csv').parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'user_id': ['u1']}).to_csv(tmp_path / 'users_in.csv', index=False)
    cfg_file = toy(tmp_path, extra={'input_pairs_file': None, 'input_users_file': str(tmp_path / 'users_in.csv'),
                                    'chats_per_user': 1, 'guestuser_randomitem_records': 1,
                                    'max_rec_attempts': 2, 'max_questions': 1, 'seed': 2025})
    rng = np.random.default_rng(0)
    rows = []
    for k in range(30):
        v = rng.normal(size=8)
        rows.append({'item_id': str(1000001 + k), 'title': f'Item {k}', 'category': 'Women Tops Basic',
                     'details': f'A soft jersey piece number {k}.', 'release_year': 2000 + k,
                     'embedding_vector': str((v / np.linalg.norm(v)).tolist())})
    pd.DataFrame(rows).to_csv(tmp_path / 'data' / 'toy' / 'items.csv', index=False)
    clients = lambda config: (FakeChat(), FakeEmbeddings())
    monkeypatch.setattr(qs, 'get_openai_clients', clients)
    monkeypatch.setattr(upb, 'get_openai_clients', clients)
    monkeypatch.setattr(qs, '_needs_profile_rebuild', lambda config, data_path: False)
    from utils.retrieval import sampling
    sample_items = sampling.sample_items                         # unseeded rng: fix it so both runs draw alike
    monkeypatch.setattr(sampling, 'sample_items', lambda backend, size, seed=None: sample_items(backend, size, 0))
    if not notify:
        monkeypatch.setattr(sim, '_note_target_not_found', lambda *a, **k: None)
    summary = qs.run_simulation([cfg_file], mode_override='free')
    files = sorted(glob.glob(str(tmp_path / 'chats' / '**' / '*.txt'), recursive=True))
    return summary, files


def transcripts(files):
    """{(strategy, user, item): bytes}; the final file name ends with the run timestamp, which differs."""
    return {(f.split(os.sep)[-3],) + tuple(os.path.basename(f).split('-')[:2]): open(f, 'rb').read()
            for f in finals(files)}


def target_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and WARN in r.getMessage()]


def test_hm_shaped_user_guest_and_items_targets_not_found(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        summary, files = hm_shaped(tmp_path / 'on', monkeypatch)
    assert summary['strategies'] == ['user_guest', 'items']
    found = target_warnings(caplog)
    assert len(found) == 2                                       # one per conversation
    assert any('strategy user_guest, user u1' in m for m in found)
    assert any('strategy items, user None' in m for m in found)
    assert all('dataset toy' in m and 'cannot succeed' in m for m in found)
    c = summary['conversations']
    assert (c['saved'], c['failed'], c['target_not_found']) == (2, 2, 2) and summary['exit_code'] == 0
    summary_file(tmp_path / 'on', summary, 'free')
    with_note = transcripts(files)
    assert len(with_note) == 2
    assert all("The target item '' (" in t.decode('utf-8') for t in with_note.values())

    # the conversations and calls are those of a run without the notification
    monkeypatch.undo()
    summary_off, files_off = hm_shaped(tmp_path / 'off', monkeypatch, notify=False)
    assert transcripts(files_off) == with_note
    assert summary_off['calls'] == summary['calls']
    assert summary_off['conversations']['target_not_found'] == 0


def test_imdb_shaped_normal_run_counts_zero(tmp_path, monkeypatch, caplog):
    with caplog.at_level(logging.WARNING):
        summary, files = run_toy(tmp_path, monkeypatch, 'free')
    assert not target_warnings(caplog)
    assert summary['conversations']['target_not_found'] == 0 and len(finals(files)) == 2


def test_notification_outside_a_conversation_does_nothing():
    import utils.llm_calls as llm_calls
    llm_calls.note_target_not_found()                            # no current conversation: no error, no count
