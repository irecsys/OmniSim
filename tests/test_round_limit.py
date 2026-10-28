"""
The max_total_rounds stop-loss in the dialogue main loop (a dialogue that never shows items again would otherwise loop forever).

Offline: fake chat / embedding clients, and _do_recommendation replaced by a script that decides per round
whether items are shown and accepted (the real one needs a retrieval backend). Everything else in
simulate_free / simulate_static runs for real.
"""
import glob
import logging
import os
import random
import types

import pandas as pd
import pytest

import utils.simulator as sim
from utils.retrieval.local_index import load_config


class FakeChat:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        msg = types.SimpleNamespace(content='Something dark and atmospheric, please.')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeEmbeddings:
    def __init__(self):
        self.embeddings = self

    def create(self, model=None, input=None, **kwargs):
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=[0.1, 0.2, 0.3])])


def make_config(tmp_path, **overrides):
    config = load_config(['configs/imdb/imdb.yaml'])
    config.update({'outputs_folder': str(tmp_path / 'chats'), 'generation_strategy': 'user_item_pairs',
                   'expand_query': False, 'chit_chat_ratio': 0.0, 'rec_explanation_ratio': 0.0})
    config.update(overrides)
    return config


def make_dataset():
    df = pd.DataFrame([{'imdb_id': 'tt1', 'title': 'Target Movie', 'genres': 'Horror',
                        'overview': 'A haunted house.', 'release_year': 1990}])
    return types.SimpleNamespace(df_items=df, df_interactions=None, itemid='imdb_id', title='title',
                                 category='genres', details='overview')


@pytest.fixture
def scripted(monkeypatch):
    """_do_recommendation that accepts at round `accept_at` (None: never shows anything — the endless loop)."""
    state = {'calls': 0, 'accept_at': None}

    def fake_do_recommendation(config, client_chat, backend, emb, target_title, target_item_id, item_row,
                               profile, chat_history, rejected_item_ids, previous_rejections, disliked_context,
                               *args, **kwargs):
        state['calls'] += 1
        if state['accept_at'] is not None and state['calls'] >= state['accept_at']:
            return True, 1, disliked_context, True, True
        return False, 0, disliked_context, False, False

    monkeypatch.setattr(sim, '_do_recommendation', fake_do_recommendation)
    monkeypatch.setattr(sim, '_build_rec_constraints', lambda user_context, backend, config: ({}, None))
    return state


def run(mode, config, seed=0):
    random.seed(seed)
    func = {'free': sim.simulate_free, 'static': sim.simulate_static}[mode]
    func(config, make_dataset(), 'u1', 'tt1', FakeChat(), FakeEmbeddings(), backend=None)
    files = glob.glob(os.path.join(config['outputs_folder'], '**', 'u1-tt1-*.txt'), recursive=True)
    assert len(files) == 1 and not glob.glob(os.path.join(config['outputs_folder'], '**', '*-tmp-*'), recursive=True)
    uid, item, turns, shown, succeed, _ts = os.path.basename(files[0])[:-4].split('-')
    return {'turns': int(turns), 'shown': int(shown), 'succeed': int(succeed),
            'text': open(files[0], encoding='utf-8').read()}


def limit_warnings(caplog):
    return [r for r in caplog.records if r.levelno == logging.WARNING and 'max_total_rounds' in r.getMessage()]


@pytest.mark.parametrize('mode', ['free', 'static'])
def test_endless_no_show_loop_stops_at_30_rounds_as_failed(mode, tmp_path, scripted, caplog):
    config = make_config(tmp_path)
    assert config['max_total_rounds'] == 30                       # system.yaml
    with caplog.at_level(logging.WARNING):
        out = run(mode, config)
    assert scripted['calls'] == 30                                # one retrieval per round
    assert out['succeed'] == 0 and out['shown'] == 0
    warns = limit_warnings(caplog)
    assert len(warns) == 1
    for part in ('30 rounds', f'mode {mode}', 'user u1', 'target tt1', 'succeed=0'):
        assert part in warns[0].getMessage(), part


def test_missing_key_defaults_to_30(tmp_path, scripted):
    config = make_config(tmp_path)
    del config['max_total_rounds']
    run('free', config)
    assert scripted['calls'] == 30


@pytest.mark.parametrize('mode', ['free', 'static'])
def test_a_small_limit_really_truncates(mode, tmp_path, scripted, caplog):
    scripted['accept_at'] = 5
    with caplog.at_level(logging.WARNING):
        out = run(mode, make_config(tmp_path, max_total_rounds=3))
    assert scripted['calls'] == 3 and out['succeed'] == 0 and len(limit_warnings(caplog)) == 1


@pytest.mark.parametrize('mode', ['free', 'static'])
def test_dialogues_below_the_limit_are_unchanged(mode, tmp_path, scripted, caplog):
    # accepted at round 6, the largest round count in the frozen corpora: identical with limit 30 and 10**9
    outs = []
    for limit, sub in ((30, 'a'), (10 ** 9, 'b')):
        scripted['calls'], scripted['accept_at'] = 0, 6
        with caplog.at_level(logging.WARNING):
            outs.append(run(mode, make_config(tmp_path / sub, max_total_rounds=limit), seed=7))
        assert scripted['calls'] == 6
    assert outs[0] == outs[1] and outs[0]['succeed'] == 1
    assert not limit_warnings(caplog)


# ── A past hang (dialogue A-imdb_fixture-d05a79c3) with the real _do_recommendation ──────────
# The target is excluded by exclude_genres (as "not the horror I asked for" did there); the other items are
# shown and rejected. From then on every retrieval passes the τ gate (max base ≥ τ) but simulator.py:331 filters
# all candidates out as already rejected, :341-342 returns items_shown=False, and the round leaves no trace.
def make_index_dataset(tmp_path):
    import numpy as np
    from utils.retrieval.backend import LocalBackend
    from utils.retrieval.local_index import build_local_index
    rows = [('tt1', 'Target Movie', 'Horror', 'Plant-eating monsters in a farmhouse.', [1.0, 0.0, 0.0]),
            ('tt2', 'Casualties', 'War|Drama', 'A war drama.', [0.9, 0.1, 0.0]),
            ('tt3', 'Lieutenant', 'Crime|Drama', 'A crime drama.', [0.8, 0.2, 0.1])]
    df = pd.DataFrame([{'imdb_id': i, 'title': t, 'genres': g, 'overview': o, 'release_year': 1990,
                        'embedding_vector': str((np.array(v) / np.linalg.norm(v)).tolist())} for i, t, g, o, v in rows])
    (tmp_path / 'data' / 'imdb').mkdir(parents=True)
    df.to_csv(tmp_path / 'data' / 'imdb' / 'items.csv', index=False)
    config = make_config(tmp_path, data_path=str(tmp_path / 'data'), index_dir=str(tmp_path / 'index'),
                         threshold_similarity=0.0, rejection_explanation_ratio=0.0)
    build_local_index(config, config['index_dir'])
    dataset = types.SimpleNamespace(df_items=df.drop(columns=['embedding_vector']), df_interactions=None,
                                    itemid='imdb_id', title='title', category='genres', details='overview')
    return config, dataset, LocalBackend(config['index_dir'])


@pytest.mark.parametrize('mode', ['free', 'static', 'adaptive'])
def test_s6_hang_gate_passes_but_everything_is_filtered(mode, tmp_path, monkeypatch, caplog):
    config, dataset, backend = make_index_dataset(tmp_path)
    monkeypatch.setattr(sim, '_build_rec_constraints',
                        lambda user_context, backend, config: ({'exclude_genres': ['Horror']}, None))
    real_search = sim.search_items_by_embedding
    rounds = []                                    # per retrieval: (candidates returned, left after the filter)

    def traced_search(config, backend, embedding_vector, top_k, **kw):
        out = real_search(config, backend, embedding_vector, top_k, **kw)
        rounds.append({'returned': [o['imdb_id'] for o in out]})
        return out
    monkeypatch.setattr(sim, 'search_items_by_embedding', traced_search)
    real_do = sim._do_recommendation

    def traced_do(*args, **kwargs):
        rejected_before = set(args[9])             # rejected_item_ids
        result = real_do(*args, **kwargs)
        rounds[-1].update(rejected_before=rejected_before, items_shown=result[4])
        return result
    monkeypatch.setattr(sim, '_do_recommendation', traced_do)

    random.seed(3)
    func = {'free': sim.simulate_free, 'static': sim.simulate_static, 'adaptive': sim.simulate_adaptive}[mode]
    with caplog.at_level(logging.WARNING):
        func(config, dataset, 'u1', 'tt1', FakeChat(), FakeEmbeddings(), backend=backend)

    files = glob.glob(os.path.join(config['outputs_folder'], '**', 'u1-tt1-*.txt'), recursive=True)
    assert len(files) == 1
    _uid, _item, turns, shown, succeed, _ts = os.path.basename(files[0])[:-4].split('-')
    assert len(rounds) == 30 and int(succeed) == 0 and int(shown) == 2
    assert len(limit_warnings(caplog)) == 1
    shown_rounds = [i for i, r in enumerate(rounds) if r['items_shown']]
    assert len(shown_rounds) == 2                  # the two non-target items, each rejected
    for r in rounds[shown_rounds[-1] + 1:]:        # afterwards: the gate passes (results come back) ...
        assert r['returned'] and 'tt1' not in r['returned']
        assert set(r['returned']) <= r['rejected_before'] and not r['items_shown']   # ... all already rejected
    text = open(files[0], encoding='utf-8').read()
    assert text.count('"Casualties"') + text.count('"Lieutenant"') == 2      # each shown once, then never again
