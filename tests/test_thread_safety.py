"""
Thread safety of the dialogue run. Offline: fake chat / embedding
clients, toy data, the real run_simulation / simulate_* / retrieval.

1. concurrent retrieval returns exactly what sequential retrieval returns; one SQLite connection per thread
2. Conversations of the same (user, item) that start in the same second and end with the
   same outcome get the same final file name — os.rename silently overwrote all but one
3. a larger end-to-end run with 8 workers: one final file per task, no tmp left, outcome consistent
"""
import datetime as _dt
import errno
import glob
import os
import random
import sqlite3
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import yaml

import utils.quick_start as qs
import utils.simulator as sim
from utils.retrieval import ref_lookup, sampling
from utils.retrieval.backend import LocalBackend
from utils.retrieval.local_index import build_local_index, load_config
from utils.retrieval.local_search import search_items_local

GENRES = ['Drama', 'Comedy', 'Horror', 'Action', 'Romance']


def toy_items(n=40, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n):
        v = rng.normal(size=dim)
        rows.append({'item_id': f'tt{k}', 'title': f'Movie {k}', 'category': f'{GENRES[k % 5]}|{GENRES[(k + 2) % 5]}',
                     'details': f'A {GENRES[k % 5].lower()} story number {k} about {["love", "war", "ghosts", "cars"][k % 4]}',
                     'embedding_vector': str((v / np.linalg.norm(v)).tolist())})
    return pd.DataFrame(rows)


def write_toy(tmp_path, df, extra=None):
    ds = tmp_path / 'data' / 'toy'
    ds.mkdir(parents=True, exist_ok=True)
    df.to_csv(ds / 'items.csv', index=False)
    cfg = {'data_path': str(tmp_path / 'data') + '/', 'dataset': 'toy', 'file_items': 'items.csv',
           'index_dir': str(tmp_path / 'index' / 'toy'), 'precomputed_embedding_col': 'embedding_vector',
           'col_itemid': 'item_id', 'col_title': 'title', 'col_category': 'category', 'col_details': 'details',
           'embedding_fields': ['title', 'category', 'details'], 'item_attributes': {'category': 'genre'},
           'bm25_details_field': 'bm25_details'}
    cfg.update(extra or {})
    path = tmp_path / 'toy.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return str(path)


# ── 1. retrieval under concurrency ──────────────────────────────────────────
def test_concurrent_retrieval_equals_sequential(tmp_path):
    cfg_file = write_toy(tmp_path, toy_items())
    config = load_config([cfg_file])
    build_local_index(config, config['index_dir'])
    backend = LocalBackend(config['index_dir'])
    rng = np.random.default_rng(1)
    jobs = []
    for k in range(2000):
        q = rng.normal(size=8)
        jobs.append(dict(lam=[0.0, 0.3, 1.0][k % 3], tau=[0.0, 0.5, 0.7][k % 3 - 1], q=(q / np.linalg.norm(q)).tolist(),
                         text=f'{GENRES[k % 5].lower()} story about {["love", "war", "ghosts"][k % 3]}',
                         exclude=[GENRES[k % 5]] if k % 4 == 0 else None,
                         refs=[f'Movie {k % 40}'] if k % 7 == 0 else None))

    def run(job):
        cfg = dict(config, lambda_bm25=job['lam'], threshold_similarity=job['tau'])
        refs = ref_lookup.lookup_reference_embeddings(backend, cfg, job['refs']) if job['refs'] else None
        out = search_items_local(cfg, backend, job['q'], 10, query_text=job['text'],
                                 exclude_genres=job['exclude'], reference_embeddings=refs or None)
        return [(o['item_id'], o['score']) for o in out]

    sequential = [run(j) for j in jobs]
    order = list(range(len(jobs)))
    random.Random(2).shuffle(order)
    with ThreadPoolExecutor(max_workers=16) as ex:
        concurrent = dict(zip(order, ex.map(lambda i: run(jobs[i]), order)))
    assert [concurrent[i] for i in range(len(jobs))] == sequential
    assert sum(bool(r) for r in sequential) > 500                  # the gate lets a good share through

    samples = [sampling.sample_items(backend, 5, seed=s) for s in range(50)]
    with ThreadPoolExecutor(max_workers=8) as ex:
        assert list(ex.map(lambda s: sampling.sample_items(backend, 5, seed=s), range(50))) == samples


def test_one_sqlite_connection_per_thread_and_no_sharing(tmp_path):
    cfg_file = write_toy(tmp_path, toy_items(n=5))
    config = load_config([cfg_file])
    build_local_index(config, config['index_dir'])
    backend = LocalBackend(config['index_dir'])
    conns = {}

    def grab(i):
        conns[i] = backend.conn()
        return backend.conn() is conns[i]                              # cached per thread
    threads = [threading.Thread(target=grab, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert len({id(c) for c in conns.values()}) == 4
    main = backend.conn()
    errors = []

    def misuse():
        try:
            main.execute('SELECT 1')
        except sqlite3.ProgrammingError as e:                          # check_same_thread stays True
            errors.append(e)
    t = threading.Thread(target=misuse)
    t.start()
    t.join()
    assert len(errors) == 1


# ── 2./3. full runs with fake clients ───────────────────────────────────────
class FakeChat:
    def __init__(self):
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **kwargs):
        msg = types.SimpleNamespace(content='Something with a clear personality would be good.')
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


class FakeEmbeddings:
    """Every query embeds to the vector of the requested item, so it is retrieved first."""
    def __init__(self, vector):
        self.embeddings = self
        self.vector = vector

    def create(self, model=None, input=None, **kwargs):
        return types.SimpleNamespace(data=[types.SimpleNamespace(embedding=list(self.vector))])


def run_dialogues(tmp_path, monkeypatch, df, overrides, query_vector):
    cfg_file = write_toy(tmp_path, df)
    ds = tmp_path / 'data' / 'toy'
    pd.DataFrame({'user_id': ['u1', 'u2', 'u3'], 'age': [30, 40, 50]}).to_csv(ds / 'users.csv', index=False)
    base = {'log_path_outputs': str(tmp_path / 'logs') + '/', 'outputs_folder': str(tmp_path / 'chats') + '/',
            'file_users': 'users.csv', 'file_interactions': None, 'chit_chat_ratio': 0.0,
            'rec_explanation_ratio': 0.0, 'rejection_explanation_ratio': 0.0, 'expand_query': False,
            'input_items_file': None, 'input_users_file': None, 'guestuser_randomitem_records': None}
    base.update(overrides)
    over = tmp_path / 'over.yaml'
    over.write_text(yaml.safe_dump(base), encoding='utf-8')
    monkeypatch.setattr(qs, 'get_openai_clients', lambda config: (FakeChat(), FakeEmbeddings(query_vector)))
    monkeypatch.setattr(qs, '_needs_profile_rebuild', lambda config, data_path: False)
    qs.run_simulation([cfg_file, str(over)], mode_override='free')
    finals = [f for f in glob.glob(str(tmp_path / 'chats' / '**' / '*.txt'), recursive=True) if '-tmp-' not in f]
    tmps = glob.glob(str(tmp_path / 'chats' / '**' / '*-tmp-*'), recursive=True)
    return finals, tmps


class FixedDateTime(_dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return cls(2026, 10, 25, 12, 0, 59)


def no_hard_links(src, dst):
    raise OSError(errno.EPERM, 'Operation not permitted (simulated: file system without hard links)')


@pytest.mark.parametrize('links', ['link', 'no-link fallback'])
def test_same_pair_same_second_keeps_every_conversation(tmp_path, monkeypatch, links):
    if links != 'link':
        monkeypatch.setattr(os, 'link', no_hard_links)
    # All six conversations start "at the same second" (clock pinned) and end the same way
    # (the target is retrieved first and accepted in round 1), so they compute the same final name.
    df = toy_items()
    target_vec = [float(x) for x in yaml.safe_load(df.loc[3, 'embedding_vector'])]
    pd.DataFrame({'user_id': ['u1'], 'item_id': ['tt3']}).to_csv(tmp_path / 'pairs.csv', index=False)
    monkeypatch.setattr(sim, 'datetime', types.SimpleNamespace(datetime=FixedDateTime, timedelta=_dt.timedelta))
    finals, tmps = run_dialogues(tmp_path, monkeypatch, df,
                                 {'input_pairs_file': str(tmp_path / 'pairs.csv'), 'chats_per_entry': 6, 'num_workers': 6},
                                 target_vec)
    names = sorted(os.path.basename(f) for f in finals)
    assert tmps == []
    assert len(finals) == 6, names
    fields = [n[:-4].split('-') for n in names]
    assert all(f[:2] == ['u1', 'tt3'] and f[4] == '1' for f in fields)
    groups = {}                                                      # same outcome → same name before the fix
    for f in fields:
        groups.setdefault(tuple(f[:5]), []).append(f[5])
    assert max(len(g) for g in groups.values()) >= 2                  # at least one real collision happened
    # in each group: the start second, then +1 s each (with the carry into the next minute)
    steps = ['20261025120059', '20261025120100', '20261025120101', '20261025120102', '20261025120103', '20261025120104']
    for g in groups.values():
        assert sorted(g) == steps[:len(g)]
    for f in finals:
        text = open(f, encoding='utf-8').read()
        assert text.rstrip().splitlines()[-1].startswith('System: <END> Session ended successfully')


def test_many_workers_one_file_per_task(tmp_path, monkeypatch):
    df = toy_items()
    target_vec = [float(x) for x in yaml.safe_load(df.loc[0, 'embedding_vector'])]
    pairs = pd.DataFrame({'user_id': [f'u{1 + k % 3}' for k in range(30)], 'item_id': [f'tt{k % 10}' for k in range(30)]})
    pairs.to_csv(tmp_path / 'pairs.csv', index=False)
    pd.DataFrame({'item_id': [f'tt{k}' for k in range(10)]}).to_csv(tmp_path / 'items_in.csv', index=False)
    pd.DataFrame({'user_id': ['u1', 'u2']}).to_csv(tmp_path / 'users_in.csv', index=False)
    finals, tmps = run_dialogues(tmp_path, monkeypatch, df, {
        'input_pairs_file': str(tmp_path / 'pairs.csv'), 'input_items_file': str(tmp_path / 'items_in.csv'),
        'input_users_file': str(tmp_path / 'users_in.csv'), 'guestuser_randomitem_records': 20,
        'chats_per_entry': 2, 'chats_per_user': 5, 'num_workers': 8}, target_vec)
    expected = 20 + 10 * 2 + 30 * 2 + 2 * 5                       # items + item_list + pairs + user_guest
    assert tmps == []
    assert len(finals) == expected
    for f in finals:
        succeed = os.path.basename(f)[:-4].split('-')[4]
        last = open(f, encoding='utf-8').read().rstrip().splitlines()[-1]
        assert last.startswith('System: <END>') if succeed == '1' else last.startswith('System: <FAILED>'), f


# ── _save_conversation itself ────────────────────────────
def _tmp(folder, k):
    path = folder / f'user_item_pairs-tt1-u1-tmp-20261025120059-{k:08x}.txt'
    path.write_text(f'Bot: hello {k}\nSystem: <END> Session ended successfully.\n', encoding='utf-8')
    return str(path)


@pytest.mark.parametrize('links', ['link', 'no-link fallback'])
def test_save_conversation_never_overwrites(tmp_path, monkeypatch, links):
    if links != 'link':
        monkeypatch.setattr(os, 'link', no_hard_links)
    tmps = [_tmp(tmp_path, k) for k in range(3)]
    finals = [sim._save_conversation(str(tmp_path), 'u1', 'tt1', 3, 1, 1, '20261025120059', t) for t in tmps]
    assert [os.path.basename(f) for f in finals] == ['u1-tt1-3-1-1-20261025120059.txt', 'u1-tt1-3-1-1-20261025120100.txt',
                                                     'u1-tt1-3-1-1-20261025120101.txt']
    assert [open(f, encoding='utf-8').read().splitlines()[0] for f in finals] == ['Bot: hello 0', 'Bot: hello 1', 'Bot: hello 2']
    assert not any(os.path.exists(t) for t in tmps)
    assert sorted(os.listdir(tmp_path)) == sorted(os.path.basename(f) for f in finals)   # no placeholder, no tmp left


def test_save_conversation_fallback_replace_failure_keeps_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'link', no_hard_links)
    tmp = _tmp(tmp_path, 9)

    def refuse(src, dst):
        raise PermissionError(errno.EACCES, 'simulated: destination is open in another process')
    monkeypatch.setattr(os, 'replace', refuse)
    with pytest.raises(PermissionError):
        sim._save_conversation(str(tmp_path), 'u1', 'tt1', 3, 1, 1, '20261025120059', tmp)
    assert os.listdir(tmp_path) == [os.path.basename(tmp)]              # placeholder removed, tmp kept


def test_save_conversation_link_path_never_leaves_an_empty_final_file(tmp_path, monkeypatch):
    # the final file must have its full content the moment it exists: watch every link / open call
    seen = []
    real_link = os.link

    def watching_link(src, dst):
        real_link(src, dst)
        seen.append(os.path.getsize(dst))
    monkeypatch.setattr(os, 'link', watching_link)
    monkeypatch.setattr(os, 'open', lambda *a, **k: pytest.fail('the O_EXCL fallback must not run when links work'))
    final = sim._save_conversation(str(tmp_path), 'u1', 'tt1', 3, 1, 1, '20261025120059', _tmp(tmp_path, 1))
    assert seen and seen[0] > 0 and os.path.getsize(final) == seen[0]
