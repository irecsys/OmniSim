"""
SQLite FTS5 availability. A Python whose sqlite3 lacks FTS5 must get a clear error (with the SQLite
version and how to fix it) from building and from opening an index, before any index file is written.
"""
import sqlite3

import numpy as np
import pandas as pd
import pytest

import utils.retrieval.local_index as li
from utils.retrieval.backend import LocalBackend


def toy_config(tmp_path, **extra):
    ds = tmp_path / 'data' / 'toy'
    ds.mkdir(parents=True)
    vecs = np.eye(3)
    pd.DataFrame({'item_id': ['a', 'b', 'c'], 'title': ['A', 'B', 'C'], 'category': ['x', 'y', 'z'],
                  'details': ['one', 'two', 'three'],
                  'embedding_vector': [str(v.tolist()) for v in vecs]}).to_csv(ds / 'items.csv', index=False)
    config = {'data_path': str(tmp_path / 'data'), 'dataset': 'toy', 'file_items': 'items.csv',
              'col_itemid': 'item_id', 'col_title': 'title', 'col_category': 'category', 'col_details': 'details',
              'precomputed_embedding_col': 'embedding_vector'}
    config.update(extra)
    return config


@pytest.fixture
def no_fts5(monkeypatch):
    def probe():
        raise sqlite3.OperationalError('no such module: fts5')
    monkeypatch.setattr(li, '_probe_fts5', probe)
    monkeypatch.setattr(li, '_FTS5_OK', False)


def test_this_python_has_fts5(monkeypatch):
    monkeypatch.setattr(li, '_FTS5_OK', False)
    li.check_fts5()
    assert li._FTS5_OK is True


def test_build_without_fts5_fails_clearly_and_writes_nothing(tmp_path, no_fts5):
    config = toy_config(tmp_path)
    out = tmp_path / 'index'
    with pytest.raises(RuntimeError) as e:
        li.build_local_index(config, str(out))
    for part in ('FTS5', f'SQLite {sqlite3.sqlite_version}', 'no such module: fts5', 'python.org', 'conda',
                 "create virtual table t using fts5(x)"):
        assert part in str(e.value), part
    assert not out.exists()


def test_open_without_fts5_fails_clearly(tmp_path, monkeypatch):
    config = toy_config(tmp_path)
    li.build_local_index(config, str(tmp_path / 'index'))           # built where FTS5 exists
    monkeypatch.setattr(li, '_probe_fts5', lambda: (_ for _ in ()).throw(sqlite3.OperationalError('no such module: fts5')))
    monkeypatch.setattr(li, '_FTS5_OK', False)
    with pytest.raises(RuntimeError, match='needs SQLite.s FTS5 extension'):
        LocalBackend(str(tmp_path / 'index'))


def test_trigram_needs_sqlite_3_34(tmp_path, monkeypatch):
    config = toy_config(tmp_path, bm25_tokenizer='trigram')
    monkeypatch.setattr(sqlite3, 'sqlite_version_info', (3, 31, 1))
    monkeypatch.setattr(sqlite3, 'sqlite_version', '3.31.1')
    with pytest.raises(RuntimeError, match=r"'trigram' needs SQLite >= 3\.34\.0, .* has SQLite 3\.31\.1"):
        li.build_local_index(config, str(tmp_path / 'index'))
    assert not (tmp_path / 'index').exists()
    assert li.fts_tokenizer({'bm25_tokenizer': 'unicode61'}) == 'unicode61'      # other tokenizers unaffected
