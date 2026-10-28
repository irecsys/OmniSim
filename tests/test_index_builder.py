"""
The local index builder computes embeddings when the items CSV has no precomputed column.
Offline: every test passes a fake embed_fn;
tests/conftest.py fails any test that reaches the real API client.
"""
import hashlib
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import utils.quick_start as qs
import utils.retrieval.local_index as li
from utils.retrieval.backend import LocalBackend
from utils.retrieval.local_search import search_items_local
from utils.utils import build_embedding_texts

LOG = logging.getLogger('test_index_builder')
DIM = 8


class FakeEmbed:
    """Deterministic texts → vectors; records every request."""
    def __init__(self, broken=False):
        self.calls = []
        self.broken = broken

    def __call__(self, texts):
        self.calls.append(list(texts))
        if self.broken:
            return [None for _ in texts]
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode('utf-8')).digest()
            out.append([(b - 127.5) / 50.0 for b in h[:DIM]])
        return out

    @property
    def n_texts(self):
        return sum(len(c) for c in self.calls)


def make_dataset(tmp_path, n=4, with_vectors=False, fields=('title', 'category', 'details'), extra=None):
    """Toy items CSV (no vector column by default) + config dict; returns (config, csv path)."""
    rows = [{'item_id': f'i{k}', 'title': f'Title {k}', 'category': ['Drama', 'Comedy'][k % 2],
             'details': f'details of item {k}'} for k in range(n)]
    df = pd.DataFrame(rows)
    if with_vectors:
        df['embedding_vector'] = [str([1.0, float(k)] + [0.0] * (DIM - 2)) for k in range(n)]
    ds = tmp_path / 'data' / 'toy'
    ds.mkdir(parents=True, exist_ok=True)
    df.to_csv(ds / 'items.csv', index=False)
    config = {'data_path': str(tmp_path / 'data'), 'dataset': 'toy', 'file_items': 'items.csv',
              'index_dir': str(tmp_path / 'index' / 'toy'),
              'col_itemid': 'item_id', 'col_title': 'title', 'col_category': 'category', 'col_details': 'details',
              'embeddings_provider': 'azure', 'embeddings_model': 'fake-model',
              'lambda_bm25': 0.3, 'weight_es_score': 0.7, 'weight_user_taste_short': 0.3,
              'threshold_similarity': 0.0}
    if fields is not None:
        config['embedding_fields'] = list(fields)
    if with_vectors:
        config['precomputed_embedding_col'] = 'embedding_vector'
    config.update(extra or {})
    return config, ds / 'items.csv'


def write_config(tmp_path, config):
    path = tmp_path / 'toy.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return str(path)


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()


def build(config, embed, **kw):
    return li.build_local_index(config, config['index_dir'], embed_fn=embed, **kw)


# vectors computed in batches, CSV left alone ─────────────────────────────
def test_computes_from_embedding_fields_in_batches_and_leaves_the_csv_alone(tmp_path):
    config, csv = make_dataset(tmp_path, n=205)
    before = sha(csv)
    embed = FakeEmbed()
    meta = build(config, embed)
    df = pd.read_csv(csv)
    assert [len(c) for c in embed.calls] == [100, 100, 5]
    assert sum(embed.calls, []) == build_embedding_texts(df, ['title', 'category', 'details'])
    assert sha(csv) == before and list(df.columns) == ['item_id', 'title', 'category', 'details']
    src = meta['source']
    assert (src['embeddings'], src['embedding_cache'], src['n_embedded'], src['n_cache_hits']) == \
        ('computed', 'embedding_cache.db', 205, 0)
    assert src['precomputed_embedding_col'] is None and meta['builder_version'] == 3
    raw = np.array(embed(sum(embed.calls, [])), dtype=np.float64)
    V = LocalBackend(config['index_dir']).vectors
    assert np.array_equal(V, (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32))


# precomputed vector column ────────────────────────────────────────────────
def test_precomputed_column_never_embeds_and_creates_no_cache(tmp_path):
    config, _ = make_dataset(tmp_path, with_vectors=True)
    embed = FakeEmbed()
    meta = build(config, embed)
    assert embed.calls == []
    assert meta['source']['embeddings'] == 'precomputed_col' and meta['source']['embedding_cache'] is None
    assert not os.path.exists(os.path.join(config['index_dir'], li.CACHE_FILE))


# configured vector column missing from the CSV ────────────────────────────
def test_configured_column_missing_from_the_csv_is_computed(tmp_path):
    config, _ = make_dataset(tmp_path, extra={'precomputed_embedding_col': 'embedding_vector'})
    embed = FakeEmbed()
    assert build(config, embed)['source']['embeddings'] == 'computed' and embed.n_texts == 4


# default embedding fields ─────────────────────────────────────────────────
def test_default_embedding_fields_are_title_category_details(tmp_path):
    config, csv = make_dataset(tmp_path, fields=None)
    embed = FakeEmbed()
    build(config, embed)
    df = pd.read_csv(csv)
    assert embed.calls[0] == build_embedding_texts(df, ['title', 'category', 'details'])
    assert embed.calls[0][0] == 'Title 0 Drama details of item 0'


# max_embed_items ──────────────────────────────────────────────────────────
def test_max_embed_items_stops_before_the_first_request(tmp_path):
    config, _ = make_dataset(tmp_path, n=5)
    embed = FakeEmbed()
    with pytest.raises(li.EmbeddingsNeeded, match='5 of 5 items need an embedding .*allows 4'):
        build(config, embed, max_embed_items=4)
    assert embed.calls == []
    assert not os.path.exists(config['index_dir'])


# dry run ──────────────────────────────────────────────────────────────────
def test_dry_run_sends_and_writes_nothing(tmp_path, monkeypatch, capsys):
    import scripts.build_local_index as cli
    config, csv = make_dataset(tmp_path, n=3)
    cfg = write_config(tmp_path, config)
    monkeypatch.setattr(li, 'build_local_index', lambda *a, **k: pytest.fail('dry run must not build'))
    monkeypatch.setattr(sys, 'argv', ['build_local_index.py', '--config', cfg, '--dry-run'])
    cli.main()
    out = capsys.readouterr().out
    assert '3 items, 0 in the cache' in out and '3 to embed = 1 requests' in out and 'nothing written' in out
    assert not os.path.exists(config['index_dir'])


# provider without vectors ─────────────────────────────────────────────────
def test_provider_without_vectors_is_an_error(tmp_path):
    config, _ = make_dataset(tmp_path)
    with pytest.raises(ValueError, match='returned no usable vectors'):
        build(config, FakeEmbed(broken=True))
    assert not os.path.exists(os.path.join(config['index_dir'], 'meta.json'))


# rebuilds use the embedding cache ─────────────────────────────────────────
def test_rebuild_uses_the_cache(tmp_path):
    config, _ = make_dataset(tmp_path, n=6)
    build(config, FakeEmbed())
    npy = os.path.join(config['index_dir'], 'vectors', 'text.npy')
    first = sha(npy)
    embed = FakeEmbed()
    meta = build(config, embed)
    assert embed.calls == [] and sha(npy) == first
    assert (meta['source']['n_embedded'], meta['source']['n_cache_hits']) == (0, 6)
    for f in ('items.db', 'meta.json', os.path.join('vectors', 'text.npy')):
        os.remove(os.path.join(config['index_dir'], f))
    build(config, embed)
    assert embed.calls == [] and sha(npy) == first


# embedding cache key ──────────────────────────────────────────────────────
def test_cache_key_is_id_text_and_model(tmp_path):
    config, csv = make_dataset(tmp_path, n=4)
    build(config, FakeEmbed())
    df = pd.read_csv(csv)
    df.loc[2, 'details'] = 'rewritten details'
    df.to_csv(csv, index=False)
    embed = FakeEmbed()
    build(config, embed)
    assert embed.calls == [['Title 2 Drama rewritten details']]
    embed2 = FakeEmbed()
    build(dict(config, embeddings_model='another-model'), embed2)
    assert embed2.n_texts == 4


# automatic build from the cache alone ─────────────────────────────────────
def test_automatic_build_works_from_the_cache_alone(tmp_path):
    config, _ = make_dataset(tmp_path, n=4)
    build(config, FakeEmbed())
    for f in ('items.db', 'meta.json'):
        os.remove(os.path.join(config['index_dir'], f))
    cfg = write_config(tmp_path, config)
    backend = qs._open_local_backend(li.load_config([cfg]), [cfg], LOG)    # max_embed_items=0, no embed_fn
    assert backend.n == 4 and backend.meta['source']['n_cache_hits'] == 4


# embedding space check when opening an index ──────────────────────────────
def test_open_refuses_an_index_of_another_embedding_model(tmp_path):
    config, _ = make_dataset(tmp_path, with_vectors=True)
    cfg = write_config(tmp_path, config)
    loaded = li.load_config([cfg])
    li.build_local_index(loaded, config['index_dir'], config_files=[cfg])
    assert qs._open_local_backend(loaded, [cfg], LOG).n == 4                 # same provider/model: opens
    other = dict(loaded, embeddings_model='text-embedding-3-large')
    with pytest.raises(ValueError) as e:
        qs._open_local_backend(other, [cfg], LOG)
    msg = str(e.value)
    for part in (config['index_dir'], "provider 'azure', model 'fake-model' (dim 8)",
                 "provider 'azure', model 'text-embedding-3-large'", '--build-index'):
        assert part in msg, part
    with pytest.raises(ValueError, match="provider 'openai'"):
        qs._open_local_backend(dict(loaded, embeddings_provider='openai'), [cfg], LOG)


def test_space_check_old_meta_without_provider_or_model_warns_once(tmp_path, caplog):
    # builder_version 2 recorded the raw config values, which may be null: no error, one WARNING
    config, _ = make_dataset(tmp_path, with_vectors=True)
    cfg = write_config(tmp_path, config)
    loaded = li.load_config([cfg])
    li.build_local_index(loaded, config['index_dir'], config_files=[cfg])
    meta_path = os.path.join(config['index_dir'], 'meta.json')
    meta = json.load(open(meta_path, encoding='utf-8'))
    meta['builder_version'] = 2
    meta['spaces']['text']['provider'] = meta['spaces']['text']['model'] = None
    json.dump(meta, open(meta_path, 'w', encoding='utf-8'))
    with caplog.at_level(logging.WARNING):
        backend = qs._open_local_backend(dict(loaded, embeddings_model='any-model'), [cfg], LOG)
    assert backend.n == 4
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and 'does not record the embedding' in r.message]
    assert len(warns) == 1
    for part in (config['index_dir'], 'builder_version 2', 'provider / model', '--build-index'):
        assert part in warns[0].message, part


def test_space_check_compares_the_recorded_fields_only(caplog):
    meta = {'builder_version': 2, 'spaces': {'text': {'provider': None, 'model': 'm', 'dim': 3}}}
    with caplog.at_level(logging.WARNING):
        li.check_embedding_space(meta, {'embeddings_provider': 'openai', 'embeddings_model': 'm'}, 'idx')
    assert sum('does not record the embedding provider' in r.message for r in caplog.records) == 1
    with pytest.raises(ValueError, match="model 'm'.*model 'other'"):
        li.check_embedding_space(meta, {'embeddings_provider': 'openai', 'embeddings_model': 'other'}, 'idx')
    meta['spaces']['text']['provider'] = 'Azure'                               # raw config value, any case
    caplog.clear()
    li.check_embedding_space(meta, {'embeddings_provider': 'azure', 'embeddings_model': 'm'}, 'idx')
    assert not caplog.records


@pytest.mark.parametrize('dataset', ['imdb', 'hm'])
def test_space_check_accepts_the_shipped_configs(dataset, caplog):
    # what index/{imdb,hm}/meta.json (builder_version 2) records: provider 'azure', model text-embedding-3-small
    meta = {'builder_version': 2, 'spaces': {'text': {'provider': 'azure', 'model': 'text-embedding-3-small', 'dim': 1536}}}
    config = li.load_config([f'configs/{dataset}/{dataset}.yaml'])
    with caplog.at_level(logging.WARNING):
        li.check_embedding_space(meta, config, f'index/{dataset}')
    assert not caplog.records


# query dimension check ────────────────────────────────────────────────────
def test_query_dimension_mismatch_is_a_clear_error(tmp_path):
    config, _ = make_dataset(tmp_path, with_vectors=True)
    li.build_local_index(config, config['index_dir'])
    b = LocalBackend(config['index_dir'])
    assert search_items_local(config, b, [1.0] + [0.0] * (DIM - 1), top_k=2)       # right dimension: works
    with pytest.raises(ValueError, match='dimension 1536, but the local index .* holds 8-dim vectors'):
        search_items_local(config, b, [0.1] * 1536, top_k=2)


# provider recorded in meta.json ───────────────────────────────────────────
def test_meta_records_the_resolved_provider(tmp_path):
    config, _ = make_dataset(tmp_path, with_vectors=True)
    del config['embeddings_provider']
    config['openai_provider'] = 'thetaedgecloud'
    meta = li.build_local_index(config, config['index_dir'])
    assert meta['spaces']['text']['provider'] == 'thetaedgecloud' and meta['spaces']['text']['dim'] == DIM


# verify_local_index.py --library-only ──────────────────────────────────────
def test_verify_library_only_on_a_computed_index(tmp_path, monkeypatch, capsys):
    import legacy_es_equivalence.verify_local_index as vli
    config, _ = make_dataset(tmp_path, n=5)
    build(config, FakeEmbed())
    cfg = write_config(tmp_path, config)
    monkeypatch.setattr(sys, 'argv', ['verify_local_index.py', '--config', cfg, '--library-only'])
    assert vli.main() == 0
    assert '== L2-normalised cache: True' in capsys.readouterr().out
