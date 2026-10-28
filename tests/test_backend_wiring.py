"""
Unit tests for the local-backend wiring (ES branches removed). No ES, no API, no repo data except the
committed config files: indexes are built from toy CSVs in temporary directories.
"""
import json
import logging
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

import utils.quick_start as qs
import utils.simulator as sim
import utils.utils as uu
from utils.retrieval import local_search, ref_lookup
from utils.retrieval.backend import LocalBackend
from utils.retrieval.local_index import index_dir_for, load_config

LOG = logging.getLogger('test_backend_wiring')


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def write_dataset(tmp_path, with_vectors=True, id_col='item_id', ids=None):
    """Toy items CSV + a dataset config file; returns (config file path, config dict)."""
    ids = ids or ['tt1', 'tt2', 'tt3', 'tt4']
    rows = [{id_col: i, 'title': t, 'category': c, 'details': f'{t} details'}
            for i, t, c in zip(ids, ['Alpha', 'Beta', 'Gamma', 'Delta'], ['Drama', 'Comedy', 'Drama', 'Horror'])]
    df = pd.DataFrame(rows)
    if with_vectors:
        df['embedding_vector'] = [str(_unit([1, k, 0, 0]).tolist()) for k in range(len(rows))]
    ds = tmp_path / 'data' / 'toy'
    ds.mkdir(parents=True)
    df.to_csv(ds / 'items.csv', index=False)
    cfg = {'data_path': str(tmp_path / 'data'), 'dataset': 'toy', 'file_items': 'items.csv',
           'index_dir': str(tmp_path / 'index' / 'toy'),
           'col_itemid': id_col, 'col_title': 'title', 'col_category': 'category', 'col_details': 'details',
           'embedding_fields': ['title', 'category', 'details'], 'item_attributes': {'category': 'genre'},
           'lambda_bm25': 0.3, 'weight_es_score': 0.7, 'weight_user_taste_short': 0.3,
           'threshold_similarity': 0.0}
    if with_vectors:
        cfg['precomputed_embedding_col'] = 'embedding_vector'
    path = tmp_path / 'toy.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return str(path), load_config([str(path)])


@pytest.fixture
def toy(tmp_path):
    cfg_file, config = write_dataset(tmp_path)
    backend = qs._open_local_backend(config, [cfg_file], LOG)          # builds the missing index
    return cfg_file, config, backend


# ── search entry point ──────────────────────────────────────────────────────
def test_search_entry_point_equals_the_library(toy):
    _, config, b = toy
    q = _unit([1, 0.5, 0, 0]).tolist()
    kw = dict(top_k=2, query_text='drama', user_long_embedding=_unit([0, 1, 0, 0]).tolist())
    expected = local_search.search_items_local(config, b, q, **kw)
    assert uu.search_items_by_embedding(config, b, q, **kw) == expected
    assert expected                                                    # non-trivial comparison


# ── leftover retrieval_backend key (absent → local, local → deprecated, elasticsearch → error) ──
def test_system_yaml_has_no_retrieval_backend_or_es_fields():
    config = load_config([])
    assert 'retrieval_backend' not in config and 'es_host' not in config and 'es_index' not in config


def test_azure_embeddings_request_url(monkeypatch):
    # embeddings_endpoint is the resource root; the deployment comes from
    # embeddings_model and the api-version from embeddings_api_version. Mock transport, no network.
    try:
        import httpx2 as httpx
    except ImportError:
        import httpx
    seen = []
    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, json={'object': 'list', 'model': 'm', 'usage': {'prompt_tokens': 1, 'total_tokens': 1},
                                         'data': [{'object': 'embedding', 'index': 0, 'embedding': [0.0]}]})
    real = uu.AzureOpenAI
    monkeypatch.setattr(uu, 'AzureOpenAI', lambda **kw: real(**kw, http_client=httpx.Client(transport=httpx.MockTransport(handler))))
    config = load_config([])
    client = uu._build_embeddings_client('azure', config, 'fake-key')
    client.embeddings.create(model=config['embeddings_model'], input='hi')
    assert seen == ['https://your_resource.openai.azure.com/openai/deployments/text-embedding-3-small/embeddings'
                    '?api-version=2024-02-01']


def test_check_retrieval_backend(caplog):
    qs.check_retrieval_backend({}, LOG)                                # absent: local, silent
    assert not caplog.records
    with caplog.at_level(logging.WARNING):
        qs.check_retrieval_backend({'retrieval_backend': 'local'}, LOG)
    assert len(caplog.records) == 1 and 'deprecated' in caplog.records[0].getMessage()
    with pytest.raises(ValueError, match='no longer supported.*v1'):
        qs.check_retrieval_backend({'retrieval_backend': 'elasticsearch'}, LOG)
    with pytest.raises(ValueError, match="must be 'local'"):
        qs.check_retrieval_backend({'retrieval_backend': 'faiss'}, LOG)


@pytest.mark.parametrize('backend', [None, 'local', 'elasticsearch'])
def test_build_index_retrieval_backend(tmp_path, monkeypatch, caplog, backend):
    import run
    import utils.retrieval.local_index as li
    calls = []
    monkeypatch.setattr(li, 'build_local_index', lambda cfg, out, **kw: calls.append(out))
    cfg = tmp_path / 'c.yaml'
    cfg.write_text(yaml.safe_dump({'dataset': 'toy', **({'retrieval_backend': backend} if backend else {})}),
                   encoding='utf-8')
    if backend == 'elasticsearch':
        with pytest.raises(ValueError, match='v1'):
            run.build_index([str(cfg)])
        assert calls == []
        return
    with caplog.at_level(logging.WARNING):
        run.build_index([str(cfg)])
    assert calls == ['index/toy']
    deprecated = [r for r in caplog.records if 'deprecated' in r.getMessage()]
    assert len(deprecated) == (1 if backend == 'local' else 0)


# ── reference-title lookup (simulator.py) ───────────────────────────────────
def test_reference_lookup_dispatches_to_local(toy):
    _, config, b = toy
    titles = ['Alpha', 'Gamma', 'not a title at all']
    expected = ref_lookup.lookup_reference_embeddings(b, config, titles)
    assert len(expected) == 2
    assert sim._lookup_reference_embeddings(b, config, titles) == expected
    assert sim._lookup_reference_embeddings(b, {}, titles) == expected      # default backend


# ── user_guest sampling (quick_start.py) ────────────────────────────────────
def test_sample_guest_items_local_structure(tmp_path):
    cfg_file, config = write_dataset(tmp_path, id_col='pid', ids=[101, 102, 103, 104])
    b = qs._open_local_backend(config, [cfg_file], LOG)
    rows, pool = qs._sample_guest_items(b, config, 'pid', 600)
    assert len(rows) == 4 and sorted(pool) == ['101', '102', '103', '104']
    assert pool == [str(r['pid']) for r in rows] and all(isinstance(r['pid'], int) for r in rows)
    assert all(list(r.keys()) == b.columns + ['embedding'] for r in rows)
    rows3, pool3 = qs._sample_guest_items(b, config, 'pid', 3)
    assert len(rows3) == 3 and len(set(pool3)) == 3


# ── local backend opening / building (quick_start.py) ───────────────────────
def test_open_local_backend_uses_existing_index(toy, monkeypatch):
    cfg_file, config, b = toy
    import utils.retrieval.local_index as li
    monkeypatch.setattr(li, 'build_local_index', lambda *a, **k: pytest.fail('index exists, must not rebuild'))
    b2 = qs._open_local_backend(config, [cfg_file], LOG)
    assert isinstance(b2, LocalBackend) and b2.item_ids == b.item_ids and b2.index_dir == config['index_dir']


def test_open_local_backend_build_failure_message(tmp_path):
    # The automatic build never calls the embedding API (conftest fails the test if
    # the real client is reached); it stops before any request and points to --build-index.
    from utils.retrieval.local_index import EmbeddingsNeeded
    cfg_file, config = write_dataset(tmp_path, with_vectors=False)    # no precomputed column, no cache
    with pytest.raises(RuntimeError) as e:
        qs._open_local_backend(config, [cfg_file], LOG)
    msg = str(e.value)
    for part in (config['index_dir'], "'toy'", '4 of 4 items need an embedding', 'this build allows 0',
                 'precomputed_embedding_col', 'the automatic build never does that', 'CSV is not modified',
                 f'python run.py --config "{cfg_file}" --build-index',
                 f'python scripts/build_local_index.py --config "{cfg_file}" --dry-run'):
        assert part in msg, part
    assert isinstance(e.value.__cause__, EmbeddingsNeeded)
    assert not (tmp_path / 'index' / 'toy' / 'meta.json').exists()
    assert not (tmp_path / 'index' / 'toy' / 'embedding_cache.db').exists()


# ── index_dir_for ───────────────────────────────────────────────────────────
def test_index_dir_for():
    assert index_dir_for({'dataset': 'imdb'}) == 'index/imdb'
    assert index_dir_for({'dataset': 'x', 'index_dir': 'somewhere/else'}) == 'somewhere/else'
    assert index_dir_for({'dataset': 'x', 'index_dir': None}) == 'index/x'


def test_index_dir_of_the_three_datasets():
    imdb = ['configs/imdb/imdb.yaml']
    assert index_dir_for(load_config(imdb)) == 'index/imdb'
    assert index_dir_for(load_config(imdb + ['data/imdb_fixture/imdb_fixture.yaml'])) == 'index/imdb'
    assert index_dir_for(load_config(['configs/hm/hm.yaml'])) == 'index/hm'


def test_dataset_configs_set_precomputed_embedding_col():
    # Both CSVs ship their vectors; building their indexes never calls an API
    for cfg in (['configs/imdb/imdb.yaml'], ['configs/hm/hm.yaml']):
        assert load_config(cfg)['precomputed_embedding_col'] == 'embedding_vector'


# ── run.py --build-index ────────────────────────────────────────────────────
def test_build_index_builds_the_local_index(tmp_path, monkeypatch):
    import run
    import utils.retrieval.local_index as li
    calls = []
    monkeypatch.setattr(li, 'build_local_index', lambda cfg, out, **kw: calls.append(('local', out, kw)))
    cfg = tmp_path / 'c.yaml'
    cfg.write_text(yaml.safe_dump({'dataset': 'toy'}), encoding='utf-8')
    files = [str(cfg)]
    run.build_index(files)
    assert calls == [('local', 'index/toy', {'config_files': files})]


# ── run_simulation hands the backend to simulate_* (es -> backend) ────────
class _FakeChat:
    """Stands in for the OpenAI chat client: run_simulation only sends one "Hello!" probe."""
    class chat:
        class completions:
            @staticmethod
            def create(**kw):
                return None


@pytest.mark.parametrize('mode, retrieval_backend', [
    ('free', None), ('static', None), ('adaptive', None), ('static', 'local'), ('static', 'elasticsearch')])
def test_run_simulation_passes_backend(tmp_path, monkeypatch, caplog, mode, retrieval_backend):
    """All four strategies build their task dicts with keys that bind to the real simulate_* signature
    (a mismatch would only be a warning swallowed by _run_parallel). Offline: fake clients, no profile build."""
    import inspect
    cfg_file, _ = write_dataset(tmp_path)
    ds = tmp_path / 'data' / 'toy'
    pd.DataFrame({'user_id': ['u1', 'u2'], 'age': [30, 40]}).to_csv(ds / 'users.csv', index=False)
    pd.DataFrame({'item_id': ['tt1', 'tt2']}).to_csv(tmp_path / 'items_in.csv', index=False)
    pd.DataFrame({'user_id': ['u1', 'u2'], 'item_id': ['tt3', 'tt4']}).to_csv(tmp_path / 'pairs.csv', index=False)
    pd.DataFrame({'user_id': ['u1', 'u2']}).to_csv(tmp_path / 'users_in.csv', index=False)
    overrides = {'data_path': str(tmp_path / 'data') + '/', 'log_path_outputs': str(tmp_path / 'logs') + '/',
                 'outputs_folder': str(tmp_path / 'chats') + '/', 'file_users': 'users.csv',
                 'file_interactions': None, 'guestuser_randomitem_records': 2, 'chats_per_entry': 1,
                 'input_items_file': str(tmp_path / 'items_in.csv'), 'input_pairs_file': str(tmp_path / 'pairs.csv'),
                 'input_users_file': str(tmp_path / 'users_in.csv'), 'num_workers': 1}
    if retrieval_backend:
        overrides['retrieval_backend'] = retrieval_backend
    over_file = tmp_path / 'over.yaml'
    over_file.write_text(yaml.safe_dump(overrides), encoding='utf-8')

    monkeypatch.setattr(qs, 'get_openai_clients', lambda config: (_FakeChat(), object()))
    monkeypatch.setattr(qs, '_needs_profile_rebuild', lambda config, data_path: False)
    real = inspect.signature(qs.SIMULATION_FUNCS[mode])
    calls = []
    monkeypatch.setitem(qs.SIMULATION_FUNCS, mode, lambda **params: calls.append(real.bind(**params).arguments))

    if retrieval_backend == 'elasticsearch':
        with pytest.raises(ValueError, match='v1'):
            qs.run_simulation([cfg_file, str(over_file)], mode_override=mode)
        assert calls == []
        return
    with caplog.at_level(logging.WARNING):
        qs.run_simulation([cfg_file, str(over_file)], mode_override=mode)
    assert len(calls) == 8                                  # items 2 + item_list 2 + pairs 2 + user_guest 2
    assert all(isinstance(c['backend'], LocalBackend) for c in calls)
    assert len({id(c['backend']) for c in calls}) == 1
    assert {'tt1', 'tt2', 'tt3', 'tt4'} <= {str(c['target_item_id']) for c in calls}   # item_list + pairs
    deprecated = [r for r in caplog.records if 'deprecated' in r.getMessage()]
    assert len(deprecated) == (1 if retrieval_backend == 'local' else 0)


# ── the library works with elasticsearch impossible to import ───────────────
def test_local_mode_works_with_elasticsearch_import_blocked(toy, tmp_path):
    cfg_file, config, _ = toy
    code = f"""
import importlib.abc, json, logging, sys
class BlockES(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in ('elasticsearch', 'elastic_transport'):
            raise ImportError('blocked: ' + name)
sys.meta_path.insert(0, BlockES())
import run, utils.quick_start as qs, utils.simulator as sim, utils.utils as uu
import legacy_es_equivalence.replay_fixtures
from utils.retrieval.local_index import load_config
config = load_config({json.dumps([cfg_file])})
run.build_index({json.dumps([cfg_file])})
b = qs._open_local_backend(config, {json.dumps([cfg_file])}, logging.getLogger())
out = uu.search_items_by_embedding(config, b, [1.0, 0.2, 0.0, 0.0], 2, query_text='drama')
refs = sim._lookup_reference_embeddings(b, config, ['Alpha'])
rows, pool = qs._sample_guest_items(b, config, config['col_itemid'], 10)
assert out and refs and rows
print(json.dumps(sorted(m for m in sys.modules if m.split('.')[0] in ('elasticsearch', 'elastic_transport'))))
"""
    res = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert json.loads(res.stdout.strip().splitlines()[-1]) == []
