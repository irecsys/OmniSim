"""Local embedding models (embeddings_provider sentence-transformers, utils/local_embeddings.py): settings validation,
the vector-space id, the index texts and the query / document roles. No model is loaded: a fake SentenceTransformer
records what it is asked to encode."""
import hashlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

import utils.local_embeddings as le
import utils.retrieval.local_index as li
from utils.configurator import Config

REAL_API_EMBED_FN = li._api_embed_fn            # captured before conftest's autouse fixture replaces it
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# space id of configs/examples/imdb_embeddinggemma.yaml (model, revision, weights, prompt, template, dim, dtype)
EXPECTED_SPACE = 'st:f4bdf7516ae99471b05a1cec559319c30e4ddfe831a5845657265cf7acc79985'
CONFIGS = [os.path.join(ROOT, 'configs', 'system', 'system.yaml'), os.path.join(ROOT, 'configs', 'imdb', 'imdb.yaml'),
           os.path.join(ROOT, 'configs', 'examples', 'imdb_embeddinggemma.yaml')]


@pytest.mark.parametrize('key, value, words', [
    ('embeddings_dtype', 'float16', 'float16'),
    ('embeddings_revision', None, 'embeddings_revision'),
    ('embeddings_document_template', 'title: {name}', 'embeddings_document_template'),
    ('embeddings_dim', 0, 'embeddings_dim'),
    ('embeddings_batch_size', -1, 'embeddings_batch_size'),
    ('embeddings_modalities', 'image', 'embeddings_modalities'),
])
def test_invalid_settings_are_config_errors(key, value, words):
    cfg = Config(CONFIGS)
    cfg[key] = value
    with pytest.raises(le.ConfigError, match=words):
        le.embedding_spec(cfg)


def test_space_id_ignores_where_and_how_the_model_runs():
    cfg = Config(CONFIGS)
    before = li.space_key(cfg)
    cfg['embeddings_local_path'] = '/somewhere/else'
    cfg['embeddings_batch_size'] = 8
    cfg['embeddings_num_threads'] = 2
    assert li.space_key(cfg) == before
    cfg['embeddings_query_prompt'] = 'query: '
    assert li.space_key(cfg) != before


def test_example_config_space_id():
    cfg = Config(CONFIGS)
    assert le.is_local(cfg)
    assert li.space_key(cfg) == EXPECTED_SPACE
    ident = le.space_identity(cfg)
    assert ident['model'] == 'google/embeddinggemma-2' and ident['revision'].startswith('f9972ac2')
    assert ident['query_prompt'] == 'task: search result | query: '
    assert ident['document_template'] == 'title: {title} | text: {text}'


class FakeST:
    instances = []

    def __init__(self, source, **kwargs):
        self.source, self.kwargs, self.seen = source, kwargs, []
        FakeST.instances.append(self)

    def encode(self, texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True):
        self.seen.append(list(texts))
        out = []
        for t in texts:
            v = np.frombuffer(hashlib.sha256(t.encode('utf-8')).digest(), dtype=np.uint8).astype(float)[:8] + 1
            out.append(v / np.linalg.norm(v))
        return np.asarray(out)


@pytest.fixture
def fake_model(monkeypatch, tmp_path):
    FakeST.instances.clear()
    monkeypatch.setattr(le, '_import_sentence_transformers', lambda: FakeST)
    # torch is an optional package (requirements-local-embeddings.txt), not installed by CI: the client only needs
    # torch.float32 and torch.set_num_threads before the (fake) model is built
    import types
    fake_torch = types.SimpleNamespace(float32='float32', bfloat16='bfloat16', set_num_threads=lambda n: None)
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    cfg = Config(CONFIGS)
    cfg['embeddings_local_path'] = str(tmp_path)        # no model.safetensors there
    cfg['embeddings_weights_sha256'] = None             # so the weights check is skipped with the fake
    return cfg


def test_dialogue_calls_without_a_role_are_queries(fake_model):
    from utils import llm_calls
    from utils.utils import SentenceTransformerEmbeddingsClient, _build_embeddings_client
    cfg = fake_model
    client = _build_embeddings_client('sentence-transformers', cfg, None)      # as get_openai_clients builds it
    assert isinstance(client, SentenceTransformerEmbeddingsClient) and client.accepts_role
    # exactly as 2.0.0's simulator calls it (utils/simulator.py: query_embedding, profile_embedding; no role)
    vec = llm_calls.embed(client, 'query_embedding', model=cfg['embeddings_model'], input='a crime film').data[0].embedding
    assert FakeST.instances[-1].seen[-1] == ['task: search result | query: a crime film']
    assert len(vec) == 8 and abs(np.linalg.norm(vec) - 1) < 1e-12
    client.create(model=cfg['embeddings_model'], input=['x'], role='document')
    assert FakeST.instances[-1].seen[-1] == ['x']


def test_index_texts_use_the_document_template_and_role(fake_model):
    cfg = fake_model
    df = pd.DataFrame({'imdb_id': ['tt1', 'tt2'], 'title': ['Alien', None], 'genres': ['Horror', 'Drama'],
                       'overview': ['In space.', 'A story.'], 'release_year': [1979.0, 2001.0]})
    cfg['embedding_fields'] = ['genres', 'title', 'overview', 'release_year']
    texts = li.embedding_texts(cfg, df)
    assert texts == ['title: Alien | text: Horror In space. 1979.0', 'title: none | text: Drama A story. 2001.0']
    embed = REAL_API_EMBED_FN(cfg)
    vecs = embed(texts)
    assert FakeST.instances[-1].seen[-1] == texts          # no query prompt on documents
    assert len(vecs) == 2
    keys = li._cache_keys(cfg, df)[1]
    assert keys[0] == ('tt1', hashlib.sha256(texts[0].encode('utf-8')).hexdigest(), li.space_key(cfg))


def test_api_providers_keep_2_0_0_texts_and_keys():
    cfg = Config([os.path.join(ROOT, 'configs', 'system', 'system.yaml'), os.path.join(ROOT, 'configs', 'imdb', 'imdb.yaml')])
    df = pd.DataFrame({'imdb_id': ['tt1'], 'title': ['Alien'], 'genres': ['Horror'], 'overview': ['In space.']})
    cfg['embedding_fields'] = ['genres', 'title', 'overview']
    assert li.embedding_texts(cfg, df) == ['Horror Alien In space.']
    assert li.space_key(cfg) == 'text-embedding-3-small'
