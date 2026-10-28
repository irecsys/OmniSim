"""
Build and load the local retrieval index (v2 replacement for Elasticsearch).

The build follows v1's scripts/build_es_index.py (the ES builder of the v1 Elasticsearch
implementation, git tag v1.0) step by step so that the stored
fields and the BM25 text are identical to what v1 put into ES:

    read file_items_for_index (default file_items) → object columns fillna('')
    → vectors: the precomputed_embedding_col column if the CSV has it (unparseable rows dropped),
      otherwise computed from embedding_fields with embeddings_provider (v1 :91-129)
    → build bm25_details → store fields with the same values/types as ES _source

Computed vectors are never written back to the CSV (v1 :131-135 did). They are kept
in embedding_cache.db in the index directory, keyed by (item id, sha256 of the embedding text,
embeddings_model), and reused by later builds. Only an explicit build may call the embedding API
(max_embed_items=None or N); max_embed_items=0 reads the cache only.

Layout of index/{dataset}/:

    items.db              SQLite: `items` table (row_idx = CSV order = v1's ES insert order)
                          + `items_fts` (FTS5 over bm25_details, external content; tokenizer from
                          config `bm25_tokenizer`, default "unicode61 remove_diacritics 0"; use
                          "trigram" for CJK data, which has no spaces between words)
    vectors/text.npy      float32, every vector L2-normalised (standard cosine)
    meta.json             dims, embedding provider/model, column types, row_idx↔item_id, config snapshot
    embedding_cache.db    only when vectors were computed: the raw vectors (kept across rebuilds)

The library uses standard components only. Elasticsearch's own vector handling (conditional
normalisation W1, raw _source vectors) is reproduced by the verification tools from the CSV
(in legacy_es_equivalence/), not stored here.
"""
import ast
import datetime
import hashlib
import json
import logging
import math
import os
import sqlite3
import sys

import numpy as np
import pandas as pd
import yaml

from utils.utils import build_embedding_texts, resolve_embeddings_provider

BUILDER_VERSION = 3
BM25_DETAILS_FIELD = "bm25_details"
CACHE_FILE = 'embedding_cache.db'
EMBED_BATCH = 100                     # v1 build_es_index.py: 100 texts per API request
DEFAULT_FTS_TOKENIZER = "unicode61 remove_diacritics 0"
FTS_TOKENIZERS = ('unicode61', 'ascii', 'porter', 'trigram')    # SQLite's built-in FTS5 tokenizers
SQL_TYPES = {'str': 'TEXT', 'int': 'INTEGER', 'float': 'REAL', 'bool': 'INTEGER'}


# ── Config ─────────────────────────────────────────────────────────────────
def load_config(config_file_list):
    """Same merge as v1 build_es_index.py: system.yaml first, then each file in order."""
    config = {}
    for path in ['configs/system/system.yaml'] + list(config_file_list):
        with open(path, 'r', encoding='utf-8') as f:
            config.update(yaml.safe_load(f) or {})
    return config


def index_dir_for(config):
    """Index directory of a dataset: config `index_dir` if set, else index/{dataset}."""
    return config.get('index_dir') or os.path.join('index', config['dataset'])


# ── Field values (same as v1 build_es_index.py:_clean) ─────────────────────
def _clean(val, is_numeric=False):
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    if is_numeric:
        try:
            return float(val)
        except (ValueError, TypeError):
            return None
    return val


def _to_native(val):
    """numpy scalar → Python scalar, so values match what v1's ES returned after the JSON round trip."""
    return val.item() if isinstance(val, np.generic) else val


def _column_type(series, is_numeric):
    if is_numeric:
        return 'float'
    if pd.api.types.is_bool_dtype(series):
        return 'bool'
    if pd.api.types.is_integer_dtype(series):
        return 'int'
    if pd.api.types.is_float_dtype(series):
        return 'float'
    bad = [v for v in series if not isinstance(v, str)]
    if bad:
        raise ValueError(f"column '{series.name}' is object dtype but holds non-str values, e.g. {bad[0]!r}")
    return 'str'


def _restore(val, typ):
    if val is None:
        return None
    if typ == 'bool':
        return bool(val)
    if typ == 'int':
        return int(val)
    if typ == 'float':
        return float(val)
    return val


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ── SQLite FTS5 availability ───────────────────────────────────────────────
_FTS5_OK = False
TRIGRAM_MIN_SQLITE = (3, 34, 0)


def _probe_fts5():
    con = sqlite3.connect(':memory:')
    try:
        con.execute('CREATE VIRTUAL TABLE temp.fts5_probe USING fts5(x)')
    finally:
        con.close()


def check_fts5():
    """Raise a clear error if Python's sqlite3 has no FTS5 (BM25 needs it). Checked once per process."""
    global _FTS5_OK
    if _FTS5_OK:
        return
    try:
        _probe_fts5()
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            f"OmniSim's local index needs SQLite's FTS5 extension (BM25), but the sqlite3 module of this Python "
            f"({sys.executable}, SQLite {sqlite3.sqlite_version}) does not provide it: {e}. "
            "Use a Python whose sqlite3 includes FTS5 — the python.org installers, conda / Miniconda and the Python "
            "of the major Linux distributions all do. Check with: "
            "python -c \"import sqlite3; sqlite3.connect(':memory:').execute('create virtual table t using fts5(x)')\"") from e
    _FTS5_OK = True


def fts_tokenizer(config):
    """The FTS5 tokenizer spec from config `bm25_tokenizer` (default unicode61 remove_diacritics 0)."""
    spec = str(config.get('bm25_tokenizer') or DEFAULT_FTS_TOKENIZER).strip()
    if spec.split()[0] not in FTS_TOKENIZERS or '"' in spec:
        raise ValueError(f"bm25_tokenizer must be one of the FTS5 tokenizers {FTS_TOKENIZERS} "
                         f"(with options), got {spec!r}")
    if spec.split()[0] == 'trigram' and sqlite3.sqlite_version_info < TRIGRAM_MIN_SQLITE:
        raise RuntimeError(
            f"bm25_tokenizer 'trigram' needs SQLite >= {'.'.join(map(str, TRIGRAM_MIN_SQLITE))}, but this Python "
            f"({sys.executable}) has SQLite {sqlite3.sqlite_version}. Use a newer Python / SQLite, or another tokenizer.")
    return spec


class EmbeddingsNeeded(ValueError):
    """Rows have no precomputed vector and no cached one, and this build may not call the API."""
    def __init__(self, plan, max_embed_items):
        self.plan = plan
        super().__init__(
            f"{plan['n_embed']} of {plan['n_rows']} items need an embedding from the API "
            f"({plan['n_requests']} requests, ~{plan['est_tokens']} tokens, provider {plan['provider']}, "
            f"model {plan['model']}); this build allows {max_embed_items}")


def _csv_df(config):
    data_path = os.path.join(config.get('data_path', 'data'), config['dataset'])
    items_file = os.path.join(data_path, config.get('file_items_for_index', config['file_items']))
    df = pd.read_csv(items_file, low_memory=False)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna('')
    return df, items_file


def _emb_col(config, precomputed_col, df):
    """(column, source) when the CSV has the precomputed column (v1 :68-69), else (None, None)."""
    emb_col, emb_col_source = config.get('precomputed_embedding_col', ''), 'config'
    if not emb_col:
        emb_col, emb_col_source = precomputed_col, 'cli --precomputed-col'
    if emb_col and emb_col in df.columns:
        return emb_col, emb_col_source
    return None, None


def embedding_texts(config, df):
    """The text each item is embedded from: embedding_fields, default title + category + details (v1 :96-97)."""
    fields = config.get('embedding_fields', [config['col_title'], config['col_category'], config['col_details']])
    return build_embedding_texts(df, fields)


def _cache_keys(config, df):
    model = str(config.get('embeddings_model') or '')
    texts = embedding_texts(config, df)
    keys = [(str(i), hashlib.sha256(t.encode('utf-8')).hexdigest(), model)
            for i, t in zip(df[config['col_itemid']], texts)]
    return texts, keys


def _open_cache(cache_path):
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    con = sqlite3.connect(cache_path)
    con.execute('CREATE TABLE IF NOT EXISTS embeddings (item_id TEXT NOT NULL, text_sha256 TEXT NOT NULL, '
                'model TEXT NOT NULL, dim INTEGER NOT NULL, vector TEXT NOT NULL, created_at TEXT NOT NULL, '
                'PRIMARY KEY (item_id, text_sha256, model))')
    return con


def _cache_lookup(cache_path, keys):
    """{key: vector} for the keys found in the cache (an absent cache file is empty)."""
    if not cache_path or not os.path.exists(cache_path):
        return {}
    found = {}
    con = sqlite3.connect(f"file:{cache_path}?mode=ro", uri=True)
    try:
        for k in set(keys):
            row = con.execute('SELECT vector FROM embeddings WHERE item_id=? AND text_sha256=? AND model=?', k).fetchone()
            if row:
                found[k] = json.loads(row[0])
    finally:
        con.close()
    return found


def plan_embeddings(config, df, cache_path):
    """What computing the vectors would take: rows, cache hits, rows to embed, requests, ~tokens."""
    texts, keys = _cache_keys(config, df)
    found = _cache_lookup(cache_path, keys)
    todo = [i for i, k in enumerate(keys) if k not in found]
    return {'n_rows': len(keys), 'n_cache_hits': len(keys) - len(todo), 'n_embed': len(todo),
            'n_requests': math.ceil(len(todo) / EMBED_BATCH),
            'est_tokens': sum(len(texts[i]) for i in todo) // 4,
            'provider': resolve_embeddings_provider(config), 'model': config.get('embeddings_model')}


def _api_embed_fn(config):
    """texts → vectors with the same client the dialogue flow uses for query embeddings."""
    from dotenv import load_dotenv
    from utils.utils import _build_embeddings_client
    load_dotenv()
    provider = resolve_embeddings_provider(config)
    key = os.getenv('AZURE_KEY') if provider == 'azure' else os.getenv('OPENAI_KEY')
    client = _build_embeddings_client(provider, config, key)
    model = config.get('embeddings_model')

    def embed(texts):
        return [d.embedding for d in client.embeddings.create(model=model, input=texts).data]
    return embed


def computed_vectors(config, df, cache_path, embed_fn=None, max_embed_items=0):
    """Raw vectors (list per row) from the cache, embedding the missing rows first.

    max_embed_items: None = no limit, N = at most N rows may be embedded (checked before the first
    request; EmbeddingsNeeded otherwise), 0 = cache only. Each batch is cached as soon as it returns.
    """
    plan = plan_embeddings(config, df, cache_path)
    if max_embed_items is not None and plan['n_embed'] > max_embed_items:
        raise EmbeddingsNeeded(plan, max_embed_items)
    texts, keys = _cache_keys(config, df)
    found = _cache_lookup(cache_path, keys)
    todo = [i for i, k in enumerate(keys) if k not in found]
    if todo:
        embed_fn = embed_fn or _api_embed_fn(config)
        con = _open_cache(cache_path)
        try:
            for b in range(0, len(todo), EMBED_BATCH):
                batch = todo[b:b + EMBED_BATCH]
                vecs = embed_fn([texts[i] for i in batch])
                if len(vecs) != len(batch) or any(
                        not isinstance(v, (list, tuple)) or not v or not all(isinstance(x, (int, float)) for x in v)
                        for v in vecs):
                    raise ValueError(f"embeddings provider {plan['provider']} returned no usable vectors "
                                     f"(got {type(vecs[0]).__name__ if vecs else 'nothing'})")
                now = datetime.datetime.now().isoformat(timespec='seconds')
                for i, v in zip(batch, vecs):
                    v = [float(x) for x in v]
                    found[keys[i]] = v
                    con.execute('INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?, ?)',
                                keys[i] + (len(v), json.dumps(v), now))
                con.commit()
        finally:
            con.close()
    return [found[k] for k in keys], plan


def read_items(config, precomputed_col=None, cache_path=None, embed_fn=None, max_embed_items=0):
    """Read the items CSV as v1 build_es_index.py does.

    Returns (df, raw vectors float64, info). With the precomputed column, rows whose vector does not
    parse are dropped (v1 :68-90). Without it, the vectors come from the cache at cache_path, and
    rows missing there are embedded only as far as max_embed_items allows (computed_vectors).
    """
    df, items_file = _csv_df(config)
    n_csv_records = len(df)

    emb_col, emb_col_source = _emb_col(config, precomputed_col, df)
    if emb_col is None:
        vectors, plan = computed_vectors(config, df, cache_path, embed_fn=embed_fn, max_embed_items=max_embed_items)
        raw = np.array(vectors, dtype=np.float64)
        if raw.ndim != 2:
            raise ValueError("embeddings do not all have the same dimension")
        info = {'items_file': items_file, 'n_csv_records': n_csv_records, 'dropped_rows': [],
                'emb_col': None, 'emb_col_source': None, 'embeddings': 'computed',
                'n_cache_hits': plan['n_cache_hits'], 'n_embedded': plan['n_embed']}
        return df, raw, info

    def _parse_emb(val):
        if isinstance(val, list):
            return [float(x) for x in val]
        if isinstance(val, str):
            try:
                return [float(x) for x in ast.literal_eval(val)]
            except Exception:
                return None
        return None
    embeddings = df[emb_col].apply(_parse_emb).tolist()
    valid_mask = [e is not None for e in embeddings]
    dropped_rows = [i for i, v in enumerate(valid_mask) if not v]
    if dropped_rows:
        df = df[valid_mask].reset_index(drop=True)
        embeddings = [e for e, v in zip(embeddings, valid_mask) if v]
    raw = np.array(embeddings, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("embeddings do not all have the same dimension")
    info = {'items_file': items_file, 'n_csv_records': n_csv_records, 'dropped_rows': dropped_rows,
            'emb_col': emb_col, 'emb_col_source': emb_col_source, 'embeddings': 'precomputed_col',
            'n_cache_hits': 0, 'n_embedded': 0}
    return df, raw, info


def l2_normalised(raw):
    """Row-wise L2 normalisation (float64), stored as float32. Zero vectors are rejected."""
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError(f"zero embedding vector at rows {np.flatnonzero(norms[:, 0] == 0).tolist()[:10]}")
    return (raw / norms).astype(np.float32)


# ── Build ──────────────────────────────────────────────────────────────────
def build_local_index(config, out_dir, precomputed_col=None, config_files=None, embed_fn=None,
                      max_embed_items=None):
    """Build index/{dataset}/ from the items CSV. Returns the meta dict.

    precomputed_col (scripts/build_local_index.py --precomputed-col) is used only when the
    config has no precomputed_embedding_col; where it came from is recorded in meta.json.
    Without a precomputed column the vectors are computed (cached in out_dir/embedding_cache.db);
    max_embed_items limits how many rows may be sent to the API (None = no limit, 0 = cache only),
    embed_fn replaces the API client (tests).
    """
    check_fts5()                                   # before anything is read or written
    col_itemid   = config['col_itemid']
    col_title    = config['col_title']
    col_details  = config['col_details']
    col_category = config['col_category']

    tokenizer = fts_tokenizer(config)
    cache_path = os.path.join(out_dir, CACHE_FILE)
    df, raw, info = read_items(config, precomputed_col, cache_path=cache_path, embed_fn=embed_fn,
                               max_embed_items=max_embed_items)
    items_file = info['items_file']
    stored = l2_normalised(raw)

    # bm25_details — verbatim from v1 build_es_index.py:145-152
    agg_fields = (
        config.get('embedding_fields')
        or ([col_title, col_category, col_details] + list(config.get('item_attributes', {}).keys()))
    )
    seen_agg = set()
    agg_fields = [f for f in agg_fields if f in df.columns and not (f in seen_agg or seen_agg.add(f))]
    df[BM25_DETAILS_FIELD] = build_embedding_texts(df, agg_fields)

    # Stored fields — same key set and order as the ES _source (minus `embedding`)
    item_attributes = config.get('item_attributes', {})
    numeric_fields  = config.get('numeric_fields', [])
    keyword_fields  = config.get('keyword_fields', [])
    all_store_fields = (
        [col_itemid, col_title, col_details, col_category, BM25_DETAILS_FIELD]
        + list(item_attributes.keys()) + numeric_fields + keyword_fields
    )
    columns = [k for k in dict.fromkeys(all_store_fields) if k in df.columns]
    if col_itemid not in columns:
        raise ValueError(f"id column '{col_itemid}' not found in {items_file}")
    col_types = {k: _column_type(df[k], k in numeric_fields) for k in columns}

    rows = []
    for i, rec in enumerate(df[columns].itertuples(index=False, name=None)):
        vals = [_to_native(_clean(v, is_numeric=(k in numeric_fields))) for k, v in zip(columns, rec)]
        rows.append([i] + vals)
    if any(r[1 + columns.index(col_itemid)] is None for r in rows):
        raise ValueError(f"'{col_itemid}' has missing values")

    # Write
    os.makedirs(os.path.join(out_dir, 'vectors'), exist_ok=True)
    db_path = os.path.join(out_dir, 'items.db')
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    try:
        col_defs = ', '.join(
            f'"{k}" {SQL_TYPES[col_types[k]]}' + (' NOT NULL' if k == col_itemid else '') for k in columns)
        con.execute(f'CREATE TABLE items (row_idx INTEGER PRIMARY KEY, {col_defs})')
        placeholders = ', '.join('?' * (len(columns) + 1))
        con.executemany(f'INSERT INTO items VALUES ({placeholders})', rows)
        con.execute(
            f'CREATE VIRTUAL TABLE items_fts USING fts5("{BM25_DETAILS_FIELD}", '
            f"content='items', content_rowid='row_idx', tokenize=\"{tokenizer}\")")
        con.execute("INSERT INTO items_fts(items_fts) VALUES('rebuild')")
        con.commit()
    finally:
        con.close()

    np.save(os.path.join(out_dir, 'vectors', 'text.npy'), stored)
    meta = {
        'builder_version': BUILDER_VERSION,
        'built_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'dataset': config['dataset'],
        'source': {
            'items_file': items_file,
            'sha256': _sha256(items_file),
            'n_csv_records': info['n_csv_records'],
            'dropped_unparseable_rows': info['dropped_rows'],
            'precomputed_embedding_col': info['emb_col'],
            'precomputed_embedding_col_from': info['emb_col_source'],
            'embeddings': info['embeddings'],
            'embedding_cache': CACHE_FILE if info['embeddings'] == 'computed' else None,
            'n_cache_hits': info['n_cache_hits'],
            'n_embedded': info['n_embedded'],
        },
        'n_items': len(rows),
        'col_itemid': col_itemid,
        'col_title': col_title,
        'col_details': col_details,
        'col_category': col_category,
        'columns': [{'name': k, 'type': col_types[k]} for k in columns],
        'item_ids': [r[1 + columns.index(col_itemid)] for r in rows],   # index = row_idx
        'bm25': {
            'field': BM25_DETAILS_FIELD,
            'agg_fields': agg_fields,
            'fts_table': 'items_fts',
            'fts_tokenizer': tokenizer,
        },
        'spaces': {
            'text': {
                'dim': int(raw.shape[1]),
                'file': 'vectors/text.npy',
                'dtype': 'float32',
                'model': config.get('embeddings_model'),
                'provider': resolve_embeddings_provider(config),
                'similarity': 'cosine',
                'normalization': 'l2 (float64), stored as float32',
            },
        },
        'config_files': list(config_files or []),
        'config': config,
    }
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    return meta


# ── Load ───────────────────────────────────────────────────────────────────
def load_local_index(index_dir):
    """Load an index fully into memory.

    Returns a dict with meta, items (list of dicts in row_idx order, values with
    the original _source types restored), vectors (float32, L2-normalised) and db_path.
    """
    with open(os.path.join(index_dir, 'meta.json'), encoding='utf-8') as f:
        meta = json.load(f)
    db_path = os.path.join(index_dir, 'items.db')
    names = [c['name'] for c in meta['columns']]
    types = [c['type'] for c in meta['columns']]
    con = sqlite3.connect(db_path)
    try:
        cols = ', '.join(f'"{n}"' for n in names)
        rows = con.execute(f'SELECT row_idx, {cols} FROM items ORDER BY row_idx').fetchall()
    finally:
        con.close()
    items = []
    for i, r in enumerate(rows):
        if r[0] != i:
            raise ValueError(f"items.row_idx is not contiguous at {i}")
        items.append({n: _restore(v, t) for n, t, v in zip(names, types, r[1:])})
    space = meta['spaces']['text']
    vectors = np.load(os.path.join(index_dir, space['file']))
    if not (len(items) == vectors.shape[0] == meta['n_items']):
        raise ValueError("items / vectors / meta row counts disagree")
    return {'meta': meta, 'items': items, 'vectors': vectors, 'db_path': db_path}


# ── Embedding space check ──────────────────────────────────────────────────
def check_embedding_space(meta, config, index_dir, logger=None):
    """Raise if the index was built with another embedding provider/model than the config uses.

    The query vectors come from the config's provider/model; vectors of another model are not
    comparable (and usually have another dimension). Indexes built before builder_version 3 recorded
    the raw config values, which may be missing (null): those fields cannot be checked, so they are
    skipped with one WARNING suggesting a rebuild; the recorded fields are still compared.
    """
    space = meta['spaces']['text']
    built = {'provider': str(space['provider']).lower() if space.get('provider') else None,
             'model': space.get('model') or None}
    current = {'provider': resolve_embeddings_provider(config), 'model': config.get('embeddings_model')}
    missing = [k for k, v in built.items() if v is None]
    if missing:
        (logger or logging.getLogger(__name__)).warning(
            f"The local index '{index_dir}' (builder_version {meta.get('builder_version')}) does not record the "
            f"embedding {' / '.join(missing)} it was built with, so it cannot be checked against the config "
            f"(provider '{current['provider']}', model '{current['model']}'). Rebuild it with "
            f"`python run.py --config \"<your config files>\" --build-index` to record them.")
    if any(built[k] is not None and built[k] != current[k] for k in built):
        raise ValueError(
            f"The local index '{index_dir}' was built with embeddings provider '{built['provider']}', model "
            f"'{built['model']}' (dim {space.get('dim')}), but the config now uses provider '{current['provider']}', "
            f"model '{current['model']}'. Query and item vectors must come from the same model. Either set "
            f"embeddings_provider / embeddings_model back to the values above, or rebuild the index for the new "
            f"model: delete '{index_dir}' and run `python run.py --config \"<your config files>\" --build-index` "
            f"(if the vectors come from a precomputed CSV column, that column must hold vectors of the new model).")
