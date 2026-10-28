"""
verify_local_index.py — offline checks of a local index (scripts/build_local_index.py) against the items CSV
and the es_golden fixture. No ES, no LLM or embedding API.

Checks (numbered as in the original migration check list; the missing numbers needed a live ES):
  4   library index vectors: float32, every row == the L2-normalised CSV vector (unit);
      verification-tool vectors (legacy_es_equivalence/es_compat/vectors.es_vectors, built from the CSV):
      W1 rows are CSV float32 or unit, raw == CSV float64
  8   verification-tool raw vectors == es_golden ref_vectors bit for bit (the vectors v1's reference-title
      lookup returned from ES _source.embedding), when the fixture has hits for this index

The checks against the temporary Elasticsearch index (2 counts, 3 fields, 5 scores, 6 FTS5 hit sets,
8 raw vectors vs ES _source, 9 bm25_details, 11 tokenizer diagnostic) were removed together with the ES
connection. Check 4 is unchanged (same code, same output).

--library-only: only the library part of check 4 — text.npy == L2(source vectors), where the
source vectors are the CSV column, or for an index whose vectors were computed, the raw vectors in its
embedding_cache.db. For indexes that are not one of the two migration datasets (no W1 / es_golden data).

Usage:
    python legacy_es_equivalence/verify_local_index.py --config "configs/system/system.yaml configs/imdb/imdb.yaml"
    python legacy_es_equivalence/verify_local_index.py --config "configs/system/system.yaml configs/hm/hm.yaml"
    python legacy_es_equivalence/verify_local_index.py --config "... my.yaml" --index <dir> --library-only

Exit code is 0 only when every check passes.
"""
import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_es_equivalence.es_compat.vectors import es_vectors
from utils.retrieval.backend import LocalBackend
from utils.retrieval.local_index import index_dir_for, load_config, load_local_index, read_items

REF_LOOKUPS = 'legacy_es_equivalence/recordings/direct_call/ref_lookups.jsonl'
FIXTURE_INDEX = {'index/imdb': 'imdb-2025', 'index/hm': 'hm-2025'}   # es_index of the es_golden records


def check_vectors(ctx):
    meta = ctx['idx']['meta']
    V, W, R = ctx['idx']['vectors'], ctx['w1'], ctx['raw']
    df = pd.read_csv(ctx['items_file'], low_memory=False)
    csv = np.array([[float(x) for x in ast.literal_eval(s)] for s in df[meta['source']['precomputed_embedding_col']]],
                   dtype=np.float64)
    dim = meta['spaces']['text']['dim']
    ok_shape = V.shape == W.shape == R.shape == (len(ctx['idx']['items']), dim) and V.dtype == np.float32
    lib_exact = bool(np.array_equal(V, (csv / np.linalg.norm(csv, axis=1, keepdims=True)).astype(np.float32)))
    lib_dev = float(np.abs(np.linalg.norm(V.astype(np.float64), axis=1) - 1.0).max())
    raw32 = csv.astype(np.float32)
    is_raw = np.all(W.astype(np.float32) == raw32, axis=1)
    unit_dev = float(np.abs(np.linalg.norm(W[~is_raw], axis=1) - 1.0).max()) if (~is_raw).any() else 0.0
    raw_exact = bool(np.array_equal(R, csv))
    print(f"[4] library: shape {V.shape} dtype {V.dtype}, == L2-normalised CSV: {lib_exact}, max |‖v‖−1| {lib_dev:.2e}; "
          f"verification W1: rows identical to CSV float32 {int(is_raw.sum())}, other rows {int((~is_raw).sum())} "
          f"with max |‖v‖−1| {unit_dev:.2e}; raw == CSV float64: {raw_exact}")
    return ok_shape and lib_exact and lib_dev < 1e-6 and raw_exact and unit_dev < 1e-6


def check_library_vectors(index_dir, idx):
    """Check 4, library part only: text.npy == L2(source vectors) — the CSV column, or the cache."""
    meta = idx['meta']
    src = meta['source']
    computed = src.get('embeddings') == 'computed'
    cache = os.path.join(index_dir, src['embedding_cache']) if computed else None
    df, raw, _ = read_items(meta['config'], src['precomputed_embedding_col'], cache_path=cache, max_embed_items=0)
    V = idx['vectors']
    ids_ok = [str(x) for x in df[meta['col_itemid']]] == [str(x) for x in meta['item_ids']]
    lib_exact = bool(np.array_equal(V, (raw / np.linalg.norm(raw, axis=1, keepdims=True)).astype(np.float32)))
    lib_dev = float(np.abs(np.linalg.norm(V.astype(np.float64), axis=1) - 1.0).max())
    print(f"[4] library: shape {V.shape} dtype {V.dtype}, rows match the CSV: {ids_ok}, "
          f"== L2-normalised {'cache' if computed else 'CSV'}: {lib_exact}, max |‖v‖−1| {lib_dev:.2e}")
    return ids_ok and V.dtype == np.float32 and lib_exact and lib_dev < 1e-6


def check_ref_vectors(ctx):
    R, id_col = ctx['raw'], ctx['id_col']
    row_of = {it[id_col]: r for r, it in enumerate(ctx['idx']['items'])}
    n_cmp = n_fail = 0
    base = os.path.join(os.path.dirname(ctx['ref_lookups']), 'ref_vectors')
    with open(ctx['ref_lookups'], encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            if rec['es_index'] != ctx['es_index'] or not rec['n_embeddings_returned']:
                continue
            emb = np.load(os.path.join(base, f"{rec['lookup_id']}.npz"))[rec['vectors']]
            hits = [t for t in rec['titles'] if t['hit']]
            for j, t in enumerate(hits):
                n_cmp += 1
                n_fail += not np.array_equal(emb[j], R[row_of[t['item_id']]])
    if n_cmp:
        print(f"[8] verification raw vectors vs es_golden ref_vectors: {n_cmp - n_fail}/{n_cmp} bit-identical")
    else:
        print(f"[8] es_golden ref_vectors: N/A, no lookup hits recorded for '{ctx['es_index']}'")
    return n_fail == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', required=True, help='space-separated YAML files, later ones override')
    parser.add_argument('--index', default=None, help='index directory (default: index_dir, else index/{dataset})')
    parser.add_argument('--ref-lookups', default=REF_LOOKUPS, help='es_golden direct_call/ref_lookups.jsonl (read only)')
    parser.add_argument('--library-only', action='store_true', help='only the library part of check 4 (see above)')
    args = parser.parse_args()

    config = load_config(args.config.strip().split(' '))
    index_dir = args.index or index_dir_for(config)
    idx = load_local_index(index_dir)
    if args.library_only:
        ok = check_library_vectors(index_dir, idx)
        print('checks:', {'4_library': ok})
        print('PASS' if ok else 'FAIL: [\'4_library\']')
        return 0 if ok else 1
    w1, raw = es_vectors(LocalBackend(index_dir))
    ctx = {'idx': idx, 'id_col': config['col_itemid'], 'dataset': config['dataset'],
           'items_file': idx['meta']['source']['items_file'], 'w1': w1, 'raw': raw,
           'es_index': FIXTURE_INDEX.get(os.path.normpath(index_dir)), 'ref_lookups': args.ref_lookups}

    results = {'4_vectors': bool(check_vectors(ctx)), '8_ref_vectors': bool(check_ref_vectors(ctx))}
    failed = [k for k, v in results.items() if not v]
    print('checks:', results)
    print('PASS' if not failed else f"FAIL: {failed}")
    return 0 if not failed else 1


if __name__ == '__main__':
    sys.exit(main())
