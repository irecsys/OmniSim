"""
Build the local retrieval index (v2, replaces Elasticsearch) from the items CSV.

Vectors: the precomputed column if the CSV has it (config precomputed_embedding_col, else
--precomputed-col); otherwise they are computed from embedding_fields with embeddings_provider
(paid API for azure / openai) and cached in <out>/embedding_cache.db, keyed by item id, embedding
text and model, so a rebuild only embeds new or changed items. The CSV is never modified.

Usage:
    python scripts/build_local_index.py --config "configs/system/system.yaml configs/imdb/imdb.yaml"
    python scripts/build_local_index.py --config "configs/system/system.yaml configs/hm/hm.yaml"
    python scripts/build_local_index.py --config "... my.yaml" --dry-run              # what would be embedded
    python scripts/build_local_index.py --config "... my.yaml" --max-embed-items 500  # refuse more than 500

--dry-run reads the CSV and the cache only: no API call, .env is not loaded, nothing is written.

Output: config index_dir, else index/{dataset}/ (override with --out). After every change to the builder, re-run the fixture
replay (legacy_es_equivalence/replay_fixtures.py): with the ES replicas injected it must stay byte-identical, and it
rebuilds the ES vectors from the CSV and checks them against the expected sha256 values (EXPECTED_SHA256 in
legacy_es_equivalence/es_compat/vectors.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.retrieval.local_index import (CACHE_FILE, _csv_df, _emb_col, build_local_index, index_dir_for,
                                         load_config, plan_embeddings)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='space-separated YAML files, later ones override')
    parser.add_argument('--precomputed-col', type=str, default=None,
                        help='embedding column to use when the config has no precomputed_embedding_col')
    parser.add_argument('--out', type=str, default=None, help='output directory (default: config index_dir, else index/{dataset})')
    parser.add_argument('--dry-run', action='store_true', help='report what would be embedded, then exit')
    parser.add_argument('--max-embed-items', type=int, default=None,
                        help='abort before the first API request if more items than this need embedding')
    args = parser.parse_args()

    config_files = args.config.strip().split(' ')
    config = load_config(config_files)
    out_dir = args.out or index_dir_for(config)

    if args.dry_run:
        df, items_file = _csv_df(config)
        emb_col, source = _emb_col(config, args.precomputed_col, df)
        if emb_col:
            print(f"dry run: {items_file} has the precomputed column '{emb_col}' (from {source}): "
                  f"{len(df)} items, 0 to embed")
        else:
            p = plan_embeddings(config, df, os.path.join(out_dir, CACHE_FILE))
            print(f"dry run: {items_file} has no precomputed column; {p['n_rows']} items, "
                  f"{p['n_cache_hits']} in the cache ({os.path.join(out_dir, CACHE_FILE)}), {p['n_embed']} to embed "
                  f"= {p['n_requests']} requests, ~{p['est_tokens']} tokens (provider {p['provider']}, model {p['model']})")
        print('dry run: nothing sent, nothing written')
        return

    meta = build_local_index(config, out_dir, precomputed_col=args.precomputed_col, config_files=config_files,
                             max_embed_items=args.max_embed_items)
    src = meta['source']
    print(f"Built {out_dir}: {meta['n_items']} items (CSV records {src['n_csv_records']}, "
          f"dropped {len(src['dropped_unparseable_rows'])}), dim {meta['spaces']['text']['dim']}")
    if src['embeddings'] == 'computed':
        print(f"  embeddings computed: {src['n_embedded']} embedded now, {src['n_cache_hits']} from the cache "
              f"({os.path.join(out_dir, src['embedding_cache'])}); "
              f"provider {meta['spaces']['text']['provider']}, model {meta['spaces']['text']['model']}")
    else:
        print(f"  embedding column '{src['precomputed_embedding_col']}' "
              f"(from {src['precomputed_embedding_col_from']})")
    print(f"  vectors: {meta['spaces']['text']['normalization']}; FTS5 tokenizer: {meta['bm25']['fts_tokenizer']}")
    print(f"  bm25_details from {meta['bm25']['agg_fields']}")
    print(f"  columns: {[(c['name'], c['type']) for c in meta['columns']]}")


if __name__ == '__main__':
    main()
