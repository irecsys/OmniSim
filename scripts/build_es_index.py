"""
Build (or rebuild) an Elasticsearch index from items.csv using
sentence-transformers embeddings.

Usage:
    python scripts/build_es_index.py --config configs/imdb/imdb.yaml
    python scripts/build_es_index.py --config configs/taw9eel/taw9eel.yaml
"""

import argparse
import os
import sys
import yaml
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import build_embedding_texts

load_dotenv()


class _SimpleConfig(dict):
    """Minimal config loader that reads YAML files without triggering circular imports."""
    def get(self, key, default=None):
        return super().get(key, default)


def _load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def build_index(config_file_list):
    config = _SimpleConfig()
    # Always load system defaults first so es_host, embeddings_model, etc. are available
    config.update(_load_yaml('configs/system/system.yaml'))
    for path in config_file_list:
        config.update(_load_yaml(path))

    es_host  = config['es_host']
    es_index = config['es_index']
    es_user  = os.getenv('ES_USER')
    es_pwd   = os.getenv('ES_PWD')
    emb_model_name = config.get('embeddings_model', 'all-MiniLM-L6-v2')

    data_path = os.path.join(config.get('data_path', 'data'), config['dataset'])
    # Allow a separate file for index building (e.g. one that includes pre-computed embeddings)
    items_file = os.path.join(data_path, config.get('file_items_for_index', config['file_items']))

    print(f"Loading items from: {items_file}")
    df = pd.read_csv(items_file, low_memory=False)
    print(f"Loaded {len(df)} items")

    # Fill missing values
    col_itemid   = config['col_itemid']
    col_title    = config['col_title']
    col_details  = config['col_details']
    col_category = config['col_category']

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna('')

    # Generate (or reuse) embeddings
    precomputed_col = config.get('precomputed_embedding_col', '')
    if precomputed_col and precomputed_col in df.columns:
        # Fast path: load pre-computed embeddings directly from CSV column
        import ast
        print(f"Loading pre-computed embeddings from column '{precomputed_col}' ...")
        def _parse_emb(val):
            if isinstance(val, list):
                return [float(x) for x in val]
            if isinstance(val, str):
                try:
                    return [float(x) for x in ast.literal_eval(val)]
                except Exception:
                    return None
            return None
        embeddings = df[precomputed_col].apply(_parse_emb).tolist()
        valid_mask = [e is not None for e in embeddings]
        if not all(valid_mask):
            n_bad = sum(1 for v in valid_mask if not v)
            print(f"  Warning: {n_bad} rows have unparseable embeddings — dropping them.")
            df = df[[v for v in valid_mask]].reset_index(drop=True)
            embeddings = [e for e, v in zip(embeddings, valid_mask) if v]
        embedding_dim = len(embeddings[0])
        print(f"Loaded {len(embeddings)} embeddings, dim={embedding_dim}")
    else:
        # Build text for embedding.
        # Use dataset-specific embedding fields if defined, otherwise fall back to
        # title + category + details (ensures category/genre is always included
        # so genre-based queries retrieve genre-matched items).
        embedding_fields = config.get('embedding_fields', [col_title, col_category, col_details])
        texts = build_embedding_texts(df, embedding_fields)
        print(f"Embedding fields: {embedding_fields}")

        print(f"Generating embeddings with '{emb_model_name}' ...")
        emb_provider = config.get('embeddings_provider', 'sentence-transformers')
        if emb_provider == 'azure':
            azure_client = AzureOpenAI(
                api_version=config.get('embeddings_api_version', '2024-02-01'),
                azure_endpoint=config['embeddings_endpoint'],
                api_key=os.getenv('AZURE_KEY'),
            )
            batch_size = 100
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = azure_client.embeddings.create(model=emb_model_name, input=batch)
                embeddings.extend([d.embedding for d in resp.data])
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        elif emb_provider == 'openai':
            oa_client = OpenAI(api_key=os.getenv('OPENAI_KEY'))
            batch_size = 100
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                resp = oa_client.embeddings.create(model=emb_model_name, input=batch)
                embeddings.extend([d.embedding for d in resp.data])
                print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(emb_model_name)
            embeddings = model.encode(texts, batch_size=64, show_progress_bar=True,
                                      normalize_embeddings=True).tolist()
        embedding_dim = len(embeddings[0])

        # Save embeddings back to CSV so future runs skip API calls
        save_col = config.get('precomputed_embedding_col', 'embedding_vector')
        df[save_col] = [str(e) for e in embeddings]
        df.to_csv(items_file, index=False)
        print(f"Saved embeddings to column '{save_col}' in {items_file}")

    print(f"Embedding dimension: {embedding_dim}")

    # ── Build the aggregated "details" field for BM25 (paper Eq. 5) ──────────
    # Eq. 5 computes S̃_BM25(q, M_i,d), where M_i,d is a single "details" field
    # defined as the aggregated textual information from ALL item attributes.
    # We concatenate every attribute column into one text field (`bm25_details`)
    # so BM25 scores a single aggregated field rather than multiple separate ones.
    BM25_DETAILS_FIELD = "bm25_details"
    agg_fields = (
        config.get('embedding_fields')
        or ([col_title, col_category, col_details] + list(config.get('item_attributes', {}).keys()))
    )
    # Keep only columns that exist; preserve order and de-duplicate.
    seen_agg = set()
    agg_fields = [f for f in agg_fields if f in df.columns and not (f in seen_agg or seen_agg.add(f))]
    df[BM25_DETAILS_FIELD] = build_embedding_texts(df, agg_fields)
    print(f"Built aggregated BM25 field '{BM25_DETAILS_FIELD}' from: {agg_fields}")

    # Connect to ES
    if es_user and es_pwd:
        es = Elasticsearch(es_host, basic_auth=(es_user, es_pwd), verify_certs=False)
    else:
        es = Elasticsearch(es_host)

    # Delete and recreate index
    if es.indices.exists(index=es_index):
        es.indices.delete(index=es_index)
        print(f"Deleted existing index '{es_index}'")

    # Build mapping: standard fields + extra numeric/keyword fields from config
    item_attributes = config.get('item_attributes', {})
    numeric_fields  = config.get('numeric_fields', [])    # e.g. ["price"] → stored as float for range filters
    keyword_fields  = config.get('keyword_fields', [])    # e.g. ["gender", "item_type"] → stored as keyword for exact filters

    properties = {
        col_itemid:          {"type": "keyword"},
        col_title:           {"type": "text"},
        col_details:         {"type": "text"},
        col_category:        {"type": "text"},
        BM25_DETAILS_FIELD:  {"type": "text"},   # aggregated all-attribute text for BM25 (Eq. 5)
        "embedding":  {
            "type": "dense_vector",
            "dims": embedding_dim,
            "index": True,
            "similarity": "cosine"
        },
    }
    for f in keyword_fields:
        if f not in properties:
            properties[f] = {"type": "keyword"}
    for f in numeric_fields:
        if f not in properties:
            properties[f] = {"type": "float"}

    # elasticsearch-py 9.x removed the `body=` parameter; pass mapping fields directly
    es.indices.create(index=es_index, mappings={"properties": properties})
    print(f"Created index '{es_index}' with {embedding_dim}-dim embedding field")

    import math

    def _clean(val, is_numeric=False):
        """Convert a pandas cell value to a JSON-safe Python object.

        pandas represents missing values as float('nan'), which serialises to
        the non-standard JSON token NaN that Elasticsearch rejects.  Replace
        any NaN / None with Python None (serialised as JSON null).
        """
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

    def gen_actions():
        all_store_fields = (
            [col_itemid, col_title, col_details, col_category, BM25_DETAILS_FIELD]
            + list(item_attributes.keys())
            + numeric_fields
            + keyword_fields
        )
        for i, (_, row) in enumerate(df.iterrows()):
            source = {}
            for k in all_store_fields:
                if k in row:
                    source[k] = _clean(row[k], is_numeric=(k in numeric_fields))
            source['embedding'] = embeddings[i]
            yield {"_index": es_index, "_source": source}

    print("Bulk inserting into Elasticsearch ...")
    ok_count = 0
    for ok, response in helpers.streaming_bulk(es, gen_actions(), chunk_size=50,
                                               raise_on_error=False):
        if ok:
            ok_count += 1
        else:
            # Use ascii() to avoid UnicodeEncodeError on narrow Windows terminals (GBK)
            print("  Failed:", ascii(str(response)))

    print(f"Done. Inserted {ok_count}/{len(df)} documents into '{es_index}'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/imdb/imdb.yaml')
    args = parser.parse_args()
    config_list = args.config.strip().split(' ')
    build_index(config_list)
