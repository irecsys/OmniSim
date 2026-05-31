"""
Build (or rebuild) the H&M Elasticsearch index.

Workflow:
  1. Run HMPreprocessor to ensure data/hm/items.csv is up to date.
  2. Generate embeddings from the `description` field (gender_type + brand + color + price + details).
  3. Bulk-insert documents into ES with a dense_vector mapping.

Usage:
    python datasets/hm/build_index.py
    python datasets/hm/build_index.py --config configs/hm/hm.yaml --skip-preprocess
"""

import argparse
import os
import sys

# ── allow running from project root ───────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import yaml
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers
from openai import AzureOpenAI, OpenAI

from utils.hm.preprocessor import HMPreprocessor
from utils.utils import build_embedding_texts

load_dotenv()


def _load_configs(*paths) -> dict:
    cfg = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            cfg.update(yaml.safe_load(f) or {})
    return cfg


def build_hm_index(config_path: str = "configs/hm/hm.yaml",
                   skip_preprocess: bool = False) -> None:

    # ── load config (system defaults + user overrides) ─────────────────────────
    cfg = _load_configs("configs/system/system.yaml", config_path)

    data_path  = os.path.join(cfg.get("data_path", "data"), cfg["dataset"])
    items_file = os.path.join(data_path, cfg.get("file_items", "items.csv"))
    raw_file   = os.path.join(data_path, "raw_handm.csv")

    # ── step 1: preprocess ─────────────────────────────────────────────────────
    if not skip_preprocess:
        HMPreprocessor(input_path=raw_file, output_path=items_file).run()
    else:
        print(f"Skipping preprocessing, using existing: {items_file}")

    # ── step 2: load items ─────────────────────────────────────────────────────
    df = pd.read_csv(items_file, low_memory=False)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("").astype(str)
    print(f"Loaded {len(df)} items from {items_file}")

    # ── step 3: embed using embedding_fields (same logic as IMDB build_index) ────
    emb_model_name = cfg.get("embeddings_model", "text-embedding-3-small")
    emb_provider   = cfg.get("embeddings_provider", "azure")
    col_title    = cfg.get("col_title", "productName")
    col_category = cfg.get("col_category", "gender_type")
    col_details  = cfg.get("col_details", "details")
    embedding_fields = cfg.get("embedding_fields", [col_title, col_category, col_details])
    texts = build_embedding_texts(df, embedding_fields)
    print(f"Generating embeddings with '{emb_model_name}' ({emb_provider}) from fields {embedding_fields} ...")

    if emb_provider == "azure":
        client = AzureOpenAI(
            api_version=cfg.get("embeddings_api_version", "2024-02-01"),
            azure_endpoint=cfg["embeddings_endpoint"],
            api_key=os.getenv("AZURE_KEY"),
        )
    else:
        client = OpenAI(api_key=os.getenv("OPENAI_KEY"))

    batch_size = 100
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=emb_model_name, input=batch)
        embeddings.extend([d.embedding for d in resp.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")

    embedding_dim = len(embeddings[0])
    print(f"Embedding dimension: {embedding_dim}")

    # ── step 4: connect to ES ──────────────────────────────────────────────────
    es_host  = cfg.get("es_host", "http://localhost:9200")
    es_index = cfg["es_index"]
    es_user  = os.getenv("ES_USER")
    es_pwd   = os.getenv("ES_PWD")

    es = (Elasticsearch(es_host, basic_auth=(es_user, es_pwd), verify_certs=False)
          if es_user and es_pwd else Elasticsearch(es_host))

    # ── step 5: recreate index with mapping ───────────────────────────────────
    if es.indices.exists(index=es_index):
        es.indices.delete(index=es_index)
        print(f"Deleted existing index '{es_index}'")

    mapping = {
        "mappings": {
            "properties": {
                "productId":   {"type": "keyword"},
                "productName": {"type": "text"},
                "brandName":   {"type": "keyword"},
                "colorName":   {"type": "keyword"},
                "price":       {"type": "float"},
                "gender":      {"type": "keyword"},
                "item_type":   {"type": "text"},
                "gender_type": {"type": "text"},
                "details":     {"type": "text"},
                "description": {"type": "text"},
                "embedding":   {"type": "dense_vector", "dims": embedding_dim},
            }
        }
    }
    es.indices.create(index=es_index, body=mapping)
    print(f"Created index '{es_index}' with {embedding_dim}-dim embedding field")

    # ── step 6: bulk insert ───────────────────────────────────────────────────
    def gen_actions():
        for i, (_, row) in enumerate(df.iterrows()):
            yield {
                "_index": es_index,
                "_source": {
                    "productId":   row.get("productId", ""),
                    "productName": row.get("productName", ""),
                    "brandName":   row.get("brandName", ""),
                    "colorName":   row.get("colorName", ""),
                    "price":       float(row.get("price", 0)),
                    "gender":      row.get("gender", ""),
                    "item_type":   row.get("item_type", ""),
                    "gender_type": row.get("gender_type", ""),
                    "details":     row.get("details", ""),
                    "description": row.get("description", ""),
                    "embedding":   embeddings[i],
                }
            }

    print("Bulk inserting into Elasticsearch ...")
    ok_count = 0
    for ok, response in helpers.streaming_bulk(es, gen_actions(), chunk_size=100,
                                               raise_on_error=False):
        if ok:
            ok_count += 1
        else:
            print("  Failed:", response)

    print(f"Done. Inserted {ok_count}/{len(df)} documents into '{es_index}'.")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build H&M Elasticsearch index")
    parser.add_argument("--config",          default="configs/hm/hm.yaml",
                        help="User config YAML (default: configs/hm/hm.yaml)")
    parser.add_argument("--skip-preprocess", action="store_true",
                        help="Skip preprocessing step if items.csv already exists")
    args = parser.parse_args()

    build_hm_index(config_path=args.config, skip_preprocess=args.skip_preprocess)
