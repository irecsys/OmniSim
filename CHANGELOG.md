# Changelog

## v2.0.0

- Retrieval uses a built-in local backend (SQLite FTS5 for BM25, NumPy for vector search); Elasticsearch is no longer needed. Scores differ slightly from Elasticsearch's (BM25 implementation, vector normalisation); `legacy_es_equivalence/` holds the recorded v1 results and a replay check of the retrieval pipeline.
- Indexes are built from the shipped precomputed item vectors without any API call. Item vectors computed through the embeddings API are cached in the index directory, so a rebuild only embeds new or changed items.
- Errors are visible: fallback text is marked in the dialogue files, every run writes a run summary, `run.py` exits with code 3 when an LLM or embedding call failed, and quota errors stop the run.
- New safety limit `max_total_rounds` (default 30) ends dialogues that can no longer show new items.
- Dialogue files are no longer overwritten when two dialogues get the same name in multi-threaded runs.
- `sentence-transformers` (and with it PyTorch) is no longer a default dependency. The `sentence-transformers` embeddings provider and the cosine-diversity metric remain; to use them, install the package separately (see README, Optional: local embeddings).
- The `es` parameter of the simulation functions is renamed `backend`; `compare_metrics.py` labels generated dialogues `OmniSim (generated)`.

## v1.0

- The version described in the UMAP '26 paper, with Elasticsearch-based retrieval.
