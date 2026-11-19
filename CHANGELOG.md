# Changelog

## v2.1.0

v2.0.0 plus a local embedding model that runs offline, a chat provider for any OpenAI-compatible endpoint, waiting out
rate limits, and an optional call log. **The dialogue logic, the prompts, the retrieval scoring and the default
parameters are those of v2.0.0**: 21 files (`run.py`, the simulator, profile builder, retrieval modules,
`configs/prompts/*.yaml`, `configs/imdb/imdb.yaml`, `configs/hm/hm.yaml`, ...) are byte for byte those of v2.0.0, and
`configs/system/system.yaml` differs only in its `library_version` line (`"2.1.0"`); `tests/test_dialogue_logic_unchanged.py`
checks both. Defaults such as `max_rec_attempts: 6` and `max_total_rounds: 30` are unchanged.

- Local embeddings (`embeddings_provider: sentence-transformers`, `utils/local_embeddings.py`), e.g. EmbeddingGemma 2:
  a fixed model revision, the weights' sha256 checked on every load, local files only (`embeddings_allow_download:
  false`), float32 on CPU, a query prompt and a document template. New keys `embeddings_revision`,
  `embeddings_local_path`, `embeddings_weights_sha256`, `embeddings_allow_download`, `embeddings_dtype`,
  `embeddings_device`, `embeddings_modalities`, `embeddings_num_threads`, `embeddings_batch_size`,
  `embeddings_query_prompt`, `embeddings_document_template`, `embeddings_dim` (see `docs/API.md`). Optional packages:
  `requirements-local-embeddings.txt`. `scripts/fetch_embedding_model.py` downloads one revision once. Example overlay:
  `configs/examples/imdb_embeddinggemma.yaml`. The index cache key of a local model is a hash of the model's identity
  (model, revision, weights, prompt, template, dim, dtype); API providers keep their item texts and cache keys.
  **Change for existing `sentence-transformers` users:** a Hugging Face model id now needs `embeddings_revision`
  (a commit sha), and the model is no longer downloaded automatically.
- `openai_provider: openai_compatible`: any OpenAI-compatible endpoint (vLLM and others); `chat_endpoint` is its base
  URL; the key comes from the environment variable named by `chat_api_key_env` (none when unset; `OPENAI_KEY` is never
  sent). `chat_extra_body` merges request fields into every chat request, e.g.
  `{chat_template_kwargs: {enable_thinking: false}}` for Qwen. Replies of thinking models are cleaned of a leading
  `<think>…</think>` block; a reply whose thinking was cut off is asked once more, then takes the call site's error
  path; signs of thinking at the startup check stop the run with a configuration error.
- Rate limits (`openai_compatible`, `utils/rate_limit.py`): a 429, 5xx, timeout or connection error is sent again
  unchanged after an exponential backoff (`chat_retry_base` 2 s, `chat_retry_max_wait` 300 s, `chat_retry_total`
  1800 s; Retry-After followed); retries are logged and counted in the run summary's `retries`. A rate gate shared by
  all processes through a locked file: `chat_rate_limit` (default `70/60`; null or `0` = off),
  `chat_rate_window_file`. The jitter never touches the global `random` state.
- Optional passive call log `llm_log_file` (`utils/llm_log.py`; absent / null: off): one JSONL line per returned chat
  request (call site, request, reply text, usage, the conversation's user / item). It changes neither the requests nor
  their order, the retries or the random state.
- `configs/examples/budget_aligned_baseline.yaml`: example configuration that limits recommendation attempts to 4.
- `requirements_locked.txt` and `requirements-local-embeddings_locked.txt`: the exact package versions of the tested
  environments, for reproduction.

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
