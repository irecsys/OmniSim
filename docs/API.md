# OmniSim API Reference

## Overview

OmniSim is a domain-agnostic, retrieval-grounded framework for generating
personalised conversational recommendation dialogues.  It combines hybrid
retrieval over a local index (BM25 with SQLite FTS5 + exact dense vector
search) with LLM-based dialogue generation across three interaction modes:
**Free**, **Static**, and **Adaptive**.

---

## Module: `utils.utils`

Core utilities: LLM clients, similarity functions, retrieval, and dialogue-act generators.

### Similarity & Scoring

#### `cosine_sim(a, b) → float`
Returns the raw cosine similarity between two vectors in **[-1, 1]**.  
Use `normalize_cosine()` before combining with other scores.

#### `normalize_cosine(s) → float`
Maps a raw cosine similarity to **[0, 1]** using `(s + 1) / 2`.

#### `normalize_bm25(scores: dict) → dict`
Min-max normalises a dict of BM25 scores to **[0, 1]** (all zeros if every score is equal).  
Returns `{item_id: normalised_score}`.

#### `search_items_by_embedding(config, backend, embedding_vector, top_k, ...)  → list[dict]`
Hybrid retrieval over the local index. `backend` is an opened
`utils.retrieval.backend.LocalBackend`; the work is done by
`utils.retrieval.local_search.search_items_local`.

| Symbol | Config Key | Default | Meaning |
|---|---|---|---|
| λ | `lambda_bm25` | 0.3 | BM25 weight in the base score |
| α | `weight_es_score` | 0.7 | Base-score weight in the hybrid score |
| β | `weight_user_taste_short` | 0.3 | Short-term preference ratio |

**Scoring pipeline:**
1. Hard filters (year range, excluded / required genres, excluded ids), then exact cosine similarity over the remaining items; the top `3 × top_k` candidates are kept, with `S̃_cos(q, i) = (1 + cos(q, i)) / 2` ∈ [0, 1]
2. BM25 (SQLite FTS5) on the same candidates, scoring the query against the **single aggregated `bm25_details` field** (`M_i,d` = all item attributes concatenated, built at index time) → min-max normalise among the candidates that match (the others count as 0) → `S̃_BM25(q, M_i,d)`
3. `S̃_base = λ · S̃_BM25 + (1 − λ) · S̃_cos`
4. `S̃_hybrid = α · S̃_base + (1 − α) · [β · S̃_cos(i, p_short) + (1 − β) · S̃_cos(i, p_long)]`; without a long-term profile `S̃_hybrid = S̃_base`, and without a short-term profile β = 0
5. **Threshold gate:** if there are no candidates, or `max(S̃_base) < τ` (`threshold_similarity`) across all candidates, return an empty list. The caller then shows no item in that round; the round does not count as a recommendation attempt, and the dialogue continues with the next round (another question if the question budget allows, otherwise another retrieval). Otherwise the top-`top_k` items ranked by `S̃_hybrid` are returned. With the defaults λ = τ = 0.3, the gate in practice only fails when the hard filters leave no candidates.

**Key parameters:**
- `embedding_vector` — encoded user preference query
- `user_long_embedding` — user long-term preference vector (optional)
- `user_short_embedding` — user short-term preference vector (optional)
- `excluded_ids` — item IDs to exclude (optional)
- `query_text` — raw text for BM25 matching (optional)
- `year_range` — `(min_year, max_year)` hard filter (optional)
- `exclude_genres` — genre strings to exclude (optional)
- `require_any_genre` — keep only items matching at least one of these genres (optional)
- `reference_embeddings` — embeddings of reference items blended into the query with weight `reference_blend_weight` (optional)

Returns a list of item dicts sorted by descending `score`, capped at `top_k`.

---

### LLM Clients

#### `get_openai_clients(config) → (client_chat, client_embeddings)`
Build LLM and embeddings clients from config.  Supports:
- `openai_provider: openai` — OpenAI native API
- `openai_provider: azure` — Azure OpenAI
- `openai_provider: azure_foundry` — Azure AI Foundry (`chat_endpoint` used as the base URL)
- `openai_provider: thetaedgecloud` — ThetaEdgeCloud (Llama etc.)
- `embeddings_provider: openai` / `azure` / `sentence-transformers` (local, no API key needed; optional package, installed separately) / `thetaedgecloud` (accepted, but returns no vectors); defaults to `openai_provider` when not set

#### `ThetaEdgeCloudClient`
Drop-in wrapper that exposes the same `.chat.completions.create()` interface as
the OpenAI client.  Retries HTTP 409, 429 and 503 with exponential back-off (up to 6 attempts).

#### `SentenceTransformerEmbeddingsClient`
Local embeddings client using `sentence-transformers`.  Runs locally; the model is downloaded from Hugging Face on first use and loaded from the local cache afterwards. The package is optional and not installed
by `requirements.txt` (`pip install "sentence-transformers>=2.2.0,<7"`); without it, creating the client raises an ImportError that says so.

---

### Dialogue-Act Generators

| Function | Purpose |
|---|---|
| `generate_chit_chat(config, client_chat, speaker, ...)` | Context-aware chit-chat (greeting / follow-up / closing) |
| `generate_rejection_explanation(config, client_chat, ...)` | Natural rejection explanation from item metadata comparison |
| `generate_recommendation_explanation(config, client_chat, ...)` | Why the bot recommends these items |
| `generate_user_responses_free_mode(config, client_chat, ...)` | Simulated user reply in Free mode |
| `generate_user_acceptance(config, client_chat, ...)` | Short acceptance phrase |
| `generate_bot_closing_failed(config, client_chat, ...)` | Bot message when max attempts reached |
| `rephrase_attribute_to_natural_language(config, client_chat, ...)` | Rephrase an attribute value as natural speech |
| `expand_query_for_search(config, client_chat, user_context)` | LLM expands user text into search keywords |

---

### Extraction Utilities

| Function | Purpose |
|---|---|
| `extract_attribute_value(config, client_chat, attribute, item_info, details)` | Get structured or derived attribute value |
| `extract_attributes_for_asking(config, client_chat, item_type, target_category, attribute_by_openai=True)` | Attributes to ask about: the configured `item_attributes` (Static), plus attributes proposed by the LLM (Adaptive) |
| `extract_dynamic_description_from_chat(config, client_chat, chat_history)` | Summarise user intent from dialogue context |
| `extract_search_constraints(user_context)` | Parse year range, excluded / required genres and reference titles from user text (regular expressions, no LLM call) |
| `extract_item_from_title(config, client_chat, title, item_category)` | Infer a descriptive item type from its title |

---

## Module: `utils.simulator`

Conversation simulation engines.  All three public functions take the same
parameters and write the completed conversation to disk under
`chats/{dataset}/{mode}/{strategy}/{run_timestamp}/`.

### `simulate_free(config, dataset, user_id, target_item_id, client_chat, client_embeddings, backend, user_profiles=None)`
Open-ended dialogue — the bot issues open-ended follow-up prompts and the
simulated user replies with free-form natural language derived from the target
item's metadata.  After each round the accumulated user messages (expanded into
search keywords when `expand_query` is on) are embedded and used for hybrid retrieval.

### `simulate_static(config, dataset, user_id, target_item_id, client_chat, client_embeddings, backend, user_profiles=None)`
Schema-driven dialogue — the bot asks questions **only** about the predefined
`item_attributes` in the dataset YAML.  User answers are rephrased into natural
language and aggregated into a structured query.  Internally delegates to
`simulate_adaptive(attribute_by_openai=False)`.

### `simulate_adaptive(config, dataset, user_id, target_item_id, client_chat, client_embeddings, backend, attribute_by_openai=True, user_profiles=None)`
Like Static, but the LLM also **discovers additional relevant attributes**
beyond the YAML schema (e.g. "music score", "animation style" for movies).
Set `attribute_by_openai=False` to restrict to the YAML schema only (Static mode).

**Shared parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config` | `Config` | Merged OmniSim configuration |
| `dataset` | `Dataset` | Loaded CSV DataFrames |
| `user_id` | str / None | Simulated user ID; `None` for anonymous guest |
| `target_item_id` | str | Ground-truth item the user secretly wants |
| `client_chat` | client | Initialised LLM chat client |
| `client_embeddings` | client | Initialised embeddings client |
| `backend` | `LocalBackend` | Opened local index (`utils.retrieval.backend.LocalBackend`) |
| `user_profiles` | dict / None | Pre-built user profile dict keyed by user_id |

**Output filename pattern:**
```
{user_id}-{item_id}-{num_turns}-{rec_attempts}-{succeed}-{timestamp}.txt
```
`user_id` is `user` for anonymous users. `succeed=1` when the target item was accepted; `0` when the dialogue
ended without success (`max_rec_attempts` or `max_total_rounds` reached). An existing file is never overwritten:
if the name is taken, the timestamp is moved forward by one second.

---

## Module: `utils.quick_start`

### `run_simulation(config_file_list=None, mode_override=None, strategies_override=None, num_workers_override=None, chats_per_entry_override=None, pairs_file_override=None)`
Main entry point. Loads config, opens the local index (building it from the
precomputed vectors if it is missing), builds or loads user profiles, and
dispatches all conversation generation tasks across a thread pool.
Returns the run summary (a dict, also written to
`chats/{dataset}/{mode}/run_summary_{run_timestamp}.json`); its `exit_code`
is the code `run.py` exits with.

---

## Module: `utils.configurator`

### `Config(config_file_list=None)`
Merges `configs/system/system.yaml` with dataset-specific YAML files.
Dataset values take precedence over system defaults.

```python
config = Config(['configs/imdb/imdb.yaml'])
dataset = config['dataset']                       # KeyError if missing
thresh = config.get('threshold_similarity', 0.3)  # with default
```

---

## Module: `utils.dataset`

### `Dataset(config)`
Loads items, users, and interactions CSVs into DataFrames.

```python
ds = Dataset(config)
ds.load_data()
items        = ds.df_items          # pd.DataFrame
users        = ds.df_users          # pd.DataFrame or None
interactions = ds.df_interactions   # pd.DataFrame or None (loaded only when users_as_guest is false)
```

---

## Module: `utils.user_profile_builder`

### `build_user_profiles(config_file_list) → (profiles_df, failures)`
For each user, constructs:
- `user_demographic` — natural-language summary from `users.csv`
- `user_likes_long` — long-term preference from all positive interactions
- `user_likes_short` — short-term preference: with a `timestamp` column, from the interactions within the last `short_term_days` days (falls back to `user_likes_long` when there are none); without one, an empty string

The profiles are written to `data/{dataset}/user_profiles.csv`, unless an LLM
call failed (`failures` lists them). The preference embeddings are not stored:
the simulator embeds `user_likes_long` / `user_likes_short` at the start of each
conversation. `run_simulation` rebuilds the profiles only when
`user_profiles.csv` is missing or the MD5 hash of the users or interactions file
has changed (stored in `.profile_hash.json`). Returns `None` (nothing written)
when neither the users file nor the interactions file exists.

---

## Module: `utils.evaluator`

### `evaluate_folder(config, folder_path, output_csv=None)`
Runs LLM-based evaluation on all `.txt` conversations in a folder (temporary
files are skipped), using the chat client and `chat_model` from `config`.
Scores three dimensions on a 1–5 scale:
- **Fluency**
- **Informativeness**
- **Coherence**

Each row also has the reason for each score, `overall_score` (the mean of the
three) and `overall_quality`. Results are saved to `output_csv` if given.

```bash
python -m utils.evaluator --config configs/imdb/imdb.yaml --folder chats/imdb/free/user_item_pairs/<run_timestamp> --output eval.csv
```

---

## Module: `utils.metrics`

### `compute_metrics(folder_path, output_csv=None) → pd.DataFrame`
Conversation-level statistics for the `.txt` files in a folder: success rate,
turns, recommendation attempts, and utterance counts per speaker (printed).
Returns one row per conversation (`user_id`, `item_id`, `num_turns`,
`rec_attempts`, `succeed`, `timestamp`), saved to `output_csv` if given.

Reference-free lexical diversity metrics are computed by
`scripts/compute_nlp_metrics.py`:

| Metric | Description |
|---|---|
| `distinct_1` | Distinct unigram ratio |
| `distinct_2` | Distinct bigram ratio |
| `ttr`, `root_ttr`, `log_ttr`, `maas_ttr`, `msttr`, `mattr` | Type-Token Ratio and its length-robust variants |
| `mtld`, `mtld_ma_w`, `mtld_ma_bid` | Measure of Textual Lexical Diversity and its moving-average variants |
| `hdd` | Hypergeometric Distribution D (D=42) |
| `cosine_diversity` | Mean pairwise semantic distance between utterance embeddings |
| `item_entropy` | Shannon entropy of the target-item distribution |

---

## Configuration Parameters

See [`configs/system/system.yaml`](../configs/system/system.yaml) for the
complete annotated parameter reference.

Key parameters related to the scoring formula:

| Parameter | Default | Description |
|---|---|---|
| `lambda_bm25` | 0.3 | λ — BM25 weight in the base retrieval score |
| `weight_es_score` | 0.7 | α — base-score weight in the hybrid score |
| `weight_user_taste_short` | 0.3 | β — short-term vs long-term preference ratio |
| `threshold_similarity` | 0.3 | τ — if `max(S̃_base)` across all candidates is below this value, no item is shown in that round and the dialogue continues with the next round |
| `mode_refinement` | free | Dialogue mode: `free` / `static` / `adaptive` |
