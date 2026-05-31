# OmniSim API Reference

## Overview

OmniSim is a domain-agnostic, retrieval-grounded framework for generating
personalised conversational recommendation dialogues.  It combines
Elasticsearch-based hybrid retrieval (BM25 + dense kNN) with LLM-based
dialogue generation across three interaction modes: **Free**, **Static**, and
**Adaptive**.

---

## Module: `utils.utils`

Core utilities: LLM clients, similarity functions, retrieval, and dialogue-act generators.

### Similarity & Scoring

#### `cosine_sim(a, b) → float`
Returns the raw cosine similarity between two vectors in **[-1, 1]** (paper Eq. 3).  
Use `normalize_cosine()` before combining with other scores.

#### `normalize_cosine(s) → float`
Maps a raw cosine similarity to **[0, 1]** using `(s + 1) / 2` (paper Eq. 4).

#### `normalize_bm25(scores: dict) → dict`
Min-max normalises a dict of BM25 scores to **[0, 1]** (paper Eq. 4).  
Returns `{item_id: normalised_score}`.

#### `search_items_by_embedding(config, es, embedding_vector, top_k, ...)  → list[dict]`
Hybrid retrieval implementing **Equations 3–6** from the paper.

| Paper Symbol | Config Key | Default | Meaning |
|---|---|---|---|
| λ | `lambda_bm25` | 0.3 | BM25 weight in Eq. 5 |
| α | `weight_es_score` | 0.7 | Base-score weight in Eq. 6 |
| β | `weight_user_taste_short` | 0.3 | Short-term preference ratio |

**Scoring pipeline:**
1. kNN search → `S̃_cos(q, i)` ∈ [0, 1] (ES normalises automatically)
2. BM25 search on the same candidates, scoring the query against the **single aggregated `bm25_details` field** (`M_i,d` = all item attributes concatenated, built at index time) → min-max normalise → `S̃_BM25(q, M_i,d)`
3. `S̃_base = λ · S̃_BM25 + (1 − λ) · S̃_cos`  (Eq. 5)
4. `S̃_hybrid = α · S̃_base + (1 − α) · [β · S̃_cos(i, p_short) + (1 − β) · S̃_cos(i, p_long)]`  (Eq. 6)
5. **Threshold gate (paper §4.2):** if `max(S̃_base) < τ` across all candidates, return an empty list — the caller stays in the preference-elicitation phase and asks another question.  Only when `max(S̃_base) ≥ τ` does the system transition to the recommendation phase and return the top-`N` items ranked by `S̃_hybrid`.

**Key parameters:**
- `embedding_vector` — encoded user preference query
- `user_long_embedding` — user long-term preference vector (optional)
- `user_short_embedding` — user short-term preference vector (optional)
- `excluded_ids` — item IDs to exclude (already shown)
- `query_text` — raw text for BM25 matching (optional)
- `year_range` — `(min_year, max_year)` hard filter (optional)
- `exclude_genres` — genre strings to exclude (optional)
- `reference_embeddings` — embeddings of reference items blended into query (optional)

Returns a list of item dicts sorted by descending `score`, capped at `top_k`.

---

### LLM Clients

#### `get_openai_clients(config) → (client_chat, client_embeddings)`
Build LLM and embeddings clients from config.  Supports:
- `openai_provider: openai` — OpenAI native API
- `openai_provider: azure` — Azure OpenAI
- `openai_provider: thetaedgecloud` — ThetaEdgeCloud (Llama etc.)
- `embeddings_provider: sentence-transformers` — local, no API key needed

#### `ThetaEdgeCloudClient`
Drop-in wrapper that exposes the same `.chat.completions.create()` interface as
the OpenAI client.  Handles rate-limit back-off (exponential retry).

#### `SentenceTransformerEmbeddingsClient`
Local embeddings client using `sentence-transformers`.  Runs offline.

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
| `extract_attribute_value(config, item_row, attribute)` | Get structured or derived attribute value |
| `extract_attributes_for_asking(config, client_chat, item_row)` | Discover relevant attributes to ask about (Adaptive mode) |
| `extract_dynamic_description_from_chat(config, client_chat, chat_history)` | Summarise user intent from dialogue context |
| `extract_search_constraints(config, client_chat, user_context)` | Parse year range, genres, reference titles from user text |
| `extract_item_from_title(config, client_chat, title, category)` | Infer a descriptive item type from its title |

---

## Module: `utils.simulator`

Conversation simulation engines.  All three public functions share the same
signature and write the completed conversation to disk under
`chats/{dataset}/{mode}/{strategy}/{timestamp}/`.

### `simulate_free(config, dataset, user_id, target_item_id, client_chat, client_embeddings, es, user_profiles=None)`
Open-ended dialogue — the bot issues open-ended follow-up prompts and the
simulated user replies with free-form natural language derived from the target
item's metadata.  Each user turn is embedded and used directly for kNN search.

### `simulate_static(config, dataset, user_id, target_item_id, client_chat, client_embeddings, es, user_profiles=None)`
Schema-driven dialogue — the bot asks questions **only** about the predefined
`item_attributes` in the dataset YAML.  User answers are rephrased into natural
language and aggregated into a structured query.  Internally delegates to
`simulate_adaptive(attribute_by_openai=False)`.

### `simulate_adaptive(config, dataset, user_id, target_item_id, client_chat, client_embeddings, es, attribute_by_openai=True, user_profiles=None)`
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
| `es` | client | Connected Elasticsearch client |
| `user_profiles` | dict / None | Pre-built user profile dict keyed by user_id |

**Output filename pattern:**
```
{user_id}-{item_id}-{num_turns}-{rec_attempts}-{succeed}-{timestamp}.txt
```
`succeed=1` when the target item was accepted; `0` when max attempts exhausted.

---

## Module: `utils.quick_start`

### `run_simulation(config_file_list, mode_override=None, strategies_override=None, num_workers_override=None, chats_per_entry_override=None, pairs_file_override=None)`
Main entry point. Loads config, connects to ES, builds user profiles, and
dispatches all conversation generation tasks across a thread pool.

---

## Module: `utils.configurator`

### `Config(config_file_list: list[str])`
Merges `configs/system/system.yaml` with dataset-specific YAML files.
Dataset values take precedence over system defaults.

```python
config = Config(['configs/imdb/imdb.yaml'])
index  = config['es_index']            # KeyError if missing
thresh = config.get('threshold_similarity', 0.3)  # with default
```

---

## Module: `utils.dataset`

### `Dataset(config)`
Loads items, users, and interactions CSVs into DataFrames.

```python
ds = Dataset(config)
items        = ds.item_feat          # pd.DataFrame
users        = ds.user_feat          # pd.DataFrame or None
interactions = ds.inter_feat         # pd.DataFrame or None
```

---

## Module: `utils.user_profile_builder`

### `build_user_profiles(config, client_chat, client_embeddings, dataset) → dict`
For each user, constructs:
- `user_demographic` — natural-language summary from `users.csv`
- `user_likes_long` — long-term preference from all positive interactions
- `user_likes_short` — short-term preference from interactions within `short_term_days`
- `long_embedding` / `short_embedding` — dense vectors for hybrid retrieval

Profiles are cached with MD5 hashing; unchanged data is not recomputed.

---

## Module: `utils.evaluator`

### `evaluate_folder(config, client_chat, folder_path, output_csv, limit=None)`
Runs LLM-as-a-Judge evaluation on all `.txt` conversations in a folder.
Scores three dimensions on a 1–5 scale:
- **Language Fluency** — grammar, naturalness, tone
- **Conversational Quality** — appropriateness, proactivity, multi-turn consistency
- **Content Quality** — factuality, coverage, coherence, relevance

Results are saved to `output_csv` with per-conversation rationale.

---

## Module: `utils.metrics`

### `compute_metrics(folder_path) → dict`
Computes reference-free lexical diversity metrics on all `.txt` files:

| Metric | Description |
|---|---|
| `distinct_1` | Distinct unigram ratio |
| `distinct_2` | Distinct bigram ratio |
| `ttr` | Type-Token Ratio |
| `log_ttr` | Log TTR (length-robust) |
| `mtld` | Measure of Textual Lexical Diversity |
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
| `lambda_bm25` | 0.3 | λ — BM25 weight in base retrieval score (Eq. 5) |
| `weight_es_score` | 0.7 | α — base-score weight in hybrid formula (Eq. 6) |
| `weight_user_taste_short` | 0.3 | β — short-term vs long-term preference ratio (Eq. 6) |
| `threshold_similarity` | 0.3 | τ — if `max(S̃_base)` across all candidates is below this value, the system stays in elicitation mode instead of recommending |
| `mode_refinement` | free | Dialogue mode: `free` / `static` / `adaptive` |
