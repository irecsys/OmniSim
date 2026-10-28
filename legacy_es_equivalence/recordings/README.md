# recordings — v1 Elasticsearch retrieval behaviour (cannot be re-recorded)

The reference for the v2 migration from Elasticsearch to the built-in local backend:
`../replay_fixtures.py` replays these records against the local backend call by call, and the
dialogues of phase A served as the dialogue-level reference. Versions, configuration, time and
checksums are in `manifest.json`.

## Why these cannot be recorded again

- They were recorded from a temporary Elasticsearch 8.13.0 / Lucene 9.10.0 instance (indices `imdb-2025`,
  `hm-2025`) that no longer exists. Elasticsearch's scores depend on its conditional vector
  normalisation, the HNSW graph, Lucene's BM25 and its analyzer, so they cannot be computed without it.
- The queries of phase A come from LLM dialogues (ThetaEdgeCloud, 823 requests), and
  `inputs/imdb_fixture/user_profiles.csv` holds LLM-generated profiles. Both are non-deterministic;
  a new recording would produce different data.
- Integrity check: `cd legacy_es_equivalence/recordings && sha256sum -c SHA256SUMS` (432 files).
  This README is not covered by `SHA256SUMS`.

## Layout

| Path | Contents | Produced by |
|---|---|---|
| `inputs/` | 4 pair lists (seeded order, used by prefix), `selection.json` (which interactions were moved into the recent past), the imdb_fixture profiles and `.profile_hash.json` | `legacy_es_equivalence/make_fixture_pairs.py` (at the time `scripts/make_fixture_pairs.py`, commit `79bdc62`¹); the profiles came from an LLM run of the recording script |
| `phase_a/` | 116 retrieval calls from 53 real dialogues (λ=0.3, τ=0.3): `calls.jsonl`, `vectors/<call_id>.npz`, `conversations.jsonl`, `runs*.jsonl`, `inputs/` (the pairs actually used in each step), `duplicates.json` (0 groups) | the recording script, through a hook in v1's retrieval function (removed) |
| `phase_b/` | Every phase A call replayed at λ ∈ {0.3, 0.0, 1.0}, τ=0 (348 records): `calls.jsonl`, `replay_summary.json` (λ=0.3 self-check: 0 mismatches), `inspect.md` (the report written at the time). No vectors of its own | the Elasticsearch grid-replay script (removed) |
| `direct_call/` | Hand-built calls, no LLM: 215 inputs / 621 records, plus `ref_lookups.jsonl` + `ref_vectors/` (25 reference-title lookups), `guest_samples.jsonl` (user_guest sampling, 100 per dataset), `summary.json` | the direct-call recording script (removed) |
| `inspect.md` | Final summary of all data: path coverage, exit_path, τ set | the fixture inspection script (removed) |
| `manifest.json`, `SHA256SUMS` | Metadata and per-file checksums | the manifest script (removed) |

The vector normalisation rule the recordings depend on (an Elasticsearch document vector is normalised
only when its squared norm, accumulated in float32, differs from 1 by more than 1e-4) was checked
document by document against the live instance during development; it is implemented in
`../es_compat/vectors.py`.

## Record format (`calls.jsonl`, the same in all three directories)

One line is one `search_items_by_embedding` call:

- `call_id`, `source` (`dialogue` | `direct_call`), `conv_id` / `seq_in_conv` (phase A only), `path` (phase A)
- `config`: configuration snapshot (es_index, col_*, lambda_bm25, threshold_similarity, weight_*, reference_blend_weight, …)
- `inputs`: top_k, query_text, excluded_ids (+ `excluded_ids_type`), year_range (+ `year_range_type`),
  exclude_genres, require_any_genre, `*_is_none`, n_reference_embeddings (+ `reference_embeddings_type`)
- `vectors`: names of the vectors stored for this record (query / user_long / user_short / reference, float64).
  An empty list together with `vectors_ref` means the vectors are taken from another record, see below
- `internal`: lam, alpha, beta, tau, fetch_k, num_candidates, bm25_ran, knn_filter, n_candidates,
  **max_base_score**, gate_passed, **exit_path** (gate_passed | gate_failed | early_return_747),
  `candidates` (in the order Elasticsearch's kNN returned them; each with item_id, cos_norm (= Elasticsearch `_score`), bm25_raw,
  bm25_norm (None = no BM25 hit, counted as 0.0 by v1), base_score, hybrid_score, item_emb_present)
- `output`: the function's return value as is. phase_b and direct_call ran at τ=0, so output is the full list
- `meta`: phase_b `{phase, src_call_id, src_label, src_path, src_mode, lambda, tau_run}`;
  direct_call `{kind, input_id, path, src_call_id, src_label, src_path, dataset, lambda, tau_run}`

### Where `vectors_ref` points

| Directory of the record | `vectors_ref` points to | Vector file |
|---|---|---|
| `phase_a/` | not used, every record has its own vectors | `phase_a/vectors/<call_id>.npz` |
| `phase_b/` | the **phase A** call_id (= `meta.src_call_id`) | `phase_a/vectors/<vectors_ref>.npz` |
| `direct_call/` | the first record (the λ=0.3 one) with the same `meta.input_id` **in direct_call itself** | `direct_call/vectors/<vectors_ref>.npz` |

- For calls with reference vectors, the blended query actually sent to Elasticsearch is stored as
  `<call_id>.blended.npz` (key `query_blended`), next to the first record.
- Calls whose vectors are all None (the H&M early return) have no npz.
- direct_call records with β=0 intentionally have no `user_short`: it was None, not lost.

## How the replay reads the records

The replay is `../replay_fixtures.py`. `../fixture_io.py` (`load_inputs(vec_dir, rec)`) rebuilds the
arguments of a record: it restores set / tuple / `[]` and None from the `*_type` fields; point `vec_dir`
at the directory from the table above.

1. Read `phase_b/calls.jsonl` and `direct_call/calls.jsonl`. Phase A is contained in the λ=0.3 records
   of phase_b (bit-identical).
2. For each record: config = the `config` snapshot + `meta.lambda` (`lambda_bm25`) + `meta.tau_run`
   (`threshold_similarity`); call the local backend with the recorded vectors. **No embedding API call.**
3. Compare `internal.candidates` (order, item_id, cos_norm, bm25_raw/norm, base_score, hybrid_score),
   `max_base_score`, `exit_path` and `output`. gate_passed for any τ is `n_candidates > 0 and max_base_score >= τ`;
   the τ set is `grid.tau_set` in `manifest.json`.
4. Grouping: `meta.lambda`; path (`meta.src_path` for phase_b, `meta.path` for direct_call); `exit_path`; `meta.kind`.
5. The two other call sites are reported separately:
   - `direct_call/ref_lookups.jsonl`: each record's `titles[]` holds title, hit, item_id, es_title, score;
     the returned vectors are in `ref_vectors/<lookup_id>.npz` (key `embeddings`)
   - `direct_call/guest_samples.jsonl`: body, n_hits, item_pool, _id, scores; distribution comparison only
6. When the phase A dialogues are used as a reference, take only `s6_use` from each group in
   `phase_a/duplicates.json` (there are 0 groups).

### Edge cases the local backend must reproduce

- **λ=1.0:** the best base score is 1.0 when the candidates have ≥ 2 distinct BM25 scores;
  otherwise (no hit, one hit, or all scores equal) `normalize_bm25` returns all zeros and it is 0.0.
  direct_call has one record with 0.0, checked separately. At λ=1.0 gate_passed carries no
  information; only the ranking is compared.
- **Vector normalisation:** document vectors accumulate ‖v‖² in float32 one dimension at a time and are
  normalised only when |‖v‖² − 1| > 1e-4 (only H&M is affected).
- **Reference-title lookup** used Elasticsearch `match` with `fuzziness: AUTO`, so phrases that are not
  titles also hit, e.g. "Death Wish" → "Killer Fish", "a feel-good movie" → "Nothing Too Good for a Cowboy".
  H&M has no `title` field, so its lookups are always empty; the blending branch can therefore only be
  tested on IMDB.
- **excluded_ids** is always empty in v1: the interactions are loaded only when `users_as_guest` is set to
  false, and no configuration sets it, so no rated item is ever excluded. Only direct_call covers it,
  including the empty id `''` of one IMDB item. H&M ids are passed as str; the Elasticsearch field was a keyword, while `_source` held an int.

## Traceability

v2 removed Elasticsearch, and the code that made these recordings was removed with it, so the recordings
cannot be made again from this repository. The recording code (the capture hook in v1's retrieval function,
the recording, direct-call, grid-replay, inspection and manifest scripts, and the Elasticsearch index
builder) last existed in full at commit `edcb598`¹. Re-recording would also need Elasticsearch 8.13.0
and paid LLM / embedding API access.

The Elasticsearch-based v1 is available on the `v1.x` branch.

¹ A commit in the private development repository; it is not part of this repository's history.
