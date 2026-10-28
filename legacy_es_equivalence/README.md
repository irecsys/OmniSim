# legacy_es_equivalence — v1 (Elasticsearch) equivalence check

OmniSim v1 retrieved items with Elasticsearch 8.13. v2 replaces it with a built-in local backend
(SQLite FTS5 + NumPy). This directory holds the evidence that the replacement is faithful: recordings
of what v1's Elasticsearch returned, and a tool that replays every recorded call against the v2 local
backend and compares the results call by call.

Nothing here is needed to use OmniSim. Elasticsearch does not have to be installed: the recordings
contain everything the replay needs. The library (`utils/`) does not import this directory;
`tests/test_local_retrieval.py::test_library_does_not_use_the_verification_tools` checks that.

## What the replay checks

The library uses standard components only (L2-normalised vectors, SQLite FTS5 BM25, …).
`es_compat/` reimplements the parts of Elasticsearch that v1's results depended on (Elasticsearch's
conditional vector normalisation, Lucene BM25 and its standard analyzer, fuzzy title matching).
In the default *injection mode* the replay installs these reimplementations into the library's
replacement points and then requires the library's retrieval pipeline to reproduce the recorded
v1 results. A mismatch therefore means that the pipeline around those components has changed.

Recorded calls: 348 calls replayed from 53 real dialogues at λ ∈ {0.3, 0.0, 1.0} (`recordings/phase_b/`)
and 621 hand-built calls covering the remaining code paths (`recordings/direct_call/`), 969 in total.

| `--part` | What is compared | Pass criterion |
|---|---|---|
| `vector` | All 969 records: the kNN candidates (independent of λ). For the 319 records at λ = 0 also scores, gate and output | candidate item ids and order identical, cosine error < 1e-5; at λ = 0 as for `full` |
| `full` | All 969 records at every recorded λ | exit path identical; top-10 item ids identical; cosine, base and hybrid score errors < 1e-5; returned entries identical (scores < 1e-5); the τ gate gives the same decision for every τ in `manifest.json` `grid.tau_set` (at λ = 1.0 only the ranking is compared, see below); a blended reference query is bit-identical |
| `w2` | The two other places v1 queried Elasticsearch: the 25 recorded reference-title lookups, and `user_guest` item sampling | reference-title lookups: real-title and typo lookups return the same top-1 item; multi-title lookups also return the same list; H&M lookups are empty; the returned vectors are bit-identical. Non-title phrases and score differences are reported only. `user_guest` sampling has no seed in v1, so only its distribution is compared (report only) |

At λ = 1.0 the gate carries no information: the best base score is 1.0 when the candidates have at
least two distinct BM25 scores and 0.0 otherwise, so only the ranking is compared.

Two kinds of difference are reported but not counted as failures: a single swap of two adjacent
candidates whose recorded Elasticsearch scores differ by less than 2e-6, and an approximate-kNN recall
miss on the Elasticsearch side. If either occurs too often at one λ (more than 5 swaps, or more than 1%
of the records), the run fails. Everything else is a failure. A run prints `PASS` or `FAIL` on its last
line and exits with 0 only on `PASS`.

`--entry library` calls the retrieval functions in `utils/retrieval/` directly. `--entry utils` goes
through the entry points the dialogue simulator uses (`utils.utils.search_items_by_embedding`,
`simulator._lookup_reference_embeddings`, `quick_start._sample_guest_items`); its results must be
identical to `--entry library`.

With `--components none` (or `--exclude <component>`) the replay runs the library exactly as shipped,
or with some components left out. These runs are report only: they show how far the standard components
alone are from v1, with no pass/fail threshold.

## Running it

From the repository root, with the indexes built from the bundled precomputed vectors (no API call):

```bash
unzip -o data/hm/hm.zip -d data/hm/
python scripts/build_local_index.py --config "configs/system/system.yaml configs/imdb/imdb.yaml"
python scripts/build_local_index.py --config "configs/system/system.yaml configs/hm/hm.yaml"
python scripts/index_digest.py --check tests/ci/index_digest.txt index/imdb index/hm

python legacy_es_equivalence/replay_fixtures.py --part vector --entry library
python legacy_es_equivalence/replay_fixtures.py --part vector --entry utils
python legacy_es_equivalence/replay_fixtures.py --part full --entry library
python legacy_es_equivalence/replay_fixtures.py --part full --entry utils
python legacy_es_equivalence/replay_fixtures.py --part w2 --entry library
python legacy_es_equivalence/replay_fixtures.py --part w2 --entry utils
```

Each command takes under a minute on a CPU-only machine. `--out <file.json>` writes the per-record results.

Integrity of the recordings:

```bash
cd legacy_es_equivalence/recordings && sha256sum -c SHA256SUMS
```

CI (`.github/workflows/test.yml`) runs the same steps on every push to `main` and on pull requests:
it unpacks `hm.csv`, builds both indexes, checks the index digest, runs pytest, and then runs the six
replay commands above; the job fails unless all six print `PASS`.

## What it does not show

The replay shows that the v2 retrieval pipeline reproduces v1's results when Elasticsearch's scoring is
substituted (injection mode); the shipped library alone differs slightly. It cannot be used to reproduce the numbers
reported in the UMAP '26 paper: those come from dialogues generated by LLMs, which differ from run
to run.

## Contents

| Path | What it is |
|---|---|
| `recordings/` | The v1 Elasticsearch recordings (see `recordings/README.md`). They cannot be recorded again: the Elasticsearch instance no longer exists, and the dialogue queries came from LLM calls |
| `es_compat/` | Reimplementations of the Elasticsearch behaviour v1 depended on: `analysis.py` (standard analyzer), `bm25_lucene.py` (Lucene BM25), `title_fuzzy.py` (fuzzy title `match`), `vectors.py` (conditional normalisation, kNN scoring, raw stored vectors), `inject.py` (installs them into the library's replacement points) |
| `replay_fixtures.py` | The replay and comparison tool described above |
| `fixture_io.py` | Reads a recorded call and rebuilds its arguments; used by `replay_fixtures.py` |
| `verify_local_index.py` | Offline checks of a local index against the items CSV and the recorded vectors |
| `make_fixture_pairs.py` | Writes the seeded user–item pair lists the recordings were made from, and the `data/imdb_fixture/` dataset (5 IMDB users with interactions moved into the recent past, so that the short-term profile differs from the long-term one) |
| `compare_dialogue_corpora.py` | Dialogue-level comparison of v1 and v2 dialogues generated for the same pairs. The v1 and v2 dialogue corpora its `main()` reads are not shipped, so it cannot be run as is; it is kept because it documents the comparison and because the tests use its `classify()` and `bot_turns()` |

## Paths inside the historical files

The recordings were made while this code lived elsewhere in the repository. The following files are
kept exactly as recorded and still show the old paths; some of them are covered by `SHA256SUMS`:
`recordings/manifest.json`, `recordings/inspect.md`, `recordings/phase_b/inspect.md`,
`recordings/phase_a/runs.jsonl`, `recordings/phase_a/runs_end.jsonl`, and the header comment of
`data/imdb_fixture/imdb_fixture.yaml`.

| Old path | Now |
|---|---|
| `tests/fixtures/es_golden/` | `legacy_es_equivalence/recordings/` |
| `scripts/es_compat/` | `legacy_es_equivalence/es_compat/` |
| `scripts/replay_fixtures.py` | `legacy_es_equivalence/replay_fixtures.py` |
| `scripts/fixture_io.py` | `legacy_es_equivalence/fixture_io.py` |
| `scripts/verify_local_index.py` | `legacy_es_equivalence/verify_local_index.py` |
| `scripts/make_fixture_pairs.py` | `legacy_es_equivalence/make_fixture_pairs.py` |
| `scripts/compare_dialogue_corpora.py` | `legacy_es_equivalence/compare_dialogue_corpora.py` |
| `scripts/record_fixtures.py`, `scripts/run_fixture_recording.sh`, `scripts/record_direct_calls.py`, `scripts/replay_grid_es.py`, `scripts/inspect_fixtures.py`, `scripts/write_fixture_manifest.py`, `utils/fixture_capture.py` | Removed: the recording code needed the live Elasticsearch instance |
| `scripts/verify_es_norm_rule.py`, `tests/fixtures/es_norm_rule/` | Removed: development-time evidence for the vector normalisation rule, which is implemented in `es_compat/vectors.py` |

The header comment that `make_fixture_pairs.py` writes is kept as it was at recording time, so it
matches the header of `data/imdb_fixture/imdb_fixture.yaml` byte for byte. The repository file has one
more line, `index_dir: index/imdb`, added after the recordings were made so that this dataset reuses
the IMDB local index.

`recordings/manifest.json` records `embeddings_endpoint`, the address of the Azure OpenAI resource that
was used when the recordings were made. The recordings contain no keys or other credentials.
