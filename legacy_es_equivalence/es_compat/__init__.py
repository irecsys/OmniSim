"""
es_compat — Elasticsearch 8.13 replicas used ONLY by the verification tools (fixture replay,
index verification). The library (utils/retrieval/) uses standard components;
these modules reproduce what v1's ES did, so that a replay with them injected must match the
v1 recordings bit for bit (design decision: library = standard components, ES replicas only in
the verification tools).

  analysis.py      ES standard analyzer (UAX#29 + lowercase)
  bm25_lucene.py   Lucene BM25 over the kNN candidates (ES `match` on bm25_details)
  title_fuzzy.py   Lucene fuzzy `match` (fuzziness AUTO) on the title field
  vectors.py       W1 document rule, W1-q query rule, ES kNN scoring, raw _source vectors
  inject.py        install(): swaps the library's replacement points R1–R5 for these

This package must not import the library at module level (inject.py does, lazily).
"""
