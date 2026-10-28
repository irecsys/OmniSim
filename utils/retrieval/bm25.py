"""
BM25 over the kNN candidates with SQLite FTS5 (replacement point R3 of local_search; the
verification tools in legacy_es_equivalence/ inject a Lucene replica here).

v1 runs an ES `match` query on the aggregated bm25_details field, restricted to the kNN candidate
ids, and keeps the raw `_score` of every candidate that matches. Here: FTS5 MATCH on items_fts
restricted to the candidate rowids; FTS5's bm25() is "lower is better", so the raw score is its
negation. Collection statistics (IDF, average length) are over the whole table, as in ES.

The tokenizer is the index's (config `bm25_tokenizer`, default unicode61 remove_diacritics 0;
trigram for CJK). The query is split into words (query_parse.py) and each word is passed as a
quoted phrase, which FTS5 tokenizes with the index tokenizer — with trigram that is a substring
match, and words shorter than 3 characters match nothing.

FTS5's bm25() (k1=1.2, b=0.75) differs from Lucene's (IDF form, tokenizer, length encoding), so
λ > 0 rankings differ from v1.
"""
from utils.retrieval.query_parse import to_fts5_query


def bm25_raw_scores(backend, config, query_text, rows):
    """Raw BM25 scores {item_id: score} for the candidate rows that match query_text."""
    field = config.get('bm25_details_field', 'bm25_details')
    if field != backend.meta['bm25']['field']:
        raise ValueError(f"bm25_details_field '{field}' is not the indexed BM25 field "
                         f"'{backend.meta['bm25']['field']}'")
    match = to_fts5_query(query_text)
    if match is None or not rows:
        return {}
    placeholders = ','.join('?' * len(rows))
    cur = backend.conn().execute(
        f'SELECT rowid, bm25(items_fts) FROM items_fts WHERE items_fts MATCH ? AND rowid IN ({placeholders})',
        [match] + [int(r) for r in rows])
    return {backend.item_ids[r]: -score for r, score in cur}
