"""
Lucene BM25 over the kNN candidates — the replica of the ES behaviour behind v1's BM25
step (utils/utils.py:793-816), as an R3 replacement for the verification tools.

`bm25_raw_scores(backend, config, query_text, rows)`, scored the way v1's ES `match` query on
bm25_details scores it:
  - query and documents analysed with the ES standard analyzer (analysis.py, verified against ES
    `_analyze` on every bm25_details value)
  - BM25 as ES 8.13 computes it, the same formula the reference-title lookup uses (ref_lookup.py,
    verified there): LegacyBM25Similarity k1=1.2, b=0.75 with the (k1+1) factor,
    idf = ln(1 + (N − df + 0.5) / (df + 0.5)), field length through Lucene's SmallFloat norm,
    N = documents with at least one token, avgdl = total tokens / N (whole-index statistics,
    the ES indices have one shard and no deleted documents)
  - one clause per query token, summed; a repeated token counts once per occurrence (match, OR)
  - the score is returned as ES returns `_score`: a float32, as its shortest decimal representation
Only the candidate rows are scored; the statistics are those of the whole index. They are computed
on the first call per process (all bm25_details analysed: H&M ~3 s); precomputing them would be an
internal optimisation of this verification tool only.
"""
import collections
import math
import threading

import numpy as np

from .analysis import standard_tokenize
from .title_fuzzy import B, K1, encoded_length


class LuceneBM25Index:
    """Per-document term frequencies and collection statistics for one tokenised text field."""

    def __init__(self, doc_tokens):
        self.tf = [collections.Counter(toks) for toks in doc_tokens]
        self.length = [encoded_length(len(toks)) for toks in doc_tokens]
        self.df = collections.Counter(t for c in self.tf for t in c)
        lengths = [len(toks) for toks in doc_tokens if toks]
        self.doc_count = len(lengths)
        self.avgdl = sum(lengths) / self.doc_count if self.doc_count else 0.0

    def idf(self, term):
        df = self.df[term]
        return math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))

    def scores(self, query_tokens, rows):
        """{row: score} for the rows that contain at least one query token."""
        out = {}
        for r in rows:
            tf_r, total, hit = self.tf[r], 0.0, False
            norm = K1 * (1 - B + B * self.length[r] / self.avgdl) if self.avgdl else K1
            for t in query_tokens:
                tf = tf_r.get(t, 0)
                if tf:
                    hit = True
                    total += self.idf(t) * (K1 + 1) * tf / (tf + norm)
            if hit:
                out[r] = total
        return out


_indexes = {}
_lock = threading.Lock()


def _field_index(backend, field):
    key = (id(backend), field)
    with _lock:
        if key not in _indexes:
            _indexes[key] = LuceneBM25Index(backend.field_tokens(field, standard_tokenize))
        return _indexes[key]


def bm25_raw_scores(backend, config, query_text, rows):
    """Raw BM25 scores {item_id: score} for the candidate rows that match query_text."""
    field = config.get('bm25_details_field', 'bm25_details')
    if field != backend.meta['bm25']['field']:
        raise ValueError(f"bm25_details_field '{field}' is not the indexed BM25 field "
                         f"'{backend.meta['bm25']['field']}'")
    tokens = standard_tokenize(query_text or '')
    if not tokens or not rows:
        return {}
    scores = _field_index(backend, field).scores(tokens, [int(r) for r in rows])
    return {backend.item_ids[r]: float(str(np.float32(s))) for r, s in scores.items()}
