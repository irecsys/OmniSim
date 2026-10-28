"""
Lucene fuzzy `match` on the "title" field — the ES behaviour behind v1's reference-title lookup
(simulator.py:93-115), as an R4 replacement (`match_title`) for the verification tools.

Scoring follows ES 8.13 / Lucene 9.10 so that the top-1 agrees:
  - query and titles analysed with the standard analyzer (analysis.py)
  - per query token, AUTO edits by code-point length: < 3 → 0, < 6 → 1, else 2 (prefix_length 0,
    transpositions on → optimal-string-alignment distance), at most 50 expansions (max_expansions)
    kept by boost, then by term
  - each expansion is boosted by 1 − ed / min(len(term), len(query token)) (FuzzyTermsEnum) and all
    expansions of one token share the largest document frequency among them
    (top_terms_blended_freqs rewrite, BlendedTermQuery)
  - BM25 as ES scores it (LegacyBM25Similarity: k1=1.2, b=0.75, (k1+1) factor,
    idf = ln(1 + (N − df + 0.5) / (df + 0.5)), field length through Lucene's SmallFloat byte4 norm)
  - clause scores summed (match operator OR); top-1 ties go to the lower row (ES doc order)
v1 hit rule: no score threshold — any document matching at least one query token within its edit
distance is a hit.
"""
import math
import threading

from .analysis import standard_tokenize

TITLE_FIELD = 'title'          # hard-coded in v1, deliberately not config['col_title']
MAX_EXPANSIONS = 50
K1, B = 1.2, 0.75


# ── Lucene SmallFloat (norm encoding of the field length) ─────────────────────
def _long_to_int4(i):
    num_bits = i.bit_length()
    if num_bits < 4:
        return i
    shift = num_bits - 4
    return ((i >> shift) & 0x07) | ((shift + 1) << 3)


def _int4_to_long(i):
    bits = i & 0x07
    shift = (i >> 3) - 1
    return bits if shift == -1 else (bits | 0x08) << shift


_NUM_FREE_VALUES = 255 - _long_to_int4(2 ** 31 - 1)


def encoded_length(n):
    """Field length as Lucene's BM25 sees it after the byte4 norm round trip."""
    b = n if n < _NUM_FREE_VALUES else _NUM_FREE_VALUES + _long_to_int4(n - _NUM_FREE_VALUES)
    return b if b < _NUM_FREE_VALUES else _NUM_FREE_VALUES + _int4_to_long(b - _NUM_FREE_VALUES)


# ── fuzzy matching ────────────────────────────────────────────────────────────
def auto_max_edits(token):
    n = len(token)
    return 0 if n < 3 else (1 if n < 6 else 2)


def osa_distance(a, b, limit):
    """Optimal-string-alignment (Damerau, adjacent transpositions) distance, or limit+1 if larger."""
    if abs(len(a) - len(b)) > limit:
        return limit + 1
    prev2, prev = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)
        if min(cur) > limit:
            return limit + 1
        prev2, prev = prev, cur
    return prev[-1]


class TitleIndex:
    """Inverted index of the "title" field, built once per backend."""

    def __init__(self, backend):
        self.postings = {}             # term → {row: tf}
        self.length = []               # encoded field length per row
        n_docs, total = 0, 0
        for r, item in enumerate(backend.items):
            toks = standard_tokenize(item.get(TITLE_FIELD) or '')
            self.length.append(encoded_length(len(toks)))
            if toks:
                n_docs += 1
                total += len(toks)
            for t in toks:
                self.postings.setdefault(t, {})
                self.postings[t][r] = self.postings[t].get(r, 0) + 1
        self.doc_count = n_docs
        self.avgdl = total / n_docs if n_docs else 0.0
        self.terms = sorted(self.postings)

    def expansions(self, token):
        """[(term, boost)] within AUTO edits of token, top MAX_EXPANSIONS by boost then term."""
        max_ed = auto_max_edits(token)
        out = []
        for term in self.terms:
            ed = 0 if term == token else (osa_distance(token, term, max_ed) if max_ed else max_ed + 1)
            if ed <= max_ed:
                out.append((term, 1.0 - ed / min(len(term), len(token))))
        out.sort(key=lambda x: (-x[1], x[0]))
        return out[:MAX_EXPANSIONS]

    def scores(self, title):
        """{row: score} of every document the title query matches."""
        total = {}
        for token in standard_tokenize(title):
            exp = self.expansions(token)
            if not exp:
                continue
            df = max(len(self.postings[t]) for t, _ in exp)
            idf = math.log(1 + (self.doc_count - df + 0.5) / (df + 0.5))
            for term, boost in exp:
                for r, tf in self.postings[term].items():
                    norm = K1 * (1 - B + B * self.length[r] / self.avgdl)
                    total[r] = total.get(r, 0.0) + boost * idf * (K1 + 1) * tf / (tf + norm)
        return total


_title_indexes = {}
_lock = threading.Lock()


def _title_index(backend):
    with _lock:
        if id(backend) not in _title_indexes:
            _title_indexes[id(backend)] = TitleIndex(backend)
        return _title_indexes[id(backend)]


def match_title(backend, title):
    """R4 replacement: (row, score) of ES's top-1 for a title query, or None when nothing matches."""
    scores = _title_index(backend).scores(title)
    if not scores:
        return None
    row = min(scores, key=lambda r: (-scores[r], r))
    return row, scores[row]
