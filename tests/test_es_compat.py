"""
Unit tests for the ES replicas used by the verification tools (legacy_es_equivalence/es_compat/). No ES, no API:
W1-q, the standard-analyzer emulation, the Lucene fuzzy pieces and the Lucene BM25.
"""
import numpy as np
import pytest

from legacy_es_equivalence.es_compat.analysis import standard_tokenize
from legacy_es_equivalence.es_compat.title_fuzzy import auto_max_edits, encoded_length, osa_distance
from legacy_es_equivalence.es_compat.vectors import es_knn_query, es_query_is_normalised, es_stored_vectors, panama_sqmag
from test_local_retrieval import movies   # noqa: F401  (fixture)
from utils.retrieval.local_search import knn_query


def test_query_norm_library_vs_es_rule():
    near = np.zeros(64)
    near[0] = np.sqrt(1 + 5e-5)                             # |‖q‖²−1| = 5e-5 < 1e-4
    far = np.zeros(64)
    far[0] = np.sqrt(1 + 2e-4)
    assert not es_query_is_normalised(near) and es_query_is_normalised(far)
    assert np.array_equal(es_knn_query(near), near.astype(np.float32).astype(np.float64))
    assert abs(np.linalg.norm(es_knn_query(far)) - 1) < 1e-6
    assert abs(np.linalg.norm(knn_query(near)) - 1) < 1e-12        # library: always normalised


def test_panama_sqmag_close_to_exact():
    v = np.random.default_rng(0).standard_normal(1536)
    assert abs(float(panama_sqmag(v)) - float((v.astype(np.float32).astype(np.float64) ** 2).sum())) < 1e-2


@pytest.mark.parametrize('text,tokens', [
    ("Don't Click", ["don't", 'click']),
    ('10,000 Days', ['10,000', 'days']),
    ('MacGyver: Lost', ['macgyver', 'lost']),
    ('U.S. army 7.5', ['u.s', 'army', '7.5']),
    ('Sci-Fi', ['sci', 'fi']),
    ('foo_bar', ['foo_bar']),
    ('日本語', ['日', '本', '語']),
    ('Bon appétit', ['bon', 'appétit']),
    ('Brand™ shirt', ['brand', '™', 'shirt']),
    ('', []),
])
def test_standard_tokenize(text, tokens):
    assert standard_tokenize(text) == tokens


def test_auto_edits_and_osa_distance():
    assert [auto_max_edits('ab'), auto_max_edits('abc'), auto_max_edits('abcde'), auto_max_edits('abcdef')] == [0, 1, 1, 2]
    assert osa_distance('wish', 'fish', 2) == 1
    assert osa_distance('blily', 'billy', 2) == 1           # adjacent transposition counts once
    assert osa_distance('abc', 'abcdef', 1) == 2            # beyond the limit → limit + 1


def test_lucene_length_encoding():
    assert [encoded_length(n) for n in (0, 1, 23)] == [0, 1, 23]   # exact below 24
    assert encoded_length(100) <= 100 and encoded_length(10 ** 6) <= 10 ** 6


def test_bm25_lucene_formula_and_repeats():
    from legacy_es_equivalence.es_compat.bm25_lucene import LuceneBM25Index
    from legacy_es_equivalence.es_compat.title_fuzzy import B as B_, K1, encoded_length
    import math
    docs = [['a', 'b', 'b'], ['b', 'c'], [], ['c'] * 300]
    ix = LuceneBM25Index(docs)
    assert ix.doc_count == 3 and ix.avgdl == (3 + 2 + 300) / 3             # empty doc not counted
    assert ix.length[3] == encoded_length(300) != 300                       # lossy SmallFloat length
    idf_b = math.log(1 + (3 - 2 + 0.5) / (2 + 0.5))
    norm0 = K1 * (1 - B_ + B_ * 3 / ix.avgdl)
    expect = idf_b * (K1 + 1) * 2 / (2 + norm0)
    assert ix.scores(['b'], [0, 1, 2, 3]).keys() == {0, 1}
    assert abs(ix.scores(['b'], [0])[0] - expect) < 1e-12
    assert abs(ix.scores(['b', 'b'], [0])[0] - 2 * expect) < 1e-12          # repeated token counts twice
    assert ix.scores(['zzz'], [0, 1, 2, 3]) == {}
    assert set(ix.scores(['c'], [1])) == {1}                               # only the given rows are scored


def test_bm25_lucene_backend_api(movies):
    from legacy_es_equivalence.es_compat import bm25_lucene
    _, b = movies
    cfg = {'bm25_details_field': 'bm25_details'}
    rows = list(range(b.n))
    assert bm25_lucene.bm25_raw_scores(b, cfg, '  ,;  ', rows) == {}        # no token → no hits
    hits = bm25_lucene.bm25_raw_scores(b, cfg, 'Fiction FICTION', rows)
    assert len(hits) == 3 and all(v > 0 for v in hits.values())         # Beta, Gamma, Epsilon
    assert all(isinstance(v, float) and v == float(str(np.float32(v))) for v in hits.values())
    with pytest.raises(ValueError):
        bm25_lucene.bm25_raw_scores(b, {'bm25_details_field': 'other'}, 'fiction', rows)


def test_w1_document_rule_threshold():
    # |‖v‖² − 1| just below / above 1e-4 → stored raw / normalised (ES 8.13)
    below = np.array([1.0, 0, 0, 0]) * np.sqrt(1 + 0.9e-4)
    above = np.array([1.0, 0, 0, 0]) * np.sqrt(1 + 1.1e-4)
    stored, normalised = es_stored_vectors(np.array([below, above]))
    assert normalised.tolist() == [False, True]
    assert np.array_equal(stored[0], below.astype(np.float32))
    assert abs(np.linalg.norm(stored[1].astype(np.float64)) - 1) < 1e-7

