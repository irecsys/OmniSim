"""
Unit tests for the local retrieval backend. No ES, no API, no repo data:
each test builds a tiny index with the real index builder in a temporary directory.
"""
import numpy as np
import pandas as pd
import pytest

from utils.retrieval.backend import LocalBackend
from utils.retrieval.filters import build_mask
from utils.retrieval.local_index import build_local_index
from utils.retrieval.local_search import search_items_local
from utils.utils import cosine_sim, normalize_bm25, normalize_cosine

DIM = 4


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def make_backend(tmp_path, rows, id_col='item_id', extra_config=None):
    """rows: list of dicts with id/title/category/details/(release_year)/vec."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k != 'vec'} for r in rows])
    df['embedding_vector'] = [str([float(x) for x in r['vec']]) for r in rows]
    ds = tmp_path / 'data' / 'toy'
    ds.mkdir(parents=True)
    df.to_csv(ds / 'items.csv', index=False)
    config = {
        'data_path': str(tmp_path / 'data'), 'dataset': 'toy', 'file_items': 'items.csv',
        'col_itemid': id_col, 'col_title': 'title', 'col_category': 'category', 'col_details': 'details',
        'precomputed_embedding_col': 'embedding_vector',
        'embedding_fields': ['title', 'category', 'details'],
        'item_attributes': {'category': 'genre', 'release_year': 'year', 'lang': 'language'},
        'lambda_bm25': 0.0, 'weight_es_score': 0.7, 'weight_user_taste_short': 0.3,
        'threshold_similarity': 0.0,
    }
    config.update(extra_config or {})
    build_local_index(config, str(tmp_path / 'index'))
    return config, LocalBackend(str(tmp_path / 'index'))


def movie_rows():
    base = [_unit([1, 0, 0, 0]), _unit([0.9, 0.1, 0, 0]), _unit([0.8, 0.2, 0, 0]),
            _unit([0.5, 0.5, 0, 0]), _unit([0, 1, 0, 0]), _unit([0, 0, 1, 0])]
    meta = [('tt1', 'Alpha', 'Drama, Crime', 1995, 'en'), ('', 'Beta', 'Science Fiction', 2001, 'fr'),
            ('tt3', 'Gamma', 'Fiction', None, ''), ('tt4', 'Delta', 'Comedy', 1999, 'en'),
            ('tt5', 'Epsilon', 'Fiction Science', 2010, 'en'), ('tt6', 'Zeta', 'Horror', 1980, 'de')]
    return [{'item_id': i, 'title': t, 'category': c, 'details': f'{t} details', 'release_year': y,
             'lang': l, 'vec': v} for (i, t, c, y, l), v in zip(meta, base)]


@pytest.fixture
def movies(tmp_path):
    return make_backend(tmp_path, movie_rows())


# ── scores ───────────────────────────────────────────────────────────────────
def test_cosine_score_is_eq4(movies):
    config, b = movies
    q = [0.3, 0.7, 0.1, 0.0]
    trace = {}
    search_items_local(config, b, q, top_k=2, trace=trace)
    qn = _unit(q)
    for c in trace['candidates']:
        r = b.item_ids.index(c['item_id'])
        expected = float((1.0 + b.vectors64[r] @ qn) / 2.0)          # plain cosine, float64
        assert c['cos_norm'] == expected


def test_index_stores_l2_normalised_float32(tmp_path):
    # every row is normalised, also one whose |‖v‖² − 1| is below ES's 1e-4 threshold
    near = np.array([1.0, 0, 0, 0]) * np.sqrt(1 + 0.9e-4)
    far = np.array([3.0, 4.0, 0, 0])
    rows = [{'item_id': 'a', 'title': 'A', 'category': 'x', 'details': '', 'vec': near},
            {'item_id': 'b', 'title': 'B', 'category': 'x', 'details': '', 'vec': far}]
    _, b = make_backend(tmp_path, rows)
    assert b.vectors.dtype == np.float32
    assert np.array_equal(b.vectors[0], np.array([1, 0, 0, 0], dtype=np.float32))
    assert np.array_equal(b.vectors[1], (far / 5.0).astype(np.float32))
    assert b.item_embedding(1) == b.vectors[1].tolist()


# ── candidate selection, gate, fallback ──────────────────────────────────────
def test_excluded_ids_apply_before_topk(movies):
    config, b = movies
    trace = {}
    search_items_local(config, b, [1, 0, 0, 0], top_k=1, excluded_ids={'tt1'}, trace=trace)
    ids = [c['item_id'] for c in trace['candidates']]
    assert 'tt1' not in ids and len(ids) == 3                # fetch_k = 3·top_k still filled


def test_excluded_ids_str_match_including_empty_id(movies):
    config, b = movies
    out = search_items_local(config, b, [0.9, 0.1, 0, 0], top_k=2, excluded_ids={''})
    assert '' not in [o['item_id'] for o in out]
    out = search_items_local(config, b, [0.9, 0.1, 0, 0], top_k=2)
    assert '' in [o['item_id'] for o in out]


def test_int_ids_keep_type_and_match_str_exclusion(tmp_path):
    rows = [{'pid': 834333001, 'title': 'Top', 'category': 'Women', 'details': '', 'vec': _unit([1, 0, 0, 0])},
            {'pid': 900000002, 'title': 'Tee', 'category': 'Men', 'details': '', 'vec': _unit([0.9, 0.1, 0, 0])}]
    config, b = make_backend(tmp_path, rows, id_col='pid')
    out = search_items_local(config, b, [1, 0, 0, 0], top_k=2)
    assert [o['pid'] for o in out] == [834333001, 900000002] and isinstance(out[0]['pid'], int)
    out = search_items_local(config, b, [1, 0, 0, 0], top_k=2, excluded_ids={'834333001'})
    assert [o['pid'] for o in out] == [900000002]


def test_gate_uses_all_fetch_k_candidates(movies):
    config, b = movies
    # λ=0, α=0: ranking follows the user profile only; the gate still looks at max(base) over all candidates
    cfg = dict(config, weight_es_score=0.0, threshold_similarity=0.99)
    q = [1, 0, 0, 0]
    trace = {}
    out = search_items_local(cfg, b, q, top_k=1, user_long_embedding=[0, 1, 0, 0], trace=trace)
    assert trace['exit_path'] == 'gate_passed' and len(trace['candidates']) == 3
    assert out[0]['item_id'] != 'tt1'                        # top hybrid item is not the max-base item
    assert search_items_local(dict(cfg, threshold_similarity=1.01), b, q, top_k=1,
                              user_long_embedding=[0, 1, 0, 0]) == []


def test_no_candidates_returns_empty(movies):
    config, b = movies
    trace = {}
    assert search_items_local(config, b, [1, 0, 0, 0], top_k=2, year_range=(1900, 1901), trace=trace) == []
    assert trace['exit_path'] == 'gate_failed' and trace['n_candidates'] == 0


def test_fallback_without_embedding(movies):
    config, b = movies
    out = search_items_local(config, b, None, top_k=3, exclude_genres=['Horror'])
    assert [o['item_id'] for o in out] == ['tt1', '', 'tt3']                 # index order, filtered
    assert 'details' in out[0] and 'score' not in out[0]
    assert out[2]['lang'] == '' and out[2]['release_year'] is None           # no empty-value filtering
    assert list(out[0].keys()) == ['item_id', 'title', 'details', 'category', 'release_year', 'lang']


def test_attribute_empty_values_dropped_in_scored_results(movies):
    config, b = movies
    out = search_items_local(config, b, [0.8, 0.2, 0, 0], top_k=6)
    gamma = next(o for o in out if o['item_id'] == 'tt3')
    assert 'lang' not in gamma and 'release_year' not in gamma


# ── base and hybrid score ────────────────────────────────────────────────────────────
def test_alpha_one_without_profile(movies):
    config, b = movies
    trace = {}
    out = search_items_local(config, b, [0.3, 0.7, 0, 0], top_k=6, trace=trace)
    for c in trace['candidates']:
        assert c['hybrid_score'] == c['base_score']
    assert [o['item_id'] for o in out] == [c['item_id'] for c in sorted(
        trace['candidates'], key=lambda c: c['base_score'], reverse=True)]


def _hybrid_expected(b, iid, base, long_e, short_e, alpha, beta):
    raw = b.item_embedding(b.item_ids.index(iid))
    long_sim = normalize_cosine(cosine_sim(raw, long_e))
    if short_e is None:
        return alpha * base + (1 - alpha) * long_sim
    short_sim = normalize_cosine(cosine_sim(raw, short_e))
    return alpha * base + (1 - alpha) * (beta * short_sim + (1 - beta) * long_sim)


def test_beta_branches_use_item_embedding(tmp_path):
    rows = movie_rows()
    rows[0]['vec'] = np.array([1.0, 0, 0, 0]) * 1.01        # non-unit in the CSV, normalised in the index
    config, b = make_backend(tmp_path, rows)
    long_e, short_e = [0.2, 0.9, 0.1, 0.0], [0.7, 0.1, 0.0, 0.3]
    for short in (short_e, None):
        trace = {}
        search_items_local(config, b, [1, 0.2, 0, 0], top_k=2, user_long_embedding=long_e,
                           user_short_embedding=short, trace=trace)
        beta = 0.3 if short is not None else 0.0
        assert trace['beta'] == beta
        for c in trace['candidates']:
            assert c['hybrid_score'] == _hybrid_expected(b, c['item_id'], c['base_score'], long_e, short, 0.7, 0.3)


def test_reference_blend_is_verbatim(movies):
    config, b = movies
    q = [0.1, 0.2, 0.3, 0.4]
    refs = [b.item_embedding(0), b.item_embedding(3)]
    trace = {}
    search_items_local(config, b, q, top_k=2, reference_embeddings=refs, trace=trace)
    ref_avg = np.mean(refs, axis=0).tolist()
    assert trace['query_blended'] == [(1 - 0.3) * a + 0.3 * r for a, r in zip(q, ref_avg)]


def test_normalize_bm25_boundaries():
    assert normalize_bm25({}) == {}
    assert normalize_bm25({'a': 3.2}) == {'a': 0.0}                           # single hit
    assert normalize_bm25({'a': 2.0, 'b': 2.0}) == {'a': 0.0, 'b': 0.0}       # all equal
    assert normalize_bm25({'a': 1.0, 'b': 3.0, 'c': 2.0}) == {'a': 0.0, 'b': 1.0, 'c': 0.5}


# ── filters ──────────────────────────────────────────────────────────────────
def _ids(b, mask):
    return [b.item_ids[r] for r in np.flatnonzero(mask)]


def test_exclude_genres_is_word_level_or(movies):
    config, b = movies
    # "Science Fiction" excludes every category containing "science" OR "fiction"
    assert _ids(b, build_mask(b, config, exclude_genres=['Science Fiction'])) == ['tt1', 'tt4', 'tt6']
    assert _ids(b, build_mask(b, config, exclude_genres=['!!!'])) == b.item_ids     # no tokens → no exclusion


def test_require_any_genre_is_phrase(movies):
    config, b = movies
    assert _ids(b, build_mask(b, config, require_any_genre=['Science Fiction'])) == ['']
    assert _ids(b, build_mask(b, config, require_any_genre=['crime', 'Comedy'])) == ['tt1', 'tt4']
    assert _ids(b, build_mask(b, config, require_any_genre=['!!!'])) == []


def test_year_range_null_and_missing_column(movies, tmp_path):
    config, b = movies
    assert _ids(b, build_mask(b, config, year_range=(1990, 2001))) == ['tt1', '', 'tt4']  # tt3 is null
    rows = [{'pid': 1, 'title': 'T', 'category': 'Women', 'details': '', 'vec': _unit([1, 0, 0, 0])}]
    cfg2, b2 = make_backend(tmp_path / 'hm', rows, id_col='pid')
    assert not build_mask(b2, cfg2, year_range=(1990, 1999)).any()               # H&M: no release_year



# ── query_parse + BM25 ───────────────────────────────────────────────────────
import sqlite3

from utils.retrieval.bm25 import bm25_raw_scores
from utils.retrieval.query_parse import fts5_tokens, to_fts5_query


def _fts5_split(text):
    """How FTS5 unicode61 (remove_diacritics 0) itself tokenizes text."""
    con = sqlite3.connect(':memory:')
    con.execute("CREATE VIRTUAL TABLE t USING fts5(x, tokenize=\"unicode61 remove_diacritics 0\")")
    con.execute("INSERT INTO t(rowid, x) VALUES (1, ?)", (text,))
    con.execute("CREATE VIRTUAL TABLE v USING fts5vocab(t, instance)")
    return [r[0] for r in con.execute("SELECT term FROM v ORDER BY offset")]


@pytest.mark.parametrize('text', [
    'Don\'t "quote" me: NEAR(a b) AND OR NOT * ^ - + ( ) col:val',
    'sci-fi 7.5 10,000 U.S. foo_bar', '日本語のテキスト', 'Bon appétit café', '2006 1990s', '   ', '',
])
def test_query_parse_matches_fts5_tokenizer(text):
    assert [t.lower() for t in fts5_tokens(text)] == _fts5_split(text)


def test_query_parse_quotes_and_keeps_duplicates():
    assert to_fts5_query('AND or NEAR') == '"AND" OR "or" OR "NEAR"'
    assert to_fts5_query('drama drama crime') == '"drama" OR "drama" OR "crime"'
    assert to_fts5_query('2006') == '"2006"'
    assert to_fts5_query('?!*"-') is None and to_fts5_query('') is None


def bm25_rows():
    vec = [_unit([1, 0, 0, 0]), _unit([0.9, 0.1, 0, 0]), _unit([0.8, 0.2, 0, 0]), _unit([0.7, 0.3, 0, 0])]
    text = [('A', 'Drama', 'a dark crime story about crime'), ('B', 'Comedy', 'a light comedy'),
            ('C', 'Drama', 'crime'), ('D', 'Horror', 'a scary night')]
    return [{'item_id': f'i{k}', 'title': t, 'category': c, 'details': d, 'vec': v}
            for k, ((t, c, d), v) in enumerate(zip(text, vec))]


@pytest.fixture
def bm25_backend(tmp_path):
    return make_backend(tmp_path, bm25_rows(), extra_config={'bm25_details_field': 'bm25_details'})


def test_bm25_sign_and_candidate_restriction(bm25_backend):
    config, b = bm25_backend
    raw = bm25_raw_scores(b, config, 'crime', [0, 1, 2, 3])
    assert set(raw) == {'i0', 'i2'} and all(v > 0 for v in raw.values())   # negated FTS5 bm25()
    assert bm25_raw_scores(b, config, 'crime', [1, 2]) == {'i2': raw['i2']}  # only candidates, global stats


def test_bm25_repeated_query_token_counts_twice(bm25_backend):
    config, b = bm25_backend
    once = bm25_raw_scores(b, config, 'crime', [0, 1, 2, 3])
    twice = bm25_raw_scores(b, config, 'crime crime', [0, 1, 2, 3])
    assert twice['i0'] == pytest.approx(2 * once['i0'])


def test_bm25_no_token_gives_base_from_cosine_only(bm25_backend):
    config, b = bm25_backend
    assert bm25_raw_scores(b, config, '?!', [0, 1]) == {}
    trace = {}
    search_items_local(dict(config, lambda_bm25=0.3), b, [1, 0, 0, 0], top_k=1, query_text='?!', trace=trace)
    for c in trace['candidates']:
        assert c['bm25_norm'] is None and c['base_score'] == 0.7 * c['cos_norm']


def test_bm25_min_max_over_hits_only(bm25_backend):
    config, b = bm25_backend
    trace = {}
    search_items_local(dict(config, lambda_bm25=1.0), b, [1, 0, 0, 0], top_k=2, query_text='crime', trace=trace)
    norm = {c['item_id']: c['bm25_norm'] for c in trace['candidates']}
    assert sorted(v for v in norm.values() if v is not None) == [0.0, 1.0]   # two hits → min-max
    assert norm['i1'] is None and trace['max_base_score'] == 1.0


def test_lambda_one_degenerate_single_hit_gives_zero(bm25_backend):
    config, b = bm25_backend
    trace = {}
    out = search_items_local(dict(config, lambda_bm25=1.0, threshold_similarity=0.3), b, [1, 0, 0, 0],
                             top_k=2, query_text='scary', trace=trace)
    assert trace['max_base_score'] == 0.0 and out == []                   # one hit → normalised 0 → gate fails


def test_bm25_field_mismatch_raises(bm25_backend):
    config, b = bm25_backend
    with pytest.raises(ValueError):
        bm25_raw_scores(b, dict(config, bm25_details_field='other'), 'crime', [0])


def test_long_query_does_not_fail(bm25_backend):
    config, b = bm25_backend
    words = ' '.join(f'w{i}' for i in range(5000)) + ' crime'
    assert set(bm25_raw_scores(b, config, words, [0, 1, 2, 3])) == {'i0', 'i2'}


# ── reference-title lookup + user_guest sampling ─────────────────────────────
from utils.retrieval.ref_lookup import lookup_reference_embeddings
from utils.retrieval.sampling import sample_items


def test_ref_lookup_hit_rules_and_vectors(tmp_path):
    rows = movie_rows()
    rows[5]['title'] = 'Killer Fish'
    config, b = make_backend(tmp_path, rows)
    trace = []
    embs = lookup_reference_embeddings(b, config, ['Alpah', 'zzqx nothing', 'gamma', 'Death Wish'], trace=trace)
    # transposition typo hits (ratio 80 = cutoff), unrelated text misses, case-insensitive,
    # one shared word is not enough (v1/ES matched "Death Wish" → "Killer Fish")
    assert [t['item_id'] for t in trace] == ['tt1', None, 'tt3', None]
    assert trace[0]['score'] == 80.0 and trace[2]['score'] == 100.0
    assert embs == [b.item_embedding(0), b.item_embedding(2)]          # list keeps title order, misses skipped


def test_match_title_cutoff(movies):
    from utils.retrieval.ref_lookup import SCORE_CUTOFF, match_title
    _, b = movies
    assert SCORE_CUTOFF == 80
    assert match_title(b, 'Epsilon') == (4, 100.0)
    assert match_title(b, 'Epsilom') is not None                       # one substitution in 7 letters
    assert match_title(b, 'Eps') is None                               # 3/7 of the title: below the cutoff


def test_ref_lookup_uses_literal_title_field(tmp_path):
    rows = [{'pid': 1, 'productName': 'Alpha', 'category': 'Women', 'details': '', 'vec': _unit([1, 0, 0, 0])}]
    df_rows = [dict(r) for r in rows]
    config, b = make_backend(tmp_path, df_rows, id_col='pid', extra_config={'col_title': 'productName'})
    assert lookup_reference_embeddings(b, config, ['Alpha']) == []    # no "title" column (H&M) → always []


def test_ref_lookup_tie_goes_to_lower_row(tmp_path):
    rows = movie_rows()
    rows[3]['title'] = 'Alpha'                                        # same title as row 0
    config, b = make_backend(tmp_path, rows)
    trace = []
    lookup_reference_embeddings(b, config, ['Alpha'], trace=trace)
    assert trace[0]['item_id'] == 'tt1'


def test_sample_items_structure(tmp_path):
    rows = [{'pid': 100 + k, 'title': f'T{k}', 'category': 'Women', 'details': '', 'vec': _unit([1, k, 0, 0])}
            for k in range(6)]
    config, b = make_backend(tmp_path, rows, id_col='pid')
    s = sample_items(b, 4, seed=1)
    assert len(s) == 4 and len({x['pid'] for x in s}) == 4 and isinstance(s[0]['pid'], int)
    assert list(s[0].keys()) == b.columns + ['embedding']
    assert s[0]['embedding'] == b.item_embedding(b.item_ids.index(s[0]['pid']))
    assert len(sample_items(b, 600)) == 6                              # size > N → all items, like ES
    assert [x['pid'] for x in sample_items(b, 6, seed=3)] == [x['pid'] for x in sample_items(b, 6, seed=3)]


# ── standard components: category words, FTS5 tokenizer option, no ES replicas in the library ──
def test_category_tokens_are_lowercase_words():
    from utils.retrieval.filters import category_tokens
    assert category_tokens('Sci-Fi, TV_Movie 2 Bon appétit') == ['sci', 'fi', 'tv_movie', '2', 'bon', 'appétit']
    assert category_tokens("Don't") == ['don', 't']                     # ES's standard analyzer keeps "don't"
    assert category_tokens('!!!') == []


def test_bm25_tokenizer_option(tmp_path):
    rows = [{'item_id': f'i{k}', 'title': t, 'category': 'x', 'details': d, 'vec': _unit([1, k, 0, 0])}
            for k, (t, d) in enumerate([('A', '東京の夜'), ('B', '大阪の朝'), ('C', 'plain english')])]
    config, b = make_backend(tmp_path / 'tri', rows, extra_config={'bm25_tokenizer': 'trigram'})
    assert b.meta['bm25']['fts_tokenizer'] == 'trigram'
    assert set(bm25_raw_scores(b, config, '東京の', [0, 1, 2])) == {'i0'}   # substring match without spaces
    assert bm25_raw_scores(b, config, '東京', [0, 1, 2]) == {}               # < 3 characters match nothing
    config_u, b_u = make_backend(tmp_path / 'uni', rows)
    assert b_u.meta['bm25']['fts_tokenizer'] == 'unicode61 remove_diacritics 0'
    assert bm25_raw_scores(b_u, config_u, '東京の', [0, 1, 2]) == {}         # unicode61: the run is one token
    with pytest.raises(ValueError):
        make_backend(tmp_path / 'bad', rows, extra_config={'bm25_tokenizer': 'icu'})


def test_library_does_not_use_the_verification_tools():
    import pathlib
    for path in pathlib.Path('utils').rglob('*.py'):
        text = path.read_text(encoding='utf-8')
        assert 'es_compat' not in text, path
        assert 'from scripts' not in text and 'import scripts' not in text, path
        assert 'from legacy_es_equivalence' not in text and 'import legacy_es_equivalence' not in text, path
