"""
Inject the ES replicas into the library's replacement points for a verification run.

  R1  utils.retrieval.local_search.knn_candidates   component 'vectors' (+ 'query_norm' for W1-q)
  R2  LocalBackend.item_embedding                   component 'vectors'
  R3  utils.retrieval.local_search.bm25_raw_scores  component 'bm25'
  R4  utils.retrieval.ref_lookup.match_title        component 'title'
  R5  utils.retrieval.filters.category_tokens       component 'category'

install() with all components reproduces v1 (the acceptance mode). 'vectors' without
'query_norm' is the `always` variant: W1 document matrix, plain L2 query normalisation.
'query_norm' without 'vectors' is meaningless and rejected. reset() restores the library.
"""
from utils.retrieval import filters, local_search, ref_lookup
from utils.retrieval.backend import LocalBackend

from . import analysis, bm25_lucene, title_fuzzy, vectors

COMPONENTS = ('vectors', 'query_norm', 'bm25', 'title', 'category')

_LIBRARY = {
    'knn_candidates': local_search.knn_candidates,
    'item_embedding': LocalBackend.item_embedding,
    'bm25_raw_scores': local_search.bm25_raw_scores,
    'match_title': ref_lookup.match_title,
    'category_tokens': filters.category_tokens,
}


def reset():
    local_search.knn_candidates = _LIBRARY['knn_candidates']
    LocalBackend.item_embedding = _LIBRARY['item_embedding']
    local_search.bm25_raw_scores = _LIBRARY['bm25_raw_scores']
    ref_lookup.match_title = _LIBRARY['match_title']
    filters.category_tokens = _LIBRARY['category_tokens']


def install(components=COMPONENTS):
    """Reset to the library, then inject the given ES components."""
    components = set(components)
    unknown = components - set(COMPONENTS)
    if unknown:
        raise ValueError(f"unknown es_compat components {sorted(unknown)}")
    if 'query_norm' in components and 'vectors' not in components:
        raise ValueError("'query_norm' (W1-q) needs 'vectors' (the W1 document matrix)")
    reset()
    if 'vectors' in components:
        local_search.knn_candidates = vectors.make_knn_candidates('es' if 'query_norm' in components else 'l2')
        LocalBackend.item_embedding = vectors.item_embedding
    if 'bm25' in components:
        local_search.bm25_raw_scores = bm25_lucene.bm25_raw_scores
    if 'title' in components:
        ref_lookup.match_title = title_fuzzy.match_title
    if 'category' in components:
        filters.category_tokens = analysis.standard_tokenize
