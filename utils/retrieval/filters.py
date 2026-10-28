r"""
Structural filters as a boolean mask over the whole index — the local equivalent of the
v1 ES kNN `filter` (utils/utils.py:712-744 in v1). Applied before the top-k selection,
because v1 pre-filters inside the kNN search.

Semantics:
  excluded_ids       must_not terms on the keyword id field → compare as str (H&M ids are int)
  tokens             category_tokens: lowercase \w+ words (replacement point R5; the verification
                     tools inject ES's standard analyzer here)
  exclude_genres     must_not match on col_category (text) → word-level OR: exclude the doc if
                     any token of the genre occurs in the category; a genre with no tokens excludes nothing
  require_any_genre  should[match_phrase], minimum_should_match 1 → the genre's token sequence
                     occurs contiguously in the category; a genre with no tokens matches nothing
  year_range         range gte/lte on release_year; null never matches; no release_year column → nothing matches

`reasons` (optional list): build_mask appends (reason, value, column) when a filter on its own leaves no item for a
reason that repeats on every retrieval — year_range without a release_year column, or require_any_genre that no
item's category matches. Reporting only; the mask is the same with or without it.
"""
import re

import numpy as np

_WORD = re.compile(r'\w+')


def category_tokens(text):
    """Replacement point R5: lowercase word tokens (\\w+) of a category value or a genre."""
    return _WORD.findall(text.lower())


def _contains_seq(tokens, seq):
    m = len(seq)
    return any(tokens[i:i + m] == seq for i in range(len(tokens) - m + 1))


def build_mask(backend, config, excluded_ids=None, year_range=None,
               exclude_genres=None, require_any_genre=None, reasons=None):
    mask = np.ones(backend.n, dtype=bool)
    col_category = config['col_category']

    if excluded_ids:
        excluded = {str(x) for x in excluded_ids}
        mask &= np.array([s not in excluded for s in backend.item_id_str], dtype=bool)

    if exclude_genres:
        doc_tokens = backend.field_tokens(col_category, category_tokens)
        if doc_tokens is not None:
            for genre in exclude_genres:
                g = set(category_tokens(str(genre)))
                if g:
                    mask &= np.array([not (g & set(t)) for t in doc_tokens], dtype=bool)

    if year_range:
        if backend.has_column('release_year'):
            lo, hi = year_range[0], year_range[1]
            mask &= np.array([v is not None and (lo is None or v >= lo) and (hi is None or v <= hi)
                              for v in (it['release_year'] for it in backend.items)], dtype=bool)
        else:
            mask[:] = False
            if reasons is not None:
                reasons.append(('year_range_without_column', tuple(year_range), 'release_year'))

    if require_any_genre:
        doc_tokens = backend.field_tokens(col_category, category_tokens)
        seqs = [s for s in (category_tokens(str(g)) for g in require_any_genre) if s]
        if doc_tokens is None or not seqs:
            mask[:] = False
            matched = False
        else:
            hit = np.array([any(_contains_seq(t, s) for s in seqs) for t in doc_tokens], dtype=bool)
            mask &= hit
            matched = bool(hit.any())
        if reasons is not None and not matched:
            reasons.append(('require_any_genre_no_match', list(require_any_genre), col_category))

    return mask
