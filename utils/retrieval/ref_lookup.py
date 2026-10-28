"""
Reference-title lookup — local replacement for simulator.py:93-115 `_lookup_reference_embeddings`.

v1 sends, per title, a fuzzy `match` on the field literally named "title" (not col_title), `size: 1`,
and keeps `_source.embedding` of the top hit. Here, per title, `match_title` (replacement point R4)
returns the best row or None; a title with no match (or any exception) is skipped; the returned list
keeps the order of the titles; the vector is `backend.item_embedding(row)` (R2). A dataset without a
"title" column (H&M) always gives [].

match_title: rapidfuzz `process.extractOne` over the titles of all rows, `fuzz.ratio` (normalised
Levenshtein similarity, 0–100) after `rapidfuzz.utils.default_process` (lowercase, non-alphanumerics
→ space, trim); a match needs score ≥ SCORE_CUTOFF; ties go to the lower row. Scorer and cutoff were
chosen on a diagnostic set (200 IMDB titles + 387 one-edit typos) by a rule fixed beforehand: all real titles
hit, ≥ 98% of typos hit their source title, then fewest non-title hits, then the highest cutoff.
Unlike v1 (any shared word within edit distance is a hit, e.g. "Death Wish" → "Killer Fish"),
phrases that are not titles mostly find nothing.
"""
from rapidfuzz import fuzz, process
from rapidfuzz import utils as rf_utils

TITLE_FIELD = 'title'          # hard-coded in v1, deliberately not config['col_title']
SCORER = fuzz.ratio
SCORE_CUTOFF = 80


def _title_choices(backend):
    """{row: title} of the rows with a non-empty title, in row order. Cached on the backend instance
    (a module-level cache keyed by id(backend) could hand a new backend the titles of a collected one)."""
    with backend._lock:
        choices = getattr(backend, '_title_choices', None)
        if choices is None:
            choices = {r: it[TITLE_FIELD] for r, it in enumerate(backend.items)
                       if isinstance(it.get(TITLE_FIELD), str) and it[TITLE_FIELD]}
            backend._title_choices = choices
        return choices


def match_title(backend, title):
    """Replacement point R4: (row, score) of the best-matching title, or None below SCORE_CUTOFF."""
    m = process.extractOne(title, _title_choices(backend), scorer=SCORER,
                           processor=rf_utils.default_process, score_cutoff=SCORE_CUTOFF)
    return None if m is None else (m[2], m[1])


def lookup_reference_embeddings(backend, config, reference_titles, trace=None):
    """Local equivalent of simulator._lookup_reference_embeddings(es, config, reference_titles).

    trace: optional list, gets one {title, hit, item_id, score} per title (for the fixture replay).
    """
    embeddings = []
    for title in reference_titles:
        info = {'title': title, 'hit': False, 'item_id': None, 'score': None}
        try:
            if backend.has_column(TITLE_FIELD):
                m = match_title(backend, title)
                if m is not None:
                    row, score = m
                    emb = backend.item_embedding(row)
                    if emb:
                        embeddings.append(emb)
                        info.update(hit=True, item_id=backend.item_ids[row], score=score)
        except Exception:
            pass
        if trace is not None:
            trace.append(info)
    return embeddings
