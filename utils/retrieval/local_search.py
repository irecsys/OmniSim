"""
search_items_local — local replacement for utils.utils.search_items_by_embedding (v1).

Same signature (`es` → `backend`) and return structure. The flow follows v1 step by step;
the hybrid scoring, the τ gate and the final sort are copied
verbatim from utils/utils.py (v1 :822-880):

  0  reference blending            q' = (1−w)·q + w·mean(refs), not re-normalised (v1 :704-710)
  1  structural filters            boolean mask over the whole index (filters.py)
  2  embedding_vector is None      fallback: first top_k filtered items in index order (v1 :747-756)
  3  kNN                           exact cosine over the masked index, top fetch_k = 3·top_k
  4  BM25                          only over those fetch_k candidates (bm25.py, SQLite FTS5)
  5  hybrid score                  base score and hybrid score, verbatim
  6  gate                          max(_base_score) over all fetch_k candidates < τ → []
  7  sort                          stable, by score descending; truncate to top_k

Standard components only: documents and query are L2-normalised and scored with the plain
cosine (1 + v̂·q̂) / 2 in float64; BM25 is SQLite FTS5's bm25(). Ranking ties are broken by index
order (= CSV order, which was ES's doc order).

Replacement points R1 `knn_candidates`, R2 `LocalBackend.item_embedding`, R3 `bm25_raw_scores`:
the verification tools (in legacy_es_equivalence/) put Elasticsearch replicas here to prove that
the pipeline around them reproduces v1 bit for bit. The library never imports them.
"""
import numpy as np

from utils import llm_calls
from utils.retrieval.bm25 import bm25_raw_scores   # replacement point R3
from utils.retrieval.filters import build_mask
from utils.utils import cosine_sim, normalize_bm25, normalize_cosine


def knn_query(embedding_vector):
    """The unit query vector the kNN score is computed with (plain L2 normalisation)."""
    q = np.asarray(embedding_vector, dtype=np.float64)
    return q / np.linalg.norm(q)


def knn_candidates(backend, embedding_vector, mask, fetch_k):
    """Replacement point R1: top fetch_k rows of the masked index by cosine score (1 + v·q̂) / 2.

    Returns (rows, scores as Python floats), ordered by (score desc, row_idx asc).
    """
    rows = np.flatnonzero(mask)
    if rows.size == 0:
        return rows, []
    q = knn_query(embedding_vector)
    s = (1.0 + backend.vectors64[rows] @ q) / 2.0
    order = np.lexsort((rows, -s))[:fetch_k]
    return rows[order], s[order].tolist()


def search_items_local(config, backend, embedding_vector, top_k,
                       user_long_embedding=None, user_short_embedding=None,
                       excluded_ids=None, query_text=None,
                       year_range=None, exclude_genres=None,
                       require_any_genre=None, reference_embeddings=None,
                       trace=None):
    """Local equivalent of utils.utils.search_items_by_embedding.

    trace: optional dict, filled with the internals (candidates, scores, exit path) for the
    fixture replay. Callers in the dialogue flow do not pass it.
    """
    col_itemid   = config['col_itemid']
    col_category = config['col_category']
    if embedding_vector is not None and len(embedding_vector) != backend.vectors.shape[1]:
        space = backend.meta['spaces']['text']
        raise ValueError(
            f"The query embedding has dimension {len(embedding_vector)}, but the local index "
            f"'{backend.index_dir}' holds {backend.vectors.shape[1]}-dim vectors (built with provider "
            f"'{space.get('provider')}', model '{space.get('model')}'). The config's embeddings_provider / "
            f"embeddings_model must be the ones the index was built with; otherwise delete the index "
            f"directory and rebuild it with `python run.py --config \"<your config files>\" --build-index`.")

    # ── Reference embedding blending (v1 :704-710, verbatim) ───────────────────
    if reference_embeddings and embedding_vector is not None:
        ref_avg   = np.mean(reference_embeddings, axis=0).tolist()
        alpha_ref = config.get('reference_blend_weight', 0.3)
        embedding_vector = [
            (1 - alpha_ref) * q + alpha_ref * r
            for q, r in zip(embedding_vector, ref_avg)
        ]
    if trace is not None:
        trace['query_blended'] = embedding_vector if reference_embeddings else None

    # ── Hard filters ──────────────────────────────────────────────────────────
    empty_reasons = []      # filters that leave nothing for a structural reason: logged, no effect
    mask = build_mask(backend, config, excluded_ids=excluded_ids, year_range=year_range,
                      exclude_genres=exclude_genres, require_any_genre=require_any_genre, reasons=empty_reasons)
    for reason, value, column in empty_reasons:
        llm_calls.note_empty_filter(reason, value, column, config.get('dataset'))

    # ── Fallback when no embedding is available (v1 :747-756) ─────────────────
    if embedding_vector is None:
        if trace is not None:
            trace['exit_path'] = 'early_return_747'
        attr_keys = list(config.get('item_attributes', {}).keys())
        return [
            {k: src.get(k)
             for k in [col_itemid, config['col_title'], config['col_details'], col_category] + attr_keys
             if k in src}
            for src in (backend.items[r] for r in np.flatnonzero(mask)[:top_k])
        ]

    # ── Step 1: kNN → S̃_cos(q, i) ─────────────────────────────────────────────
    fetch_k = top_k * 3
    rows, scores = knn_candidates(backend, embedding_vector, mask, fetch_k)
    knn_scores  = {}   # item_id → S̃_cos(q, i)
    knn_sources = {}   # item_id → stored fields
    knn_rows    = {}   # item_id → row_idx
    for r, s in zip(rows.tolist(), scores):
        iid = backend.item_ids[r]
        if iid is not None:
            knn_scores[iid]  = s
            knn_sources[iid] = backend.items[r]
            knn_rows[iid]    = r

    # ── Step 2: BM25 over the candidates → S̃_BM25 ─────────────────────────────
    bm25_scores_norm = {}
    raw_bm25_scores = {}
    if query_text and knn_scores:
        raw_bm25_scores = bm25_raw_scores(backend, config, query_text, list(knn_rows.values()))
        bm25_scores_norm = normalize_bm25(raw_bm25_scores)

    # ── Step 3 & 4: S̃_base and S̃_hybrid — v1 :822-864 verbatim ──
    lam   = config.get('lambda_bm25', 0.3)
    alpha = config.get('weight_es_score', 0.7)
    beta  = config.get('weight_user_taste_short', 0.3) if user_short_embedding is not None else 0.0

    attr_keys = list(config.get('item_attributes', {}).keys())
    candidates = []

    for iid, cos_score in knn_scores.items():
        source = knn_sources[iid]

        # base score: S̃_base = λ · S̃_BM25 + (1 − λ) · S̃_cos
        bm25_score = bm25_scores_norm.get(iid, 0.0)
        base_score = lam * bm25_score + (1 - lam) * cos_score

        # hybrid score: add user preference signal when profile is available
        if user_long_embedding is not None:
            item_embedding = backend.item_embedding(knn_rows[iid])
            if item_embedding:
                long_sim = normalize_cosine(cosine_sim(item_embedding, user_long_embedding))
                if user_short_embedding is not None:
                    short_sim = normalize_cosine(cosine_sim(item_embedding, user_short_embedding))
                    profile_sim = beta * short_sim + (1 - beta) * long_sim
                else:
                    profile_sim = long_sim
                score = alpha * base_score + (1 - alpha) * profile_sim
            else:
                score = base_score  # no stored embedding — fall back to base score
        else:
            score = base_score  # no user profile → pure retrieval score

        entry = {
            config['col_itemid']:   iid,
            config['col_title']:    source.get(config['col_title']),
            config['col_category']: source.get(config['col_category']),
            "score": score,
            "_base_score": base_score,
        }
        for key in attr_keys:
            val = source.get(key)
            if val is not None and str(val).strip() not in ('', 'nan', 'NaN', 'None'):
                entry[key] = val
        candidates.append(entry)

    if trace is not None:
        trace.update({
            'lam': lam, 'alpha': alpha, 'beta': beta, 'fetch_k': fetch_k,
            'n_candidates': len(candidates),
            'max_base_score': max((c['_base_score'] for c in candidates), default=None),
            'candidates': [{'item_id': c[col_itemid], 'cos_norm': knn_scores[c[col_itemid]],
                            'bm25_raw': raw_bm25_scores.get(c[col_itemid]),
                            'bm25_norm': bm25_scores_norm.get(c[col_itemid]),
                            'base_score': c['_base_score'], 'hybrid_score': c['score']}
                           for c in candidates],
        })

    # ── τ gate on max(S̃_base) over all fetch_k candidates (v1 :876-878) ────────
    tau = config['threshold_similarity']
    if not candidates or max(c["_base_score"] for c in candidates) < tau:
        if trace is not None:
            trace['exit_path'] = 'gate_failed'
        return []
    if trace is not None:
        trace['exit_path'] = 'gate_passed'

    # Strip the internal bookkeeping field before returning
    for c in candidates:
        c.pop("_base_score", None)

    # Final ranking by hybrid score (descending)
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_k]
