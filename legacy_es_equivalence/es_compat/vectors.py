"""
ES 8.13 dense-vector behaviour (W1 / W1-q) and the kNN scoring v1 saw.

W1    ES normalises a cosine document vector only when its squared magnitude, accumulated in
      float32 one dimension at a time, satisfies |‖v‖² − 1| > 1e-4; otherwise the raw float32
      vector is used as if it were unit.
W1-q  ES normalises a cosine kNN query only when |‖q‖² − 1| > 1e-4 (float32 comparison), and
      computes ‖q‖² with Lucene's Panama `VectorUtil.dotProduct(q, q)`: 4 accumulators of
      FLOAT_SPECIES lanes, fused multiply-add, then `acc1+acc2` and `acc3+acc4` lane-wise, their
      sum, and the lanes added in order. FLOAT_SPECIES has 8 lanes on the AVX2 machine the v1
      fixtures were recorded on (ES log: "preferredBitSize=256; FMA enabled"); an AVX-512 machine
      has 16 lanes and rounds differently. Verified against a live ES 8.13: 0 mismatches on 2668
      probes.
Score v1 reads `_score` = (1 + cos) / 2 as a float32 from the JSON response, i.e. its shortest
      decimal representation parsed into a Python float.
Raw   v1 reads `_source.embedding` (the CSV values, float64) for the hybrid score, the reference-title lookup
      and the user_guest sample.
"""
import hashlib
import io

import numpy as np

NORM_EPS = 1e-4                                    # W1 threshold on |‖v‖² − 1|
ES_LANES = 8
_EPS32 = np.float32(1e-4)
_ONE32 = np.float32(1.0)


# ── W1 (documents) ───────────────────────────────────────────────────────────
def _sqmag_f32_sequential(V32):
    """‖v‖² per row, accumulated in float32 one dimension at a time (the order ES uses)."""
    acc = np.zeros(V32.shape[0], dtype=np.float32)
    for j in range(V32.shape[1]):
        col = V32[:, j]
        acc = np.add(acc, np.multiply(col, col, dtype=np.float32), dtype=np.float32)
    return acc


def es_stored_vectors(raw):
    """Return (float32 matrix as ES stores it, bool mask of normalised rows)."""
    V32 = raw.astype(np.float32)
    sq = _sqmag_f32_sequential(V32).astype(np.float64)
    normalise = np.abs(sq - 1.0) > NORM_EPS
    out = V32.copy()
    if normalise.any():
        sub = V32[normalise].astype(np.float64)
        out[normalise] = (sub / np.linalg.norm(sub, axis=1, keepdims=True)).astype(np.float32)
    return out, normalise


# ── W1-q (kNN query) ─────────────────────────────────────────────────────────
def _fma32(a, b, c):
    """float32 fused multiply-add; a*b of two float32 values is exact in float64."""
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def panama_sqmag(q, lanes=ES_LANES):
    """‖q‖² as Lucene's PanamaVectorUtilSupport.dotProduct(q, q) computes it (float32)."""
    v = np.asarray(q, dtype=np.float32)
    n = len(v)
    res = np.float32(0)
    i = 0
    if n > 2 * lanes:
        limit = n - n % lanes
        acc = [np.zeros(lanes, dtype=np.float32) for _ in range(4)]
        unrolled_limit = limit - 3 * lanes
        while i < unrolled_limit:
            for k in range(4):
                seg = v[i + k * lanes: i + (k + 1) * lanes]
                acc[k] = _fma32(seg, seg, acc[k])
            i += 4 * lanes
        while i < limit:
            seg = v[i: i + lanes]
            acc[0] = _fma32(seg, seg, acc[0])
            i += lanes
        total = ((acc[0] + acc[1]).astype(np.float32) + (acc[2] + acc[3]).astype(np.float32)).astype(np.float32)
        red = np.float32(0)
        for x in total:
            red = np.float32(red + x)
        res = np.float32(res + red)
    for j in range(i, n):
        res = _fma32(np.float32(v[j]), np.float32(v[j]), np.float32(res))
    return np.float32(res)


def es_query_is_normalised(q, lanes=ES_LANES):
    """True when ES 8.13 kNN would normalise this query vector."""
    return bool(abs(panama_sqmag(q, lanes) - _ONE32) > _EPS32)


def es_knn_query(q, lanes=ES_LANES):
    """The query vector as ES scores it: parsed as float32, divided by (float) sqrt(‖q‖²) when not unit."""
    v = np.asarray(q, dtype=np.float32)
    sq = panama_sqmag(v, lanes)
    if abs(sq - _ONE32) > _EPS32:
        v = (v / np.float32(np.sqrt(np.float64(sq)))).astype(np.float32)
    return v.astype(np.float64)


def l2_query(q):
    """Plain L2 normalisation (what ES's script_score does; the `always` replay variant)."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)


# ── vectors as v1 saw them ───────────────────────────────────────────────────
# sha256 (np.save format) these arrays must reproduce: the vectors/text.npy (W1) and vectors/text_raw.npy
# files of the first local index, which stored the ES vectors before the library switched to plain L2
# normalisation.
EXPECTED_SHA256 = {
    'imdb': {'w1': 'be769cf32cbd09f77c05162a65dd8b2311ac5e413a70e2fb01c39d4d764e2dbc',
             'raw': 'b69bd745090fc2cb4d82779df67bfa3556fd3224d5354f017311cd913f9c226e'},
    'hm':   {'w1': 'ad3b882474bbf6122e1090ad3d7f2e5708a56cc50b11bf3163df993d3636fcc5',
             'raw': '8a2aca8798b8bca0c2757a544f39de0fb4060aa9565b2807518e129fabef143c'},
}
_cache = {}


def npy_sha256(arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def es_vectors(backend):
    """(W1 matrix as float64, raw _source vectors float64) for a backend, cached per backend.

    Built from the items CSV with the library's reader (the same rows, in the same order, as the
    index); for the two migration datasets both arrays are checked against EXPECTED_SHA256.
    """
    key = id(backend)
    if key not in _cache:
        from utils.retrieval.local_index import read_items
        meta = backend.meta
        df, raw, _ = read_items(meta['config'], meta['source']['precomputed_embedding_col'])
        if [str(x) for x in df[meta['col_itemid']]] != backend.item_id_str:
            raise ValueError(f"{backend.index_dir}: CSV rows do not match the index rows")
        w1, _ = es_stored_vectors(raw)
        expected = EXPECTED_SHA256.get(meta['dataset'])
        if expected:
            got = {'w1': npy_sha256(w1), 'raw': npy_sha256(raw)}
            if got != expected:
                raise ValueError(f"{meta['dataset']}: ES vectors rebuilt from the CSV differ from EXPECTED_SHA256 ({got})")
        _cache[key] = (w1.astype(np.float64), raw)
    return _cache[key]


def es_score(s32):
    """float32 score → the Python float v1 gets from the ES JSON response."""
    return float(str(s32))


def make_knn_candidates(query_rule):
    """R1 replacement: ES kNN over the W1 matrix. query_rule 'es' (W1-q) or 'l2' (plain normalisation)."""
    to_query = {'es': es_knn_query, 'l2': l2_query}[query_rule]

    def knn_candidates(backend, embedding_vector, mask, fetch_k):
        rows = np.flatnonzero(mask)
        if rows.size == 0:
            return rows, []
        w1, _ = es_vectors(backend)
        q = to_query(embedding_vector)
        s = ((1.0 + w1[rows] @ q) / 2.0).astype(np.float32)
        order = np.lexsort((rows, -s))[:fetch_k]
        return rows[order], [es_score(x) for x in s[order]]
    return knn_candidates


def item_embedding(backend, row):
    """R2 replacement: the raw _source.embedding of a row."""
    return es_vectors(backend)[1][row].tolist()
