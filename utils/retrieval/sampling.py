"""
user_guest item sampling — local replacement for the ES call at quick_start.py:435.

v1: `function_score` + `random_score` (no seed) with size = min(total·3, 1000); ES returns
min(size, N) distinct documents in random order, and the caller turns their `_source` into the
items DataFrame. Here: a uniform random sample without replacement of min(size, N) items, in random
order, each as a dict with the same keys as `_source` (stored fields in index order, then
`embedding` = backend.item_embedding, replacement point R2) and the same value types (H&M ids stay int).

`embedding` holds the index's L2-normalised float32 values, not the CSV values v1 returned; the caller
(quick_start.py:449, items DataFrame) reads the other columns by name and never computes with it.

v1 is non-deterministic (no seed); seed=None keeps that, a seed makes the sample reproducible.
Only the distribution is compared with v1.
"""
import numpy as np


def sample_items(backend, size, seed=None):
    n = min(int(size), backend.n)
    rows = np.random.default_rng(seed).permutation(backend.n)[:n]
    return [dict(backend.items[r], embedding=backend.item_embedding(r)) for r in rows.tolist()]
