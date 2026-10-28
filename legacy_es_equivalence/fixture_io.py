"""
fixture_io.py — read the es_golden fixture records and rebuild the keyword arguments of a recorded
retrieval call. Used by the replay tool (replay_fixtures.py).

Copied verbatim from the fixture recording script, which was removed together with the
recording code.
"""
import json
import os

import numpy as np


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def load_inputs(a_dir, rec):
    """Rebuild the exact keyword arguments of the recorded call."""
    inp = rec['inputs']
    z = np.load(os.path.join(a_dir, 'vectors', f"{rec['call_id']}.npz")) if rec['vectors'] else {}
    vec = lambda k: z[k].tolist() if k in rec['vectors'] else None
    ex = inp['excluded_ids']
    if ex is not None and inp['excluded_ids_type'] == 'set':
        ex = set(ex)
    yr = inp['year_range']
    if yr is not None and inp.get('year_range_type') == 'tuple':
        yr = tuple(yr)
    refs = vec('reference')
    if refs is None and inp['reference_embeddings_type'] == 'list':
        refs = []
    return dict(embedding_vector=vec('query'), top_k=inp['top_k'],
                user_long_embedding=vec('user_long'), user_short_embedding=vec('user_short'),
                excluded_ids=ex, query_text=inp['query_text'], year_range=yr,
                exclude_genres=inp['exclude_genres'], require_any_genre=inp['require_any_genre'],
                reference_embeddings=refs)
