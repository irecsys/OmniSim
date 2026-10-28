"""
replay_fixtures.py — replay the v1 Elasticsearch recordings (legacy_es_equivalence/recordings/) against
the local retrieval backend and compare call by call. Never calls an LLM / embedding API and
never touches ES: the recorded query / user / reference vectors are used as-is.

Acceptance (λ > 0 is held to the same standard as λ = 0 — it was reached with the Lucene BM25
replica, es_compat/bm25_lucene.py, and is locked as a regression guard):

  vector part, every record (kNN is independent of λ):
    candidate item_ids and order identical, cos_norm error < 1e-5
  every λ (λ > 0 only with --part full):
    exit_path identical; top-10 item_id sequence identical; cos / base error < 1e-5;
    hybrid error < 1e-5 when α < 1; output entries identical (scores < 1e-5);
    early_return: item_id sequence and dict structure identical; ref_blend: blended query bit-identical
    gate_passed identical for every τ in the manifest's tau_set — except λ = 1.0, where the gate only
    depends on whether there are ≥ 2 distinct BM25 scores (max_base is 1.0 or 0.0), so
    only the ranking is compared; the degenerate record (ES max_base 0.0) is listed separately
  W2 (--part w2, the other two ES call sites):
    reference-title lookup, tiered acceptance: real / typo top-1 (hit and item_id)
    100%; multi additionally returns the same list length and order; H&M empty; non_title report only;
    returned vectors bit-identical to the recorded ones; score differences report only
    user_guest sampling: distribution comparison only (v1 has no seed)

Mismatch classification (decided before the results were seen):
  numeric_tie          the only difference is one adjacent pair swapped and the two ES scores differ
                       by < 2e-6 → reported, not a failure; more than 5 such records at one λ → STOP
  es_ann_recall_miss   candidate set differs and the ann_miss_check conditions hold (ES missed a true top item)
                       → reported, not a failure; more than 1% of the records at one λ → STOP
  (both rules were defined for λ = 0 and apply to every λ since λ > 0 was tightened)
  everything else      failure

Modes (design decision: the library uses standard components, the ES replicas live in
legacy_es_equivalence/es_compat/ and are injected into the library's replacement points R1–R5 for verification):

  injection mode   --components all (default): every ES replica injected (vectors, query_norm, bm25,
                   title, category). The acceptance run and the CI regression guard: the strict
                   standard above applies to every λ, and W2 uses the tiered reference-title standard.
  partial          --exclude <components> (or an explicit --components list): the named components
                   use the library's own implementation. Report only. `--exclude query_norm`
                   normalises every query as the library does, `--exclude bm25` uses the library's FTS5 BM25.
  default mode     --components none: the library exactly as shipped. Report only; W2 uses the
                   library's reference-title standard (real titles hit their own title, H&M empty,
                   the rest reported).

Usage:
    python legacy_es_equivalence/replay_fixtures.py --part vector [--out <summary.json>]          # injection mode
    python legacy_es_equivalence/replay_fixtures.py --part full   [--out <summary.json>]          # injection mode
    python legacy_es_equivalence/replay_fixtures.py --part w2                                     # injection mode
    python legacy_es_equivalence/replay_fixtures.py --part full --exclude query_norm              # report only
    python legacy_es_equivalence/replay_fixtures.py --part full --components none                 # default mode
    python legacy_es_equivalence/replay_fixtures.py --part full --entry utils                     # through the dialogue entry points

--entry utils replays through the dialogue flow's real entry points instead of the library
functions: utils.utils.search_items_by_embedding, simulator._lookup_reference_embeddings and
quick_start._sample_guest_items (retrieval_backend local, the default). The traces come from wrapping
local_search.search_items_local / ref_lookup.lookup_reference_embeddings, the guest sample gets seed=k
by wrapping sampling.sample_items; the output must be byte-identical to --entry library.
Exit code 0 only when there is no failure and no STOP condition (always 0 in report-only modes).
"""
import argparse
import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from legacy_es_equivalence.es_compat import inject as es_inject
from legacy_es_equivalence.fixture_io import load_inputs
from utils.retrieval import bm25, local_search, ref_lookup, sampling
from utils.retrieval.backend import LocalBackend
from utils.retrieval.filters import build_mask
from utils.retrieval.local_search import search_items_local
from utils.retrieval.ref_lookup import lookup_reference_embeddings
from utils.retrieval.sampling import sample_items

FIX = 'legacy_es_equivalence/recordings'
INDEX_DIRS = {'imdb-2025': 'index/imdb', 'hm-2025': 'index/hm'}
TOL = 1e-5
TIE_EPS = 2e-6
MAX_NUMERIC_TIES = 5
MAX_ANN_MISS_RATE = 0.01
ENTRY = 'library'          # --entry: 'library' (utils/retrieval functions) or 'utils' (dialogue entry points)


def run_search(config, backend, kwargs, trace):
    """search_items_local(config, backend, **kwargs, trace=trace), or the same call through
    utils.utils.search_items_by_embedding with --entry utils."""
    if ENTRY == 'library':
        return search_items_local(config, backend, **kwargs, trace=trace)
    import utils.utils as uu
    orig = local_search.search_items_local
    calls = []
    def traced(*a, **kw):
        calls.append(1)
        return orig(*a, **kw, trace=trace)
    local_search.search_items_local = traced
    try:
        out = uu.search_items_by_embedding(config, backend, **kwargs)
    finally:
        local_search.search_items_local = orig
    assert calls == [1], 'search_items_by_embedding did not dispatch to search_items_local once'
    return out


def run_ref_lookup(backend, titles, trace):
    """lookup_reference_embeddings(backend, {}, titles, trace=trace), or through
    simulator._lookup_reference_embeddings with --entry utils."""
    if ENTRY == 'library':
        return lookup_reference_embeddings(backend, {}, titles, trace=trace)
    import utils.simulator as sim
    orig = ref_lookup.lookup_reference_embeddings
    calls = []
    def traced(*a, **kw):
        calls.append(1)
        return orig(*a, **kw, trace=trace)
    ref_lookup.lookup_reference_embeddings = traced
    try:
        out = sim._lookup_reference_embeddings(backend, {}, titles)
    finally:
        ref_lookup.lookup_reference_embeddings = orig
    assert calls == [1], '_lookup_reference_embeddings did not dispatch to the local lookup once'
    return out


def run_guest_sample(backend, size, seed):
    """(items, item_pool) of sample_items(backend, size, seed), or of quick_start._sample_guest_items
    with --entry utils (sampling.sample_items wrapped to pass the seed)."""
    if ENTRY == 'library':
        items = sample_items(backend, size, seed=seed)
        return items, [str(x[backend.col_itemid]) for x in items]
    import utils.quick_start as qs
    orig = sampling.sample_items
    calls = []
    def seeded(b, n):
        calls.append(1)
        return orig(b, n, seed=seed)
    sampling.sample_items = seeded
    try:
        items, pool = qs._sample_guest_items(backend, {}, backend.col_itemid, size)
    finally:
        sampling.sample_items = orig
    assert calls == [1], '_sample_guest_items did not dispatch to sample_items once'
    return items, pool


def parse_components(components, exclude):
    """--components / --exclude → the set of injected ES components."""
    comps = (set(es_inject.COMPONENTS) if components == 'all' else
             set() if components == 'none' else {c for c in components.split(',') if c})
    comps -= {c for c in (exclude or '').split(',') if c}
    if 'vectors' not in comps:
        comps.discard('query_norm')               # W1-q only exists on top of the W1 matrix
    return comps


def components_label(comps):
    if comps == set(es_inject.COMPONENTS):
        return 'all (injection mode)'
    return ','.join(c for c in es_inject.COMPONENTS if c in comps) or 'none (default mode)'


def set_query_norm(mode):
    """Inject the ES replicas (legacy_es_equivalence/es_compat). 'es_compat' → all components (v1 behaviour, the
    acceptance run); 'always' → all but the W1-q query rule (plain L2 query normalisation)."""
    components = set(es_inject.COMPONENTS)
    if mode == 'always':
        components.discard('query_norm')
    elif mode != 'es_compat':
        raise ValueError(mode)
    es_inject.install(components)


def set_bm25(impl):
    """'lucene' → keep the ES-replica BM25 installed by set_query_norm; 'fts5' → the library's FTS5
    version (bm25.py), for comparison."""
    if impl == 'fts5':
        local_search.bm25_raw_scores = bm25.bm25_raw_scores
    elif impl != 'lucene':
        raise ValueError(impl)


def read_jsonl(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_fixture():
    phase_a = {r['call_id']: r for r in read_jsonl(os.path.join(FIX, 'phase_a', 'calls.jsonl')) if 'inputs' in r}
    phase_b = read_jsonl(os.path.join(FIX, 'phase_b', 'calls.jsonl'))
    direct = read_jsonl(os.path.join(FIX, 'direct_call', 'calls.jsonl'))
    direct_by_id = {r['call_id']: r for r in direct}
    with open(os.path.join(FIX, 'manifest.json'), encoding='utf-8') as f:
        tau_set = json.load(f)['grid']['tau_set']
    out = []
    for sub, recs in (('phase_b', phase_b), ('direct_call', direct)):
        for r in recs:
            if r['vectors'] or not r.get('vectors_ref'):
                owner, vdir = r, os.path.join(FIX, sub)
            elif sub == 'phase_b':
                owner, vdir = phase_a[r['vectors_ref']], os.path.join(FIX, 'phase_a')
            else:
                owner, vdir = direct_by_id[r['vectors_ref']], os.path.join(FIX, 'direct_call')
            out.append((sub, r, owner, vdir))
    return out, tau_set


def replay_config(rec):
    config = dict(rec['config'])
    config['lambda_bm25'] = rec['meta']['lambda']
    config['threshold_similarity'] = rec['meta']['tau_run']
    attrs = config.get('item_attributes') or []
    config['item_attributes'] = {k: '' for k in attrs} if isinstance(attrs, list) else attrs
    return config


def single_adjacent_swap(a, b):
    """Index i if b equals a with exactly a[i], a[i+1] swapped, else None."""
    if len(a) != len(b) or a == b:
        return None
    diff = [i for i in range(len(a)) if a[i] != b[i]]
    if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
        return diff[0]
    return None


def ann_miss_check(backend, config, kwargs, es_cands, fetch_k):
    """Is a candidate-set mismatch explained by an ES approximate-kNN miss?"""
    q = kwargs['embedding_vector']
    refs = kwargs['reference_embeddings']
    if refs:
        ref_avg = np.mean(refs, axis=0).tolist()
        w = config.get('reference_blend_weight', 0.3)
        q = [(1 - w) * a + w * b for a, b in zip(q, ref_avg)]
    mask = build_mask(backend, config, excluded_ids=kwargs['excluded_ids'], year_range=kwargs['year_range'],
                      exclude_genres=kwargs['exclude_genres'], require_any_genre=kwargs['require_any_genre'])
    rows, s = local_search.knn_candidates(backend, q, mask, backend.n)   # injected R1
    R = [backend.item_ids[r] for r in rows.tolist()]
    exact = dict(zip(R, s))
    E = [c['item_id'] for c in es_cands]
    es_score = {c['item_id']: c['cos_norm'] for c in es_cands}
    c1 = all(e in exact and abs(exact[e] - es_score[e]) < TOL for e in E)
    M = [x for x in R[:fetch_k] if x not in set(E)]
    allowed = set(R[:fetch_k + len(M)])
    min_e = min(es_score.values()) if E else float('inf')
    c2 = set(E) <= allowed and all(exact[m] >= min_e for m in M)
    rank = {x: i for i, x in enumerate(R)}
    c3 = all(e in rank for e in E) and [rank[e] for e in E] == sorted(rank[e] for e in E)
    return c1 and c2 and c3, {'missed': M, 'c1': c1, 'c2': c2, 'c3': c3}


def compare(sub, rec, owner, vdir, backend, tau_set, part, keep=False):
    """keep=True: also return the ES and local candidate lists (for statistics over the candidate lists)."""
    config = replay_config(rec)
    col = config['col_itemid']
    kwargs = load_inputs(vdir, {**rec, 'call_id': owner['call_id'], 'vectors': owner['vectors']})
    trace = {}
    out = run_search(config, backend, kwargs, trace)
    lam = rec['meta']['lambda']
    group = rec['meta'].get('kind') or rec['meta'].get('src_path') or rec.get('path')
    res = {'call_id': rec['call_id'], 'sub': sub, 'es_index': rec['config']['es_index'], 'lambda': lam,
           'group': group, 'exit_path': rec['internal']['exit_path'], 'problems': [], 'class': 'match'}
    P = res['problems']
    if keep:
        res['_es_cands'] = rec['internal'].get('candidates') or []
        res['_lo_cands'] = trace.get('candidates') or []
        res['_rec'] = rec

    if trace.get('exit_path') != rec['internal']['exit_path']:
        P.append(f"exit_path local {trace.get('exit_path')} != ES {rec['internal']['exit_path']}")

    es_out = rec['output'] or []
    if rec['internal']['exit_path'] == 'early_return_747':
        if [o.get(col) for o in out] != [o.get(col) for o in es_out]:
            P.append('early_return item_id sequence differs')
        elif any(list(a.keys()) != list(b.keys()) or any(type(a[k]) is not type(b[k]) or a[k] != b[k] for k in a)
                 for a, b in zip(out, es_out)):
            P.append('early_return dict structure / values differ')
        return res

    # ── vector part (all λ) ──
    es_c = rec['internal'].get('candidates') or []
    lo_c = trace.get('candidates') or []
    es_ids, lo_ids = [c['item_id'] for c in es_c], [c['item_id'] for c in lo_c]
    lo_by = {c['item_id']: c for c in lo_c}
    common = [c for c in es_c if c['item_id'] in lo_by]
    res['cos_err'] = max((abs(lo_by[c['item_id']]['cos_norm'] - c['cos_norm']) for c in common), default=0.0)
    res['cos_bit_equal'] = sum(lo_by[c['item_id']]['cos_norm'] == c['cos_norm'] for c in common)
    res['n_common'] = len(common)
    if res['cos_err'] >= TOL:
        P.append(f"cos error {res['cos_err']:.3e}")
    blended_path = os.path.join(vdir, 'vectors', f"{owner['call_id']}.blended.npz")
    if os.path.exists(blended_path):
        ref = np.load(blended_path)['query_blended']
        res['blended_bit_equal'] = bool(np.array_equal(np.asarray(trace['query_blended'], dtype=np.float64), ref))
        if not res['blended_bit_equal']:
            P.append('blended query not bit-identical')

    cand_issue = None
    if es_ids != lo_ids:
        if set(es_ids) != set(lo_ids) or len(es_ids) != len(lo_ids):
            ok, info = ann_miss_check(backend, config, kwargs, es_c, trace['fetch_k'])
            cand_issue = ('es_ann_recall_miss', info) if ok else ('fail', info)
        else:
            i = single_adjacent_swap(es_ids, lo_ids)
            if i is not None and abs(es_c[i]['cos_norm'] - es_c[i + 1]['cos_norm']) < TIE_EPS:
                cand_issue = ('numeric_tie', {'pos': i, 'es_diff': abs(es_c[i]['cos_norm'] - es_c[i + 1]['cos_norm'])})
            else:
                cand_issue = ('fail', {'order': 'candidate order differs'})
        res['candidates'] = {'class': cand_issue[0], **cand_issue[1]}
        if cand_issue[0] == 'fail':
            P.append(f"candidates differ: {cand_issue[1]}")

    if part == 'vector' and lam != 0.0:
        res['class'] = cand_issue[0] if cand_issue and not P else ('fail' if P else 'match')
        return res

    # ── score / gate / output part ──
    res['base_err'] = max((abs(lo_by[c['item_id']]['base_score'] - c['base_score']) for c in common), default=0.0)
    alpha_lt1 = not rec['inputs']['user_long_is_none']
    res['hybrid_err'] = max((abs(lo_by[c['item_id']]['hybrid_score'] - c['hybrid_score']) for c in common),
                            default=0.0) if alpha_lt1 else None
    if res['base_err'] >= TOL:
        P.append(f"base error {res['base_err']:.3e}")
    if alpha_lt1 and res['hybrid_err'] >= TOL:
        P.append(f"hybrid error {res['hybrid_err']:.3e}")
    es_n, es_max = rec['internal']['n_candidates'], rec['internal']['max_base_score']
    lo_n, lo_max = trace.get('n_candidates', 0), trace.get('max_base_score')
    gates = {t: (es_n > 0 and es_max >= t, lo_n > 0 and lo_max is not None and lo_max >= t) for t in tau_set}
    res['gate_mismatch_taus'] = [t for t, (a, b) in gates.items() if a != b]
    eo, lo_o = [o[col] for o in es_out], [o[col] for o in out]
    res['top10_equal'] = eo == lo_o
    union = set(eo) | set(lo_o)
    res['jaccard10'] = len(set(eo) & set(lo_o)) / len(union) if union else 1.0
    res['max_base_es'], res['max_base_local'] = es_max, lo_max
    if res['gate_mismatch_taus'] and lam != 1.0:          # λ = 1.0: ranking only (see docstring)
        P.append(f"gate differs at τ {res['gate_mismatch_taus']}")
    if not res['top10_equal']:
        i = single_adjacent_swap(eo, lo_o)
        if i is not None and abs(es_out[i]['score'] - es_out[i + 1]['score']) < TIE_EPS:
            res['top10'] = {'class': 'numeric_tie', 'pos': i,
                            'es_diff': abs(es_out[i]['score'] - es_out[i + 1]['score'])}
        elif cand_issue and cand_issue[0] == 'es_ann_recall_miss':
            res['top10'] = {'class': 'es_ann_recall_miss'}
        else:
            res['top10'] = {'class': 'fail'}
            P.append('top-10 sequence differs')
    else:
        for a, b in zip(out, es_out):
            if list(a.keys()) != list(b.keys()):
                P.append('output keys differ')
                break
            if abs(a['score'] - b['score']) >= TOL or any(
                    type(a[k]) is not type(b[k]) or a[k] != b[k] for k in a if k != 'score'):
                P.append('output entry differs')
                break

    classes = {x.get('class') for x in (res.get('candidates'), res.get('top10')) if x}
    if P:
        res['class'] = 'fail'
    elif 'es_ann_recall_miss' in classes:
        res['class'] = 'es_ann_recall_miss'
    elif 'numeric_tie' in classes:
        res['class'] = 'numeric_tie'
    return res


def check_ref_lookups(backends, injected=True):
    """Reference-title lookup against direct_call/ref_lookups.jsonl.

    injected (title + vectors replicas): the tiered acceptance against ES.
    otherwise: the library's own standard — see check_ref_lookups_library.
    """
    if not injected:
        return check_ref_lookups_library(backends)
    gated = {'real', 'typo', 'multi', 'hm'}
    by_kind, fails, reports, score_diff = {}, [], [], []
    vec_eq = vec_n = 0
    for rec in read_jsonl(os.path.join(FIX, 'direct_call', 'ref_lookups.jsonl')):
        b = backends[rec['es_index']]
        trace = []
        embs = run_ref_lookup(b, [t['title'] for t in rec['titles']], trace)
        es_hits = [t for t in rec['titles'] if t['hit']]
        per_title_ok = all((e['hit'], e.get('item_id')) == (lo['hit'], lo['item_id'])
                           for e, lo in zip(rec['titles'], trace))
        ok = per_title_ok and len(embs) == len(es_hits)
        if rec['kind'] == 'hm':
            ok = ok and embs == []
        if rec['n_embeddings_returned']:
            ref = np.load(os.path.join(FIX, 'direct_call', 'ref_vectors', f"{rec['lookup_id']}.npz"))[rec['vectors']]
            for j, e in enumerate(embs[:len(ref)]):
                vec_n += 1
                vec_eq += bool(np.array_equal(np.asarray(e, dtype=np.float64), ref[j]))
            ok = ok and len(embs) == len(ref)
        for e, lo in zip(rec['titles'], trace):
            if e['hit'] and lo['hit'] and e.get('item_id') == lo['item_id']:
                score_diff.append(abs(e['score'] - lo['score']))
        by_kind.setdefault(rec['kind'], [0, 0])
        by_kind[rec['kind']][0] += ok
        by_kind[rec['kind']][1] += 1
        if not ok:
            target = fails if rec['kind'] in gated else reports
            target.append((rec['lookup_id'], rec['kind'],
                           [(e['title'], e.get('item_id'), lo['item_id']) for e, lo in zip(rec['titles'], trace)]))
    print('[ref_lookup] ' + ', '.join(f"{k} {a}/{n}" for k, (a, n) in by_kind.items())
          + f"; returned vectors bit-identical {vec_eq}/{vec_n}; max |Δscore| (same top-1) "
          + (f"{max(score_diff):.2e}" if score_diff else '-'))
    for r in reports:
        print(f"    non_title difference (report only): {r}")
    for f in fails:
        print(f"    FAIL {f}")
    return not fails and vec_eq == vec_n


def check_ref_lookups_library(backends):
    """Library standard (fixed before the matcher was chosen): real titles — including the real titles
    of the multi calls — hit their own title (= ES's item); H&M returns []; typos, non-title phrases and
    the multi call's non-title are reported (typos should hit their source title, non-titles should miss)."""
    fails, reports, by = [], [], collections.Counter()
    for rec in read_jsonl(os.path.join(FIX, 'direct_call', 'ref_lookups.jsonl')):
        b = backends[rec['es_index']]
        trace = []
        embs = run_ref_lookup(b, [t['title'] for t in rec['titles']], trace)
        if rec['kind'] == 'hm':
            by[('hm', embs == [])] += 1
            if embs:
                fails.append((rec['lookup_id'], 'hm', embs))
            continue
        if len(embs) != sum(lo['hit'] for lo in trace):
            fails.append((rec['lookup_id'], 'returned list length differs from the hits'))
        for e, lo in zip(rec['titles'], trace):
            kind = rec['kind'] if rec['kind'] != 'multi' else ('real' if e['hit'] else 'non_title')
            ok = (lo['item_id'] == e.get('item_id')) if kind in ('real', 'typo') else not lo['hit']
            by[(kind, ok)] += 1
            row = (rec['lookup_id'], kind, e['title'], e.get('item_id'), lo['item_id'], lo['score'])
            if not ok:
                (fails if kind == 'real' else reports).append(row)
    print('[ref_lookup] library standard: ' + ', '.join(
        f"{k} {by[(k, True)]}/{by[(k, True)] + by[(k, False)]}" for k in ('real', 'typo', 'non_title', 'hm'))
        + ' (real and hm gated; typo should hit its source title, non_title should miss — report only)')
    for r in reports:
        print(f"    report: {r}")
    for f in fails:
        print(f"    FAIL {f}")
    return not fails


def check_guest_samples(backends, repeats=100):
    """user_guest sampling: distribution preview against direct_call/guest_samples.jsonl (no threshold)."""
    from scipy.stats import chisquare
    recs = read_jsonl(os.path.join(FIX, 'direct_call', 'guest_samples.jsonl'))
    for label in sorted({r['label'] for r in recs}):
        g = [r for r in recs if r['label'] == label]
        b = backends[g[0]['es_index']]
        size = g[0]['body']['size']
        runs = [run_guest_sample(b, size, k) for k in range(repeats)]
        local = [items for items, _ in runs]
        lo_pools = [pool for _, pool in runs]
        n_expected = min(size, b.n)
        def stats(pools):
            first10 = collections.Counter(i for p in pools for i in p[:10])
            counts = [first10.get(i, 0) for i in b.item_id_str]
            return (all(len(p) == n_expected for p in pools), all(len(set(p)) == len(p) for p in pools),
                    chisquare(counts).pvalue)
        es_s, lo_s = stats([r['item_pool'] for r in g]), stats(lo_pools)
        keys_es = sorted(g[0]['source_keys'])
        keys_lo = sorted(local[0][0].keys())
        type_lo = type(local[0][0][b.col_itemid]).__name__
        print(f"[guest] {label}: size {size} → n_hits {n_expected}; ES  n ok {es_s[0]}, no dup {es_s[1]}, "
              f"first-10 uniformity χ² p={es_s[2]:.3f}; local n ok {lo_s[0]}, no dup {lo_s[1]}, p={lo_s[2]:.3f}; "
              f"_source keys equal {keys_es == keys_lo}; id type ES {g[0]['itemid_type_in_source']} local {type_lo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--part', choices=['vector', 'full', 'w2'], default='vector')
    ap.add_argument('--components', default='all',
                    help=f"ES replicas to inject: all (default, injection mode), none (default mode) or a "
                         f"comma list of {','.join(es_inject.COMPONENTS)}")
    ap.add_argument('--exclude', default='', help='comma list of components NOT to inject')
    ap.add_argument('--out', default=None, help='write per-record results + summary as JSON')
    ap.add_argument('--entry', choices=['library', 'utils'], default='library',
                    help="library: call utils/retrieval directly; utils: through the dialogue flow's entry "
                         "points (utils.search_items_by_embedding, simulator, quick_start)")
    args = ap.parse_args()
    global ENTRY
    ENTRY = args.entry
    comps = parse_components(args.components, args.exclude)
    es_inject.install(comps)
    if args.part == 'w2':
        backends = {k: LocalBackend(v) for k, v in INDEX_DIRS.items()}
        ok = check_ref_lookups(backends, injected={'title', 'vectors'} <= comps)
        check_guest_samples(backends)
        print('PASS' if ok else 'FAIL')
        return 0 if ok else 1
    report_only = comps != set(es_inject.COMPONENTS)

    records, tau_set = load_fixture()
    backends = {k: LocalBackend(v) for k, v in INDEX_DIRS.items()}
    results = [compare(sub, r, owner, vdir, backends[r['config']['es_index']], tau_set, args.part)
               for sub, r, owner, vdir in records]

    lam0 = [x for x in results if x['lambda'] == 0.0]
    cls = collections.Counter(x['class'] for x in results)
    cls0 = collections.Counter(x['class'] for x in lam0)
    fails = [x for x in results if x['class'] == 'fail']
    lambdas = sorted({x['lambda'] for x in results})
    stops = []
    for lam in lambdas:
        g = [x for x in results if x['lambda'] == lam]
        c = collections.Counter(x['class'] for x in g)
        if c['numeric_tie'] > MAX_NUMERIC_TIES:
            stops.append(f"numeric_tie records at λ={lam}: {c['numeric_tie']} > {MAX_NUMERIC_TIES}")
        if c['es_ann_recall_miss'] / len(g) > MAX_ANN_MISS_RATE:
            stops.append(f"es_ann_recall_miss at λ={lam}: {c['es_ann_recall_miss']}/{len(g)} > {MAX_ANN_MISS_RATE:.0%}")

    print(f"records {len(results)} (λ=0: {len(lam0)}), part={args.part}, components={components_label(comps)}"
          + ("  [REPORT ONLY]" if report_only else ''))
    print(f"classes all: {dict(cls)}   λ=0: {dict(cls0)}")
    groups = collections.defaultdict(list)
    for x in results:
        groups[(x['es_index'], x['lambda'], x['sub'], x['group'], x['exit_path'])].append(x)
    print(f"{'index':10} {'λ':>4} {'source':11} {'group':14} {'exit_path':16} {'n':>4} {'cand=':>5} "
          f"{'cos_err':>9} {'base_err':>9} {'hyb_err':>9} {'top10=':>6} {'class'}")
    for key in sorted(groups, key=str):
        g = groups[key]
        def mx(f):
            v = [x[f] for x in g if x.get(f) is not None]
            return f"{max(v):.2e}" if v else '-'
        cand_eq = sum('candidates' not in x for x in g if x['exit_path'] != 'early_return_747')
        top = [x['top10_equal'] for x in g if 'top10_equal' in x]
        c = collections.Counter(x['class'] for x in g)
        print(f"{key[0]:10} {key[1]:>4} {key[2]:11} {str(key[3]):14} {key[4]:16} {len(g):>4} {cand_eq:>5} "
              f"{mx('cos_err'):>9} {mx('base_err'):>9} {mx('hybrid_err'):>9} "
              f"{(str(sum(top)) if top else '-'):>6} {dict(c)}")
    allc = [x for x in results if 'cos_err' in x]
    print(f"vector part: max cos error {max(x['cos_err'] for x in allc):.3e}; cos bit-identical "
          f"{sum(x['cos_bit_equal'] for x in allc)}/{sum(x['n_common'] for x in allc)}; "
          f"blended bit-identical {sum(x.get('blended_bit_equal', False) for x in results)}/"
          f"{sum('blended_bit_equal' in x for x in results)}")
    for lam in (lambdas if args.part == 'full' else [0.0]):
        g = [x for x in results if x['lambda'] == lam]
        def m0(f):
            v = [x[f] for x in g if x.get(f) is not None]
            return max(v) if v else 0.0
        gate = ('not compared (ranking only)' if lam == 1.0
                else sum(bool(x.get('gate_mismatch_taus')) for x in g))
        print(f"λ={lam}: max cos {m0('cos_err'):.3e}  base {m0('base_err'):.3e}  hybrid {m0('hybrid_err'):.3e}; "
              f"top-10 identical {sum(x.get('top10_equal', True) for x in g)}/{len(g)}; gate mismatches {gate}")
    for x in results:
        if x['class'] in ('numeric_tie', 'es_ann_recall_miss'):
            print(f"  {x['class']}: {x['call_id']} λ={x['lambda']} {x.get('candidates') or ''} {x.get('top10') or ''}")
    for x in fails[:30]:
        print(f"  FAIL {x['call_id']} λ={x['lambda']} {x['group']}: {x['problems']}")
    if args.part == 'full':
        for x in results:
            if x['lambda'] == 1.0 and x.get('max_base_es') == 0.0:
                print(f"λ=1.0 degenerate record {x['call_id']} ({x['group']}): ES max_base 0.0, "
                      f"local max_base {x['max_base_local']}")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump({'summary': {'classes': dict(cls), 'classes_lambda0': dict(cls0), 'stops': stops,
                                   'n_fail': len(fails)}, 'records': results}, f, indent=1, default=str)
    if report_only:
        report_agreement(results, lam0, tau_set)
        print(f'REPORT ONLY (components={components_label(comps)}): no pass/fail threshold')
        return 0
    for s in stops:
        print('STOP:', s)
    ok = not fails and not stops
    print('PASS' if ok else 'FAIL')
    return 0 if ok else 1


def report_agreement(results, lam0, tau_set):
    """Agreement rates and maximum deviations, for the report-only run."""
    vec = [x for x in results if x['exit_path'] != 'early_return_747']
    same_set = sum('candidates' not in x or x['candidates'].get('class') == 'numeric_tie'
                   or 'order' in x['candidates'] for x in vec)          # 'order': same set, other order
    same_order = sum('candidates' not in x for x in vec)
    print(f"[report] candidates: same set {same_set}/{len(vec)}, same set and order {same_order}/{len(vec)}")
    scored = [x for x in results if 'top10_equal' in x]
    for lam in sorted({x['lambda'] for x in scored}):
        g = [x for x in scored if x['lambda'] == lam]
        gate_ok = {t: sum(t not in x['gate_mismatch_taus'] for x in g) for t in tau_set}
        def mx(f):
            v = [x[f] for x in g if x.get(f) is not None]
            return f"{max(v):.3e}" if v else '-'
        print(f"[report] λ={lam}: top-10 identical {sum(x['top10_equal'] for x in g)}/{len(g)}; "
              f"gate agreement per τ " + ', '.join(f"{t}: {c}/{len(g)}" for t, c in gate_ok.items())
              + f"; max |Δcos| {mx('cos_err')}, |Δbase| {mx('base_err')}, |Δhybrid| {mx('hybrid_err')}")
    early = [x for x in results if x['exit_path'] == 'early_return_747']
    print(f"[report] early_return identical {sum(not x['problems'] for x in early)}/{len(early)}")


if __name__ == '__main__':
    sys.exit(main())
