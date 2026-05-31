"""
Comparison Metrics for SimuConv vs. Real Conversations.

Supports three categories of metrics:

1. No-reference diversity metrics (run on any folder):
   - Distinct-1, Distinct-2   : n-gram uniqueness rate
   - TTR                       : Type-Token Ratio
   - MTLD                      : Measure of Textual Lexical Diversity (length-independent)
   - Cosine Diversity (SBERT)  : Mean pairwise semantic diversity across utterances

2. Group-level comparison (requires two folders: --sim and --real):
   - MTLD Gap                  : abs(MTLD_sim - MTLD_real)
   - Cosine Diversity Gap      : abs(CosDiv_sim - CosDiv_real)
   - Item Mention Entropy      : Shannon entropy over mentioned item IDs
   - Entropy Gap               : abs(Entropy_sim - Entropy_real)

Usage:
    # Single-folder diversity analysis
    python -m utils.compare_metrics --folder chats/imdb/free/20260310131810

    # Compare simulated vs. real
    python -m utils.compare_metrics \\
        --sim   chats/imdb/free/20260310131810 \\
        --real  chats/redial/real \\
        --output compare.csv

Dependencies:
    pip install sentence-transformers scikit-learn numpy
"""

import os
import re
import math
import argparse
import warnings
from collections import Counter

import numpy as np

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _load_utterances(folder: str, speakers: tuple = ('Bot', 'User')) -> list[str]:
    """Return all utterance texts from a folder of SimuConv TXT files."""
    texts = []
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt') or 'tmp' in fname:
            continue
        with open(os.path.join(folder, fname), encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('- '):
                    continue
                if ': ' in line:
                    speaker, _, text = line.partition(': ')
                    if speaker.strip() in speakers:
                        texts.append(text.strip())
    return texts


def _tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


# ---------------------------------------------------------------------------
# lexical_diversity library wrapper (Root TTR, Log TTR, Maas, MSTTR, MATTR, MTLD-MA)
# ---------------------------------------------------------------------------

def _ld_metrics(utterances: list[str]) -> dict:
    """Compute metrics from the lexical_diversity library."""
    try:
        from lexical_diversity import lex_div as ld
    except ImportError:
        return {}

    all_tokens = []
    for utt in utterances:
        all_tokens.extend(_tokenize(utt))

    if len(all_tokens) < 10:
        return {}

    return {
        'root_ttr':   ld.root_ttr(all_tokens),
        'log_ttr':    ld.log_ttr(all_tokens),
        'maas_ttr':   ld.maas_ttr(all_tokens),
        'msttr':      ld.msttr(all_tokens, window_length=50),
        'mattr':      ld.mattr(all_tokens, window_length=50),
        'mtld_ma_w':  ld.mtld_ma_wrap(all_tokens),
        'mtld_ma_bid':ld.mtld_ma_bid(all_tokens),
    }


# ---------------------------------------------------------------------------
# Distinct-n
# ---------------------------------------------------------------------------

def distinct_n(utterances: list[str], n: int) -> float:
    """Ratio of unique n-grams to total n-grams across all utterances."""
    total, unique = 0, set()
    for utt in utterances:
        tokens = _tokenize(utt)
        grams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        total += len(grams)
        unique.update(grams)
    return len(unique) / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# TTR
# ---------------------------------------------------------------------------

def ttr(utterances: list[str]) -> float:
    """Type-Token Ratio over all utterances combined."""
    all_tokens = []
    for utt in utterances:
        all_tokens.extend(_tokenize(utt))
    if not all_tokens:
        return 0.0
    return len(set(all_tokens)) / len(all_tokens)


# ---------------------------------------------------------------------------
# MTLD — Measure of Textual Lexical Diversity
# ---------------------------------------------------------------------------

def _mtld_forward(tokens: list[str], threshold: float = 0.72) -> float:
    """One-direction MTLD pass."""
    if not tokens:
        return 0.0
    factors = 0
    types: set = set()
    token_count = 0
    for token in tokens:
        token_count += 1
        types.add(token)
        ttr_val = len(types) / token_count
        if ttr_val <= threshold:
            factors += 1
            types = set()
            token_count = 0
    # Partial factor
    if token_count > 0:
        ttr_val = len(types) / token_count
        partial = (1.0 - ttr_val) / (1.0 - threshold) if threshold < 1.0 else 0.0
        factors += partial
    return len(tokens) / factors if factors > 0 else len(tokens)


def mtld(utterances: list[str], threshold: float = 0.72) -> float:
    """Bidirectional MTLD score."""
    all_tokens = []
    for utt in utterances:
        all_tokens.extend(_tokenize(utt))
    if len(all_tokens) < 10:
        return 0.0
    forward = _mtld_forward(all_tokens, threshold)
    backward = _mtld_forward(list(reversed(all_tokens)), threshold)
    return (forward + backward) / 2.0


# ---------------------------------------------------------------------------
# HDD — Hypergeometric Distribution D
# ---------------------------------------------------------------------------

def hdd(utterances: list[str], sample_size: int = 42) -> float:
    """
    HD-D (Hypergeometric Distribution D): measures how much lexical diversity
    you would expect in a random sample of `sample_size` tokens drawn from the
    corpus without replacement.

    Formula: HDD = sum_w P(X_w >= 1) = sum_w [1 - C(N-n_w, D) / C(N, D)]
    where N = total tokens, n_w = count of type w, D = sample_size.

    Ref: McCarthy & Jarvis (2010). HD-D: A valid alternative to MTLD.
    """
    from math import comb

    all_tokens = []
    for utt in utterances:
        all_tokens.extend(_tokenize(utt))

    N = len(all_tokens)
    if N < sample_size:
        return 0.0

    counts = {}
    for t in all_tokens:
        counts[t] = counts.get(t, 0) + 1

    D = sample_size
    denom = comb(N, D)
    if denom == 0:
        return 0.0

    score = 0.0
    for n_w in counts.values():
        if n_w <= N - D:
            score += 1.0 - comb(N - n_w, D) / denom
        else:
            score += 1.0  # type always appears in any sample of size D
    return score


# ---------------------------------------------------------------------------
# Cosine Diversity (SBERT)
# ---------------------------------------------------------------------------

def cosine_diversity(utterances: list[str], sample_size: int = 200) -> float:
    """
    Mean pairwise cosine distance between utterance embeddings.
    Uses sentence-transformers (all-MiniLM-L6-v2) locally.
    Returns 0.0 if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("  [WARN] sentence-transformers not installed — skipping Cosine Diversity")
        return float('nan')

    if not utterances:
        return 0.0

    # Sample for speed
    if len(utterances) > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(utterances), sample_size, replace=False)
        sample = [utterances[i] for i in idx]
    else:
        sample = utterances

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(sample, show_progress_bar=False, normalize_embeddings=True)

    # Mean pairwise cosine distance = 1 - mean cosine similarity (off-diagonal)
    sim_matrix = cosine_similarity(embeddings)
    n = len(sim_matrix)
    if n < 2:
        return 0.0
    off_diag = sim_matrix[~np.eye(n, dtype=bool)]
    return float(1.0 - off_diag.mean())


# ---------------------------------------------------------------------------
# Item Mention Entropy
# ---------------------------------------------------------------------------

_ITEM_RE = re.compile(r'\((\w[\w\-]*)\)')   # e.g. (tt0009932) or (203371)


def item_mention_entropy(folder: str) -> float:
    """
    Shannon entropy over item IDs mentioned in all conversations.
    Higher = more diverse item coverage; lower = concentrated on popular items.
    """
    counts: Counter = Counter()
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt') or 'tmp' in fname:
            continue
        with open(os.path.join(folder, fname), encoding='utf-8') as f:
            for line in f:
                for item_id in _ITEM_RE.findall(line):
                    counts[item_id] += 1
    if not counts:
        return 0.0
    total = sum(counts.values())
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ---------------------------------------------------------------------------
# Core report
# ---------------------------------------------------------------------------

def _conv_stats(folder: str) -> dict:
    """Parse success_rate and avg_turns from conversation filenames."""
    import re as _re
    pattern = _re.compile(r'-(\d+)-(\d+)-([01])-\d+\.txt$')
    turns_list, succeed_list = [], []
    for fname in os.listdir(folder):
        m = pattern.search(fname)
        if m:
            turns_list.append(int(m.group(1)))
            succeed_list.append(int(m.group(3)))
    if not turns_list:
        return {'success_rate': float('nan'), 'avg_turns': float('nan')}
    return {
        'success_rate': sum(succeed_list) / len(succeed_list),
        'avg_turns': sum(turns_list) / len(turns_list),
    }


def analyze_folder(folder: str, label: str = '') -> dict:
    """Compute all no-reference metrics for one folder."""
    label = label or os.path.basename(folder)
    utterances = _load_utterances(folder)
    print(f"\n{'='*60}")
    print(f"Folder  : {folder}  [{label}]")
    print(f"Utterances loaded: {len(utterances)}")

    d1 = distinct_n(utterances, 1)
    d2 = distinct_n(utterances, 2)
    t   = ttr(utterances)
    m   = mtld(utterances)
    h   = hdd(utterances)
    cd  = cosine_diversity(utterances)
    ie  = item_mention_entropy(folder)
    ld_ = _ld_metrics(utterances)

    print(f"  Distinct-1          : {d1:.4f}")
    print(f"  Distinct-2          : {d2:.4f}")
    print(f"  TTR                 : {t:.4f}")
    print(f"  Root TTR            : {ld_.get('root_ttr', float('nan')):.4f}")
    print(f"  Log TTR             : {ld_.get('log_ttr',  float('nan')):.4f}")
    print(f"  Maas TTR            : {ld_.get('maas_ttr', float('nan')):.4f}")
    print(f"  MSTTR               : {ld_.get('msttr',    float('nan')):.4f}")
    print(f"  MATTR               : {ld_.get('mattr',    float('nan')):.4f}")
    print(f"  MTLD                : {m:.2f}")
    print(f"  MTLD-MA (wrap)      : {ld_.get('mtld_ma_w',   float('nan')):.2f}")
    print(f"  MTLD-MA (bi-dir)    : {ld_.get('mtld_ma_bid', float('nan')):.2f}")
    print(f"  HDD                 : {h:.4f}")
    print(f"  Cosine Diversity    : {cd:.4f}  (human redial ≥ 0.39)")
    print(f"  Item Mention Entropy: {ie:.4f}  (bits)")

    stats = _conv_stats(folder)
    return {
        'label': label,
        'folder': folder,
        'num_utterances': len(utterances),
        'success_rate': stats['success_rate'],
        'avg_turns': stats['avg_turns'],
        'distinct_1': d1,
        'distinct_2': d2,
        'ttr': t,
        'root_ttr':    ld_.get('root_ttr',    float('nan')),
        'log_ttr':     ld_.get('log_ttr',     float('nan')),
        'maas_ttr':    ld_.get('maas_ttr',    float('nan')),
        'msttr':       ld_.get('msttr',       float('nan')),
        'mattr':       ld_.get('mattr',       float('nan')),
        'mtld': m,
        'mtld_ma_w':   ld_.get('mtld_ma_w',   float('nan')),
        'mtld_ma_bid': ld_.get('mtld_ma_bid', float('nan')),
        'hdd': h,
        'cosine_diversity': cd,
        'item_entropy': ie,
    }


def compare_folders(sim_folder: str, real_folder: str, output_csv: str | None = None):
    """Compute metrics for both folders and print a gap report."""
    sim_r  = analyze_folder(sim_folder,  label='SimuConv (generated)')
    real_r = analyze_folder(real_folder, label='ReDial (real human)')

    print(f"\n{'='*60}")
    print("=== Gap Report (|SimuConv − Real|) ===")

    def gap(key, name, higher_better=True):
        s, r = sim_r[key], real_r[key]
        diff = s - r
        direction = '↑ sim higher' if diff > 0 else '↓ sim lower'
        note = '(good)' if (diff > 0) == higher_better else '(gap)'
        print(f"  {name:<24}: sim={s:.4f}  real={r:.4f}  diff={diff:+.4f}  {direction} {note}")

    gap('distinct_1',       'Distinct-1',           higher_better=True)
    gap('distinct_2',       'Distinct-2',           higher_better=True)
    gap('ttr',              'TTR',                  higher_better=True)
    gap('root_ttr',         'Root TTR',             higher_better=True)
    gap('log_ttr',          'Log TTR',              higher_better=True)
    gap('maas_ttr',         'Maas TTR',             higher_better=False)
    gap('msttr',            'MSTTR',                higher_better=True)
    gap('mattr',            'MATTR',                higher_better=True)
    gap('mtld',             'MTLD',                 higher_better=True)
    gap('mtld_ma_w',        'MTLD-MA (wrap)',        higher_better=True)
    gap('mtld_ma_bid',      'MTLD-MA (bi-dir)',      higher_better=True)
    gap('hdd',              'HDD',                  higher_better=True)
    gap('cosine_diversity', 'Cosine Diversity',      higher_better=True)
    gap('item_entropy',     'Item Mention Entropy', higher_better=True)

    if output_csv:
        import csv
        rows = [sim_r, real_r]
        fieldnames = list(rows[0].keys())
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"\nResults saved to: {output_csv}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Diversity & comparison metrics for SimuConv')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--folder', type=str,
                       help='Single folder to analyze (no-reference metrics only)')
    group.add_argument('--sim', type=str,
                       help='Simulated conversation folder (use with --real)')

    parser.add_argument('--real', type=str, default=None,
                        help='Real conversation folder (ReDial), required when --sim is used')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: save results as CSV')
    args = parser.parse_args()

    if args.folder:
        analyze_folder(args.folder)
    else:
        if not args.real:
            parser.error('--real is required when using --sim')
        compare_folders(args.sim, args.real, args.output)
