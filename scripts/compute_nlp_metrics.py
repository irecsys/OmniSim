"""
compute_nlp_metrics.py — Compute all NLP metrics for OmniSim vs ReDial redial.

Computes: Distinct-1/2, TTR variants (Root/Log/Maas/MSTTR/MATTR),
          MTLD/MTLD-MA, HDD, Cosine Diversity, Item Entropy,
          Avg Turns, Success Rate.

Results saved to results/nlp_metrics_cache.csv (read by dashboard).

Usage:
    python scripts/compute_nlp_metrics.py
    python scripts/compute_nlp_metrics.py --dataset hm
    python scripts/compute_nlp_metrics.py --modes free adaptive
    python scripts/compute_nlp_metrics.py --cosine-sample 300
"""
import os
import re
import math
import argparse
from collections import Counter

import pandas as pd

# ── Default paths ──────────────────────────────────────────────────────────
RESULTS_DIR = "results"
NLP_CACHE_CSV = os.path.join(RESULTS_DIR, "nlp_metrics_cache.csv")
MODES = ["free", "static", "adaptive"]


# ── Utterance loading ──────────────────────────────────────────────────────
def load_utterances(folder: str) -> list[str]:
    utts = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if not fname.endswith('.txt'):
                continue
            try:
                with open(os.path.join(root, fname), encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('Bot:') or line.startswith('User:'):
                            text = line.split(':', 1)[1].strip()
                            if text:
                                utts.append(text)
            except Exception:
                pass
    return utts


def tokenize(text: str) -> list[str]:
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


# ── Metric computations ────────────────────────────────────────────────────
def compute_distinct(utts: list[str]) -> tuple[float, float]:
    tokens = [t for u in utts for t in tokenize(u)]
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
    d1 = len(set(tokens)) / len(tokens) if tokens else 0
    d2 = len(set(bigrams)) / len(bigrams) if bigrams else 0
    return d1, d2


def compute_hdd(tokens: list[str], D: int = 42) -> float:
    from math import comb
    N = len(tokens)
    if N < D:
        return 0.0
    cnt = Counter(tokens)
    return sum(
        1.0 - comb(N - n, D) / comb(N, D) if n <= N - D else 1.0
        for n in cnt.values()
    )


def compute_ld_metrics(tokens: list[str]) -> dict:
    try:
        from lexical_diversity import lex_div as ld
        flt = ld.flemmatize(' '.join(tokens))
        return {
            'ttr':         ld.ttr(flt),
            'root_ttr':    ld.root_ttr(flt),
            'log_ttr':     ld.log_ttr(flt),
            'maas_ttr':    ld.maas_ttr(flt),
            'msttr':       ld.msttr(flt),
            'mattr':       ld.mattr(flt),
            'mtld':        ld.mtld(flt),
            'mtld_ma_w':   ld.mtld_ma_wrap(flt),
            'mtld_ma_bid': ld.mtld_ma_bid(flt),
        }
    except Exception as e:
        print(f"    [warn] lexical_diversity error: {e}")
        return {}


def compute_cosine_diversity(utts: list[str], n_sample: int = 200) -> float:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sample = utts[:n_sample]
        embs = model.encode(sample, normalize_embeddings=True, show_progress_bar=False)
        sim = embs @ embs.T
        n = len(sample)
        upper = sim[__import__('numpy').triu_indices(n, k=1)]
        return float(1 - upper.mean())
    except Exception as e:
        print(f"    [warn] cosine diversity error: {e}")
        return 0.0


def compute_item_entropy(folder: str) -> float:
    cnt = Counter()
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.endswith('.txt'):
                parts = fname.replace('.txt', '').split('-')
                if len(parts) >= 2:
                    cnt[parts[1]] += 1
    total = sum(cnt.values())
    if not total:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in cnt.values())


def compute_turns_and_success(folder: str) -> tuple[float, float]:
    turns_list, success_list = [], []
    fname_re = re.compile(r'-(\d+)-(\d+)-([01])-\d+\.txt$')
    for root, _, files in os.walk(folder):
        for fname in files:
            m = fname_re.search(fname)
            if m:
                turns_list.append(int(m.group(1)))
                success_list.append(int(m.group(3)))
    avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0.0
    success_rate = sum(success_list) / len(success_list) * 100 if success_list else 0.0
    return round(avg_turns, 2), round(success_rate, 1)


# ── Main compute ───────────────────────────────────────────────────────────
def compute_folder(folder: str, label: str, cosine_sample: int = 200) -> dict | None:
    utts = load_utterances(folder)
    if not utts:
        print(f"  [{label}] no utterances found in {folder}")
        return None

    tokens = [t for u in utts for t in tokenize(u)]

    print(f"  [{label}] {len(utts)} utterances, {len(tokens)} tokens — computing...")

    d1, d2 = compute_distinct(utts)
    hdd    = compute_hdd(tokens)
    ld     = compute_ld_metrics(tokens)
    cos    = compute_cosine_diversity(utts, cosine_sample)
    ent    = compute_item_entropy(folder)
    turns, succ = compute_turns_and_success(folder)

    result = {
        'label':            label,
        'avg_turns':        turns,
        'success_rate':     succ,
        'distinct_1':       round(d1, 4),
        'distinct_2':       round(d2, 4),
        'hdd':              round(hdd, 4),
        'cosine_diversity': round(cos, 4),
        'item_entropy':     round(ent, 4),
        **{k: round(v, 4) for k, v in ld.items()},
    }

    print(f"    D1={d1:.3f}  D2={d2:.3f}  MTLD={ld.get('mtld', 0):.1f}  "
          f"HDD={hdd:.3f}  CosDiv={cos:.3f}  Entropy={ent:.3f}  "
          f"Turns={turns}  Succ={succ}%")
    return result


# ── Entry point ────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute NLP metrics for OmniSim vs ReDial")
    parser.add_argument('--dataset',       default='imdb',  help='Dataset name (imdb/hm)')
    parser.add_argument('--modes',         nargs='+', default=MODES, help='Modes to compute')
    parser.add_argument('--redial',      default='chats/redial/real', help='ReDial redial folder')
    parser.add_argument('--cosine-sample', type=int, default=200, help='Max utterances for cosine diversity')
    parser.add_argument('--output',        default=NLP_CACHE_CSV, help='Output CSV path')
    parser.add_argument('--strategy',      default='user_item_pairs', help='Conversation strategy subfolder')
    parser.add_argument('--folder',        default=None, help='Exact run folder to compute (skips mode loop and redial)')
    parser.add_argument('--folders',       default=None, help='Comma-separated list of run folders; metrics computed over combined file set')
    parser.add_argument('--label',         default=None, help='Label for the row when --folder/--folders is used')
    args = parser.parse_args()

    rows = []

    # Multi-folder mode: pool all txt files from specified folders and compute metrics on the merged set
    if args.folders:
        folder_list = [f.strip() for f in args.folders.split(',') if f.strip()]
        label = args.label or '+'.join(os.path.basename(f.rstrip('/')) for f in folder_list)
        print(f"Computing {label} over {len(folder_list)} folder(s)...")
        all_files = []
        for fd in folder_list:
            for root, _, files in os.walk(fd):
                for fname in files:
                    if fname.endswith('.txt') and 'tmp' not in fname:
                        all_files.append(os.path.join(root, fname))
        print(f"  Total files across folders: {len(all_files)}")
        utts = []
        for fpath in all_files:
            try:
                with open(fpath, encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('Bot:') or line.startswith('User:'):
                            text = line.split(':', 1)[1].strip()
                            if text:
                                utts.append(text)
            except Exception:
                pass
        if not utts:
            print("No utterances found across specified folders.")
            return
        tokens = [t for u in utts for t in __import__('re').findall(r'\b[a-zA-Z]+\b', u.lower())]
        print(f"  {len(utts)} utterances, {len(tokens)} tokens — computing...")
        d1, d2 = compute_distinct(utts)
        hdd    = compute_hdd(tokens)
        ld     = compute_ld_metrics(tokens)
        cos    = compute_cosine_diversity(utts, args.cosine_sample)
        # item entropy from filenames across all folders
        from collections import Counter as _Counter
        import math as _math
        item_cnt = _Counter()
        for fp in all_files:
            parts = os.path.basename(fp).replace('.txt','').split('-')
            if len(parts) >= 2:
                item_cnt[parts[1]] += 1
        total_items = sum(item_cnt.values())
        ent = -sum((c/total_items)*_math.log2(c/total_items) for c in item_cnt.values()) if total_items else 0.0
        # turns/success from filenames
        import re as _re
        fn_pat = _re.compile(r'-(\d+)-(\d+)-([01])-\d+\.txt$')
        turns_list, succ_list = [], []
        for fp in all_files:
            m2 = fn_pat.search(os.path.basename(fp))
            if m2:
                turns_list.append(int(m2.group(1)))
                succ_list.append(int(m2.group(3)))
        avg_turns = round(sum(turns_list)/len(turns_list), 2) if turns_list else 0.0
        succ_rate = round(sum(succ_list)/len(succ_list)*100, 1) if succ_list else 0.0
        r = {
            'label': label,
            'avg_turns': avg_turns, 'success_rate': succ_rate,
            'distinct_1': round(d1,4), 'distinct_2': round(d2,4),
            'hdd': round(hdd,4), 'cosine_diversity': round(cos,4), 'item_entropy': round(ent,4),
            **{k: round(v,4) for k,v in ld.items()},
        }
        rows.append(r)
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
        avail = [c for c in ['label','distinct_1','distinct_2','mtld','hdd','cosine_diversity','item_entropy','avg_turns','success_rate'] if c in df.columns]
        print(df[avail].to_string(index=False))
        return

    # Single-folder mode: compute only the specified folder, skip redial and mode loop
    if args.folder:
        label = args.label or os.path.basename(args.folder.rstrip('/'))
        print(f"Computing {label} ({args.folder})...")
        r = compute_folder(args.folder, label, args.cosine_sample)
        if r:
            rows.append(r)
        if not rows:
            print("No data computed.")
            return
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")
        avail = [c for c in ['label', 'distinct_1', 'distinct_2', 'mtld', 'hdd',
                  'cosine_diversity', 'item_entropy', 'avg_turns', 'success_rate'] if c in df.columns]
        print(df[avail].to_string(index=False))
        return

    # ReDial redial
    print(f"Computing ReDial redial ({args.redial})...")
    r = compute_folder(args.redial, 'ReDial', args.cosine_sample)
    if r:
        rows.append(r)

    # OmniSim modes
    for mode in args.modes:
        folder = os.path.join('chats', args.dataset, mode, args.strategy)
        if not os.path.isdir(folder):
            print(f"  [{mode}] folder not found: {folder} — skipping")
            continue
        print(f"Computing {mode} ({folder})...")
        r = compute_folder(folder, mode, args.cosine_sample)
        if r:
            rows.append(r)

    if not rows:
        print("No data computed.")
        return

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    avail = [c for c in ['label', 'distinct_1', 'distinct_2', 'mtld', 'hdd',
              'cosine_diversity', 'item_entropy', 'avg_turns', 'success_rate'] if c in df.columns]
    print(df[avail].to_string(index=False))


if __name__ == '__main__':
    main()
