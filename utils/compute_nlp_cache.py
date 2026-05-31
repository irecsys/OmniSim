"""
Precompute NLP metrics for all chat folders and save to results/nlp_metrics_cache.csv.
Run once from the OmniSim root directory:

    python utils/compute_nlp_cache.py

The Dashboard will then load instantly from cache on next open.
"""

import os
import re
import random
import math
import pandas as pd
from collections import Counter
from math import comb

RESULTS_DIR = "results"
NLP_CACHE_CSV = os.path.join(RESULTS_DIR, "nlp_metrics_cache.csv")
MODES = ["free", "static", "adaptive"]
SEED = 2025

random.seed(SEED)

# Load SentenceTransformer once globally
_ST_MODEL = None
def _get_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _ST_MODEL


def _tok(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


def compute_metrics(folder: str, label: str, file_sample=None) -> dict | None:
    file_list = file_sample if file_sample is not None else [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files
        if f.endswith('.txt') and 'tmp' not in f
    ]
    if not file_list:
        return None

    utts = []
    for fpath in file_list:
        try:
            with open(fpath, encoding='utf-8') as fh:
                for line in fh:
                    line = line.strip()
                    if line.startswith('Bot:') or line.startswith('User:'):
                        utts.append(line.split(':', 1)[1].strip())
        except Exception:
            pass
    if not utts:
        return None

    tokens = [t for u in utts for t in _tok(u)]
    # Cap tokens for LD metrics to keep MTLD-MA tractable on large corpora
    MAX_TOKENS = 5_000
    tokens_ld = tokens[:MAX_TOKENS] if len(tokens) > MAX_TOKENS else tokens
    N = len(tokens)
    cnt = Counter(tokens)

    ng1 = tokens
    ng2 = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    d1 = len(set(ng1)) / len(ng1) if ng1 else 0
    d2 = len(set(ng2)) / len(ng2) if ng2 else 0

    D = 42
    hdd_val = sum(1.0 - comb(N-n, D)/comb(N, D) if n <= N-D else 1.0
                  for n in cnt.values()) if N >= D else 0.0

    ld_vals = {}
    try:
        from lexical_diversity import lex_div as ld
        ld_vals = {
            'ttr':         ld.ttr(tokens_ld),
            'root_ttr':    ld.root_ttr(tokens_ld),
            'log_ttr':     ld.log_ttr(tokens_ld),
            'maas_ttr':    ld.maas_ttr(tokens_ld),
            'msttr':       ld.msttr(tokens_ld),
            'mattr':       ld.mattr(tokens_ld),
            'mtld':        ld.mtld(tokens_ld),
            'mtld_ma_w':   ld.mtld_ma_wrap(tokens_ld),
            'mtld_ma_bid': ld.mtld_ma_bid(tokens_ld),
        }
    except Exception as e:
        print(f"  [warn] lexical_diversity failed for {label}: {e}")

    cosine_div = 0.0
    try:
        import numpy as np
        _model = _get_st_model()
        sample_utts = utts[:200]
        embs = _model.encode(sample_utts, normalize_embeddings=True, show_progress_bar=False)
        sim_matrix = np.dot(embs, embs.T)
        n = len(sample_utts)
        upper = sim_matrix[np.triu_indices(n, k=1)]
        cosine_div = float(1 - upper.mean())
    except Exception as e:
        print(f"  [warn] cosine diversity failed for {label}: {e}")

    item_entropy = 0.0
    try:
        item_cnt = Counter()
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith('.txt') and 'tmp' not in f:
                    parts = f.replace('.txt', '').split('-')
                    if len(parts) >= 2:
                        item_cnt[parts[1]] += 1
        total_items = sum(item_cnt.values())
        if total_items > 0:
            item_entropy = -sum((c/total_items)*math.log2(c/total_items) for c in item_cnt.values())
    except Exception:
        pass

    _fn_pat = re.compile(r'-(\d+)-(\d+)-([01])-(\d{10,})\.txt$')
    stat_files = [os.path.basename(fp) for fp in file_list] if file_sample is not None else \
                 [f for _, _, fs in os.walk(folder) for f in fs if f.endswith('.txt') and 'tmp' not in f]
    total = len(stat_files)
    succ, turns_list = 0, []
    for f in stat_files:
        m = _fn_pat.search(f)
        if m:
            turns_list.append(int(m.group(1)))
            succ += int(m.group(3))
    avg_turns = sum(turns_list) / len(turns_list) if turns_list else 0.0

    return {
        'label': label, 'folder': folder,
        'num_files': total,
        'num_utterances': len(utts),
        'success_rate': f"{succ/total:.0%}" if total else "—",
        'avg_turns': round(avg_turns, 1),
        'distinct_1': d1, 'distinct_2': d2,
        'hdd': hdd_val,
        'cosine_diversity': cosine_div,
        'item_entropy': item_entropy,
        **ld_vals,
    }


def all_txt_files(folder):
    return [
        os.path.join(root, f)
        for root, _, files in os.walk(folder)
        for f in files if f.endswith('.txt') and 'tmp' not in f
    ]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}

    # ── ReDial ────────────────────────────────────────────────
    real_folder = os.path.join("chats", "redial", "real_low_turn")
    real_all = all_txt_files(real_folder)
    print(f"[1/14] ReDial-500 sample ({len(real_all)} available)…")
    sample_500 = random.sample(real_all, min(500, len(real_all)))
    results["ReDial"] = compute_metrics(real_folder, "ReDial", file_sample=sample_500)

    # ── ReDial Full (all 11348 conversations) ─────────────────
    real_full_folder = os.path.join("chats", "redial", "real_full")
    print(f"[2/14] ReDial-Full ({len(all_txt_files(real_full_folder))} files)…")
    results["ReDial-Full"] = compute_metrics(real_full_folder, "ReDial-Full")

    # ── LLama + GPT ──────────────────────────────────────────
    idx = 3
    for model_prefix in ["llama", "gpt"]:
        for mode in MODES:
            folder = os.path.join("chats", model_prefix, "imdb", mode)
            all_files = all_txt_files(folder) if os.path.isdir(folder) else []
            n = len(all_files)

            key_full = f"{model_prefix}_{mode}"
            key_200  = f"{model_prefix}_{mode}_200"

            print(f"[{idx}/14] {key_full} ({n} files)…")
            results[key_full] = compute_metrics(folder, key_full) if n else None
            idx += 1

            print(f"[{idx}/14] {key_200} (200-sample)…")
            results[key_200] = compute_metrics(
                folder, key_200,
                file_sample=random.sample(all_files, min(200, n))
            ) if n else None
            idx += 1

    # ── Save ──────────────────────────────────────────────────
    rows = []
    for key, r in results.items():
        if r:
            row = {"key": key}
            row.update({k: v for k, v in r.items() if k != "key"})
            rows.append(row)
    pd.DataFrame(rows).to_csv(NLP_CACHE_CSV, index=False)
    print(f"\nDone — saved {len(rows)} rows to {NLP_CACHE_CSV}")


if __name__ == "__main__":
    main()
