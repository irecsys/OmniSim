"""
Sample redial (ReDial) conversations to match the structural distribution of
simulated conversations (num_turns, rec_attempts).

Strategy:
  1. Parse ALL ReDial conversations from the raw JSONL files (11k+).
  2. Parse the target simulated folder(s) to get the reference distribution.
  3. Use histogram-matched weighted sampling so the sampled real conversations
     have a similar num_turns (and optionally rec_attempts) distribution.
  4. Convert and write the sampled conversations in SimuConv TXT format.

Usage:
    # Match distribution of one sim folder
    python scripts/sample_baseline.py \
        --sim chats/imdb/adaptive/user_item_pairs/20260321074527

    # Match combined distribution of all three modes (recommended)
    python scripts/sample_baseline.py \
        --sim chats/imdb/free/user_item_pairs/20260321045333 \
              chats/imdb/static/user_item_pairs/20260321060337 \
              chats/imdb/adaptive/user_item_pairs/20260321074527 \
        --n 200 \
        --output chats/redial/real
"""

import os
import re
import json
import csv
import random
import argparse
import datetime
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_DATA_DIR   = "data/redial"
_MOVIES_CSV = os.path.join(_DATA_DIR, "movies_with_mentions.csv")
_JSONL_FILES = [
    os.path.join(_DATA_DIR, "train_data.jsonl"),
    os.path.join(_DATA_DIR, "test_data.jsonl"),
]
_SIM_FNAME_RE = re.compile(
    r'^(?:[^-]+-)?[^-]+-(\d+)-(\d+)-([01])-\d+\.txt$'
)


# ---------------------------------------------------------------------------
# Movie title lookup
# ---------------------------------------------------------------------------
def _load_movies(csv_path: str) -> dict:
    movies = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            raw = row['movieName'].strip()
            clean = re.sub(r'\s*\(\d{4}\)\s*$', '', raw).strip()
            movies[row['movieId']] = {'title': clean, 'full': raw}
    return movies


def _replace_refs(text: str, mentions: dict, movies: dict) -> str:
    def rep(m):
        mid = m.group(1)
        if mid in mentions:
            return f'"{re.sub(r"\\s*\\(\\d{{4}}\\)\\s*$", "", mentions[mid]).strip()}"'
        if mid in movies:
            return f'"{movies[mid]["title"]}"'
        return f'movie#{mid}'
    return re.sub(r'@(\d+)', rep, text)


# ---------------------------------------------------------------------------
# Parse raw ReDial JSONL to get quick stats + full data
# ---------------------------------------------------------------------------
def _parse_redial_all(movies: dict) -> list[dict]:
    """Return list of dicts with keys: conv_id, num_turns, rec_attempts, succeed, lines."""
    records = []
    for jsonl_path in _JSONL_FILES:
        if not os.path.exists(jsonl_path):
            continue
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                d = json.loads(line)
                msgs = d.get('messages', [])
                if not msgs:
                    continue

                init_id = d['initiatorWorkerId']
                resp_id = d['respondentWorkerId']
                mentions = d.get('movieMentions', {})
                rq = d.get('respondentQuestions', {})
                if not isinstance(rq, dict):
                    rq = {}

                suggested      = [mid for mid, q in rq.items() if q.get('suggested', 0) == 1]
                liked_sug      = [mid for mid in suggested if rq[mid].get('liked', 0) == 1]
                succeed        = 1 if liked_sug else 0
                target_id      = liked_sug[0] if liked_sug else (suggested[0] if suggested else None)
                rec_attempts   = len(suggested)

                if target_id and target_id in mentions:
                    tgt_title = re.sub(r'\s*\(\d{4}\)\s*$', '', mentions[target_id]).strip()
                elif target_id and target_id in movies:
                    tgt_title = movies[target_id]['title']
                else:
                    tgt_title = f'movie#{target_id}' if target_id else 'unknown'

                lines = []
                for msg in msgs:
                    text   = _replace_refs(msg.get('text', '').strip(), mentions, movies)
                    sender = msg.get('senderWorkerId')
                    speaker = 'User' if sender == init_id else 'Bot'
                    lines.append(f'{speaker}: {text}')

                num_turns = len(lines)

                if succeed and target_id:
                    lines.append(
                        f"System: <END> Session ended successfully. "
                        f"The target item '{tgt_title}' ({target_id}) was accepted by the user."
                    )
                else:
                    lines.append(
                        f"System: <END> Session ended. "
                        f"The target item '{tgt_title}' ({target_id}) was NOT found."
                    )

                records.append({
                    'conv_id':      d['conversationId'],
                    'target_id':    target_id or d['conversationId'],
                    'num_turns':    num_turns,
                    'rec_attempts': rec_attempts,
                    'succeed':      succeed,
                    'lines':        lines,
                })
    return records


# ---------------------------------------------------------------------------
# Parse simulated folder(s) for reference distribution
# ---------------------------------------------------------------------------
def _parse_sim_folders(folders: list[str]) -> pd.DataFrame:
    rows = []
    for folder in folders:
        for fname in os.listdir(folder):
            if not fname.endswith('.txt'):
                continue
            m = _SIM_FNAME_RE.match(fname)
            if m:
                rows.append({
                    'num_turns':    int(m.group(1)),
                    'rec_attempts': int(m.group(2)),
                    'succeed':      int(m.group(3)),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Histogram-matched sampling
# ---------------------------------------------------------------------------
def _matched_sample(pool_df: pd.DataFrame, ref_df: pd.DataFrame,
                    n: int, seed: int = 42) -> pd.DataFrame:
    """
    Sample n rows from pool_df whose num_turns distribution matches ref_df.
    Uses histogram-bin weighting.
    """
    rng = np.random.default_rng(seed)

    # Determine bins covering both distributions
    all_turns = pd.concat([pool_df['num_turns'], ref_df['num_turns']])
    lo, hi = int(all_turns.min()), int(all_turns.max())
    bins = list(range(lo, hi + 2))  # [lo, lo+1, ..., hi+1]

    # Reference histogram (normalised)
    ref_counts, _ = np.histogram(ref_df['num_turns'], bins=bins)
    ref_freq = ref_counts / ref_counts.sum()

    # Assign weight to each pool row based on its bin
    pool_bins = np.digitize(pool_df['num_turns'].values, bins) - 1
    pool_bins = np.clip(pool_bins, 0, len(ref_freq) - 1)
    weights = ref_freq[pool_bins]

    # Zero-weight rows in bins that don't appear in the reference
    weights_sum = weights.sum()
    if weights_sum == 0:
        # Fallback: uniform
        weights = np.ones(len(pool_df))
        weights_sum = weights.sum()

    weights = weights / weights_sum

    idx = rng.choice(len(pool_df), size=min(n, len(pool_df)),
                     replace=(n > len(pool_df)), p=weights)
    return pool_df.iloc[idx].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Sample ReDial redial conversations to match simulated distribution'
    )
    parser.add_argument(
        '--sim', nargs='+', required=True,
        help='One or more simulated conversation folders (used as reference distribution)'
    )
    parser.add_argument('--n',         type=int, default=200,  help='Number of redial conversations to sample')
    parser.add_argument('--output',    type=str, default='chats/redial/real', help='Output folder')
    parser.add_argument('--seed',      type=int, default=42,   help='Random seed')
    parser.add_argument('--max-turns', type=int, default=None,
                        help='Hard cap on ReDial num_turns (overrides range filter upper bound)')
    args = parser.parse_args()

    print(f"Loading movies from {_MOVIES_CSV} ...")
    movies = _load_movies(_MOVIES_CSV)

    print("Parsing reference simulated folders ...")
    ref_df = _parse_sim_folders(args.sim)
    if ref_df.empty:
        print("ERROR: no valid simulated files found."); return

    print(f"  Reference: N={len(ref_df)}  turns={ref_df['num_turns'].mean():.1f}±{ref_df['num_turns'].std():.1f}"
          f"  rec={ref_df['rec_attempts'].mean():.1f}±{ref_df['rec_attempts'].std():.1f}"
          f"  succeed={ref_df['succeed'].mean():.1%}")

    print("Parsing full ReDial dataset ...")
    redial_all = _parse_redial_all(movies)
    pool_df = pd.DataFrame([{k: r[k] for k in ('conv_id','target_id','num_turns','rec_attempts','succeed')}
                             for r in redial_all])
    conv_lookup = {r['conv_id']: r for r in redial_all}

    print(f"  ReDial pool: N={len(pool_df)}  turns={pool_df['num_turns'].mean():.1f}±{pool_df['num_turns'].std():.1f}"
          f"  rec={pool_df['rec_attempts'].mean():.1f}±{pool_df['rec_attempts'].std():.1f}"
          f"  succeed={pool_df['succeed'].mean():.1%}")

    # Filter pool to range ± some slack around reference
    ref_min = max(1, int(ref_df['num_turns'].min()) - 2)
    ref_max = int(ref_df['num_turns'].max()) + 2
    if args.max_turns is not None:
        ref_max = min(ref_max, args.max_turns)
    filtered = pool_df[(pool_df['num_turns'] >= ref_min) & (pool_df['num_turns'] <= ref_max)]
    if len(filtered) < args.n:
        print(f"  WARNING: only {len(filtered)} conversations in turns range [{ref_min},{ref_max}]. "
              f"Expanding (max-turns cap kept).")
        filtered = pool_df[pool_df['num_turns'] <= (args.max_turns or pool_df['num_turns'].max())]
    if len(filtered) < args.n:
        print(f"  WARNING: still only {len(filtered)} conversations. Using full pool.")
        filtered = pool_df

    print(f"  After range filter [{ref_min},{ref_max}]: {len(filtered)} available")

    sampled = _matched_sample(filtered, ref_df, args.n, seed=args.seed)

    print(f"\nSampled {len(sampled)} conversations:")
    print(f"  turns={sampled['num_turns'].mean():.1f}±{sampled['num_turns'].std():.1f}"
          f"  rec={sampled['rec_attempts'].mean():.1f}±{sampled['rec_attempts'].std():.1f}"
          f"  succeed={sampled['succeed'].mean():.1%}")

    # Write output
    os.makedirs(args.output, exist_ok=True)
    # Clear existing txt files
    for f in os.listdir(args.output):
        if f.endswith('.txt'):
            os.remove(os.path.join(args.output, f))

    written = 0
    ts_base = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    for i, row in sampled.iterrows():
        conv = conv_lookup[row['conv_id']]
        ts = f"{ts_base}{i:06d}"
        fname = (
            f"redial-{conv['target_id']}"
            f"-{conv['num_turns']}"
            f"-{conv['rec_attempts']}"
            f"-{conv['succeed']}"
            f"-{ts}.txt"
        )
        with open(os.path.join(args.output, fname), 'w', encoding='utf-8') as fout:
            fout.write('\n'.join(conv['lines']) + '\n')
        written += 1

    print(f"\nWrote {written} files → {args.output}")
    print("\nDistribution comparison (reference vs sampled):")
    print(f"  {'Metric':<15} {'Sim (ref)':<20} {'ReDial (sampled)':<20}")
    for col, label in [('num_turns', 'Num Turns'), ('rec_attempts', 'Rec Attempts')]:
        sim_s  = f"{ref_df[col].mean():.1f}±{ref_df[col].std():.1f}"
        samp_s = f"{sampled[col].mean():.1f}±{sampled[col].std():.1f}"
        print(f"  {label:<15} {sim_s:<20} {samp_s:<20}")


if __name__ == '__main__':
    main()
