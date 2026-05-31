"""
evaluation/compare_judge.py
─────────────────────────────────────────────────────────────────────
Compare LLM-as-Judge scores between SimuConv simulated conversations
and matched ReDial real conversations, across all three modes.

For each mode (free / static / adaptive):
  1. Score N simulated conversations from the latest run folder
  2. Sample N real conversations from chats/redial/real/ (matching succeed ratio)
  3. Score the real conversations
  4. Save per-conversation CSVs and a summary comparison CSV

Output files:
  results/judge_{mode}_sim.csv      — per-conv scores (simulated)
  results/judge_{mode}_real.csv     — per-conv scores (real)
  results/judge_comparison.csv      — descriptive statistics comparison

Usage:
    python evaluation/compare_judge.py --modes free static adaptive --limit 20
    python evaluation/compare_judge.py --modes adaptive --limit 10 --provider thetaedgecloud
"""

import os
import sys
import json
import random
import argparse
import pandas as pd

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.judge import evaluate_folder, _build_client

CHATS_ROOT   = "chats/imdb"
REAL_DIR     = "chats/redial/real"
RESULTS_DIR  = "results"
STRATEGY     = "user_item_pairs"

SCORE_COLS = ["language_fluency", "conversational_quality", "content_quality", "overall_score"]
SCORE_LABELS = {
    "language_fluency":        "Language Fluency",
    "conversational_quality":  "Conv. Quality",
    "content_quality":         "Content Quality",
    "overall_score":           "Overall",
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def latest_run_folder(mode: str) -> str | None:
    base = os.path.join(CHATS_ROOT, mode, STRATEGY)
    if not os.path.isdir(base):
        return None
    runs = sorted(
        [r for r in os.listdir(base)
         if os.path.isdir(os.path.join(base, r))
         and any(f.endswith(".txt") and "tmp" not in f
                 for f in os.listdir(os.path.join(base, r)))],
        reverse=True,
    )
    return os.path.join(base, runs[0]) if runs else None


def sample_real(n: int, succeed_ratio: float, seed: int = 42) -> list[str]:
    """Return N file paths from REAL_DIR matching succeed_ratio as closely as possible."""
    random.seed(seed)
    all_files = [f for f in os.listdir(REAL_DIR) if f.endswith(".txt")]
    succeed  = [f for f in all_files if "-1-" in f.rsplit("-", 2)[1] or f.endswith("-1-" + f.split("-")[-1])]
    fail     = [f for f in all_files if f not in succeed]

    # Parse succeed flag from filename (second-to-last numeric segment)
    def is_success(fname):
        parts = fname.replace(".txt", "").split("-")
        try:
            return int(parts[-2]) == 1
        except (ValueError, IndexError):
            return False

    succeed = [f for f in all_files if is_success(f)]
    fail    = [f for f in all_files if not is_success(f)]

    n_succ = round(n * succeed_ratio)
    n_fail = n - n_succ
    sampled = (
        random.sample(succeed, min(n_succ, len(succeed))) +
        random.sample(fail,    min(n_fail, len(fail)))
    )
    random.shuffle(sampled)
    return [os.path.join(REAL_DIR, f) for f in sampled[:n]]


def descriptive_stats(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Return descriptive statistics for given columns."""
    stats = []
    for col in cols:
        s = df[col].describe()
        stats.append({
            "Dimension": SCORE_LABELS.get(col, col),
            "Mean":   round(s["mean"], 3),
            "Std":    round(s["std"],  3),
            "Min":    round(s["min"],  2),
            "Q1":     round(s["25%"], 2),
            "Median": round(s["50%"], 2),
            "Q3":     round(s["75%"], 2),
            "Max":    round(s["max"],  2),
            "N":      int(s["count"]),
        })
    return pd.DataFrame(stats)


def make_comparison_table(sim_df: pd.DataFrame, real_df: pd.DataFrame,
                          mode: str) -> pd.DataFrame:
    """Create side-by-side descriptive stats for sim vs real."""
    rows = []
    for col in SCORE_COLS:
        s_sim  = sim_df[col].describe()
        s_real = real_df[col].describe()
        rows.append({
            "Mode":      mode.capitalize(),
            "Dimension": SCORE_LABELS.get(col, col),
            # SimuConv
            "Sim Mean":   round(s_sim["mean"], 3),
            "Sim Std":    round(s_sim["std"],  3),
            "Sim Median": round(s_sim["50%"],  2),
            "Sim Min":    round(s_sim["min"],   2),
            "Sim Max":    round(s_sim["max"],   2),
            # ReDial
            "Real Mean":   round(s_real["mean"], 3),
            "Real Std":    round(s_real["std"],  3),
            "Real Median": round(s_real["50%"],  2),
            "Real Min":    round(s_real["min"],   2),
            "Real Max":    round(s_real["max"],   2),
            # Gap
            "Δ Mean": round(s_sim["mean"] - s_real["mean"], 3),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_comparison(modes: list[str], limit: int, provider: str, model: str,
                   force: bool = False):
    client = _build_client(provider, model)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_comparisons = []

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"MODE: {mode.upper()}")
        print(f"{'='*60}")

        sim_folder = latest_run_folder(mode)
        if not sim_folder:
            print(f"  [SKIP] No run folder found for mode '{mode}'")
            continue
        print(f"  Sim folder : {sim_folder}")

        sim_csv  = os.path.join(RESULTS_DIR, f"judge_{mode}_sim.csv")
        real_csv = os.path.join(RESULTS_DIR, f"judge_{mode}_real.csv")

        # ── Evaluate simulated ──────────────────────────────────
        if os.path.exists(sim_csv) and not force:
            print(f"  [CACHED] Loading sim scores from {sim_csv}")
            sim_df = pd.read_csv(sim_csv)
        else:
            print(f"  Evaluating simulated conversations (limit={limit or 'all'})…")
            sim_df = evaluate_folder(sim_folder, client, model,
                                     output_csv=sim_csv, limit=limit)
            if sim_df is None or sim_df.empty:
                print(f"  [SKIP] No sim results for mode '{mode}'")
                continue

        n = len(sim_df)
        succeed_ratio = sim_df["succeed"].mean() if "succeed" in sim_df.columns else 0.5

        # ── Sample & evaluate real ──────────────────────────────
        if os.path.exists(real_csv) and not force:
            print(f"  [CACHED] Loading real scores from {real_csv}")
            real_df = pd.read_csv(real_csv)
        else:
            print(f"  Sampling {n} real conversations (succeed_ratio={succeed_ratio:.2f})…")
            real_files = sample_real(n, succeed_ratio)

            # Write temp folder structure for evaluate_folder
            import tempfile, shutil
            tmpdir = tempfile.mkdtemp(prefix="simuconv_real_")
            try:
                for fpath in real_files:
                    shutil.copy(fpath, tmpdir)
                real_df = evaluate_folder(tmpdir, client, model,
                                          output_csv=real_csv, limit=0)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            if real_df is None or real_df.empty:
                print(f"  [SKIP] No real results for mode '{mode}'")
                continue

        # ── Descriptive stats ───────────────────────────────────
        print(f"\n  ── Descriptive Statistics: SimuConv vs ReDial ({mode}) ──")
        sim_stats  = descriptive_stats(sim_df,  SCORE_COLS)
        real_stats = descriptive_stats(real_df, SCORE_COLS)

        merged = sim_stats.copy()
        for col in ["Mean", "Std", "Median", "Min", "Max"]:
            merged[f"Real {col}"] = real_stats[col].values
            merged[f"Sim {col}"]  = sim_stats[col].values

        print("\n  SimuConv:")
        print(sim_stats.to_string(index=False))
        print("\n  ReDial (real):")
        print(real_stats.to_string(index=False))

        comp = make_comparison_table(sim_df, real_df, mode)
        all_comparisons.append(comp)

    if all_comparisons:
        full_comp = pd.concat(all_comparisons, ignore_index=True)
        out = os.path.join(RESULTS_DIR, "judge_comparison.csv")
        full_comp.to_csv(out, index=False)
        print(f"\n{'='*60}")
        print(f"Full comparison saved to: {out}")
        print(f"{'='*60}")
        print(full_comp.to_string(index=False))
        return full_comp

    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM-as-Judge scores: SimuConv vs ReDial per mode"
    )
    parser.add_argument("--modes",    nargs="+", default=["free", "static", "adaptive"],
                        help="Modes to compare (default: all three)")
    parser.add_argument("--limit",    type=int, default=20,
                        help="Max conversations per folder (0 = all)")
    parser.add_argument("--provider", default="thetaedgecloud",
                        choices=["thetaedgecloud", "openai", "azure"])
    parser.add_argument("--model",    default="meta-llama/Meta-Llama-3.1-70B-Instruct")
    parser.add_argument("--force",    action="store_true",
                        help="Re-evaluate even if cached CSV exists")
    args = parser.parse_args()

    run_comparison(
        modes=args.modes,
        limit=args.limit,
        provider=args.provider,
        model=args.model,
        force=args.force,
    )
