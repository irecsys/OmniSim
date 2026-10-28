"""
Run LLM-as-Judge evaluation (utils/judge.py) for several groups of conversations with several judge models.

GROUPS maps a group name to a folder of conversation files, and JUDGES maps a judge name to its provider,
model and number of parallel workers; edit both for your own runs. The values in this file are an example
(the paper's setting: six simulated groups, two simulator models x three modes, plus the ReDial baseline).

Usage:
    python scripts/run_judge_all.py
    python scripts/run_judge_all.py --limit 50
    python scripts/run_judge_all.py --judges <judge> [<judge> ...]
    python scripts/run_judge_all.py --groups <group> [<group> ...]

Outputs:
    results/judge_{group}_{judge}.csv     (all conversations of one group, scored by one judge)
    <group folder>/{judge}/eval_*.csv     (one file per conversation)
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from utils.judge import evaluate_folder, _build_client

# ── Configuration ─────────────────────────────────────────────

GROUPS = {
    "llama_free":     "chats/llama/imdb/free/user_item_pairs/all",
    "llama_static":   "chats/llama/imdb/static/user_item_pairs/all",
    "llama_adaptive": "chats/llama/imdb/adaptive/user_item_pairs/all",
    "gpt_free":       "chats/gpt/imdb/free/user_item_pairs/all",
    "gpt_static":     "chats/gpt/imdb/static/user_item_pairs/all",
    "gpt_adaptive":   "chats/gpt/imdb/adaptive/user_item_pairs/all",
    "redial":         "chats/redial/real",
}

JUDGES = {
    "gpt": {
        "provider": "azure_foundry",
        "model":    "gpt-4o-mini",
        "workers":  16,   # Azure has high rate limits
    },
    "deepseek": {
        "provider": "deepseek",
        "model":    "deepseek-chat",
        "workers":  8,
    },
    "llama": {
        "provider": "thetaedgecloud",
        "model":    "llama3.1-70b",
        "workers":  4,    # ThetaEdgeCloud more conservative
    },
    "gemini": {
        "provider": "gemini",
        "model":    "gemini-2.0-flash",
        "workers":  8,
    },
}

RESULTS_DIR = "results"


def main():
    parser = argparse.ArgumentParser(description="Batch LLM-as-Judge evaluation")
    parser.add_argument("--limit",  type=int, default=0,
                        help="Limit to first N conversations per group (0 = all)")
    parser.add_argument("--judges", nargs="+", default=list(JUDGES.keys()),
                        choices=list(JUDGES.keys()),
                        help="Which judges to run (default: all)")
    parser.add_argument("--groups", nargs="+", default=list(GROUPS.keys()),
                        choices=list(GROUPS.keys()),
                        help="Which groups to evaluate (default: all)")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    total = len(args.groups) * len(args.judges)
    done  = 0

    for group_key in args.groups:
        folder = GROUPS[group_key]
        if not os.path.isdir(folder):
            print(f"[SKIP] Folder not found: {folder}")
            continue

        for judge_key in args.judges:
            done += 1
            cfg    = JUDGES[judge_key]
            output = os.path.join(RESULTS_DIR, f"judge_{group_key}_{judge_key}.csv")

            print(f"\n{'#'*70}")
            print(f"  [{done}/{total}]  Group={group_key}  Judge={judge_key}")
            print(f"  Folder  : {folder}")
            print(f"  Output  : {output}")
            print(f"  Provider: {cfg['provider']}  Model: {cfg['model']}")
            print(f"{'#'*70}")

            try:
                client = _build_client(cfg["provider"], cfg["model"])
                evaluate_folder(
                    folder_path=folder,
                    client=client,
                    model=cfg["model"],
                    output_csv=output,
                    limit=args.limit,
                    eval_subdir=judge_key,
                    workers=cfg.get("workers", 8),
                )
            except Exception as e:
                print(f"[ERROR] {group_key} / {judge_key}: {e}")

    print(f"\n{'='*70}")
    print("All done.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
