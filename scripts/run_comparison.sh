#!/usr/bin/env bash
# ============================================================
# run_comparison.sh
# Runs Free / Static / Adaptive modes for IMDB and compares
# each against a matched ReDial redial sample.
#
# Usage:  bash scripts/run_comparison.sh
# ============================================================

set -e
CONFIG="configs/imdb/imdb_compare.yaml"
N_SAMPLE=150  # match number of generated conversations

echo "========================================================"
echo " SimuConv — Free / Static / Adaptive Comparison"
echo "========================================================"

for MODE in free static adaptive; do
    echo ""
    echo "════════════════════════════════════════════════════════"
    echo " Mode: $MODE"
    echo "════════════════════════════════════════════════════════"

    # 1. Generate conversations
    echo "[1/4] Generating $MODE conversations..."
    python3 run.py --config "$CONFIG" --mode "$MODE"

    # 2. Find the latest run folder for this mode
    RUN_FOLDER=$(ls -1dt chats/imdb/$MODE/user_item_pairs/*/ 2>/dev/null | head -1 | sed 's|/$||')
    if [ -z "$RUN_FOLDER" ]; then
        echo "[WARN] No run folder found for $MODE — skipping."
        continue
    fi
    echo "    Run folder: $RUN_FOLDER"

    # 3. Descriptive statistics (turns, attempts, succeed)
    echo "[2/4] Computing conversation statistics..."
    python3 -m utils.metrics --folder "$RUN_FOLDER"

    # 4. Sample matching redial
    REDIAL_SAMPLE="chats/redial/${MODE}_sample"
    echo "[3/4] Sampling ReDial to match $MODE distribution..."
    python3 scripts/sample_baseline.py --sim "$RUN_FOLDER" --n $N_SAMPLE --output "$REDIAL_SAMPLE"

    # 5. NLP metric comparison
    echo "[4/4] Comparing NLP metrics..."
    python3 -m utils.compare_metrics \
        --sim  "$RUN_FOLDER" \
        --real "$REDIAL_SAMPLE" \
        --output "results/${MODE}_metrics.csv"

    echo "    Results saved to results/${MODE}_metrics.csv"
done

echo ""
echo "========================================================"
echo " All modes done. Results in results/"
echo "========================================================"
