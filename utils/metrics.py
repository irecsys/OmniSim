"""
Metrics — Requirement [6]

Computes conversation-level and utterance-level metrics for a run folder.

Filename pattern:  {item_id}-{number_turns}-{rec_attempts}-{succeed}-{timestamp}.txt

Conversation-level metrics:
  - total conversations, success rate, avg turns, avg rec_attempts
  - turns/attempts distribution

Utterance-level metrics (parsed from txt files):
  - avg utterances per speaker
  - avg utterance length (words)

Usage:
    python utils/metrics.py --folder chats/imdb/adaptive/20250904102301
    python utils/metrics.py --folder chats/imdb/adaptive/20250904102301 --output metrics.csv
"""

import os
import re
import argparse
import pandas as pd
from collections import defaultdict


# ---------------------------------------------------------------------------
# Filename parser
# ---------------------------------------------------------------------------

# Format: {user_id}-{item_id}-{turns}-{attempts}-{succeed}-{timestamp}.txt
# user_id = "user" for anonymous sessions
_FILENAME_RE = re.compile(r'^([^-]+)-([^-]+)-(\d+)-(\d+)-([01])-(\d+)\.txt$')


def parse_filename(fname: str) -> dict | None:
    m = _FILENAME_RE.match(fname)
    if m:
        return {
            'user_id': m.group(1),
            'item_id': m.group(2),
            'num_turns': int(m.group(3)),
            'rec_attempts': int(m.group(4)),
            'succeed': int(m.group(5)),
            'timestamp': m.group(6),
        }
    return None


# ---------------------------------------------------------------------------
# Utterance-level parser
# ---------------------------------------------------------------------------

def parse_utterances(file_path: str) -> list[dict]:
    """Parse a chat txt file into a list of {speaker, text, word_count} dicts."""
    utterances = []
    with open(file_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Skip recommendation list items (lines starting with "- ")
            if line.startswith('- '):
                continue
            # Format: "SpeakerName: message text"
            if ': ' in line:
                speaker, _, text = line.partition(': ')
                utterances.append({
                    'speaker': speaker.strip(),
                    'text': text.strip(),
                    'word_count': len(text.split()),
                })
    return utterances


# ---------------------------------------------------------------------------
# Main metrics computation
# ---------------------------------------------------------------------------

def compute_metrics(folder_path: str, output_csv: str | None = None):
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    conv_records = []
    utt_records = []

    for fname in sorted(os.listdir(folder_path)):
        if not fname.endswith('.txt') or 'tmp' in fname:
            continue
        parsed = parse_filename(fname)
        if parsed is None:
            continue
        conv_records.append(parsed)

        fpath = os.path.join(folder_path, fname)
        utts = parse_utterances(fpath)
        for u in utts:
            u['item_id'] = parsed['item_id']
            u['succeed'] = parsed['succeed']
        utt_records.extend(utts)

    if not conv_records:
        print(f"No valid conversation files found in: {folder_path}")
        return

    # ---- Conversation-level ----
    conv_df = pd.DataFrame(conv_records)
    total = len(conv_df)
    success_rate = conv_df['succeed'].mean()
    avg_turns = conv_df['num_turns'].mean()
    avg_attempts = conv_df['rec_attempts'].mean()

    print("\n=== Conversation-Level Metrics ===")
    print(f"Folder           : {folder_path}")
    print(f"Total convs      : {total}")
    print(f"Success rate     : {success_rate:.1%}  ({conv_df['succeed'].sum()} / {total})")
    print(f"Avg turns        : {avg_turns:.2f}")
    print(f"Avg rec attempts : {avg_attempts:.2f}")
    print("\nTurns distribution:")
    print(conv_df['num_turns'].describe().to_string())
    print("\nRec attempts distribution:")
    print(conv_df['rec_attempts'].describe().to_string())

    # ---- Utterance-level ----
    if utt_records:
        utt_df = pd.DataFrame(utt_records)
        print("\n=== Utterance-Level Metrics ===")

        by_speaker = utt_df.groupby('speaker').agg(
            total_utterances=('text', 'count'),
            avg_word_count=('word_count', 'mean'),
            total_words=('word_count', 'sum'),
        ).reset_index()
        print(by_speaker.to_string(index=False))

        avg_utts_per_conv = len(utt_df) / total
        print(f"\nAvg utterances per conversation : {avg_utts_per_conv:.2f}")

    # ---- Save ----
    if output_csv:
        conv_df.to_csv(output_csv, index=False)
        print(f"\nConversation records saved to: {output_csv}")

    return conv_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute metrics for a SimuConv run folder')
    parser.add_argument('--folder', type=str, required=True,
                        help='Path to the run folder (e.g. chats/imdb/adaptive/20250904102301)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: path to save conversation records as CSV')
    args = parser.parse_args()
    compute_metrics(args.folder, args.output)
