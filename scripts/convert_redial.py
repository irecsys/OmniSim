"""
Convert ReDial real conversations → SimuConv-compatible TXT files.

Each ReDial conversation becomes one .txt file in the same format as chats/:
    Bot: ...
    User: ...
    ...
    System: <END> Session ended successfully. The target item '...' (movieId) was accepted by the user.

Filename pattern (same as SimuConv):
    {movieId}-{num_turns}-{rec_attempts}-{succeed}-{timestamp}.txt

Mapping:
    initiatorWorkerId (senderWorkerId == initiatorId) → User (movie seeker)
    respondentWorkerId (senderWorkerId == respondentId) → Bot (movie recommender)

rec_attempts  = number of movies suggested by the respondent
succeed       = 1 if any suggested movie was liked by the initiator
target item   = first movie that was suggested AND liked (or first suggested if none liked)

Usage:
    python scripts/convert_redial.py
    python scripts/convert_redial.py --split train          # train only
    python scripts/convert_redial.py --split test           # test only
    python scripts/convert_redial.py --limit 500            # first N conversations
    python scripts/convert_redial.py --output chats/redial/real
"""

import json
import re
import os
import csv
import argparse
import datetime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_movies(movies_csv: str) -> dict:
    """Load movieId → clean title mapping from movies_with_mentions.csv."""
    movies = {}
    with open(movies_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_name = row['movieName'].strip()
            # Strip trailing year "(2001)" to get clean title
            clean = re.sub(r'\s*\(\d{4}\)\s*$', '', raw_name).strip()
            movies[row['movieId']] = {'title': clean, 'full': raw_name}
    return movies


def replace_movie_refs(text: str, movie_mentions: dict, movies: dict) -> str:
    """Replace @movieId tokens with movie titles."""
    def replacer(m):
        mid = m.group(1)
        if mid in movie_mentions:
            full = movie_mentions[mid]          # e.g. "Final Fantasy: The Spirits Within (2001)"
            clean = re.sub(r'\s*\(\d{4}\)\s*$', '', full).strip()
            return f'"{clean}"'
        if mid in movies:
            return f'"{movies[mid]["title"]}"'
        return f'movie#{mid}'
    return re.sub(r'@(\d+)', replacer, text)


def convert_conversation(d: dict, movies: dict) -> dict | None:
    """
    Convert one ReDial conversation dict into SimuConv fields.
    Returns None if the conversation has no messages.
    """
    initiator_id = d['initiatorWorkerId']   # int, the seeker
    respondent_id = d['respondentWorkerId'] # int, the recommender
    messages = d.get('messages', [])
    if not messages:
        return None

    movie_mentions = d.get('movieMentions', {})
    rq = d.get('respondentQuestions', {})
    if not isinstance(rq, dict):
        rq = {}

    # Suggested = recommended by the bot; liked = seeker liked it
    suggested = [mid for mid, q in rq.items() if q.get('suggested', 0) == 1]
    liked_and_suggested = [mid for mid in suggested if rq[mid].get('liked', 0) == 1]

    succeed = 1 if liked_and_suggested else 0
    target_movie_id = liked_and_suggested[0] if liked_and_suggested else (suggested[0] if suggested else None)

    if target_movie_id and target_movie_id in movie_mentions:
        raw_name = movie_mentions[target_movie_id]
        target_title = re.sub(r'\s*\(\d{4}\)\s*$', '', raw_name).strip()
    elif target_movie_id and target_movie_id in movies:
        target_title = movies[target_movie_id]['title']
    else:
        target_title = f'movie#{target_movie_id}' if target_movie_id else 'unknown'

    rec_attempts = len(suggested)

    # Build utterance lines
    lines = []
    for msg in messages:
        text = msg.get('text', '').strip()
        text = replace_movie_refs(text, movie_mentions, movies)
        sender = msg.get('senderWorkerId')
        if sender == initiator_id:
            speaker = 'User'
        elif sender == respondent_id:
            speaker = 'Bot'
        else:
            # Fallback: 0 = initiator, 1 = respondent
            speaker = 'User' if sender == 0 else 'Bot'
        lines.append(f'{speaker}: {text}')

    num_turns = len(lines)

    # Closing lines
    if succeed and target_movie_id:
        lines.append(
            f"System: <END> Session ended successfully. "
            f"The target item '{target_title}' ({target_movie_id}) was accepted by the user."
        )
    else:
        lines.append(
            f"System: <END> Session ended. "
            f"The target item '{target_title}' ({target_movie_id}) was NOT found."
        )

    return {
        'conversation_id': d['conversationId'],
        'target_movie_id': target_movie_id or d['conversationId'],
        'num_turns': num_turns,
        'rec_attempts': rec_attempts,
        'succeed': succeed,
        'lines': lines,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Convert ReDial → SimuConv TXT format')
    parser.add_argument('--data_dir', default='data/redial',
                        help='Folder containing train_data.jsonl, test_data.jsonl, movies_with_mentions.csv')
    parser.add_argument('--output', default='chats/redial/real',
                        help='Output folder for converted TXT files')
    parser.add_argument('--split', choices=['train', 'test', 'both'], default='both',
                        help='Which split(s) to convert')
    parser.add_argument('--limit', type=int, default=None,
                        help='Max number of conversations to convert (None = all)')
    args = parser.parse_args()

    movies_csv = os.path.join(args.data_dir, 'movies_with_mentions.csv')
    movies = load_movies(movies_csv)
    print(f"Loaded {len(movies)} movies from {movies_csv}")

    splits = []
    if args.split in ('train', 'both'):
        splits.append(os.path.join(args.data_dir, 'train_data.jsonl'))
    if args.split in ('test', 'both'):
        splits.append(os.path.join(args.data_dir, 'test_data.jsonl'))

    os.makedirs(args.output, exist_ok=True)

    total = 0
    skipped = 0

    for jsonl_path in splits:
        split_name = os.path.basename(jsonl_path).replace('_data.jsonl', '')
        print(f"\nProcessing {jsonl_path} ...")
        with open(jsonl_path, encoding='utf-8') as f:
            for line in f:
                if args.limit and total >= args.limit:
                    break
                d = json.loads(line)
                result = convert_conversation(d, movies)
                if result is None:
                    skipped += 1
                    continue

                ts = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:18]
                fname = (
                    f"redial-{result['target_movie_id']}"
                    f"-{result['num_turns']}"
                    f"-{result['rec_attempts']}"
                    f"-{result['succeed']}"
                    f"-{ts}.txt"
                )
                out_path = os.path.join(args.output, fname)
                with open(out_path, 'w', encoding='utf-8') as fout:
                    fout.write('\n'.join(result['lines']) + '\n')

                total += 1

    print(f"\nDone. Converted {total} conversations → {args.output}")
    print(f"Skipped {skipped} empty conversations.")


if __name__ == '__main__':
    main()
