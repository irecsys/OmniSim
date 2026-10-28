"""
Evaluator — LLM-based quality scoring for generated conversations.

Evaluates each conversation file on three dimensions:
  - Fluency (1–5)
  - Informativeness (1–5)
  - Coherence (1–5)

Usage:
    python -m utils.evaluator --config configs/imdb/imdb.yaml --folder chats/imdb/free/user_item_pairs/20260310200250
    python -m utils.evaluator --config configs/imdb/imdb.yaml --folder chats/imdb/free/user_item_pairs/20260310200250 --output eval.csv
"""

import os
import re
import json
import argparse
import pandas as pd

from utils.configurator import Config
from utils.utils import get_openai_clients
from utils.metrics import parse_filename
from utils.dialogue_files import is_tmp_dialogue, split_fallback_line


# ---------------------------------------------------------------------------
# Evaluation prompt
# ---------------------------------------------------------------------------

_EVAL_PROMPT = """You are a strict evaluator for Conversational Recommender Systems (CRS).

Your task is to evaluate the quality of a generated conversation between a USER and a RECOMMENDER SYSTEM.

---

Evaluation Dimensions

1. Fluency (1–5)
Evaluate the grammatical correctness and naturalness of the language.
1 = Very unnatural, broken sentences, difficult to understand
2 = Frequent grammar mistakes and awkward phrasing
3 = Understandable but somewhat unnatural
4 = Mostly natural with minor issues
5 = Fully fluent and natural human-like conversation

2. Informativeness (1–5)
Evaluate how much useful recommendation information is exchanged.
Consider whether the conversation contains:
* user preferences
* item attributes
* explanations
* meaningful responses instead of generic replies
1 = No useful recommendation information
2 = Very little useful information
3 = Some information but shallow
4 = Informative and helpful
5 = Very informative with detailed preferences or attributes

3. Coherence (1–5)
Evaluate whether the conversation is logically consistent.
Specifically check:
* whether the recommender adapts to the user's feedback
* whether the user's rejection reasons match the item attributes
1 = Completely inconsistent logic
2 = Mostly inconsistent
3 = Partially coherent
4 = Mostly coherent
5 = Strong logical consistency

---

Evaluation Procedure
Step 1. Read the full conversation.
Step 2. Identify user preferences and rejection reasons.
Step 3. Identify how the system adapts its recommendations.
Step 4. Evaluate Fluency, Informativeness, and Coherence independently.

---

Return ONLY valid JSON in the following format (no markdown, no extra text):

{{
  "fluency": {{
    "score": <1-5>,
    "reason": "<brief explanation>"
  }},
  "informativeness": {{
    "score": <1-5>,
    "reason": "<brief explanation>"
  }},
  "coherence": {{
    "score": <1-5>,
    "reason": "<brief explanation>"
  }},
  "overall_score": <average of the three scores, one decimal>,
  "overall_quality": "<very_bad | bad | average | good | excellent>"
}}

---

Conversation to evaluate:

{dialogue}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_dialogue(file_path: str) -> str:
    with open(file_path, encoding='utf-8') as f:
        return f.read().strip()


def quality_label(score: float) -> str:
    if score >= 4.5:
        return "excellent"
    elif score >= 3.5:
        return "good"
    elif score >= 2.5:
        return "average"
    elif score >= 1.5:
        return "bad"
    return "very_bad"


def evaluate_dialogue(dialogue: str, client_chat, model: str) -> dict:
    prompt = _EVAL_PROMPT.format(dialogue=dialogue)
    completion = client_chat.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a strict CRS evaluator. Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=600,
    )
    raw = completion.choices[0].message.content.strip()

    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw)
    raw = re.sub(r'\s*```$', '', raw)

    result = json.loads(raw)

    # Ensure overall_score and overall_quality are consistent
    scores = [result['fluency']['score'], result['informativeness']['score'], result['coherence']['score']]
    result['overall_score'] = round(sum(scores) / 3, 1)
    result['overall_quality'] = quality_label(result['overall_score'])
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate_folder(config: Config, folder_path: str, output_csv: str | None = None):
    """LLM-based evaluation of the conversation .txt files in folder_path (tmp files skipped), with the chat
    client and chat_model from config: fluency, informativeness and coherence on a 1-5 scale, each with a
    reason, plus overall_score and overall_quality. Prints a summary and returns a DataFrame with one row per
    conversation, saved to output_csv if given; None if the folder does not exist or nothing was evaluated."""
    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    client_chat, _ = get_openai_clients(config)
    model = config['chat_model']

    rows = []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt') and not is_tmp_dialogue(f)])

    print(f"\nEvaluating {len(files)} conversations in: {folder_path}")
    print(f"Model: {model}\n")

    for fname in files:
        parsed = parse_filename(fname)
        if parsed is None:
            continue

        fpath = os.path.join(folder_path, fname)
        # the <FALLBACK> line is not shown to the evaluator; its count goes into the 'fallbacks' column
        dialogue, n_fallbacks = split_fallback_line(read_dialogue(fpath))
        dialogue = dialogue.strip()

        try:
            result = evaluate_dialogue(dialogue, client_chat, model)
            row = {
                'file': fname,
                'user_id': parsed.get('user_id', ''),
                'item_id': parsed['item_id'],
                'succeed': parsed['succeed'],
                'num_turns': parsed['num_turns'],
                'rec_attempts': parsed['rec_attempts'],
                'fallbacks': n_fallbacks,
                'fluency': result['fluency']['score'],
                'fluency_reason': result['fluency']['reason'],
                'informativeness': result['informativeness']['score'],
                'informativeness_reason': result['informativeness']['reason'],
                'coherence': result['coherence']['score'],
                'coherence_reason': result['coherence']['reason'],
                'overall_score': result['overall_score'],
                'overall_quality': result['overall_quality'],
            }
            rows.append(row)
            print(f"  [{fname}]  fluency={row['fluency']}  info={row['informativeness']}  coherence={row['coherence']}  overall={row['overall_score']} ({row['overall_quality']})")
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}")

    if not rows:
        print("No results.")
        return

    df = pd.DataFrame(rows)

    print("\n=== Evaluation Summary ===")
    print(f"Total evaluated   : {len(df)}")
    print(f"Avg fluency       : {df['fluency'].mean():.2f}")
    print(f"Avg informativeness: {df['informativeness'].mean():.2f}")
    print(f"Avg coherence     : {df['coherence'].mean():.2f}")
    print(f"Avg overall score : {df['overall_score'].mean():.2f}")
    print("\nQuality distribution:")
    print(df['overall_quality'].value_counts().to_string())

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\nResults saved to: {output_csv}")

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM-based CRS conversation evaluator')
    parser.add_argument('--config', type=str, default='configs/imdb/imdb.yaml')
    parser.add_argument('--folder', type=str, required=True,
                        help='Path to run folder (e.g. chats/imdb/free/user_item_pairs/20260310200250)')
    parser.add_argument('--output', type=str, default=None,
                        help='Optional: save results as CSV')
    args = parser.parse_args()

    config = Config(config_file_list=args.config.strip().split())
    evaluate_folder(config, args.folder, args.output)
