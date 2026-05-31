"""
evaluation/judge.py — LLM-as-Judge evaluator for SimuConv simulated conversations.

Evaluates the simulated USER's behaviour across three dimensions:
  1. Language Fluency      (Grammar, Fluency, Naturalness, Tone)
  2. Conversational Quality (Appropriateness, Proactivity, Multi-turn consistency)
  3. Content Quality       (Factuality, Coverage, Coherence, Relevance)

Each dimension is scored 1–5 with a Chain-of-Thought rationale before the score.

Usage:
    python evaluation/judge.py \\
        --folder chats/imdb/adaptive/user_item_pairs/20260320192656

    python evaluation/judge.py \\
        --folder chats/imdb/adaptive/user_item_pairs/20260320192656 \\
        --output results/judge_adaptive.csv \\
        --limit 20

    python evaluation/judge.py \\
        --folder chats/imdb/free/user_item_pairs/20260319003702 \\
        --model gpt-4o \\
        --output results/judge_free.csv
"""

import os
import re
import json
import argparse
import requests
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# ThetaEdgeCloud thin client (mirrors utils/utils.py without the circular import)
# ---------------------------------------------------------------------------

class _ThetaChoice:
    def __init__(self, content):
        self.message = type("msg", (), {"content": content})()

class _ThetaCompletion:
    def __init__(self, content):
        self.choices = [_ThetaChoice(content)]

class _ThetaCompletions:
    def __init__(self, api_key, model_slug):
        self._key = api_key
        self._slug = model_slug

    def create(self, model, messages, temperature=0.0, max_tokens=900, **kwargs):
        url = f"https://ondemand.thetaedgecloud.com/infer_request/{self._slug}/completions"
        payload = {"input": {"messages": messages, "max_tokens": max_tokens,
                              "temperature": temperature, "stream": False}}
        resp = requests.post(url,
                             headers={"Content-Type": "application/json",
                                      "Authorization": f"Bearer {self._key}"},
                             json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("body", {})
                       .get("infer_requests", [{}])[0]
                       .get("output", {})
                       .get("message", ""))
        return _ThetaCompletion(content)

class _ThetaChat:
    def __init__(self, api_key, model_slug):
        self.completions = _ThetaCompletions(api_key, model_slug)

class ThetaEdgeCloudClient:
    def __init__(self, api_key, model_slug):
        self.chat = _ThetaChat(api_key, model_slug)

_THETA_SLUG_MAP = {
    "llama3.1-70b": "llama_3_1_70b",
    "llama3.1-8b":  "llama_3_1_8b",
    "meta-llama/meta-llama-3.1-70b-instruct": "llama_3_1_70b",
    "meta-llama/meta-llama-3.1-8b-instruct":  "llama_3_1_8b",
    "meta-llama/meta-llama-3.1-70b": "llama_3_1_70b",
}

def _build_client(provider: str, model: str):
    if provider == "thetaedgecloud":
        theta_key = os.getenv("THETA_KEY", "")
        slug = _THETA_SLUG_MAP.get(model.lower(),
               model.lower().replace("/", "_").replace("-", "_").replace(".", "_"))
        return ThetaEdgeCloudClient(theta_key, slug)
    elif provider == "azure":
        from openai import AzureOpenAI
        return AzureOpenAI(
            api_key=os.getenv("OPENAI_KEY", ""),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
        )
    elif provider == "azure_foundry":
        return OpenAI(
            base_url=os.getenv("AZURE_FOUNDRY_ENDPOINT", "https://conv.services.ai.azure.com/api/projects/conv/openai/v1"),
            api_key=os.getenv("AZURE_KEY", ""),
        )
    elif provider == "deepseek":
        return OpenAI(
            api_key=os.getenv("DEEPSEEK_KEY", ""),
            base_url="https://api.deepseek.com",
        )
    elif provider == "gemini":
        return OpenAI(
            api_key=os.getenv("GEMINI_KEY", ""),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    else:  # openai or openai-compatible
        kwargs = {"api_key": os.getenv("OPENAI_KEY", "")}
        base = os.getenv("OPENAI_API_BASE")
        if base:
            kwargs["base_url"] = base
        return OpenAI(**kwargs)


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an expert in Human-Computer Interaction (HCI), User Simulation and Linguistics \
specializing in generating and evaluating training dialogs for Conversational Recommender Systems (CRS). \
Your task is to act as an objective judge to evaluate how realistically and logically a simulated user \
agent behaves when interacting with a search and recommendation system.

Below is the full transcript of a conversation between a Recommender System (Agent) and a Simulated User (User).

Note: You must judge the simulated user's behavior strictly based on their conversational logic, \
naturalness, and interaction quality throughout the transcript.

<Dialogue_Transcript>
{dialogue_transcript}
</Dialogue_Transcript>

### Evaluation Criteria:
Please evaluate the Simulated User's performance across the following three dimensions \
on a scale of 1 to 5 (1 = Very Poor, 5 = Excellent):

1. Language Fluency (Grammar, Fluency, Naturalness, Tone) (1-5)
- Does the user's language sound natural, fluent, and grammatically correct?
- Does the tone resemble a real human chatting with a shopping assistant, rather than a machine?
- Score 1 if it sounds like a rigid robot, uses highly repetitive templates, or outputs unnatural formats. \
Score 5 if it uses natural human-like expressions, varied vocabulary, and appropriate conversational tone.

2. Conversational Quality (Appropriateness of responses, Proactivity, Multi-turn consistency) (1-5)
- Does the user react appropriately and logically to the system's actions within the interaction rounds?
- Does the user maintain multi-turn consistency (e.g., sticking to their initial preferences \
without contradicting themselves later in the dialogue)?
- Score 1 if the user ignores the system's questions, frequently changes their mind randomly, \
or gets stuck in an unnatural loop. Score 5 if the user interacts smoothly, maintains a consistent goal, \
and provides highly appropriate feedback (e.g., accepting or rejecting) to recommendations.

3. Content Quality (Factuality, Coverage, Coherence, Relevance) (1-5)
- Is the content of the user's responses relevant to the ongoing context?
- When rejecting a recommendation or clarifying a need, are the user's explanations logical, coherent, \
and grounded in realistic item attributes (factuality)?
- Score 1 if the user's feedback is contradictory, hallucinates conflicting constraints, \
or rejects items without any logical reason. Score 5 if the user clearly and coherently articulates \
their constraints and provides informative, relevant explanations for their rejections.

### Output Format Instructions:
To ensure an objective evaluation, you MUST write a brief justification (rationale / Chain-of-Thought) \
for EACH dimension BEFORE giving the final numerical score.
Output your evaluation STRICTLY in the following JSON format (no markdown, no extra text):

{{
  "Language_Fluency": {{
    "rationale": "<your step-by-step reasoning here>",
    "score": <int 1-5>
  }},
  "Conversational_Quality": {{
    "rationale": "<your step-by-step reasoning here>",
    "score": <int 1-5>
  }},
  "Content_Quality": {{
    "rationale": "<your step-by-step reasoning here>",
    "score": <int 1-5>
  }},
  "Overall_Assessment": "<1-2 sentences summarizing the realism of this simulated user>"
}}
"""


# ---------------------------------------------------------------------------
# Filename parsing  (pattern: {user_id}-{item_id}-{turns}-{attempts}-{succeed}-{ts}.txt
#                         or: {item_id}-{turns}-{attempts}-{succeed}-{ts}.txt)
# ---------------------------------------------------------------------------

def _parse_filename(fname: str) -> dict | None:
    name = fname.replace('.txt', '')
    parts = name.split('-')
    try:
        # Try 6-part: user_id, item_id, turns, attempts, succeed, ts
        if len(parts) == 6:
            return {
                'user_id': parts[0],
                'item_id': parts[1],
                'num_turns': int(parts[2]),
                'rec_attempts': int(parts[3]),
                'succeed': int(parts[4]),
            }
        # Try 5-part: item_id, turns, attempts, succeed, ts
        if len(parts) == 5:
            return {
                'user_id': '',
                'item_id': parts[0],
                'num_turns': int(parts[1]),
                'rec_attempts': int(parts[2]),
                'succeed': int(parts[3]),
            }
    except (ValueError, IndexError):
        pass
    return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def _normalize_keys(d: dict) -> dict:
    """Normalize top-level keys to expected format regardless of model casing."""
    KEY_MAP = {
        "language_fluency":      "Language_Fluency",
        "conversational_quality": "Conversational_Quality",
        "content_quality":       "Content_Quality",
        "overall_assessment":    "overall_assessment",
    }
    normalized = {}
    for k, v in d.items():
        canonical = KEY_MAP.get(k.lower().replace(" ", "_"), k)
        normalized[canonical] = v
    return normalized


def _parse_json(raw: str) -> dict:
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)
    result = json.loads(raw)
    return _normalize_keys(result)


def judge_dialogue(dialogue: str, client: OpenAI, model: str) -> dict:
    prompt = _JUDGE_PROMPT.format(dialogue_transcript=dialogue)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict HCI and CRS evaluator. "
                    "Always return valid JSON exactly matching the requested format."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=900,
    )
    raw = completion.choices[0].message.content.strip()
    result = _parse_json(raw)
    scores = [
        result["Language_Fluency"]["score"],
        result["Conversational_Quality"]["score"],
        result["Content_Quality"]["score"],
    ]
    result["overall_score"] = round(sum(scores) / len(scores), 2)
    return result


def _quality_label(score: float) -> str:
    if score >= 4.5:
        return "excellent"
    elif score >= 3.5:
        return "good"
    elif score >= 2.5:
        return "average"
    elif score >= 1.5:
        return "bad"
    return "very_bad"


# ---------------------------------------------------------------------------
# Folder-level evaluation
# ---------------------------------------------------------------------------

def evaluate_folder(folder_path: str, client: OpenAI, model: str,
                    output_csv: str | None = None, limit: int = 0,
                    eval_subdir: str = "llama", workers: int = 8) -> pd.DataFrame | None:
    """Evaluate all conversations in a folder (concurrent requests).

    Per-conversation results are saved to:
        <folder_path>/<eval_subdir>/eval_<filename>.csv

    Aggregate results (all rows) are saved to output_csv if provided.
    Already-evaluated files are skipped (resume support).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading

    if not os.path.isdir(folder_path):
        print(f"Folder not found: {folder_path}")
        return None

    eval_dir = os.path.join(folder_path, eval_subdir)
    os.makedirs(eval_dir, exist_ok=True)

    txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])
    if limit and limit > 0:
        txt_files = txt_files[:limit]

    print(f"\n{'='*60}")
    print(f"LLM-as-Judge Evaluation")
    print(f"Folder    : {folder_path}")
    print(f"Eval dir  : {eval_dir}")
    print(f"Model     : {model}")
    print(f"Files     : {len(txt_files)}")
    print(f"Workers   : {workers}")
    print(f"{'='*60}\n")

    lock = threading.Lock()
    rows = []
    skipped = 0

    # Pre-load already completed files
    todo_files = []
    for fname in txt_files:
        parsed = _parse_filename(fname)
        if parsed is None:
            continue
        eval_fname = f"eval_{fname.replace('.txt', '.csv')}"
        eval_path  = os.path.join(eval_dir, eval_fname)
        if os.path.exists(eval_path):
            try:
                existing = pd.read_csv(eval_path)
                rows.append(existing.iloc[0].to_dict())
                skipped += 1
                continue
            except Exception:
                pass
        todo_files.append((fname, parsed))

    print(f"  Loaded from cache: {skipped}  |  To evaluate: {len(todo_files)}\n")

    def _process(fname, parsed):
        fpath = os.path.join(folder_path, fname)
        eval_fname = f"eval_{fname.replace('.txt', '.csv')}"
        eval_path  = os.path.join(eval_dir, eval_fname)
        with open(fpath, encoding='utf-8') as f:
            dialogue = f.read().strip()
        try:
            result = judge_dialogue(dialogue, client, model)
            lf  = result["Language_Fluency"]["score"]
            cq  = result["Conversational_Quality"]["score"]
            cnq = result["Content_Quality"]["score"]
            overall = result["overall_score"]
            row = {
                "file":                             fname,
                "user_id":                          parsed.get("user_id", ""),
                "item_id":                          parsed["item_id"],
                "succeed":                          parsed["succeed"],
                "num_turns":                        parsed["num_turns"],
                "rec_attempts":                     parsed["rec_attempts"],
                "language_fluency":                 lf,
                "language_fluency_rationale":       result["Language_Fluency"]["rationale"],
                "conversational_quality":           cq,
                "conversational_quality_rationale": result["Conversational_Quality"]["rationale"],
                "content_quality":                  cnq,
                "content_quality_rationale":        result["Content_Quality"]["rationale"],
                "overall_score":                    overall,
                "overall_quality":                  _quality_label(overall),
                "overall_assessment":               result.get("Overall_Assessment", ""),
            }
            pd.DataFrame([row]).to_csv(eval_path, index=False)
            print(f"  {fname}\n    Fluency={lf}  Conv.Quality={cq}  Content={cnq}  Overall={overall} ({row['overall_quality']})\n")
            return row
        except Exception as e:
            print(f"  [ERROR] {fname}: {e}\n")
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_process, fname, parsed): fname for fname, parsed in todo_files}
        for future in as_completed(futures):
            row = future.result()
            if row:
                with lock:
                    rows.append(row)

    if not rows:
        print("No results.")
        return None

    df = pd.DataFrame(rows)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"  Evaluated (new)           : {len(df) - skipped}")
    print(f"  Loaded from cache         : {skipped}")
    print(f"  Total                     : {len(df)}")
    print(f"  Avg Language Fluency      : {df['language_fluency'].mean():.2f}")
    print(f"  Avg Conversational Quality: {df['conversational_quality'].mean():.2f}")
    print(f"  Avg Content Quality       : {df['content_quality'].mean():.2f}")
    print(f"  Avg Overall Score         : {df['overall_score'].mean():.2f}")
    print(f"\nQuality distribution:")
    print(df["overall_quality"].value_counts().to_string())

    if output_csv:
        if os.path.dirname(output_csv):
            os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nAggregate results saved to: {output_csv}")
    print(f"Per-file results saved to  : {eval_dir}/eval_*.csv")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LLM-as-Judge evaluator for SimuConv simulated user conversations"
    )
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to conversation folder (e.g. chats/imdb/adaptive/user_item_pairs/20260320192656)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: save results to CSV (e.g. results/judge_adaptive.csv)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit evaluation to first N conversations (0 = all)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: reads JUDGE_MODEL or CHAT_MODEL env var, fallback: gpt-4o)",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="API provider: openai | azure | thetaedgecloud (default: reads OPENAI_PROVIDER env var)",
    )
    args = parser.parse_args()

    model    = args.model    or os.getenv("JUDGE_MODEL") or os.getenv("CHAT_MODEL", "gpt-4o")
    provider = args.provider or os.getenv("OPENAI_PROVIDER", "openai")

    client = _build_client(provider, model)
    evaluate_folder(args.folder, client, model, args.output, args.limit)
