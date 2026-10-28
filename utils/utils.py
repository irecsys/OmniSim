# @Time   : January, 2023
# @Author : Dr. Yong Zheng
# @Email  : yzheng66@illinoistech.edu

"""
utils.utils
################################################
"""

import datetime
import os
import json
import ast
import re
import random
import numpy as np
import pandas as pd
import requests as _requests
import yaml
from openai import AzureOpenAI, OpenAI
from utils import llm_calls
from utils.configurator import Config
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Prompt template loader
# ---------------------------------------------------------------------------
_PROMPT_CACHE: dict = {}

def load_prompts(config: Config) -> dict:
    """Load prompt templates from YAML. Cached per file path."""
    prompt_file = config.get('prompts_file', 'configs/prompts/default.yaml')
    if prompt_file not in _PROMPT_CACHE:
        with open(prompt_file, encoding='utf-8') as f:
            _PROMPT_CACHE[prompt_file] = yaml.safe_load(f)
    return _PROMPT_CACHE[prompt_file]

def P(config: Config, key: str, **kwargs) -> str:
    """Get a prompt template by key and format with kwargs."""
    tmpl = load_prompts(config).get(key, '')
    if not tmpl:
        raise KeyError(f"Prompt key '{key}' not found in prompts file.")
    if kwargs:
        return tmpl.format(**kwargs)
    return tmpl


# ---------------------------------------------------------------------------
# ThetaEdgeCloud wrapper — mimics the OpenAI client interface
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

    def create(self, model, messages, temperature=0.7, max_tokens=500, **kwargs):
        import time as _time
        url = f"https://ondemand.thetaedgecloud.com/infer_request/{self._slug}/completions"
        payload = {"input": {"messages": messages, "max_tokens": max_tokens,
                              "temperature": temperature, "stream": False}}
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"}
        for attempt in range(6):
            resp = _requests.post(url, headers=headers, json=payload, timeout=120)
            if resp.status_code not in (409, 429, 503):
                break
            llm_calls.note_retry(resp.status_code)   # retries inside one logical call (counted apart)
            wait = 2 ** attempt  # 1, 2, 4, 8, 16, 32 seconds
            _time.sleep(wait)
        resp.raise_for_status()
        data = resp.json()
        # ThetaEdgeCloud returns {"body": {"infer_requests": [{"output": {"message": "..."}}]}}
        content = (data.get("body", {})
                       .get("infer_requests", [{}])[0]
                       .get("output", {})
                       .get("message", ""))
        return _ThetaCompletion(content)

class _ThetaChat:
    def __init__(self, api_key, model_slug):
        self.completions = _ThetaCompletions(api_key, model_slug)

class ThetaEdgeCloudClient:
    """Wrapper that exposes the same interface as the OpenAI client for chat."""
    def __init__(self, api_key, model_slug):
        self.chat = _ThetaChat(api_key, model_slug)

    # Embeddings not supported — return None so callers degrade gracefully
    class _NoEmbeddings:
        class _Inner:
            def create(self, **kwargs):
                return type("r", (), {"data": [type("d", (), {"embedding": None})()]})()
        def __init__(self):
            self.embeddings = self._Inner()
    embeddings = _NoEmbeddings().embeddings


class SentenceTransformerEmbeddingsClient:
    """Local embeddings client using sentence-transformers. Mimics client_embeddings interface."""
    def __init__(self, model_name: str):
        import os
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "embeddings_provider 'sentence-transformers' needs the optional package sentence-transformers, "
                "which requirements.txt does not install: pip install \"sentence-transformers>=2.2.0,<7\""
            ) from e
        os.environ.setdefault('HF_HUB_OFFLINE', '1')
        os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
        self._model = SentenceTransformer(model_name)
        self.embeddings = self

    def create(self, model=None, input=None, **kwargs):
        texts = [input] if isinstance(input, str) else input
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        data = [type("d", (), {"embedding": v})() for v in vecs]
        return type("r", (), {"data": data})()


def _truncate_msg(text: str, max_chars: int = 250) -> str:
    """Truncate a message to max_chars, cutting at the last sentence boundary if possible."""
    if not text or len(text) <= max_chars:
        return text
    # Try to cut at last sentence-ending punctuation within limit
    cut = text[:max_chars]
    for punct in reversed(range(len(cut))):
        if cut[punct] in '.!?':
            return cut[:punct + 1]
    # No sentence boundary — hard truncate at last word boundary
    last_space = cut.rfind(' ')
    return (cut[:last_space] + '…') if last_space > 0 else cut


def _ensure_complete(text: str) -> str:
    """Return text only if it ends with sentence-final punctuation.
    Otherwise, cut at the last sentence boundary found.
    If no sentence boundary exists, return empty string (caller should discard)."""
    if not text:
        return ""
    # Strip trailing ellipsis (... or unicode …) — they indicate incomplete thoughts
    text = text.rstrip()
    while text.endswith('...'):
        text = text[:-3].rstrip()
    while text.endswith('…'):
        text = text[:-1].rstrip()
    if not text:
        return ""
    if text[-1] in '.!?':
        return text
    # Find last sentence boundary
    for i in reversed(range(len(text))):
        if text[i] in '.!?':
            return text[:i + 1]
    return ""  # No complete sentence found — discard


def cosine_sim(a, b) -> float:
    """Return raw cosine similarity in [-1, 1].

    S_cos(x, y) = cos(e_x, e_y)
    Use ``normalize_cosine()`` to map the result to [0, 1] before combining
    with other scores.
    """
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_cosine(s: float) -> float:
    """Map a raw cosine similarity in [-1, 1] to [0, 1].

    S̃ = (s + 1) / 2
    The retrieval backend's kNN scores are already in this range, so this function
    is only needed when calling ``cosine_sim()`` directly (e.g. for user-profile
    comparisons in the hybrid score).
    """
    return (s + 1.0) / 2.0


def normalize_bm25(scores: dict) -> dict:
    """Min-max normalise a dict of BM25 scores to [0, 1].

    S̃_BM25 = (s − s_min) / (s_max − s_min)
    If all scores are equal the function returns 0 for every item.
    """
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def init_seed(seed):
    """ init random seed for random functions in numpy, etc
        Args:
            seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)


def get_local_time():
    """ Get current time
        Returns:
            str: current time
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def ensure_dir(dir_path):
    """ Make sure the directory exists, if it does not exist, create it
        Args:
            dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def _wrap_chat_client(client, default_timeout: float = 60.0):
    """Inject default request timeout and replace None content with empty string.

    Why: GPT-4o-mini via Azure occasionally returns content=None and the SDK has no
    built-in request timeout, causing hangs + downstream .strip() AttributeErrors.
    """
    original_create = client.chat.completions.create

    def safe_create(*args, **kwargs):
        kwargs.setdefault("timeout", default_timeout)
        resp = original_create(*args, **kwargs)
        for ch in getattr(resp, "choices", []) or []:
            msg = getattr(ch, "message", None)
            if msg is not None and getattr(msg, "content", None) is None:
                msg.content = ""
        return resp

    client.chat.completions.create = safe_create
    return client


def _build_chat_client(provider: str, config, api_key: str):
    if provider == "azure":
        return _wrap_chat_client(AzureOpenAI(
            api_version=config["chat_api_version"],
            azure_endpoint=config["chat_endpoint"],
            api_key=api_key,
        ))
    elif provider == "azure_foundry":
        azure_key = os.getenv("AZURE_KEY")
        return _wrap_chat_client(OpenAI(
            base_url=config["chat_endpoint"],
            api_key=azure_key,
        ))
    elif provider == "openai":
        return _wrap_chat_client(OpenAI(api_key=api_key))
    elif provider == "thetaedgecloud":
        theta_key = os.getenv("THETA_KEY")
        model_slug_map = {
            "llama3.1-70b": "llama_3_1_70b",
            "llama3.1-8b":  "llama_3_1_8b",
            "meta-llama/meta-llama-3.1-70b-instruct": "llama_3_1_70b",
            "meta-llama/meta-llama-3.1-8b-instruct":  "llama_3_1_8b",
        }
        raw_model = str(config.get("chat_model", "llama_3_1_70b")).lower()
        model_slug = model_slug_map.get(raw_model, raw_model.replace("/", "_").replace("-", "_").replace(".", "_"))
        return ThetaEdgeCloudClient(theta_key, model_slug)
    else:
        raise ValueError(f"Unknown chat provider: {provider}")


def _build_embeddings_client(provider: str, config, api_key: str):
    if provider == "azure":
        return AzureOpenAI(
            api_version=config["embeddings_api_version"],
            azure_endpoint=config["embeddings_endpoint"],
            api_key=api_key,
        )
    elif provider == "openai":
        return OpenAI(api_key=api_key)
    elif provider == "thetaedgecloud":
        theta_key = os.getenv("THETA_KEY")
        return ThetaEdgeCloudClient(theta_key, "")  # returns None gracefully
    elif provider == "sentence-transformers":
        model_name = config.get("embeddings_model", "all-MiniLM-L6-v2")
        return SentenceTransformerEmbeddingsClient(model_name)
    else:
        raise ValueError(f"Unknown embeddings provider: {provider}")


def resolve_embeddings_provider(config) -> str:
    """embeddings_provider, defaulting to the chat provider (openai_provider, default "openai").
    Shared by the query client below and the local index builder (utils/retrieval/local_index.py)."""
    chat_provider = str(config.get("openai_provider", "openai")).lower()
    return str(config.get("embeddings_provider") or chat_provider).lower()


def get_openai_clients(config: Config):
    """
    Build the chat client and the embeddings client from config; returns (client_chat, client_embeddings).
    Chat (openai_provider): openai (OPENAI_KEY), azure (AZURE_KEY; chat_endpoint, chat_api_version),
    azure_foundry (AZURE_KEY; chat_endpoint is the base URL) or thetaedgecloud (THETA_KEY; chat_model is
    turned into the Theta model slug). Embeddings (embeddings_provider, defaults to openai_provider):
    openai (OPENAI_KEY), azure (AZURE_KEY; embeddings_endpoint, embeddings_api_version),
    sentence-transformers (local model embeddings_model, no key) or thetaedgecloud (returns no vectors).
    Keys are read from the environment or .env. An unknown provider raises ValueError.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_KEY")
    azure_key = os.getenv("AZURE_KEY")

    chat_provider = str(config.get("openai_provider", "openai")).lower()
    # embeddings_provider defaults to chat provider if not set
    embeddings_provider = resolve_embeddings_provider(config)

    chat_key = azure_key if chat_provider == "azure" else api_key
    emb_key = azure_key if embeddings_provider == "azure" else api_key

    client_chat = _build_chat_client(chat_provider, config, chat_key)
    client_embeddings = _build_embeddings_client(embeddings_provider, config, emb_key)

    return client_chat, client_embeddings

# rephrase item attributes into natural language
def rephrase_attribute_to_natural_language(config: Config, client_chat, attribute, value):
    """
    Rephrase the attribute value into human-like natural language.
    Uses the attribute's semantic meaning from item_attributes config to guide generation.
    """
    if not value or str(value).lower() == "nan":
        return "I have no preference."

    # Look up the human-readable meaning of this attribute from config
    item_attributes = config.get('item_attributes', {})
    attribute_meaning = item_attributes.get(attribute, attribute)

    if attribute in ("features", "details", "overview"):
        prompt = P(config, 'rephrase_features_prompt', role_user=config['role_user'], value=value)
    else:
        prompt = P(config, 'rephrase_attribute_prompt',
                   role_user=config['role_user'],
                   attribute=attribute,
                   attribute_meaning=attribute_meaning,
                   value=value)

    completion = llm_calls.chat(client_chat, 'rephrase_attribute',
        model=config['chat_model'],
        messages=[
            {"role": "system", "content": P(config, 'rephrase_features_system')},
            {"role": "user", "content": prompt}
        ],
        temperature=config.get('temp_rephrase', 0.3)
    )

    return completion.choices[0].message.content.strip().strip('"').strip("'")



# extract attribute value from knowledge base
def extract_attribute_value(config: Config, client_chat, attribute, item_info, details):
    """
    Try to get attribute value from structured fields, 
    otherwise extract from details using GPT.
    """
    # 1. Structured lookup
    value = None
    if item_info is not None and hasattr(item_info, "empty") and not item_info.empty and attribute in item_info.columns:
        value = item_info.iloc[0].get(attribute)
    elif hasattr(item_info, "get"):
        value = item_info.get(attribute)
    if value is not None and str(value).strip() not in ["", "nan", "NaN", "None"]:
        return str(value).strip()

    # 2. Try to extract from details (if present)
    if details is not None:
        prompt = P(config, 'extract_attribute_prompt', attribute=attribute, details=details)

        response = llm_calls.chat(client_chat, 'extract_attribute_value',
            model=config['chat_model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            timeout=60,
        )
        raw = (response.choices[0].message.content or "").strip()
        if not raw:
            return None

        # Step 1: Ensure it's JSON-shaped, fallback otherwise
        if not raw.startswith("{"):
            # Try to extract JSON substring if embedded in chatty text
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                raw = match.group(0)
            else:
                # clearly a chatty answer, discard
                return None

        try:
            parsed = json.loads(raw)
            val = parsed.get("value")
            if val and str(val).strip().lower() not in ["none", "null", "not mentioned", ""]:
                return str(val).strip()
        except Exception:
            # If JSON parsing fails → fallback only if looks like a plain value
            cleaned = raw.split("\n")[0].strip()
            if (
                cleaned.lower() not in ["none", "null", "not mentioned", ""]
                and not cleaned.lower().startswith("sure")
                and not cleaned.lower().startswith("please provide")
            ):
                return cleaned

    return None


# extract appropriate attributes for asking
def extract_attributes_for_asking(config: Config, client_chat, item_type: str, target_category: str, attribute_by_openai=True):
    """
    Attributes the bot may ask about, as a list of attribute keys ("details" is never included).
    Static mode (attribute_by_openai=False): the keys of the configured item_attributes; no LLM call.
    Adaptive mode: attributes proposed by the LLM (default_attributes if it proposes none) plus the
    configured item_attributes, without duplicates; a second LLM call keeps those that make sense for
    target_category (if it keeps none, the combined list is returned). Configured attributes are
    returned as their column keys, so their values can be looked up in the item record.
    """
    def _is_details_attr(key_or_label: str) -> bool:
        return str(key_or_label).strip().lower().replace("_", " ") in {"detail", "details"}

    def _norm_attr(attr: str) -> str:
        return str(attr).strip().lower().replace("_", " ")

    predefined_attrs = config.get('item_attributes', {}) or {}
    predefined_entries = [
        {"key": key, "display": label, "source": "yaml"}
        for key, label in predefined_attrs.items()
        if not _is_details_attr(key) and not _is_details_attr(label)
    ]

    # Static mode: ask only about configured YAML item_attributes, using column keys.
    if not attribute_by_openai:
        return [entry["key"] for entry in predefined_entries]

    candidate_attributes = []
    if attribute_by_openai:
        # Step 1: get candidate attributes from OpenAI
        attr_prompt = P(config, 'extract_candidate_attrs_prompt',
                        role_bot=config['role_bot'], item_type=item_type, target_category=target_category)

        response = llm_calls.chat(client_chat, 'extract_candidate_attributes',
            model=config['chat_model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": attr_prompt}
            ],
            temperature=0.5
        )

        raw_output = response.choices[0].message.content
        candidate_attributes = parse_attribute_list_from_json(raw_output)

        if not candidate_attributes:  # fallback
            candidate_attributes = config['default_attributes']

    # Adaptive mode: combine LLM-generated attributes with YAML item_attributes.
    # YAML attributes are kept as keys so downstream structured lookup works.
    entries = []
    seen = set()
    yaml_norms = {_norm_attr(entry["key"]) for entry in predefined_entries}
    yaml_norms.update(_norm_attr(entry["display"]) for entry in predefined_entries)

    for attr in candidate_attributes:
        if not isinstance(attr, str) or _is_details_attr(attr):
            continue
        norm_attr = _norm_attr(attr)
        if norm_attr in seen or norm_attr in yaml_norms:
            continue
        seen.add(norm_attr)
        entries.append({"key": attr.strip(), "display": attr.strip(), "source": "llm"})

    for entry in predefined_entries:
        norm_attr = _norm_attr(entry["key"])
        if norm_attr not in seen:
            seen.add(norm_attr)
            entries.append(entry)

    final_attributes = [entry["key"] for entry in entries]
    if entries:
        display_to_key = {_norm_attr(entry["display"]): entry["key"] for entry in entries}
        key_norms = {_norm_attr(entry["key"]): entry["key"] for entry in entries}
        displays = [entry["display"] for entry in entries]

        # Step 3: filter out attributes that don't make sense for this category
        filter_prompt = P(config, 'filter_attrs_prompt',
                          role_bot=config['role_bot'], item_type=item_type,
                          target_category=target_category, final_attributes=displays)
        response = llm_calls.chat(client_chat, 'filter_attributes',
            model=config['chat_model'],
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": filter_prompt}
            ],
            temperature=0
        )
        raw_filtered = response.choices[0].message.content
        filtered_attributes = parse_attribute_list_from_json(raw_filtered)

        mapped = []
        mapped_seen = set()
        for attr in filtered_attributes:
            norm_attr = _norm_attr(attr)
            key = display_to_key.get(norm_attr) or key_norms.get(norm_attr)
            if key and not _is_details_attr(key) and _norm_attr(key) not in mapped_seen:
                mapped.append(key)
                mapped_seen.add(_norm_attr(key))
        if mapped:
            final_attributes = mapped

    return final_attributes

# extract key information from user responses in order to perform information retrieval
def extract_dynamic_description_from_chat(config: Config, client_chat, chat_history):
    """
    Summarize the user's desired items into a single natural-language description.
    No predefined attributes. GPT dynamically captures key descriptors.
    """
    prompt = P(config, 'extract_description_prompt',
               role_bot=config['role_bot'], chat_history=chat_history)
    
    resp = llm_calls.chat(client_chat, 'extract_dynamic_description',
        model=config['chat_model'],
        messages=[{"role": "user", "content": prompt}]
    )
    # Defensive handling
    if not resp or not resp.choices or not resp.choices[0].message or resp.choices[0].message.content is None:
        return ""  # or some fallback string

    return resp.choices[0].message.content.strip()

# extract the item or item type from the item title
def extract_item_from_title(config: Config, client_chat, title: str, item_category: str) -> str:
    """
    Use LLM to infer a descriptive item type (e.g. 'dark comedy film', 'gritty action thriller')
    from the title and category. Returns a 2-5 word phrase — no genre list appended.
    """
    full_title = f"{title} ({item_category})" if item_category else title
    prompt = P(config, 'extract_item_type_prompt', title=full_title)
    response = llm_calls.chat(client_chat, 'extract_item_from_title',
        model=config['chat_model'],
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    # Ensure message exists
    message_obj = response.choices[0].message
    if message_obj is None or message_obj.content is None:
        return f"something in {item_category}"  # fallback if API returned nothing

    item_type = message_obj.content.strip()
    return item_type

def parse_attribute_list_from_json(raw_content):
    """
    Try to parse GPT output into a Python list of strings.
    Handles JSON, Python literal lists, and fenced code blocks.
    """
    if not raw_content:
        return []

    text = raw_content.strip()

    # 1. Remove markdown fences if present
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)  # remove leading ```json or ```python
        text = re.sub(r"\n?```$", "", text)           # remove trailing ```
        text = text.strip()

    # 2. Try JSON parsing
    try:
        return json.loads(text)
    except Exception:
        pass

    # 3. Try Python literal parsing
    try:
        return ast.literal_eval(text)
    except Exception:
        pass

    # 4. Fallback: split by commas or newlines
    return [x.strip("-• \n") for x in re.split(r"[,;\n]", text) if x.strip()]


def extract_search_constraints(user_context: str):
    """
    Parse user chat history for structural constraints:
    - year_range: decade/year mentions → (min_year, max_year)
    - exclude_genres: negative genre mentions ("no horror", "not sci-fi")
    - reference_titles: "like MovieX" / "similar to MovieX" patterns
    Returns dict with any found constraints.
    """
    import re
    constraints = {}

    # --- Year / decade ---
    decade_m = re.search(r'\b(?:19)?([4-9]0)s\b', user_context, re.I)
    if decade_m:
        d = int(decade_m.group(1))
        base = (1900 + d) if d < 100 else d
        constraints['year_range'] = (base, base + 9)
    else:
        # 4-digit year: 1950-2029
        year_m = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', user_context)
        if year_m:
            y = int(year_m.group(1))
            constraints['year_range'] = (y - 2, y + 2)
        else:
            # 2-digit year with apostrophe: '76 → 1976
            short_m = re.search(r"'([5-9]\d|[01]\d)\b", user_context)
            if short_m:
                n = int(short_m.group(1))
                y = (1900 + n) if n >= 30 else (2000 + n)
                constraints['year_range'] = (y - 2, y + 2)

    # --- Negative genres ---
    genre_vocab = {
        'horror': 'Horror', 'sci-fi': 'Science Fiction', 'science fiction': 'Science Fiction',
        'animation': 'Animation', 'animated': 'Animation', 'musical': 'Music',
        'western': 'Western', 'documentary': 'Documentary', 'romance': 'Romance',
        'comedy': 'Comedy', 'drama': 'Drama', 'thriller': 'Thriller',
        'action': 'Action', 'fantasy': 'Fantasy', 'family': 'Family',
    }
    neg_pattern = re.compile(
        r"\b(?:no|not|without|avoid|don'?t want|except|rather not)\s+([\w\s\-]+?)(?:\s*,|\s+or\b|\s+and\b|\.|\!|\?|$)",
        re.I
    )
    exclude = set()
    for m in neg_pattern.finditer(user_context):
        phrase = m.group(1).strip().lower()
        for key, val in genre_vocab.items():
            if key in phrase:
                exclude.add(val)
    if exclude:
        constraints['exclude_genres'] = list(exclude)

    # --- Positive genre requirements: "I need X in Genre", "looking for Genre" ---
    # Match "in Genre1, Genre2" or standalone genre declarations
    pos_genre_vocab = {
        'horror': 'Horror', 'sci-fi': 'Science Fiction', 'science fiction': 'Science Fiction',
        'animation': 'Animation', 'animated': 'Animation',
        'western': 'Western', 'documentary': 'Documentary', 'romance': 'Romance',
        'comedy': 'Comedy', 'drama': 'Drama', 'thriller': 'Thriller',
        'action': 'Action', 'fantasy': 'Fantasy', 'family': 'Family',
        'adventure': 'Adventure', 'crime': 'Crime', 'mystery': 'Mystery',
    }
    # Pattern: "in Action, Thriller" or "I need Action movie"
    pos_in = re.search(r'\bin\s+((?:[A-Z][a-z]+(?:,\s*)?)+)', user_context)
    if pos_in:
        phrase = pos_in.group(1).lower()
        required = set()
        for key, val in pos_genre_vocab.items():
            if key in phrase:
                required.add(val)
        if required:
            constraints['require_any_genre'] = list(required)

    # --- Reference movie titles: "like X", "similar to X", "in the style of X" ---
    ref_pattern = re.compile(
        r'\b(?:like|similar to|in the style of|reminiscent of)\s+["\']?([A-Z][A-Za-z0-9 \':,\-]+?)["\']?(?:\s*[\.,\?!]|\s+but\b|\s+or\b|\s*$)',
        re.M
    )
    refs = [m.group(1).strip() for m in ref_pattern.finditer(user_context)]
    if refs:
        constraints['reference_titles'] = refs

    return constraints


def search_items_by_embedding(config, backend, embedding_vector, top_k,
                              user_long_embedding=None, user_short_embedding=None,
                              excluded_ids=None, query_text=None,
                              year_range=None, exclude_genres=None,
                              require_any_genre=None, reference_embeddings=None):
    """Retrieve and rank candidate items with the local index: hybrid BM25 + cosine base score,
    optionally blended with the user's short/long-term preference vectors
    (utils.retrieval.local_search.search_items_local; `backend` is a LocalBackend)."""
    kwargs = dict(user_long_embedding=user_long_embedding, user_short_embedding=user_short_embedding,
                  excluded_ids=excluded_ids, query_text=query_text,
                  year_range=year_range, exclude_genres=exclude_genres,
                  require_any_genre=require_any_genre, reference_embeddings=reference_embeddings)
    from utils.retrieval import local_search   # lazy: local_search imports this module
    return local_search.search_items_local(config, backend, embedding_vector, top_k, **kwargs)

def expand_query_for_search(config, client_chat, user_context: str) -> str:
    """
    Use LLM to expand a user preference description into richer search keywords
    that better match the vocabulary used in item descriptions (overviews, tags).
    Returns the expanded query string, or the original if expansion fails.
    """
    role_bot = config.get('role_bot', 'recommender')
    category = config.get('col_category', 'category')
    prompt = (
        f"You are helping a {role_bot} search a database. "
        f"A user described what they're looking for: \"{user_context}\"\n\n"
        f"Expand this into 15-20 specific search keywords that would appear in item titles, "
        f"category labels, and descriptions of matching items. Include: item type, style, "
        f"attributes, themes, and descriptive terms relevant to this kind of item. "
        f"Output ONLY a comma-separated list of keywords, nothing else."
    )
    try:
        resp = llm_calls.chat(client_chat, 'expand_query_for_search',
            model=config['chat_model'],
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.3,
        )
        expanded = resp.choices[0].message.content.strip()
        # Return original + expanded so we don't lose user's exact terms
        return f"{user_context} {expanded}"
    except Exception as e:
        return llm_calls.fallback('expand_query_for_search', e, user_context, config)


# Fallback text where an LLM-generated line is a whole bot turn: domain-neutral and
# not misleading. The template pools of the normal path (bot_greeting, bot_closing_success) use film words.
NEUTRAL_GREETING = "Hi! What are you looking for today?"
NEUTRAL_CLOSING = "Great, glad we found the right one."


def generate_chit_chat(config: Config, client_chat, speaker: str, user_profile: dict = None, context: str = "pre_recommendation", context_data: dict = None, chat_history: list = None) -> str:
    """
    Generate a context-aware chit-chat phrase for the bot.
    context options:
      - "greeting": warm opening when conversation starts
      - "category_react": natural reaction to user's stated genre/category
      - "pre_recommendation": light transition before showing recommendations (first attempt)
      - "follow_up": after user rejected previous recommendations, encourage and keep going
      - "transition": between conversation rounds, mood check or humorous remark
      - "closing": after successful recommendation, send-off message
    context_data: optional dict with extra info (e.g. {"category": "Animation, Comedy"})
    If user_profile is provided, the bot personalizes based on user demographic/preferences.
    """
    user_context = ""
    if user_profile and speaker == "bot":
        demographic = user_profile.get('user_demographic', '') or ''
        likes = user_profile.get('user_likes_long', '') or user_profile.get('user_likes_short', '') or ''
        if demographic:
            user_context += f"User profile (for tone only, do NOT mention in output): {demographic}. "
        if likes:
            user_context += f"User interests (for tone only, do NOT mention in output): {likes}. "

    context_data = context_data or {}
    role = config['role_bot']

    if speaker == "bot":
        if context == "greeting":
            prompt = P(config, 'chitchat_greeting', role=role, user_context=user_context)
        elif context == "category_react":
            category = context_data.get("category", "")
            prompt = P(config, 'chitchat_category_react', role=role, user_context=user_context, category=category)
        elif context == "pre_recommendation":
            prompt = P(config, 'chitchat_pre_recommendation', role=role, user_context=user_context)
        elif context == "follow_up":
            prompt = P(config, 'chitchat_follow_up', role=role, user_context=user_context)
        elif context == "closing":
            prompt = P(config, 'chitchat_closing', role=role, user_context=user_context)
        else:  # transition
            prompt = P(config, 'chitchat_transition', role=role, user_context=user_context)
    else:
        prompt = P(config, 'chitchat_user', role_user=config['role_user'])

    system_msg = {
        "role": "system",
        "content": P(config, 'chitchat_system', role_bot=config['role_bot']),
    }
    instruction_msg = {"role": "user", "content": prompt}
    if chat_history:
        messages = [system_msg] + list(chat_history[-6:]) + [instruction_msg]
    else:
        messages = [system_msg, instruction_msg]

    try:
        response = llm_calls.chat(client_chat, 'generate_chit_chat',
            model=config['chat_model'],
            messages=messages,
            temperature=config.get('temp_chitchat', 1.2),
            max_tokens=60
        )
        return _ensure_complete(response.choices[0].message.content.strip())
    except Exception as e:
        # a prefix elsewhere: no text (the caller writes its own turn); greeting / closing: a whole bot turn
        text = {'greeting': NEUTRAL_GREETING, 'closing': NEUTRAL_CLOSING}.get(context, "")
        return llm_calls.fallback('generate_chit_chat', e, text, config)


def generate_rejection_explanation(config: Config, client_chat, target_item_row, recommended_items: list, previous_rejections: list = None, chat_history: list = None) -> str:
    """
    Generate a natural rejection explanation when the user rejects recommendations.
    Compares target item attributes with recommended items to explain the mismatch.
    previous_rejections: list of prior rejection strings to avoid contradictions.
    Called with probability rejection_explanation_ratio.
    """
    # Build target item attribute summary — include core fields + item_attributes
    # NOTE: title is intentionally excluded so the user never mentions the item name
    target_props = {}
    for col_key, label in [(config['col_category'], 'category')]:
        val = target_item_row.get(col_key)
        if val is not None and str(val).strip() not in ['', 'nan', 'NaN', 'None']:
            target_props[label] = str(val)[:100]
    details_val = target_item_row.get(config['col_details'])
    if details_val is not None and str(details_val).strip() not in ['', 'nan', 'NaN', 'None']:
        target_props['details'] = str(details_val)[:200]
    for attr_key, attr_label in config['item_attributes'].items():
        val = target_item_row.get(attr_key)
        if val is not None and str(val).strip() not in ['', 'nan', 'NaN', 'None']:
            target_props[attr_label] = str(val)[:60]

    # Build recommended items summary with key attributes (not just titles)
    attr_labels = config.get('item_attributes', {})  # key -> label
    rec_summaries = []
    for r in recommended_items:
        parts = [r.get(config['col_title'], str(r.get(config['col_itemid'], '')))]
        cat = r.get(config['col_category'])
        if cat:
            parts.append(f"category: {cat}")
        for attr_key, attr_label in attr_labels.items():
            val = r.get(attr_key)
            if val and str(val).strip() not in ['', 'nan', 'NaN', 'None']:
                parts.append(f"{attr_label}: {val}")
                break  # include only the first available attribute to keep prompt short
        rec_summaries.append(" | ".join(parts))

    # Include prior rejections to prevent contradictions
    prior_context = ""
    if previous_rejections:
        prior_lines = "\n".join(f"- {r}" for r in previous_rejections[-3:])
        prior_context = f"\nYou have already said the following — do NOT contradict these:\n{prior_lines}\n"

    prompt = P(config, 'rejection_prompt',
               role_user=config['role_user'], target_props=target_props,
               rec_summaries=rec_summaries, prior_context=prior_context)

    system_msg = {"role": "system", "content": P(config, 'rejection_system', role_user=config['role_user'])}
    instruction_msg = {"role": "user", "content": prompt}
    if chat_history:
        messages = [system_msg] + list(chat_history[-6:]) + [instruction_msg]
    else:
        messages = [system_msg, instruction_msg]

    try:
        response = llm_calls.chat(client_chat, 'generate_rejection_explanation',
            model=config['chat_model'],
            messages=messages,
            temperature=config.get('temp_rejection', 0.9),
            max_tokens=80
        )
        return _ensure_complete(response.choices[0].message.content.strip())
    except Exception as e:
        return llm_calls.fallback('generate_rejection_explanation', e, "None of these match what I'm looking for.", config)


def generate_recommendation_explanation(config: Config, client_chat, recommended_items: list, chat_history: list = None) -> str:
    """
    Generate a brief explanation of WHY the bot is recommending these items,
    connecting them to the user's stated preferences from the conversation.
    Called with probability rec_explanation_ratio.
    """
    attr_labels = config.get('item_attributes', {})
    rec_summaries = []
    for r in recommended_items:
        parts = [r.get(config['col_title'], str(r.get(config['col_itemid'], '')))]
        cat = r.get(config['col_category'])
        if cat:
            parts.append(f"category: {cat}")
        for attr_key, attr_label in attr_labels.items():
            val = r.get(attr_key)
            if val and str(val).strip() not in ['', 'nan', 'NaN', 'None']:
                parts.append(f"{attr_label}: {val}")
                break
        rec_summaries.append(" | ".join(parts))

    items_str = "; ".join(rec_summaries)
    role_bot = config.get('role_bot', 'recommender')

    prompt = P(config, 'rec_explanation_prompt', role_bot=role_bot, items_str=items_str)

    system_msg = {"role": "system", "content": P(config, 'rec_explanation_system', role_bot=role_bot)}
    instruction_msg = {"role": "user", "content": prompt}
    if chat_history:
        messages = [system_msg] + list(chat_history[-6:]) + [instruction_msg]
    else:
        messages = [system_msg, instruction_msg]

    try:
        response = llm_calls.chat(client_chat, 'generate_recommendation_explanation',
            model=config['chat_model'],
            messages=messages,
            temperature=config.get('temp_rec_explain', 0.9),
            max_tokens=60
        )
        return _ensure_complete(response.choices[0].message.content.strip())
    except Exception as e:
        return llm_calls.fallback('generate_recommendation_explanation', e, "", config)


_USER_ACCEPTANCE_PHRASES = [
    "Yes!", "Yeah!", "Perfect.", "That's it!", "Exactly.",
    "Yep, that one.", "That works.", "Yes, that's the one.",
    "Sounds good.", "Yeah, that's it.", "That's perfect.",
    "Yep!", "That's the one.", "Yes, exactly.", "Great, thanks.",
    "That's what I was looking for.", "Yes please.", "Okay yes.",
    "That one, yes.", "Yep that works.",
]

def generate_user_acceptance(config: Config, client_chat, target_title: str, chat_history: list = None) -> str:
    """Return a short, natural user acceptance phrase. May include the title."""
    import random
    rec_top_n = int(config.get('rec_top_n', 2))
    if rec_top_n == 1:
        # Mix of with-title and without-title phrases for variety
        no_title_phrases = [
            "Yes, that sounds perfect!",
            "That's exactly what I was looking for!",
            "Yes! Great suggestion.",
            "Perfect, I'll watch that.",
            "That works for me, yes.",
            "Yes, I'll go with that.",
            "Sounds great, I'm in.",
            "That's the one, thanks!",
        ]
        title_phrases_single = [
            f"Yes, \"{target_title}\" — that's exactly what I wanted!",
            f"Oh, \"{target_title}\"! Yes, that's the one.",
            f"\"{target_title}\" sounds perfect, I'll go with that.",
            f"Yes! \"{target_title}\" is just what I had in mind.",
            f"Perfect — \"{target_title}\" is exactly it.",
        ]
        pool = no_title_phrases + title_phrases_single
        return random.choice(pool)
    # Multiple items shown — user needs to specify which one
    title_phrases = [
        f"Yes, \"{target_title}\" sounds perfect!",
        f"Yes, I'll go with \"{target_title}\".",
        f"\"{target_title}\" — yes, that's exactly what I was looking for!",
        f"Oh, \"{target_title}\"! Yes, that one.",
        f"Yes, \"{target_title}\" — that's the one.",
        f"\"{target_title}\" sounds great, I'll watch that.",
        f"That's it — \"{target_title}\". I'll go with that.",
        f"Yes! \"{target_title}\" is perfect.",
        f"\"{target_title}\" is exactly what I had in mind.",
        f"I'll take \"{target_title}\", yes.",
        f"Perfect, \"{target_title}\" it is!",
        f"Yes, \"{target_title}\" works for me.",
        f"\"{target_title}\" — that's what I was looking for.",
        f"Yes, \"{target_title}\". Great suggestion!",
        f"\"{target_title}\", definitely. Let's go with that.",
    ]
    return random.choice(title_phrases)


def generate_bot_closing_failed(config: Config, client_chat, chat_history: list = None) -> str:
    """Generate a natural bot closing message when max attempts are exhausted."""
    role_bot = config.get('role_bot', 'recommender')
    prompt = P(config, 'bot_closing_failed_prompt', role_bot=role_bot)
    messages = [{"role": "user", "content": prompt}]
    if chat_history:
        messages = list(chat_history[-4:]) + messages
    try:
        response = llm_calls.chat(client_chat, 'generate_bot_closing_failed',
            model=config['chat_model'],
            messages=messages,
            temperature=config.get('temp_chitchat', 1.0),
            max_tokens=60,
        )
        return _ensure_complete(response.choices[0].message.content.strip().strip('"').strip("'"))
    except Exception as e:
        return llm_calls.fallback('generate_bot_closing_failed', e, "Didn't get there today — but I'm sure you'll find it.", config)


def rephrase_phrase(config: Config, client_chat, phrase: str, fallback_text: str = None) -> str:
    """Use LLM to rephrase a template phrase so it sounds unique every time.

    If the call fails, `fallback_text` is used when given (greetings: the templates use film words),
    otherwise the template phrase itself."""
    prompt = (
        f"Rephrase the following sentence in a natural, conversational way. "
        f"Rules: exactly 1 short sentence, max 12 words, same meaning, different wording. "
        f"Do NOT add examples, movie titles, extra questions, or new content. "
        f"Return only the rephrased sentence, nothing else.\n\nOriginal: {phrase}"
    )
    try:
        response = llm_calls.chat(client_chat, 'rephrase_phrase',
            model=config['chat_model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=30,
        )
        result = response.choices[0].message.content.strip().strip('"').strip("'")
        return result if result else phrase
    except Exception as e:
        return llm_calls.fallback('rephrase_phrase', e, phrase if fallback_text is None else fallback_text, config)


def generate_user_responses_free_mode(config: Config, client_chat, item_row, user_profile: dict = None, chat_history: list = None):
    """
    Generate a realistic user response for free-mode simulation.
    The user always provides more descriptive details about the item they want —
    never answers questions about specific attributes.

    Returns:
        user_msg: str, simulated user message.
    """
    user_msg = ""

    # Build user profile context string
    profile_context = ""
    if user_profile:
        demographic = user_profile.get('user_demographic', '') or ''
        likes = user_profile.get('user_likes_long', '') or user_profile.get('user_likes_short', '') or ''
        if demographic:
            profile_context += f"User background: {demographic}. "
        if likes:
            profile_context += f"User preferences: {likes}. "

    # Build target item context so LLM stays consistent with what the user is looking for
    col_category = config.get('col_category', '')
    target_category = str(item_row.get(col_category, '')) if col_category and col_category in item_row.index else ''
    target_context = ""
    if target_category:
        target_context += f"The user is specifically looking for: {target_category}. "
    for key in ['release_year', 'spoken_languages', 'original_language']:
        if key in item_row.index:
            val = item_row.get(key)
            if val and str(val).strip() not in ['', 'nan', 'NaN', 'None']:
                if key == 'release_year':
                    try:
                        target_context += f"The item is from {int(float(val))}. "
                    except (ValueError, TypeError):
                        pass
                elif key in ('spoken_languages', 'original_language'):
                    target_context += f"Language: {val}. "
                    break

    system_content = P(config, 'free_user_system',
                       role_user=config['role_user'], role_bot=config['role_bot'],
                       profile_context=profile_context, target_context=target_context)

    def _build_messages(prompt):
        system_msg = {"role": "system", "content": system_content}
        instruction_msg = {"role": "user", "content": prompt}
        if chat_history:
            return [system_msg] + list(chat_history[-6:]) + [instruction_msg]
        return [system_msg, instruction_msg]

    # Short but still informative free-mode reply.
    if random.random() < config.get('short_reply_ratio', 0.45):
        short_prompts = load_prompts(config).get('free_user_short_prompts', [])
        try:
            response = llm_calls.chat(client_chat, 'free_user_short_reply',
                model=config['chat_model'],
                messages=_build_messages(random.choice(short_prompts)),
                temperature=config.get('temp_free_user', 1.0),
                max_tokens=15,
            )
            short_reply = response.choices[0].message.content.strip().strip('"').strip("'")
            if short_reply and len(short_reply.split()) <= 8:
                return short_reply
        except Exception as e:
            llm_calls.fallback('free_user_short_reply', e, None, config)   # go on with the detailed reply

    # Primary: use item details field to generate a descriptive user response
    details_col = config.get('col_details', 'details')
    if details_col in item_row.index and pd.notna(item_row[details_col]) and str(item_row[details_col]).strip() != "":
        target_details = str(item_row[details_col])
        prompt = P(config, 'free_user_details_prompt', target_details=target_details[:300])
        try:
            response = llm_calls.chat(client_chat, 'free_user_details_reply',
                model=config['chat_model'],
                messages=_build_messages(prompt),
                temperature=config.get('temp_free_user', 1.0),
                max_tokens=40,
            )
            user_msg = _truncate_msg(response.choices[0].message.content.strip())
        except Exception as e:
            # v1 used the first two sentences of the target's details here: the user turn gave the answer away.
            # Empty → the neutral free_user_fallback_prompts below.
            user_msg = llm_calls.fallback('free_user_details_reply', e, "", config)

    # Fallback: generic open-ended description request
    if not user_msg:
        fallback_prompts = load_prompts(config).get('free_user_fallback_prompts', [
            "I'm not sure how to describe it exactly — something that just feels right.",
        ])
        user_msg = random.choice(fallback_prompts)

    user_msg = user_msg.replace('"', '')
    return user_msg


def build_embedding_texts(df, embedding_fields: list) -> list:
    """Concatenate embedding_fields into one string per row, skipping missing columns."""
    available = [f for f in embedding_fields if f in df.columns]
    return df[available].fillna("").astype(str).agg(" ".join, axis=1).tolist()


