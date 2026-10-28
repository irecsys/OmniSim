"""
User Profile Builder

Generates user_profiles.csv by:
  users.csv        -> LLM produces the {user_demographic} summary
  interactions.csv -> from the items rated above rating_threshold, produces:
          {user_likes_long}     (all-time)
          {user_likes_short}    (last short_term_days days, if the interactions have timestamps)

Usage:
    python utils/user_profile_builder.py --config configs/imdb/imdb.yaml
"""

import argparse
import os
import sys
import pandas as pd
from datetime import datetime, timedelta

# allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import llm_calls
from utils.configurator import Config
from utils.utils import get_openai_clients


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _llm(client_chat, config, prompt: str, max_tokens: int = 80, site: str = 'profile_likes_long') -> str:
    try:
        response = llm_calls.chat(client_chat, site,
            model=config['chat_model'],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [LLM error] {llm_calls.sanitize(e)}")
        return llm_calls.fallback(site, e, "", config)


def _build_demographic(user_row: pd.Series) -> str:
    """Build a plain-text demographic string from a user row without LLM (rule-based)."""
    parts = []
    if 'birthyear' in user_row.index and pd.notna(user_row['birthyear']):
        age = datetime.now().year - int(user_row['birthyear'])
        parts.append(f"{age} years old")
    for col in ('gender', 'profession', 'occupation'):
        if col in user_row.index and pd.notna(user_row[col]):
            parts.append(str(user_row[col]))
    for col in ('nationality', 'country'):
        if col in user_row.index and pd.notna(user_row[col]):
            parts.append(f"from {user_row[col]}")
    return ", ".join(parts)


def generate_demographic_summary(client_chat, config, user_row: pd.Series) -> str:
    """Convert a user's demographic columns into a 1-sentence natural language profile."""
    candidate_cols = [
        'birthyear', 'age', 'gender', 'nationality', 'country',
        'profession', 'occupation', 'education',
    ]
    demo_info = {}
    for col in candidate_cols:
        if col in user_row.index and pd.notna(user_row[col]):
            demo_info[col] = user_row[col]

    if not demo_info:
        return ""

    prompt = f"""Based on this user demographic information: {demo_info}
Write a concise 1-sentence natural language profile.
Example: "A 30-year-old female software engineer from Kuwait."
Return only the sentence."""
    return _llm(client_chat, config, prompt, max_tokens=60, site='profile_demographic')


def compute_feature_stats(subset: pd.DataFrame, col_category: str) -> dict:
    """Count feature frequencies from liked item metadata."""
    stats = {}

    # Genre / category counts (split by comma if multiple genres per item)
    if col_category in subset.columns:
        genre_series = subset[col_category].dropna().astype(str)
        genre_counts = {}
        for val in genre_series:
            for g in [x.strip() for x in val.split(',')]:
                if g:
                    genre_counts[g] = genre_counts.get(g, 0) + 1
        if genre_counts:
            stats['genres'] = dict(sorted(genre_counts.items(), key=lambda x: -x[1])[:8])

    # Language counts
    if 'original_language' in subset.columns:
        lang_counts = subset['original_language'].dropna().value_counts().head(5).to_dict()
        if lang_counts:
            stats['languages'] = lang_counts

    # Sample titles (top 8)
    title_col = next((c for c in ['title', 'name', 'item_name'] if c in subset.columns), None)
    if title_col:
        stats['sample_titles'] = subset[title_col].dropna().tolist()[:8]

    return stats


def generate_preference_summary(client_chat, config, liked_ids: list, items_df: pd.DataFrame,
                                 col_itemid: str, col_title: str, col_category: str,
                                 site: str = 'profile_likes_long') -> str:
    """Preference summary of the liked items: join their metadata, count features, summarise with the LLM."""
    if not liked_ids:
        return ""

    subset = items_df[items_df[col_itemid].isin(liked_ids)]
    if subset.empty:
        return ""

    # compute feature statistics
    stats = compute_feature_stats(subset, col_category)
    if not stats:
        return ""

    # pass the structured stats to the LLM
    domain = config.get("role_bot", "recommender")
    prompt = f"""Based on a user's liked items with the following feature statistics:

Categories (name: count): {stats.get('genres', {})}
Languages (lang: count): {stats.get('languages', {})}
Sample titles: {stats.get('sample_titles', [])}

Summarize the user's preferences in 1-2 sentences for a {domain} system.
Return only the summary."""
    return _llm(client_chat, config, prompt, max_tokens=80, site=site)


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_user_profiles(config_file_list: list):
    """Build the user profiles of the dataset in the config: user_demographic from the users file, and
    user_likes_long / user_likes_short from the liked items (rating > rating_threshold) in the interactions
    file, summarised by the LLM. With a timestamp column, user_likes_short covers the last short_term_days
    days and falls back to user_likes_long when that window has no liked item; without one, it is an empty
    string. Returns (profiles_df, failures), failures being (user id, call site, error type) of failed LLM
    calls; data/{dataset}/user_profiles.csv is written only when there are none. Returns None when neither
    the users file nor the interactions file exists."""
    config = Config(config_file_list=config_file_list)
    client_chat, _ = get_openai_clients(config)

    data_path = os.path.join(config['data_path'], config['dataset'])
    col_userid = config['col_userid']
    col_itemid = config['col_itemid']
    col_rating = config['col_rating']
    col_title = config['col_title']
    col_category = config['col_category']
    rating_threshold = config.get('rating_threshold') or 3
    short_term_days = config.get('short_term_days') or 7

    # Load items
    items_df = pd.read_csv(os.path.join(data_path, config['file_items']), low_memory=False)

    # Load users (optional)
    users_df = None
    file_users = config.get('file_users') or ''
    if str(file_users).strip().lower() not in ['', 'none', '~', 'null']:
        users_path = os.path.join(data_path, file_users)
        if os.path.exists(users_path):
            users_df = pd.read_csv(users_path, low_memory=False)
            print(f"Loaded {len(users_df)} users from {users_path}")
        else:
            print(f"[Warning] users file not found: {users_path}")

    # Load interactions (optional)
    interactions_df = None
    file_interactions = config.get('file_interactions') or ''
    if str(file_interactions).strip().lower() not in ['', 'none', '~', 'null']:
        inter_path = os.path.join(data_path, file_interactions)
        if os.path.exists(inter_path):
            interactions_df = pd.read_csv(inter_path, low_memory=False)
            print(f"Loaded {len(interactions_df)} interactions from {inter_path}")
        else:
            print(f"[Warning] interactions file not found: {inter_path}")

    if users_df is None and interactions_df is None:
        print("No user or interaction files found. Cannot build profiles.")
        return

    # Determine user ID set
    if users_df is not None:
        user_ids = users_df[col_userid].tolist()
    else:
        user_ids = interactions_df[col_userid].unique().tolist()

    has_timestamp = interactions_df is not None and 'timestamp' in interactions_df.columns
    if has_timestamp:
        # Try to parse timestamps (unix seconds or ISO strings)
        interactions_df['_ts'] = pd.to_datetime(
            interactions_df['timestamp'], unit='s', errors='coerce'
        )
        # If coerce produced all NaT, try ISO format
        if interactions_df['_ts'].isna().all():
            interactions_df['_ts'] = pd.to_datetime(
                interactions_df['timestamp'], errors='coerce'
            )
        short_term_cutoff = datetime.now() - timedelta(days=short_term_days)

    profiles = []
    failures = []                      # [(user id, call site, error type)] of failed LLM calls
    log = llm_calls.fallback_log()
    total = len(user_ids)
    for i, uid in enumerate(user_ids):
        print(f"  [{i+1}/{total}] user {uid}", end="  ")
        n_logged = len(log)

        profile = {
            'userid': uid,
            'user_demographic': '',
            'user_likes_long': '',
            'user_likes_short': '',
        }

        # --- Demographic summary ---
        if users_df is not None:
            row_df = users_df[users_df[col_userid] == uid]
            if not row_df.empty:
                profile['user_demographic'] = generate_demographic_summary(
                    client_chat, config, row_df.iloc[0]
                )
                print(f"demo=OK", end="  ")

        # --- Preference summary from interactions ---
        if interactions_df is not None:
            user_inter = interactions_df[interactions_df[col_userid] == uid]

            if col_rating in user_inter.columns:
                # Long-term: all-time liked items (rating > threshold)
                liked_long = user_inter[user_inter[col_rating] > rating_threshold][col_itemid].tolist()
                profile['user_likes_long'] = generate_preference_summary(
                    client_chat, config, liked_long, items_df, col_itemid, col_title, col_category
                )
                print(f"long=OK", end="  ")

                # Short-term: last N days (only if timestamp exists)
                if has_timestamp:
                    short_inter = user_inter[user_inter['_ts'] >= short_term_cutoff]
                    liked_short = short_inter[short_inter[col_rating] > rating_threshold][col_itemid].tolist()
                    profile['user_likes_short'] = generate_preference_summary(
                        client_chat, config, liked_short, items_df, col_itemid, col_title, col_category,
                        site='profile_likes_short'
                    )
                    # Fallback: no recent ratings → copy long-term preference
                    if not profile['user_likes_short']:
                        profile['user_likes_short'] = profile['user_likes_long']
                        print(f"short=fallback(long)", end="  ")
                    else:
                        print(f"short=OK", end="  ")

        print()
        failures += [(uid, site, err) for site, err in log[n_logged:]]
        profiles.append(profile)

    # Save — only complete profiles: a failed LLM call leaves a field empty, and a cached empty field would be
    # reused by every later run
    profiles_df = pd.DataFrame(profiles)
    output_path = os.path.join(data_path, 'user_profiles.csv')
    if failures:
        print(f"\n{len(failures)} LLM call(s) failed for users {sorted({str(u) for u, _, _ in failures})}: "
              f"{output_path} not written")
    else:
        profiles_df.to_csv(output_path, index=False)
        print(f"\nSaved {len(profiles)} user profiles -> {output_path}")
    return profiles_df, failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build user_profiles.csv from users.csv and interactions.csv')
    parser.add_argument('--config', type=str, default='user.yaml',
                        help='Config YAML file (default: user.yaml)')
    args, _ = parser.parse_known_args()
    config_file_list = args.config.strip().split(' ')
    _, failures = build_user_profiles(config_file_list) or (None, [])
    sys.exit(3 if failures else 0)          # 3: LLM calls failed, nothing written
