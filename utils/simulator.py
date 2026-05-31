"""
utils.simulator
===============
Conversation simulation engines for OmniSim.

Three modes are provided, each corresponding to a different interaction strategy:

* ``simulate_free``     — open-ended; user describes preferences in natural language.
* ``simulate_static``  — schema-driven; bot asks about predefined metadata attributes.
* ``simulate_adaptive`` — adaptive; bot infers additional relevant attributes via LLM
                          and asks about both predefined and discovered attributes.

All three modes share the same hybrid retrieval and user-profile scoring logic
(see ``utils.utils.search_items_by_embedding`` and paper Equations 3–6).
"""

import random
import datetime
import re
import os
import uuid
import yaml
import pandas as pd

from utils.configurator import Config
from utils.dataset import Dataset
from utils.utils import (
    generate_chit_chat, generate_rejection_explanation, generate_recommendation_explanation,
    generate_user_responses_free_mode, generate_user_acceptance, generate_bot_closing_failed,
    extract_item_from_title, extract_attribute_value, extract_attributes_for_asking,
    extract_dynamic_description_from_chat, rephrase_attribute_to_natural_language,
    search_items_by_embedding, expand_query_for_search, extract_search_constraints,
    rephrase_phrase, _truncate_msg,
)

_PHRASE_CACHE: dict = {}

def _load_phrases(key: str) -> list:
    """Load phrase list from phrase_templates.yaml, cached."""
    path = 'configs/prompts/phrase_templates.yaml'
    if path not in _PHRASE_CACHE:
        with open(path, encoding='utf-8') as f:
            _PHRASE_CACHE[path] = yaml.safe_load(f)
    return _PHRASE_CACHE[path].get(key, [])


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def write_chat_to_file(file_path, name, message, config=None, role=None):
    """Append one utterance line to the conversation file.

    Creates parent directories on first write.  When *config* and *role* are
    supplied the message is first passed through ``_clean_for_output`` to strip
    optional annotation tags (profile notes, chit-chat markers) according to
    the show_profile_note / show_chit_chat_tag config flags.

    Args:
        file_path: Absolute or relative path to the ``.txt`` conversation file.
        name: Speaker label written before the colon (e.g. ``"Bot"``, ``"User"``).
        message: Utterance text to append.
        config: OmniSim config object (optional; enables output cleaning).
        role: ``"bot"`` or ``"user"`` (optional; used by ``_clean_for_output``).
    """
    if config is not None and role is not None:
        message = _clean_for_output(config, role, message)
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(f"{name}: {message}\n")


def _clean_for_output(config, role, message):
    """Strip optional annotation tags from an utterance before writing to disk.

    - ``show_profile_note=False`` → removes ``(Note: ...)`` suffixes from user turns.
    - ``show_chit_chat_tag=False`` → removes the ``[chit-chat]`` prefix from bot turns.
    """
    result = message
    if role == "user" and not config.get('show_profile_note', True):
        result = re.sub(r'\s*\(Note:[^)]*\)', '', result).strip()
    if role == "bot" and not config.get('show_chit_chat_tag', True):
        result = re.sub(r'^\[chit-chat\]\s*', '', result)
    return result


# ---------------------------------------------------------------------------
# Constraint & reference movie helpers
# ---------------------------------------------------------------------------

def _lookup_reference_embeddings(es, config, reference_titles: list) -> list:
    """
    Given a list of movie title strings extracted from user context (e.g. ["Death Wish", "Amélie"]),
    look each up in ES and return their pre-computed embedding vectors.
    """
    embeddings = []
    for title in reference_titles:
        try:
            res = es.search(
                index=config['es_index'],
                body={
                    "query": {"match": {"title": {"query": title, "fuzziness": "AUTO"}}},
                    "_source": ["embedding", "title"],
                    "size": 1,
                }
            )
            if res['hits']['hits']:
                emb = res['hits']['hits'][0]['_source'].get('embedding')
                if emb:
                    embeddings.append(emb)
        except Exception:
            pass
    return embeddings


def _build_rec_constraints(user_context: str, es, config) -> tuple:
    """
    Extract search_constraints dict and reference_embeddings list from user_context.
    Returns (constraints_dict, reference_embeddings_list).
    """
    constraints = extract_search_constraints(user_context)
    ref_embeddings = []
    if constraints.get('reference_titles'):
        ref_embeddings = _lookup_reference_embeddings(es, config, constraints['reference_titles'])
    return constraints, ref_embeddings


# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------

def _setup_output(config, target_item_id, user_id=None):
    """Create output folder and tmp file for a conversation."""
    run_ts = config.get('run_timestamp', '')
    strategy = config.get('generation_strategy', 'unknown')
    base = os.path.join(config['outputs_folder'], config['dataset'], config['mode_refinement'], strategy)
    chat_folder = (os.path.join(base, run_ts) if run_ts else base) + os.sep
    os.makedirs(chat_folder, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    uid_part = f"-{user_id}" if user_id is not None else ""
    unique = uuid.uuid4().hex[:8]
    tmp_file = os.path.join(chat_folder, f"{strategy}-{target_item_id}{uid_part}-tmp-{ts}-{unique}.txt")
    return chat_folder, tmp_file, ts


def _get_item_info(config, dataset, target_item_id):
    """Extract title, category, and merged attribute details for the target item."""
    item_info = dataset.df_items[dataset.df_items[dataset.itemid] == str(target_item_id)]
    if item_info.empty:
        return item_info, pd.Series(), "", "", ""
    row = item_info.iloc[0]
    title    = row[dataset.title]
    category = str(row[dataset.category]) if pd.notna(row[dataset.category]) else ""
    raw_details = str(row[dataset.details]) if pd.notna(row[dataset.details]) else ""
    attr_parts = [
        f"{label}: {row.get(key)}"
        for key, label in config['item_attributes'].items()
        if pd.notna(row.get(key)) and str(row.get(key)).strip()
    ]
    details = (" | ".join(attr_parts) + (" | " + raw_details if raw_details else "")) if attr_parts else raw_details
    return item_info, row, title, category, details


def _get_user_embeddings(config, client_embeddings, profile):
    """Encode user long/short preference texts into embeddings."""
    long_emb = short_emb = None
    for field in ('user_likes_long', 'user_likes_short'):
        val = profile.get(field, '')
        text = str(val).strip() if val and str(val).strip().lower() not in ('nan', 'none', '') else ''
        if text:
            emb = client_embeddings.embeddings.create(
                model=config['embeddings_model'], input=text
            ).data[0].embedding
            if field == 'user_likes_long':
                long_emb = emb
            else:
                short_emb = emb
    return long_emb, short_emb


def _get_excluded_ids(config, dataset, user_id, target_item_id):
    """Return item IDs already rated by the user (target is never excluded)."""
    if not user_id or not hasattr(dataset, 'df_interactions') or dataset.df_interactions is None:
        return set()
    rated = dataset.df_interactions[
        dataset.df_interactions[config['col_userid']] == user_id
    ][config['col_itemid']].tolist()
    excluded = {str(x) for x in rated}
    excluded.discard(str(target_item_id))
    return excluded


_INITIAL_USER_TEMPLATES = [
    "I need a {t}",
    "I'm looking for a {t}",
    "I'm in the mood for a {t}",
    "Can you recommend a good {t}?",
    "Any recommendations for a {t}?",
    "I want to watch a {t}",
    "I'd love to see a {t}",
    "Looking for a {t} to watch",
    "I feel like watching a {t}",
    "Got any {t} suggestions?",
    "Help me find a {t}",
    "I'd like to find a {t}",
    "What {t} would you suggest?",
    "I've been wanting to watch a {t}",
    "Something like a {t} would be great",
    "Do you have any good {t} recommendations?",
    "I'm after a {t}",
    "Any good {t} you'd suggest?",
    "I want something — a {t} ideally",
    "Looking to watch a {t}, any ideas?",
]


def _build_initial_user_msg(config, client_chat, target_title, target_category, profile):
    """Build the user's first demand message, with optional profile note.
    item_type is LLM-inferred (e.g. 'dark comedy film', 'action thriller', 'romantic drama').
    Genres are NOT appended — they come out naturally in follow-up turns.
    """
    item_type = extract_item_from_title(config, client_chat, target_title, target_category)
    templates = config.get('initial_user_templates') or _INITIAL_USER_TEMPLATES
    template = random.choice(templates)
    base_msg = template.format(t=item_type)
    if profile and (profile.get('user_demographic') or profile.get('user_likes_long')):
        hint = f"{profile.get('user_demographic', '')} {profile.get('user_likes_long', '') or profile.get('user_likes_short', '')}".strip()
        return f"{base_msg} (Note: {hint})", item_type
    return base_msg, item_type


# ---------------------------------------------------------------------------
# Recommendation round helper
# ---------------------------------------------------------------------------

_REC_INTROS = [
    "Here are some recommendations for you:",
    "Based on what you've shared, here are my top picks:",
    "I think you might enjoy these:",
    "Check out these options:",
    "Here's what I found for you:",
    "These might be a great match:",
    "Here are a few suggestions:",
    "Take a look at these:",
    "You might want to consider these:",
    "Here's what I came up with:",
    "These could be right up your alley:",
    "Let me share a few picks with you:",
    "Based on your taste, how about these?",
    "I've lined up a few options for you:",
    "Here's my shortlist for you:",
    "After thinking it over, here's what I'd suggest:",
    "These look like strong contenders:",
    "Here are some that stood out to me:",
    "I think these are worth a look:",
    "Let me put these on your radar:",
    "How do these sound to you?",
    "Here's a selection I think you'll like:",
    "I've narrowed it down to these:",
    "These caught my eye for you:",
    "Here are a few that seem like a good fit:",
    "Based on what you're looking for, try these:",
    "I pulled together a few ideas — take a look:",
    "Here's what I'd recommend based on our chat:",
    "These are my picks for you right now:",
    "Have a look at what I found:",
    "I think these are worth considering:",
    "Here's a curated set just for you:",
    "Going off what you've told me, here are some options:",
    "These seem to match your vibe:",
    "Let me throw a few ideas your way:",
]

_REJECTION_MESSAGES = [
    "None of these are what I want.",
    "These don't quite match what I'm looking for.",
    "Hmm, not really what I had in mind.",
    "I don't think any of these are right for me.",
    "Not quite — can we try again?",
    "These aren't what I'm after.",
    "I was hoping for something different.",
    "None of these feel like a good fit.",
    "I don't think these suit my taste.",
    "Not really feeling any of these.",
    "These aren't quite hitting the mark.",
    "I had something else in mind, honestly.",
    "None of these are grabbing me.",
    "I'm not sold on any of these.",
    "Something feels off — can you try again?",
    "These don't really appeal to me.",
    "I was looking for something a bit different.",
    "Hmm, I don't think these are for me.",
    "Not what I was hoping for.",
    "Can we try a different direction?",
    "None of these feel right.",
    "I'm not sure any of these work for me.",
    "These miss the mark a little.",
    "I'd like to see some other options.",
    "I don't think these match my vibe.",
    "Not quite there yet — any other ideas?",
    "These feel a bit off from what I described.",
    "I was picturing something different.",
    "None of these are clicking for me.",
    "Let's keep looking — these aren't it.",
    "I appreciate the effort, but these aren't quite right.",
    "Maybe we can refine the search a bit more?",
    "I'm not really drawn to any of these.",
    "These are close, but not quite what I need.",
    "I think we're not quite on the same page yet.",
]


def _do_recommendation(
    config, client_chat, es, embedding_vector,
    target_title, target_item_id, item_row, profile,
    chat_history, rejected_item_ids, previous_rejections, disliked_context,
    rec_attempts, user_long_emb, user_short_emb, excluded_ids,
    tmp_file, name_bot, name_user, used_chit_chat: set,
    query_text=None, search_constraints=None, reference_embeddings=None,
):
    """
    One recommendation round: search -> present -> user reacts.
    Mutates chat_history, rejected_item_ids, previous_rejections in place.
    Returns: (conversation_ended, succeed, updated_disliked_context, had_results, items_shown)
    items_shown=True only when actual item titles were presented to the user (gate-fired rounds = False).
    query_text: plain-text user context used for BM25 genre boost in hybrid search.
    search_constraints: dict from extract_search_constraints() with year_range/exclude_genres.
    reference_embeddings: list of reference movie embeddings to blend into query.
    """
    constraints = search_constraints or {}
    es_results = search_items_by_embedding(
        config, es, embedding_vector,
        top_k=config['max_similar_items'],
        user_long_embedding=user_long_emb,
        user_short_embedding=user_short_emb,
        excluded_ids=excluded_ids,
        query_text=query_text,
        year_range=constraints.get('year_range'),
        exclude_genres=constraints.get('exclude_genres'),
        require_any_genre=constraints.get('require_any_genre'),
        reference_embeddings=reference_embeddings,
    )
    es_results = [r for r in es_results if str(r[config['col_itemid']]) not in rejected_item_ids]
    # Deduplicate by item ID within this round (ES may return the same item multiple times)
    seen_ids, deduped = set(), []
    for r in es_results:
        iid = str(r[config['col_itemid']])
        if iid not in seen_ids:
            seen_ids.add(iid)
            deduped.append(r)
    topN = deduped[:min(config['rec_top_n'], len(deduped))]

    if not topN:
        return False, 0, disliked_context, False, False

    # No relevance gate — threshold_similarity already filters low-score results.
    # If topN is empty (all below threshold), we already returned above.

    # Optional explanation: why we're recommending these items
    explanation = ""
    if random.random() < config.get('rec_explanation_ratio', 0.0):
        explanation = generate_recommendation_explanation(config, client_chat, topN, chat_history)

    # Build recommendation message
    # When rec_top_n == 1: conversational single-item format (like ReDial)
    # When rec_top_n > 1: bulleted list format
    show_id = config.get('show_item_id', False)
    if config['rec_top_n'] == 1:
        item = topN[0]
        title = item[config['col_title']]
        item_id = item[config['col_itemid']]
        id_suffix = f" ({item_id})" if show_id else ""
        intro = random.choice([
            f"How about \"{title}\"{id_suffix}?",
            f"I think you might enjoy \"{title}\"{id_suffix}.",
            f"Have you seen \"{title}\"{id_suffix}?",
            f"What about \"{title}\"{id_suffix}?",
            f"You might like \"{title}\"{id_suffix}.",
            f"I'd suggest \"{title}\"{id_suffix}.",
            f"Have you tried \"{title}\"{id_suffix}?",
            f"\"{title}\"{id_suffix} could be worth a look.",
            f"Something like \"{title}\"{id_suffix} comes to mind.",
            f"I keep coming back to \"{title}\"{id_suffix} for this.",
            f"One that stands out is \"{title}\"{id_suffix}.",
            f"My pick would be \"{title}\"{id_suffix}.",
            f"\"{title}\"{id_suffix} — that one fits the bill.",
            f"I'd recommend \"{title}\"{id_suffix}.",
            f"This one jumped out at me: \"{title}\"{id_suffix}.",
            f"Give \"{title}\"{id_suffix} a shot.",
            f"\"{title}\"{id_suffix} seems like a solid match.",
            f"I was thinking \"{title}\"{id_suffix}.",
            f"\"{title}\"{id_suffix} might be right up your alley.",
            f"Hear me out — \"{title}\"{id_suffix}.",
        ])
        combined = f"{intro} {explanation}".strip() if explanation else intro
        rec_msg = _truncate_msg(combined)
    else:
        if show_id:
            rec_items = [f"{r[config['col_title']]} ({r[config['col_itemid']]})" for r in topN]
        else:
            rec_items = [r[config['col_title']] for r in topN]
        items_block = "\n".join(f"- {item}" for item in rec_items)
        intro = random.choice(_REC_INTROS)
        rec_msg = f"{intro}\n{items_block}"
        if explanation:
            combined = f"{explanation}\n{rec_msg}"
            rec_msg = _truncate_msg(combined)

    if random.random() < config.get('chit_chat_ratio', 0.0):
        ctx = "pre_recommendation" if rec_attempts == 0 else "follow_up"
        chit = _unique_chit_chat(config, client_chat, "bot", profile, ctx, {}, chat_history, used_chit_chat)
        bot_msg = f"[chit-chat] {chit}\n{rec_msg}" if chit else rec_msg
    else:
        bot_msg = rec_msg
    chat_history.append({"role": "assistant", "content": bot_msg})
    write_chat_to_file(tmp_file, name_bot, bot_msg, config, "bot")

    rec_titles = [r[config['col_title']] for r in topN]

    if target_title in rec_titles:
        user_msg = generate_user_acceptance(config, client_chat, target_title, chat_history)
        chat_history.append({"role": "user", "content": user_msg})
        write_chat_to_file(tmp_file, name_user, user_msg, config, "user")
        bot_closing = generate_chit_chat(config, client_chat, "bot", profile, "closing", {}, chat_history) \
            or random.choice(_load_phrases('bot_closing_success') or _BOT_CLOSING_SUCCESS)
        chat_history.append({"role": "assistant", "content": bot_closing})
        write_chat_to_file(tmp_file, name_bot, bot_closing, config, "bot")
        write_chat_to_file(
            tmp_file, "System",
            f"<END> Session ended successfully. The target item '{target_title}' ({target_item_id}) was accepted by the user."
        )
        return True, 1, disliked_context, True, True

    # Rejection
    recommended_ids = [r[config['col_itemid']] for r in topN]
    if random.random() < config.get('rejection_explanation_ratio', 0.0):
        user_msg = generate_rejection_explanation(
            config, client_chat, item_row, topN,
            previous_rejections=previous_rejections, chat_history=chat_history,
        )
    else:
        user_msg = random.choice(_REJECTION_MESSAGES)
    previous_rejections.append(user_msg)
    disliked_context += f" I do not want: {user_msg}"
    rejected_item_ids.update(str(pid) for pid in recommended_ids)
    chat_history.append({"role": "user", "content": user_msg})
    write_chat_to_file(tmp_file, name_user, user_msg, config, "user")
    return False, 0, disliked_context, True, True


_FALLBACK_GREETINGS = [
    "What are you in the mood for today?",
    "What kind of movie are you after?",
    "What are you looking for?",
    "What can I help you find today?",
    "Tell me what you're after and I'll find something great.",
    "What's on your watchlist tonight?",
    "What kind of film sounds good right now?",
    "What would you like to watch?",
    "What are you hoping to find today?",
    "Got something in mind, or want me to suggest?",
]


def _start_conversation(config, client_chat, tmp_file, name_bot, name_user,
                        target_title, target_category, profile, used_chit_chat: set):
    """Write bot greeting + first user demand. Returns (chat_history, item_type, number_turns=1)."""
    chat_history = []
    _greetings = _load_phrases('bot_greeting') or _FALLBACK_GREETINGS
    if random.random() < config.get('chit_chat_ratio', 0.0):
        chit = _unique_chit_chat(config, client_chat, "bot", profile, "greeting", {}, chat_history, used_chit_chat)
        bot_msg = chit if chit else rephrase_phrase(config, client_chat, random.choice(_greetings))
    else:
        bot_msg = rephrase_phrase(config, client_chat, random.choice(_greetings))
    chat_history.append({"role": "assistant", "content": bot_msg})
    write_chat_to_file(tmp_file, name_bot, bot_msg, config, "bot")

    user_msg, item_type = _build_initial_user_msg(config, client_chat, target_title, target_category, profile)
    chat_history.append({"role": "user", "content": user_msg})
    write_chat_to_file(tmp_file, name_user, user_msg, config, "user")
    return chat_history, item_type


def _unique_chit_chat(config, client_chat, speaker, user_profile, context, context_data,
                      chat_history, used_chit_chat: set) -> str | None:
    """Generate chit-chat, returning None if the exact phrase was already used."""
    chit = generate_chit_chat(config, client_chat, speaker, user_profile=user_profile,
                              context=context, context_data=context_data, chat_history=chat_history)
    if chit in used_chit_chat:
        return None
    used_chit_chat.add(chit)
    return chit


def _save_conversation(chat_folder, user_id, target_item_id, number_turns, rec_attempts, succeed, timestamp, tmp_file):
    """Rename the temporary conversation file to the final standardised filename.

    Final filename pattern::

        {user_id}-{item_id}-{num_turns}-{rec_attempts}-{succeed}-{timestamp}.txt

    ``succeed`` is 1 if the target item was accepted by the simulated user, 0 otherwise.
    """
    uid = str(user_id) if user_id is not None else "user"
    final_file = os.path.join(chat_folder, f"{uid}-{target_item_id}-{number_turns}-{rec_attempts}-{succeed}-{timestamp}.txt")
    os.rename(tmp_file, final_file)


_BOT_CLOSING_FAILED = [
    "I'm sorry I wasn't able to find exactly what you were looking for. I hope you find the perfect match — feel free to come back anytime!",
    "I wasn't able to nail down the right recommendation this time, but I appreciate your patience. Hope to help you better next time!",
    "Sorry I couldn't find what you had in mind today. Don't hesitate to reach out again — I'd love another chance to help!",
    "It seems like I couldn't quite hit the mark this time. Thank you for chatting with me, and I hope you find what you're looking for!",
    "I wish I could have found the perfect match for you. Thank you for your time, and have a wonderful day!",
    "Apologies for not finding the right one this time. Hope you track it down soon!",
    "Didn't get there today — but I'm sure you'll find it. Come back anytime!",
    "Looks like I struck out on this one. Thanks for your patience!",
    "I couldn't quite land it this session. Hope you find what you're after!",
    "Not my best showing today — sorry about that. Try again anytime!",
]

_BOT_CLOSING_SUCCESS = [
    "Great choice! I hope you enjoy it — have a wonderful time watching!",
    "Excellent! I'm sure you'll love it. Enjoy!",
    "Fantastic! That's a great pick. Hope you have a great time!",
    "Wonderful! Enjoy the experience. Feel free to come back for more recommendations anytime!",
    "Perfect! I'm glad I could help. Enjoy, and have a great day!",
    "Enjoy the watch! Come back anytime you need another recommendation.",
    "Happy watching! Let me know how it goes.",
    "I think you'll really enjoy it. Have a great time!",
    "That's a solid pick. Sit back and enjoy!",
    "Hope it lives up to the hype — happy watching!",
    "Glad we found it! Enjoy every minute.",
    "That should be a great watch. Enjoy!",
    "A great choice — you're in for a treat!",
    "Hope you love it as much as I think you will!",
    "Awesome! Grab some popcorn and enjoy.",
]


def _end_failed(tmp_file, name_bot, name_user, target_title, target_item_id, config=None, client_chat=None, chat_history=None):
    """Write the bot's closing apology and a FAILED system tag to the conversation file.

    If *config* and *client_chat* are provided the closing message is generated
    by the LLM for naturalness; otherwise a phrase is sampled from
    ``phrase_templates.yaml`` (``bot_closing_failed`` key) or the hardcoded
    fallback pool ``_BOT_CLOSING_FAILED``.
    """
    if config and client_chat:
        closing = generate_bot_closing_failed(config, client_chat, chat_history)
    else:
        _closing_failed = _load_phrases('bot_closing_failed') or _BOT_CLOSING_FAILED
        closing = random.choice(_closing_failed)
    write_chat_to_file(tmp_file, name_bot, closing)
    write_chat_to_file(
        tmp_file, "System",
        f"<FAILED> Session ended. The target item '{target_title}' ({target_item_id}) was not found within the maximum recommendation attempts."
    )


# ---------------------------------------------------------------------------
# simulate_free
# ---------------------------------------------------------------------------

def simulate_free(config: Config, dataset: Dataset, user_id, target_item_id,
                  client_chat, client_embeddings, es, user_profiles=None):
    """Run one **Free-mode** conversational recommendation session.

    In Free mode the bot issues open-ended follow-up prompts (e.g. "Tell me
    more about what you're looking for") and the simulated user responds with
    free-form natural-language descriptions derived from the target item's
    metadata.  After each user turn the full conversation context is embedded
    and used for hybrid BM25 + kNN retrieval (Equations 3–6).

    The session ends when:

    * The target item appears in a recommendation round and the user accepts it
      (``succeed=1``), **or**
    * ``max_rec_attempts`` recommendation rounds are exhausted (``succeed=0``).

    The completed conversation is saved under::

        chats/{dataset}/free/user_item_pairs/{timestamp}/{user_id}-{item_id}-...txt

    Args:
        config: Merged OmniSim configuration.
        dataset: Loaded CSV DataFrames (items, users, interactions).
        user_id: Simulated user ID (``None`` for anonymous guest sessions).
        target_item_id: Ground-truth item the user secretly wants.
        client_chat: Initialised LLM chat client.
        client_embeddings: Initialised embeddings client.
        es: Connected Elasticsearch client.
        user_profiles: Pre-built user profile dict keyed by user_id (optional).
    """
    name_bot, name_user = config['name_bot'], config['name_user']
    profile = (user_profiles or {}).get(str(user_id), {}) if user_id is not None else {}

    item_info, item_row, target_title, target_category, _ = _get_item_info(config, dataset, target_item_id)
    user_long_emb, user_short_emb = _get_user_embeddings(config, client_embeddings, profile)
    excluded_ids = _get_excluded_ids(config, dataset, user_id, target_item_id)
    chat_folder, tmp_file, timestamp = _setup_output(config, target_item_id, user_id=user_id)
    strategy = config.get('generation_strategy', 'unknown')

    used_chit_chat: set = set()
    chat_history, _ = _start_conversation(config, client_chat, tmp_file, name_bot, name_user,
                                           target_title, target_category, profile, used_chit_chat)
    rejected_item_ids, previous_rejections = set(), []
    disliked_context = ""
    rec_attempts = 0
    shown_recs = 0   # counts only rounds where item titles were actually shown to the user
    number_turns = 1
    conversation_ended = succeed = 0

    item_row_local = item_info.iloc[0] if not item_info.empty else pd.Series({col: None for col in dataset.df_items.columns})

    general_bot_msgs = [
        "Could you tell me more about the item you want?",
        "Got it. Could you add a bit more detail about the item you're looking for?",
        "Thanks, let's refine it a little more. What else should this item be like?",
        "Could you describe the item you have in mind a bit more?",
        "Thanks for helping narrow it down. Can you give me more item details?",
        "Great, that helps. Could you elaborate on the item a bit more?",
        "What else can you tell me about what you have in mind?",
        "Help me understand what you're after. Any more details?",
        "Can you paint a clearer picture of what you're looking for?",
        "Any other details that might help me narrow it down?",
        "Tell me more about the item so I can get this right.",
        "What else comes to mind when you picture the right item?",
        "What would make this item feel like the right match?",
        "Give me one more detail and I think I can nail it.",
        "What's been missing from the options so far?",
        "Could you add another detail about the item you're imagining?",
        "Let's refine and try again. Could you tell me a bit more?",
        "What other details should the item have?",
        "Say a bit more about the product you're hoping to buy.",
    ]

    from utils.utils import load_prompts
    _bot_short_followups = load_prompts(config).get('bot_short_followups', [])

    def _pick_bot_followup():
        """Return a short follow-up question bot_short_followup_ratio% of the time."""
        if _bot_short_followups and random.random() < config.get('bot_short_followup_ratio', 0.4):
            phrase = random.choice(_bot_short_followups)
        else:
            phrase = random.choice(general_bot_msgs)
        return rephrase_phrase(config, client_chat, phrase)

    question_count = 0
    skip_question = False

    while rec_attempts < config['max_rec_attempts'] and not conversation_ended:
        # Bot asks clarifying question — enforce max_questions limit
        max_questions = int(config.get('max_questions') or 99)
        if not skip_question and question_count >= max_questions:
            skip_question = True

        if skip_question:
            # Pure recommendation — keep looping until accepted or max_rec_attempts
            user_context = " ".join(m["content"] for m in chat_history if m["role"] == "user")
            if disliked_context:
                user_context += disliked_context
            search_query = expand_query_for_search(config, client_chat, user_context) if config.get('expand_query') else user_context
            emb = client_embeddings.embeddings.create(
                model=config['embeddings_model'], input=search_query
            ).data[0].embedding
            s_constraints, s_ref_embs = _build_rec_constraints(user_context, es, config)
            conversation_ended, succeed, disliked_context, had_results, items_shown = _do_recommendation(
                config, client_chat, es, emb, target_title, target_item_id, item_row,
                profile, chat_history, rejected_item_ids, previous_rejections, disliked_context,
                rec_attempts, user_long_emb, user_short_emb, excluded_ids, tmp_file, name_bot, name_user, used_chit_chat,
                query_text=search_query, search_constraints=s_constraints, reference_embeddings=s_ref_embs,
            )
            if items_shown:
                rec_attempts += 1
                shown_recs += 1
            if had_results:
                number_turns += 1
            continue  # keep looping until max_rec_attempts or accepted

        if question_count == 0:
            base_msg = _pick_bot_followup()
            ctx, ctx_data = "category_react", {"category": target_category}
        else:
            base_msg = _pick_bot_followup()
            ctx, ctx_data = "transition", {}
        if random.random() < config.get('chit_chat_ratio', 0.0):
            chit = _unique_chit_chat(config, client_chat, "bot", profile, ctx, ctx_data, chat_history, used_chit_chat)
            bot_msg = f"[chit-chat] {chit} {base_msg}" if chit else base_msg
        else:
            bot_msg = base_msg
        question_count += 1
        chat_history.append({"role": "assistant", "content": bot_msg})
        write_chat_to_file(tmp_file, name_bot, bot_msg, config, "bot")

        # User responds with more descriptive details about the item they want
        user_msg = generate_user_responses_free_mode(
            config, client_chat, item_row_local,
            user_profile=profile, chat_history=chat_history,
        )
        chat_history.append({"role": "user", "content": user_msg})
        write_chat_to_file(tmp_file, name_user, user_msg, config, "user")
        number_turns += 1

        # Recommend based on accumulated context
        user_context = " ".join(m["content"] for m in chat_history if m["role"] == "user")
        if disliked_context:
            user_context += disliked_context
        search_query = expand_query_for_search(config, client_chat, user_context) if config.get('expand_query') else user_context
        emb = client_embeddings.embeddings.create(
            model=config['embeddings_model'], input=search_query
        ).data[0].embedding
        s_constraints, s_ref_embs = _build_rec_constraints(user_context, es, config)
        conversation_ended, succeed, disliked_context, had_results, items_shown = _do_recommendation(
            config, client_chat, es, emb, target_title, target_item_id, item_row,
            profile, chat_history, rejected_item_ids, previous_rejections, disliked_context,
            rec_attempts, user_long_emb, user_short_emb, excluded_ids, tmp_file, name_bot, name_user, used_chit_chat,
            query_text=search_query, search_constraints=s_constraints, reference_embeddings=s_ref_embs,
        )
        if items_shown:
            rec_attempts += 1
            shown_recs += 1
        if had_results:
            number_turns += 1

    if not conversation_ended:
        _end_failed(tmp_file, name_bot, name_user, target_title, target_item_id, config=config, client_chat=client_chat, chat_history=chat_history)
        number_turns += 1

    _save_conversation(chat_folder, user_id, target_item_id, number_turns, shown_recs, succeed, timestamp, tmp_file)


# ---------------------------------------------------------------------------
# simulate_static / simulate_adaptive
# ---------------------------------------------------------------------------

def simulate_static(config, dataset, user_id, target_item_id,
                    client_chat, client_embeddings, es, user_profiles=None):
    """Run one **Static-mode** conversational recommendation session.

    Static mode is schema-driven: the bot asks questions **only** about the
    predefined ``item_attributes`` specified in the dataset YAML (e.g. genre,
    release year, spoken language for movies).  User answers are rephrased into
    natural language and accumulated into a structured query for retrieval.

    This is a thin wrapper around :func:`simulate_adaptive` with
    ``attribute_by_openai=False``, which disables LLM-based attribute discovery
    and restricts the bot to the YAML-defined attribute list.

    Termination conditions and output format are identical to
    :func:`simulate_free`.

    Args:
        config: Merged OmniSim configuration.
        dataset: Loaded CSV DataFrames.
        user_id: Simulated user ID (``None`` for guest).
        target_item_id: Ground-truth target item.
        client_chat: LLM chat client.
        client_embeddings: Embeddings client.
        es: Elasticsearch client.
        user_profiles: Pre-built user profiles (optional).
    """
    simulate_adaptive(config, dataset, user_id, target_item_id,
                      client_chat, client_embeddings, es,
                      attribute_by_openai=False, user_profiles=user_profiles)


def simulate_adaptive(config: Config, dataset: Dataset, user_id, target_item_id,
                      client_chat, client_embeddings, es,
                      attribute_by_openai=True, user_profiles=None):
    """Run one **Adaptive-mode** conversational recommendation session.

    Adaptive mode extends Static mode by letting the LLM **discover additional
    relevant attributes** beyond those defined in the YAML schema.  For example,
    in a movie domain the bot may spontaneously ask about "animation style" or
    "music score" even if those fields are not in the config, because the LLM
    judges them contextually relevant for the item type.

    When ``attribute_by_openai=False`` this function behaves identically to
    Static mode (no LLM-driven attribute discovery); that code path is used by
    :func:`simulate_static`.

    **Dialogue loop per turn:**

    1. If unused attributes remain and the question quota is not exhausted, ask
       one clarifying question about a randomly selected attribute.
    2. Embed the full accumulated user context and run hybrid retrieval.
    3. If ``max(S̃_base) ≥ threshold_similarity`` present top-N recommendations;
       otherwise loop back to step 1.
    4. Check whether the target item was recommended; if so, generate user
       acceptance and end the session (``succeed=1``).
    5. Otherwise generate a rejection and continue.

    Args:
        config: Merged OmniSim configuration.
        dataset: Loaded CSV DataFrames.
        user_id: Simulated user ID (``None`` for guest).
        target_item_id: Ground-truth target item.
        client_chat: LLM chat client.
        client_embeddings: Embeddings client.
        es: Elasticsearch client.
        attribute_by_openai: If ``True``, use LLM to discover extra attributes
            beyond the YAML schema (Adaptive mode). If ``False``, restrict to
            the YAML-defined schema only (Static mode).
        user_profiles: Pre-built user profiles (optional).
    """
    from utils.utils import load_prompts
    name_bot, name_user = config['name_bot'], config['name_user']
    profile = (user_profiles or {}).get(str(user_id), {}) if user_id is not None else {}

    item_info, item_row, target_title, target_category, target_details = _get_item_info(config, dataset, target_item_id)
    user_long_emb, user_short_emb = _get_user_embeddings(config, client_embeddings, profile)
    excluded_ids = _get_excluded_ids(config, dataset, user_id, target_item_id)
    chat_folder, tmp_file, timestamp = _setup_output(config, target_item_id, user_id=user_id)
    strategy = config.get('generation_strategy', 'unknown')

    used_chit_chat: set = set()
    chat_history, item_type = _start_conversation(config, client_chat, tmp_file, name_bot, name_user,
                                                   target_title, target_category, profile, used_chit_chat)
    collected_attributes = {}
    rejected_item_ids, asked_attributes, previous_rejections = set(), set(), []
    disliked_context = ""
    rec_attempts = 0
    shown_recs = 0   # counts only rounds where item titles were actually shown to the user
    number_turns = 1
    conversation_ended = succeed = 0

    # Attribute discovery
    candidate_attributes = extract_attributes_for_asking(
        config, client_chat, item_type, target_category, attribute_by_openai
    )

    # Build reverse map: raw column key → human-readable label
    _attr_label_map = config.get('item_attributes', {})
    _attr_key_lower = {k.lower().replace('_', ' '): v for k, v in _attr_label_map.items()}
    _no_pref_replies = load_prompts(config).get('adaptive_no_preference', [])
    _no_pref_pool = random.sample(_no_pref_replies, len(_no_pref_replies)) if _no_pref_replies else []

    while rec_attempts < config['max_rec_attempts'] and not conversation_ended:
        # Ask a clarifying question if quota not exceeded and attributes remain
        remaining = [a for a in candidate_attributes if a not in asked_attributes]
        max_questions = int(config.get('max_questions') or 99)
        can_ask = remaining and len(asked_attributes) < max_questions

        if can_ask:
            attribute = random.choice(remaining[:1])
            asked_attributes.add(attribute)
            attr_display = (
                _attr_label_map.get(attribute)
                or _attr_key_lower.get(attribute.lower().replace('_', ' '))
                or attribute
            )
            # Generate contextually aware question using LLM + conversation history
            try:
                _ctx_msgs = list(chat_history) + [{
                    "role": "user",
                    "content": (
                        f"You are a conversational recommender. The user wants a '{item_type}'. "
                        f"Ask a natural follow-up question ONLY about their preference for '{attr_display}'. "
                        f"Rules: max 15 words, casual tone, no quotes, no 'Do you prefer X or Y' binary format. "
                        f"Use open-ended questions. "
                        f"IMPORTANT: Do NOT mention any genres, themes, or moods other than what the user already said. "
                        f"Do NOT treat 'dark', 'light', or other adjectives as genre names. "
                        f"Output only the question, no extra text."
                    )
                }]
                bot_msg = client_chat.chat.completions.create(
                    model=config['chat_model'],
                    messages=_ctx_msgs,
                    temperature=0.7,
                    max_tokens=50,
                ).choices[0].message.content.strip().strip('"').strip("'")
            except Exception:
                _pref_questions = load_prompts(config).get('adaptive_pref_questions', [])
                _tmpl = random.choice(_pref_questions) if _pref_questions else "What is your preference for {attribute}?"
                bot_msg = _tmpl.format(attribute=attr_display)
            chat_history.append({"role": "assistant", "content": bot_msg})
            write_chat_to_file(tmp_file, name_bot, bot_msg, config, "bot")

            attr_value = extract_attribute_value(config, client_chat, attribute, item_info, target_details)
            user_msg = rephrase_attribute_to_natural_language(config, client_chat, attribute, attr_value)
            if not user_msg or not attr_value or not target_details:
                if not _no_pref_pool:
                    _no_pref_pool = random.sample(_no_pref_replies, len(_no_pref_replies)) if _no_pref_replies else []
                user_msg = _no_pref_pool.pop(0) if _no_pref_pool else "I have no preference."
            chat_history.append({"role": "user", "content": user_msg})
            write_chat_to_file(tmp_file, name_user, user_msg, config, "user")
            number_turns += 1
            if user_msg not in (_no_pref_replies or []):
                collected_attributes[attribute] = attr_value

        # Always recommend after each Q&A round (consistent with free mode)
        user_context_text = " ".join(m["content"] for m in chat_history if m["role"] == "user")
        dynamic_desc = extract_dynamic_description_from_chat(config, client_chat, chat_history)
        if disliked_context:
            dynamic_desc += disliked_context
        search_query = expand_query_for_search(config, client_chat, dynamic_desc) if config.get('expand_query') else dynamic_desc
        emb = client_embeddings.embeddings.create(
            model=config['embeddings_model'], input=search_query
        ).data[0].embedding
        s_constraints, s_ref_embs = _build_rec_constraints(user_context_text, es, config)
        # Skip the relevance gate when no more questions can be asked — forces a real recommendation
        # instead of repeated "no match" messages, which creates consecutive Bot messages.
        conversation_ended, succeed, disliked_context, had_results, items_shown = _do_recommendation(
            config, client_chat, es, emb, target_title, target_item_id, item_row,
            profile, chat_history, rejected_item_ids, previous_rejections, disliked_context,
            rec_attempts, user_long_emb, user_short_emb, excluded_ids, tmp_file, name_bot, name_user, used_chit_chat,
            query_text=search_query, search_constraints=s_constraints, reference_embeddings=s_ref_embs,
        )
        if items_shown:
            rec_attempts += 1
            shown_recs += 1
        if had_results:
            number_turns += 1

    if not conversation_ended:
        _end_failed(tmp_file, name_bot, name_user, target_title, target_item_id, config=config, client_chat=client_chat, chat_history=chat_history)
        number_turns += 1

    _save_conversation(chat_folder, user_id, target_item_id, number_turns, shown_recs, succeed, timestamp, tmp_file)
