"""
Names of the conversation files the simulator writes.

The tmp-file name format and the pattern that recognises it live together here, so that the simulator
(which writes tmp files) and every tool that skips them (metrics, judge, NLP metrics) agree.

    tmp file    {strategy}-{item_id}[-{user_id}]-tmp-{YYYYmmddHHMMSS}-{8 hex digits}.txt
    final file  {user_id}-{item_id}-{num_turns}-{rec_attempts}-{succeed}-{YYYYmmddHHMMSS}.txt

A final name always ends in "-{0|1}-{14 digits}.txt", so it can never match TMP_NAME_RE, whatever its
user or item id contains (e.g. "u_tmp3" or an id with "-tmp-" in it).
"""
import os
import re
import uuid

TS_FORMAT = '%Y%m%d%H%M%S'
TMP_NAME_RE = re.compile(r'-tmp-\d{14}-[0-9a-f]{8}\.txt$')


def tmp_dialogue_name(strategy, item_id, user_id, ts, unique=None):
    """The tmp-file name of a conversation in progress (ts: TS_FORMAT string; unique: 8 hex digits)."""
    uid_part = f"-{user_id}" if user_id is not None else ""
    unique = unique if unique is not None else uuid.uuid4().hex[:8]
    return f"{strategy}-{item_id}{uid_part}-tmp-{ts}-{unique}.txt"


def is_tmp_dialogue(fname):
    """True for a tmp file of a conversation in progress (or left behind by a crash); False for final files."""
    return bool(TMP_NAME_RE.search(os.path.basename(fname)))


# A conversation whose LLM calls were replaced by fallback text ends with one line
#   System: <FALLBACK> fallbacks=N sites=site:n,... errors=Type:n,...
# Conversations stopped by an LLM / embedding error are kept as "<tmp name>.aborted" (not .txt: no tool reads them).
FALLBACK_PREFIX = 'System: <FALLBACK>'
_FALLBACK_COUNT_RE = re.compile(r'\bfallbacks=(\d+)')


def is_fallback_line(line):
    return line.lstrip().startswith(FALLBACK_PREFIX)


def split_fallback_line(text):
    """(text without the <FALLBACK> line, number of fallbacks it reports; 0 without the line)."""
    kept, count = [], 0
    for line in text.splitlines(keepends=True):
        if is_fallback_line(line):
            m = _FALLBACK_COUNT_RE.search(line)
            count += int(m.group(1)) if m else 1
        else:
            kept.append(line)
    return ''.join(kept), count
