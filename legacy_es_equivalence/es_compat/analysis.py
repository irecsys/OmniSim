"""
Python emulation of the Elasticsearch `standard` analyzer (Lucene StandardTokenizer,
UAX#29 word boundaries, then lowercase; no stop words).

Verification tool only (the library tokenises with standard components). Used where v1 relied on
ES text analysis:
  - structural filters on col_category (exclude_genres → `match`, require_any_genre → `match_phrase`),
    R5 replacement
  - reference-title fuzzy lookup on the `title` field (title_fuzzy.py)
  - BM25 on bm25_details (bm25_lucene.py)

Verified against a live ES 8.13 `_analyze` during the migration (100% on titles, categories,
bm25_details and the genre vocabulary).

Simplifications (no such text in those fields): Hebrew-letter rules, complex-context scripts
(Thai, Lao, Khmer, Myanmar) and emoji sequences (ZWJ, modifiers, flags) are approximated.
"""
import unicodedata

MAX_TOKEN_LENGTH = 255

# Word_Break property values that matter here (Unicode UAX#29)
_MID_LETTER = set(':··՟״‧︓﹕：')
_MID_NUM_LET = set('.‘’․﹒＇．')
_SINGLE_QUOTE = "'"
_MID_NUM = set(',;;։،؍٬߸⁄︐︔﹐﹔，；')
_EXTEND_NUM_LET_EXTRA = set(' ')

# Characters Lucene emits as single <EMOJI> tokens (Extended_Pictographic, the common ranges)
_EMOJI_SINGLES = set('©®‼⁉™ℹⓂ〰〽㊗㊙')
_EMOJI_RANGES = [(0x2194, 0x2199), (0x21a9, 0x21aa), (0x231a, 0x231b), (0x2328, 0x2328), (0x23cf, 0x23cf),
                 (0x23e9, 0x23f3), (0x23f8, 0x23fa), (0x25aa, 0x25ab), (0x25b6, 0x25b6), (0x25c0, 0x25c0),
                 (0x25fb, 0x25fe), (0x2600, 0x27bf), (0x2934, 0x2935), (0x2b05, 0x2b07), (0x2b1b, 0x2b1c),
                 (0x2b50, 0x2b50), (0x2b55, 0x2b55), (0x1f000, 0x1faff)]

ALETTER, NUMERIC, KATAKANA, EXTEND_NUM_LET, IDEO, EMOJI, EXTEND, MID_LETTER, MID_NUM_LET, \
    SINGLE_QUOTE, MID_NUM, OTHER = range(12)


def _is_emoji(cp):
    return any(lo <= cp <= hi for lo, hi in _EMOJI_RANGES)


def char_class(ch):
    cp = ord(ch)
    if ch == _SINGLE_QUOTE:
        return SINGLE_QUOTE
    if ch in _MID_LETTER:
        return MID_LETTER
    if ch in _MID_NUM_LET:
        return MID_NUM_LET
    if ch in _MID_NUM:
        return MID_NUM
    if ch in _EXTEND_NUM_LET_EXTRA:
        return EXTEND_NUM_LET
    if ch in _EMOJI_SINGLES or _is_emoji(cp):
        return EMOJI
    cat = unicodedata.category(ch)
    if cat in ('Mn', 'Me', 'Mc', 'Cf'):
        return EXTEND
    if cat == 'Pc':
        return EXTEND_NUM_LET
    if cat == 'Nd':
        return NUMERIC
    if 0x30a0 <= cp <= 0x30ff or 0x31f0 <= cp <= 0x31ff or 0xff66 <= cp <= 0xff9f:
        return KATAKANA if cp != 0x30fb else OTHER
    if (0x3040 <= cp <= 0x309f or 0x3400 <= cp <= 0x4dbf or 0x4e00 <= cp <= 0x9fff
            or 0xf900 <= cp <= 0xfaff or 0x20000 <= cp <= 0x3134f):
        return IDEO
    if cat[0] == 'L' or cat == 'Nl':
        return ALETTER
    return OTHER


_WORD = {ALETTER, NUMERIC, KATAKANA}


def _joins(prev, cur):
    """Whether two adjacent word characters stay in one token (WB5, WB8–10, WB13, WB13a/b)."""
    if cur in (ALETTER, NUMERIC):
        return prev in (ALETTER, NUMERIC, EXTEND_NUM_LET)
    if cur == KATAKANA:
        return prev in (KATAKANA, EXTEND_NUM_LET)
    if cur == EXTEND_NUM_LET:
        return prev in (ALETTER, NUMERIC, KATAKANA, EXTEND_NUM_LET)
    return False


def standard_tokenize(text):
    """Token list as ES `standard` would produce it (lowercased)."""
    if not text:
        return []
    chars = list(text)
    classes = [char_class(c) for c in chars]
    n = len(chars)
    tokens = []
    i = 0
    while i < n:
        c = classes[i]
        if c in (IDEO, EMOJI):
            j = i + 1
            while j < n and classes[j] == EXTEND:
                j += 1
            tokens.append(''.join(chars[i:j]))
            i = j
            continue
        if c not in _WORD and c != EXTEND_NUM_LET:
            i += 1
            continue
        last = c
        j = i + 1
        while j < n:
            cj = classes[j]
            if cj == EXTEND:
                j += 1
                continue
            if _joins(last, cj):
                last = cj
                j += 1
                continue
            if cj in (MID_LETTER, MID_NUM_LET, SINGLE_QUOTE, MID_NUM):
                k = j + 1
                while k < n and classes[k] == EXTEND:
                    k += 1
                if k < n:
                    ck = classes[k]
                    if last == ALETTER and ck == ALETTER and cj in (MID_LETTER, MID_NUM_LET, SINGLE_QUOTE):
                        last, j = ck, k + 1
                        continue
                    if last == NUMERIC and ck == NUMERIC and cj in (MID_NUM, MID_NUM_LET, SINGLE_QUOTE):
                        last, j = ck, k + 1
                        continue
            break
        word = ''.join(chars[i:j])
        if any(classes[p] in _WORD for p in range(i, j)):
            for s in range(0, len(word), MAX_TOKEN_LENGTH):
                tokens.append(word[s:s + MAX_TOKEN_LENGTH].lower())
        i = j
    return tokens
