"""
query_text → FTS5 MATCH expression for the BM25 step (v1: ES `match` on bm25_details, operator OR).

The text is split as FTS5's default `unicode61` tokenizer splits it: runs of token characters,
Unicode categories L*, N* and Co. (With another index tokenizer, e.g. trigram, each quoted word is
re-tokenized by FTS5 itself.) Every token is wrapped in double quotes,
so FTS5 operators and syntax characters in the query (AND OR NOT NEAR * : ^ - + ( ) ") are taken
literally, and the tokens are joined with OR. Tokens are NOT de-duplicated: Lucene's `match` scores a
repeated token once per occurrence, and so does an FTS5 OR of repeated phrases.

Returns None when the text has no token (then BM25 is skipped: no hits, all normalised scores 0).
"""
import unicodedata


def _is_token_char(ch):
    cat = unicodedata.category(ch)
    return cat[0] in ('L', 'N') or cat == 'Co'


def fts5_tokens(text):
    """Tokens of text under unicode61's default token-character classes (case not folded)."""
    tokens, cur = [], []
    for ch in text or '':
        if _is_token_char(ch):
            cur.append(ch)
        elif cur:
            tokens.append(''.join(cur))
            cur = []
    if cur:
        tokens.append(''.join(cur))
    return tokens


def to_fts5_query(text):
    """FTS5 MATCH expression: '"tok1" OR "tok2" ...', or None if the text has no token."""
    tokens = fts5_tokens(text)
    if not tokens:
        return None
    return ' OR '.join('"' + t.replace('"', '""') + '"' for t in tokens)
