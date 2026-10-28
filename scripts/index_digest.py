"""
index_digest.py — a digest of what a local index contains, independent of the SQLite file bytes.

items.db is not byte-reproducible across SQLite versions (the file header records the writer's version,
and page layout can differ), so CI compares this instead:

  items      sha256 of every row of the `items` table, ordered by row_idx (values as JSON)
  fts        FTS5 'integrity-check' passes, and sha256 of the FTS5 vocabulary (term, documents, occurrences)
  vectors    sha256 of vectors/text.npy (the file itself is byte-reproducible)

Usage:
    python scripts/index_digest.py index/imdb index/hm                         # print
    python scripts/index_digest.py --check tests/ci/index_digest.txt index/imdb index/hm
"""
import argparse
import hashlib
import json
import os
import sqlite3
import sys


def digest(index_dir):
    con = sqlite3.connect(f"file:{os.path.join(index_dir, 'items.db')}?mode=ro", uri=True)
    try:
        h = hashlib.sha256()
        for row in con.execute('SELECT * FROM items ORDER BY row_idx'):
            h.update(json.dumps(row, ensure_ascii=False).encode('utf-8') + b'\n')
        items = h.hexdigest()
    finally:
        con.close()
    con = sqlite3.connect(os.path.join(index_dir, 'items.db'))   # integrity-check and the vocab table need a writable handle
    try:
        con.execute("INSERT INTO items_fts(items_fts) VALUES('integrity-check')")
        con.execute("CREATE VIRTUAL TABLE temp.v USING fts5vocab(main, 'items_fts', 'row')")
        h = hashlib.sha256()
        for row in con.execute('SELECT term, doc, cnt FROM temp.v ORDER BY term'):
            h.update(json.dumps(row, ensure_ascii=False).encode('utf-8') + b'\n')
        fts = 'integrity-check ok, vocab ' + h.hexdigest()
    finally:
        con.close()
    vectors = hashlib.sha256(open(os.path.join(index_dir, 'vectors', 'text.npy'), 'rb').read()).hexdigest()
    return [f"{index_dir} items {items}", f"{index_dir} fts {fts}", f"{index_dir} vectors {vectors}"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('index_dirs', nargs='+')
    parser.add_argument('--check', default=None, help='expected digest file; exit 1 on any difference')
    args = parser.parse_args()
    lines = [l for d in args.index_dirs for l in digest(d)]
    print('\n'.join(lines))
    if args.check:
        expected = open(args.check, encoding='utf-8').read().split('\n')
        expected = [l for l in expected if l.strip()]
        if lines != expected:
            print('FAIL: index digest differs from', args.check)
            for a, b in zip(expected, lines):
                if a != b:
                    print(f"  expected {a}\n  got      {b}")
            return 1
        print('OK: index digest matches', args.check)
    return 0


if __name__ == '__main__':
    sys.exit(main())
