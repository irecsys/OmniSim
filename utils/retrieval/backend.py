"""
LocalBackend — the in-memory local index plus the lookups the retrieval
functions need. One instance per index directory; safe to share between threads
(numpy arrays are read-only, each thread gets its own SQLite connection).
"""
import sqlite3
import threading

import numpy as np

from utils.retrieval.local_index import check_fts5, load_local_index


class LocalBackend:
    """An opened local index (index_dir): the item records in row order, the L2-normalised float32 vectors
    and the SQLite database used for BM25. Raises if SQLite has no FTS5. Passed as `backend` to the
    simulation and retrieval functions."""

    def __init__(self, index_dir):
        check_fts5()                                   # BM25 queries need FTS5
        idx = load_local_index(index_dir)
        self.index_dir = index_dir
        self.meta = idx['meta']
        self.items = idx['items']                     # list of dicts, _source types, row_idx order
        self.vectors = idx['vectors']                 # float32, L2-normalised
        self.vectors64 = self.vectors.astype(np.float64)
        self.db_path = idx['db_path']
        self.n = len(self.items)
        self.col_itemid = self.meta['col_itemid']
        self.columns = [c['name'] for c in self.meta['columns']]
        self.item_ids = [it[self.col_itemid] for it in self.items]
        self.item_id_str = [str(i) for i in self.item_ids]
        self._local = threading.local()
        self._tokens = {}
        self._lock = threading.Lock()

    def conn(self):
        """Per-thread read-only SQLite connection."""
        con = getattr(self._local, 'con', None)
        if con is None:
            con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._local.con = con
        return con

    def has_column(self, name):
        return name in self.columns

    def item_embedding(self, row):
        """Replacement point R2: the item vector used for the hybrid score, returned by the reference-title
        lookup and by sample_items (the verification tools inject v1's raw _source vectors here)."""
        return self.vectors[row].tolist()

    def field_tokens(self, field, tokenize):
        """Per row, tokenize(text) of a text field, cached per (field, tokenize). None if the field is absent."""
        if field not in self.columns:
            return None
        key = (field, tokenize)
        with self._lock:
            if key not in self._tokens:
                self._tokens[key] = [tokenize(it[field]) if isinstance(it[field], str) else []
                                     for it in self.items]
            return self._tokens[key]
