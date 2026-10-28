"""Array-backed id -> row map for agents held in row order.

A bulk-built rank holds its agents' ids as an array in row order; a Python dict over them
costs ~1 GB and ~1-3.5 s at 12.7 M agents and is turned back into arrays by the first
simulate(). ``AgentIndex`` keeps the array and answers the same questions.
"""
from collections.abc import Mapping

import numpy as np


class AgentIndex(Mapping):
    """Read-only id -> row map over agent ids held in row order, backed by arrays.

    The row of an id is its position in ``ids``. Lookups go through a dense offset table
    when the id range is compact (span at most 3x the count, the rule
    ``convert_agent_ids_to_indices`` uses) and through the sorted ids otherwise. Measured
    at 12.7 M ids: built in 0.12 s (compact; a dict took 0.86-3.6 s), 0.28 GB against
    1.08 GB, a scalar ``get`` 270 ns (compact; 2.1 us sparse) against a dict's ~800 ns at
    that size. ``rows(ids)`` resolves many ids at once (32-150 ns per id); per-id Python
    loops over many agents should call it (see ``rows_of``). Duplicate ids are rejected.
    """

    __slots__ = ("ids", "_base", "_table", "_sorted_ids", "_sorted_rows")

    def __init__(self, ids):
        ids = np.ascontiguousarray(ids, dtype=np.int64)
        self.ids = ids
        n = len(ids)
        self._table = self._sorted_ids = self._sorted_rows = None
        self._base = 0
        if n == 0:
            self._table = np.zeros(0, dtype=np.int64)
            return
        lo, hi = int(ids.min()), int(ids.max())
        span = hi - lo + 1
        if span <= 3 * n:
            table = np.full(span, -1, dtype=np.int64)
            table[ids - lo] = np.arange(n, dtype=np.int64)
            if np.count_nonzero(table >= 0) != n:
                raise ValueError("AgentIndex: duplicate agent ids")
            self._base, self._table = lo, table
        else:
            order = np.argsort(ids, kind="stable")
            sorted_ids = ids[order]
            if n > 1 and not np.all(sorted_ids[1:] != sorted_ids[:-1]):
                raise ValueError("AgentIndex: duplicate agent ids")
            self._sorted_ids, self._sorted_rows = sorted_ids, order

    def rows(self, query, default=-1):
        """Row of each id in ``query`` (an int64 array), ``default`` where not held."""
        q = np.asarray(query, dtype=np.int64)
        out = np.full(q.shape, default, dtype=np.int64)
        if len(self.ids) == 0 or q.size == 0:
            return out
        if self._table is not None:
            k = q - self._base
            ok = (k >= 0) & (k < len(self._table))
            hit = self._table[k[ok]]
            sub = out[ok]
            sub[hit >= 0] = hit[hit >= 0]
            out[ok] = sub
        else:
            pos = np.minimum(np.searchsorted(self._sorted_ids, q), len(self._sorted_ids) - 1)
            found = self._sorted_ids[pos] == q
            out[found] = self._sorted_rows[pos[found]]
        return out

    def get(self, key, default=None):
        try:
            k = int(key)
        except (TypeError, ValueError):
            return default
        if self._table is not None:
            i = k - self._base
            if 0 <= i < len(self._table):
                r = int(self._table[i])
                if r >= 0:
                    return r
            return default
        pos = int(np.searchsorted(self._sorted_ids, k))
        if pos < len(self._sorted_ids) and int(self._sorted_ids[pos]) == k:
            return int(self._sorted_rows[pos])
        return default

    def __getitem__(self, key):
        r = self.get(key)
        if r is None:
            raise KeyError(key)
        return r

    def __contains__(self, key):
        return self.get(key) is not None

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        return iter(self.ids.tolist())

    # Mapping's default views would call __getitem__ per id; these read the arrays.
    def keys(self):
        return self.ids.tolist()

    def values(self):
        return range(len(self.ids))

    def items(self):
        return zip(self.ids.tolist(), range(len(self.ids)))


def rows_of(mapping, ids, default=-1):
    """Row of each id in ``ids`` (array, list or set) through an id -> row ``mapping``
    (an AgentIndex or a dict), as an int64 array in the order of ``ids`` with ``default``
    where the id is not held."""
    if isinstance(ids, (set, frozenset)):
        flat = np.fromiter(ids, dtype=np.int64, count=len(ids))
    else:
        flat = np.asarray(ids, dtype=np.int64).ravel()
    if isinstance(mapping, AgentIndex):
        return mapping.rows(flat, default)
    return np.fromiter((mapping.get(i, default) for i in flat.tolist()),
                       dtype=np.int64, count=len(flat))
