"""
ArrayColumn: a property column stored as one padded numpy array.

The AgentFactory keeps one column per property. By default a column is a Python
list with one entry per local agent, which is what per-agent construction
(`create_agent`) and genuinely ragged properties (variable-length per agent, like
neighbour lists) need. For a *rectangular* property -- every row has the same
shape, or is padded to it -- that list is only an expensive way of holding what
the GPU tensor holds anyway: a float32 rectangle padded with NaN. Bulk loaders
(`Model.build_from_local_columns`) may therefore pass a numpy array, and the
GPU->host sync stores what it downloads as one, so the first tick uploads the
column in one copy instead of walking millions of Python rows.

An ArrayColumn behaves like the list it replaces at every access the framework
performs (index/slice read and write, len, truthiness, iteration, append, `+`
with a list of rows, pickling). Per-row `lengths` record the width each row was
written with, so a read returns exactly what was stored (a copy, never a view
into the array) even though storage is padded. A write the array cannot hold
grows it; a write that is not array-like at all (ragged inner lists) degrades the
column to list storage inside the same object, so references held elsewhere stay
valid.
"""
from __future__ import annotations

import numpy as np


class ArrayColumn:
    """Padded array storage for a rectangular property column (see module doc)."""

    def __init__(self, values, lengths=None, dtype=None):
        values = np.asarray(values)
        if dtype is None:
            # Keep the caller's numeric dtype (a loader may hold float64 for exact
            # read-back, or float32 to halve host memory); the device copy is float32.
            dtype = values.dtype if values.dtype.kind in "iuf" else np.float32
        values = np.ascontiguousarray(values, dtype=dtype)
        if values.ndim == 0:
            raise ValueError("ArrayColumn needs at least one axis (agents)")
        self.values = values                      # shape (capacity, *row_shape)
        self._n = values.shape[0]                 # rows in use (capacity may be larger)
        if values.ndim == 1:
            lengths = np.ones(self._n, dtype=np.int32)
        elif lengths is None:
            lengths = np.full(self._n, values.shape[1], dtype=np.int32)
        else:
            lengths = np.asarray(lengths, dtype=np.int32)
            if lengths.shape != (self._n,):
                raise ValueError("lengths must have one entry per row")
        self.lengths = lengths
        self._rows = None                         # list storage once degraded

    # -- basic protocol ------------------------------------------------------

    @property
    def fill(self):
        return np.nan if self.values.dtype.kind == "f" else 0

    @property
    def ndim(self):
        return self.values.ndim

    @property
    def width(self):
        """Padded width along axis 1 (1 for scalar columns)."""
        return int(self.values.shape[1]) if self.values.ndim >= 2 else 1

    @property
    def degraded(self):
        return self._rows is not None

    def __len__(self):
        return len(self._rows) if self._rows is not None else self._n

    def __bool__(self):
        return len(self) > 0

    def __iter__(self):
        if self._rows is not None:
            yield from self._rows
        else:
            for i in range(self._n):
                yield self._row(i)

    def _row(self, i):
        v = self.values[i]
        if self.values.ndim == 1:
            return v.item()
        return v[: self.lengths[i]].tolist()

    def __getitem__(self, key):
        if self._rows is not None:
            return self._rows[key]
        if isinstance(key, slice):
            return [self._row(i) for i in range(*key.indices(self._n))]
        i = int(key)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("ArrayColumn index out of range")
        return self._row(i)

    def __setitem__(self, key, value):
        if self._rows is not None:
            self._rows[key] = value
            return
        if isinstance(key, slice):
            idx = range(*key.indices(self._n))
            if len(idx) != len(value):
                raise ValueError("slice assignment needs one row per position")
            try:
                arr = self._as_rows(value)
            except ValueError:
                self._degrade()
                self._rows[key] = list(value)
                return
            self._fit(arr.shape[1:])
            self.values[idx.start:idx.stop:idx.step] = self.fill
            self._write_block(idx, arr)
            return
        i = int(key)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("ArrayColumn index out of range")
        try:
            arr = self._as_row(value)
        except ValueError:
            self._degrade()
            self._rows[i] = value
            return
        self._fit(arr.shape)
        self.values[i] = self.fill
        self._write_row(i, arr)

    def append(self, value):
        if self._rows is not None:
            self._rows.append(value)
            return
        try:
            arr = self._as_row(value)
        except ValueError:
            self._degrade()
            self._rows.append(value)
            return
        if self._n == self.values.shape[0]:
            new_cap = max(16, 2 * self.values.shape[0])
            grown = np.full((new_cap,) + self.values.shape[1:], self.fill, dtype=self.values.dtype)
            grown[: self._n] = self.values[: self._n]
            self.values = grown
            self.lengths = np.concatenate(
                [self.lengths, np.zeros(new_cap - len(self.lengths), dtype=np.int32)])
        self._fit(arr.shape)
        self.values[self._n] = self.fill
        self._write_row(self._n, arr)
        self._n += 1

    def __add__(self, other):
        """`column + [rows]` (ghost placeholders): a new column with the rows appended."""
        other = list(other)
        if self._rows is not None:
            return self._rows + other
        if not other:
            return ArrayColumn(self.values[: self._n].copy(), self.lengths[: self._n].copy())
        out = ArrayColumn(self.values[: self._n].copy(), self.lengths[: self._n].copy())
        for row in other:
            out.append(row)
        return out

    def __eq__(self, other):
        return list(self) == list(other)

    def __array__(self, dtype=None, copy=None):
        if self._rows is not None:
            arr = np.asarray(self._rows)
        else:
            arr = self.values[: self._n]
        return arr.astype(dtype) if dtype is not None else arr

    def tolist(self):
        return list(self)

    def __repr__(self):
        state = "list" if self._rows is not None else f"{self.values.dtype}{self.values.shape[1:]}"
        return f"ArrayColumn(n={len(self)}, {state})"

    # -- bulk operations used by the framework -------------------------------

    def fill_rows(self, row):
        """Set every row to `row` (one broadcast write; `col[:] = [row] * n` on a list)."""
        if self._rows is not None:
            self._rows[:] = [row] * len(self._rows)
            return
        arr = self._as_row(row)
        self._fit(arr.shape)
        self.values[: self._n] = self.fill
        self.values[(slice(0, self._n),) + tuple(slice(0, s) for s in arr.shape)] = arr
        if self.values.ndim >= 2:
            self.lengths[: self._n] = arr.shape[0]

    def max_length(self):
        """Largest row length (axis 1), without materialising rows."""
        if self._rows is not None:
            return max(map(len, self._rows)) if self._rows else 0
        return int(self.lengths[: self._n].max()) if self._n else 0

    def permute(self, perm):
        """Reorder rows in place (sort_by_breed)."""
        perm = np.asarray(perm)
        if self._rows is not None:
            self._rows = [self._rows[p] for p in perm]
            return
        self.values[: self._n] = self.values[perm]
        self.lengths[: self._n] = self.lengths[perm]

    def pad_width(self, width, fill=0.0):
        """Widen axis 1 to `width` (MPI width sync; the framework pads with 0.0 there)."""
        if self._rows is not None or self.values.ndim < 2 or self.width >= width:
            return
        shape = list(self.values.shape)
        shape[1] = width
        grown = np.full(shape, fill, dtype=self.values.dtype)
        grown[:, : self.values.shape[1]] = self.values
        self.values = grown

    def try_intern(self, min_ratio=8):
        """An IndexedColumn with the same rows if at most 1/`min_ratio` of them are distinct,
        else None. Rows are compared by their exact padded bytes plus stored width (NaN-safe),
        via a vectorised row hash verified against the first occurrence of each hash."""
        if self._rows is not None or self.values.ndim != 2 or self._n == 0 or self.width < 2:
            return None
        vals = np.ascontiguousarray(self.values[: self._n])
        words = vals.view(np.uint32) if vals.dtype.itemsize == 4 else \
            vals.astype(np.float32).view(np.uint32)
        h = self.lengths[: self._n].astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
        for j in range(words.shape[1]):
            h ^= words[:, j].astype(np.uint64) * np.uint64(0xBF58476D1CE4E5B9 + 2 * j)
            h ^= h >> np.uint64(31)
        uniq, first, inverse = np.unique(h, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        if len(uniq) * min_ratio > self._n:
            return None
        # verify: every row equals the first row that hashed the same (no collisions)
        same = np.all(words == words[first][inverse], axis=1) & \
            (self.lengths[: self._n] == self.lengths[: self._n][first][inverse])
        if not bool(same.all()):
            return None
        return IndexedColumn(vals[first], inverse.astype(np.int32), self.lengths[: self._n][first])

    def to_device(self, capacity):
        """The padded device tensor: `capacity` rows, NaN (float) / 0 (int) beyond the data."""
        import cupy as cp
        if self._rows is not None:
            raise TypeError("degraded ArrayColumn must go through the list conversion path")
        if self.values.ndim == 1:
            out = cp.zeros(capacity, dtype=cp.float32)
        else:
            out = cp.full((capacity,) + self.values.shape[1:], cp.nan, dtype=cp.float32)
        # Copy host -> device straight into the leading rows (no device-side temporary).
        src = np.ascontiguousarray(self.values[: self._n], dtype=np.float32)
        if self._n:
            out[: self._n].set(src)
        return out

    # -- internals -----------------------------------------------------------

    def _as_row(self, value):
        arr = np.asarray(value, dtype=self.values.dtype)
        if arr.ndim != self.values.ndim - 1:
            raise ValueError("row depth does not match the column")
        return arr

    def _as_rows(self, value):
        arr = np.asarray(value, dtype=self.values.dtype)
        if arr.ndim != self.values.ndim:
            raise ValueError("rows depth does not match the column")
        return arr

    def _fit(self, row_shape):
        """Grow the trailing axes so a row of `row_shape` fits."""
        if self.values.ndim == 1:
            return
        cur = self.values.shape[1:]
        if all(r <= c for r, c in zip(row_shape, cur)):
            return
        new_shape = (self.values.shape[0],) + tuple(max(r, c) for r, c in zip(row_shape, cur))
        grown = np.full(new_shape, self.fill, dtype=self.values.dtype)
        grown[tuple(slice(0, s) for s in self.values.shape)] = self.values
        self.values = grown

    def _write_row(self, i, arr):
        if self.values.ndim == 1:
            self.values[i] = arr
            return
        self.values[(i,) + tuple(slice(0, s) for s in arr.shape)] = arr
        self.lengths[i] = arr.shape[0]

    def _write_block(self, idx, arr):
        sel = (slice(idx.start, idx.stop, idx.step),) + tuple(slice(0, s) for s in arr.shape[1:])
        self.values[sel] = arr
        if self.values.ndim >= 2:
            self.lengths[idx.start:idx.stop:idx.step] = arr.shape[1]

    def _degrade(self):
        self._rows = list(self)
        self.values = self.values[:0]
        self.lengths = self.lengths[:0]


class IndexedColumn:
    """A rectangular property column stored as a small table of distinct rows plus one
    int32 code per agent: row i is `table[codes[i]]`.

    The natural representation for per-type parameters (a few parameter sets shared by
    millions of agents): host memory is one int per agent, and when the framework
    decides the property is *interned* (no kernel writes it, see Model.setup) the device
    holds the same table + codes and kernels read `table[codes[i]]`, so a 12.5 M x 11
    float32 tensor becomes 12.5 M ints plus a handful of rows. Otherwise it is expanded
    to the dense padded tensor on upload, so results are identical either way.

    List protocol as ArrayColumn: reads return copies of the row at its stored width,
    writes look the row up in the table (appending a new distinct row when needed, growing
    the width when needed), so per-agent writes stay correct -- the table just grows.
    """

    def __init__(self, table, codes, lengths=None):
        table = np.asarray(table)
        if table.ndim != 2:
            raise ValueError("IndexedColumn table must be 2-D (distinct rows x width)")
        dtype = table.dtype if table.dtype.kind in "iuf" else np.float32
        self.table = np.ascontiguousarray(table, dtype=dtype)
        codes = np.asarray(codes, dtype=np.int32).ravel()
        if codes.size and (codes.min() < 0 or codes.max() >= len(self.table)):
            raise ValueError("codes must index rows of the table")
        self._codes = codes.copy()
        self._n = codes.size
        if lengths is None:
            lengths = np.full(len(self.table), self.table.shape[1], dtype=np.int32)
        self.lengths = np.asarray(lengths, dtype=np.int32).copy()
        if self.lengths.shape != (len(self.table),):
            raise ValueError("lengths must have one entry per table row")
        self._index = {self._key(k): k for k in range(len(self.table))}

    # -- protocol -------------------------------------------------------------

    @property
    def codes(self):
        return self._codes[: self._n]

    @property
    def fill(self):
        return np.nan if self.table.dtype.kind == "f" else 0

    @property
    def width(self):
        return int(self.table.shape[1])

    @property
    def ndim(self):
        return 2

    @property
    def degraded(self):
        return False

    @property
    def n_distinct(self):
        return int(len(self.table))

    def __len__(self):
        return self._n

    def __bool__(self):
        return self._n > 0

    def _row_of_code(self, k):
        return self.table[k, : self.lengths[k]].tolist()

    def __iter__(self):
        for i in range(self._n):
            yield self._row_of_code(self._codes[i])

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self._row_of_code(self._codes[i]) for i in range(*key.indices(self._n))]
        i = int(key)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("IndexedColumn index out of range")
        return self._row_of_code(self._codes[i])

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            idx = range(*key.indices(self._n))
            if len(idx) != len(value):
                raise ValueError("slice assignment needs one row per position")
            for i, row in zip(idx, value):
                self._codes[i] = self._code_for(row)
            return
        i = int(key)
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("IndexedColumn index out of range")
        self._codes[i] = self._code_for(value)

    def append(self, value):
        k = self._code_for(value)
        if self._n == len(self._codes):
            grown = np.zeros(max(16, 2 * len(self._codes)), dtype=np.int32)
            grown[: self._n] = self._codes[: self._n]
            self._codes = grown
        self._codes[self._n] = k
        self._n += 1

    def fill_rows(self, row):
        self._codes[: self._n] = self._code_for(row)

    def max_length(self):
        return int(self.lengths[np.unique(self.codes)].max()) if self._n else 0

    def permute(self, perm):
        self._codes[: self._n] = self._codes[: self._n][np.asarray(perm)]

    def __add__(self, other):
        out = IndexedColumn(self.table.copy(), self.codes.copy(), self.lengths.copy())
        for row in other:
            out.append(row)
        return out

    def __eq__(self, other):
        return list(self) == list(other)

    def __array__(self, dtype=None, copy=None):
        arr = self.table[self.codes]
        return arr.astype(dtype) if dtype is not None else arr

    def tolist(self):
        return list(self)

    def pad_width(self, width, fill=0.0):
        if self.width >= width:
            return
        grown = np.full((len(self.table), width), fill, dtype=self.table.dtype)
        grown[:, : self.width] = self.table
        self.table = grown
        self._index = {self._key(k): k for k in range(len(self.table))}

    def to_array_column(self):
        """Dense equivalent (same rows, same widths)."""
        return ArrayColumn(self.table[self.codes].copy(), self.lengths[self.codes].copy())

    def to_device(self, capacity):
        """Dense device tensor, `capacity` rows: table[codes] expanded on the device."""
        import cupy as cp
        table_gpu, codes_gpu = self.to_device_pair(capacity)
        out = cp.full((capacity, self.width), cp.nan, dtype=cp.float32)
        if self._n:
            cp.take(table_gpu, codes_gpu[: self._n], axis=0, out=out[: self._n])
        return out

    def to_device_pair(self, capacity):
        """(table float32 (k, w), codes int32 (capacity,)) for an interned property.
        Padding rows carry code 0 (a row that exists); kernels never read them."""
        import cupy as cp
        table_gpu = cp.asarray(np.ascontiguousarray(self.table, dtype=np.float32))
        codes_gpu = cp.zeros(capacity, dtype=cp.int32)
        if self._n:
            codes_gpu[: self._n].set(np.ascontiguousarray(self.codes))
        return table_gpu, codes_gpu

    # -- internals ------------------------------------------------------------

    def _key(self, k):
        return self.table[k, : self.lengths[k]].tobytes()

    def _code_for(self, row):
        arr = np.asarray(row, dtype=self.table.dtype).ravel()
        if arr.size > self.width:
            self.pad_width(arr.size, fill=self.fill)
        key = arr.tobytes()
        k = self._index.get(key)
        if k is None:
            k = len(self.table)
            new_row = np.full((1, self.width), self.fill, dtype=self.table.dtype)
            new_row[0, : arr.size] = arr
            self.table = np.concatenate([self.table, new_row])
            self.lengths = np.append(self.lengths, np.int32(arr.size))
            self._index[key] = k
        return k

    def __repr__(self):
        return f"IndexedColumn(n={self._n}, distinct={len(self.table)}, width={self.width})"
