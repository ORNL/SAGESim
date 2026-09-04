from typing import List, Any

import awkward as ak
import numpy as np
import cupy as cp


def convert_to_equal_side_tensor(ragged_list: List[Any]) -> cp.array:
    """
    Convert ragged list to equal-size padded tensor.
    Uses fast NumPy for depth 1-2, falls back to awkward for depth 3+.

    Optimization: If data is already padded (from previous tick), skip padding.
    """
    if not ragged_list:
        return cp.array([], dtype=np.float32)

    # Quick check: is data already padded? (all rows same length AND elements are scalars)
    # This happens after first tick when .tolist() keeps padded structure
    # Only works for depth 2 data (rows of scalars, not rows of lists)
    if isinstance(ragged_list[0], (list, tuple)):
        # Investigate row lengths - check ALL rows (not just sample)
        row_lengths = [len(row) if isinstance(row, (list, tuple)) else 1 for row in ragged_list]

        unique_lengths = set(row_lengths)
        first_len = len(ragged_list[0])
        all_same_len = len(unique_lengths) == 1

        if all_same_len:
            # Check if elements are scalars (not lists) - distinguishes depth 2 from depth 3
            if first_len > 0:
                is_scalar = not isinstance(ragged_list[0][0], (list, tuple))
                if is_scalar:
                    # Already padded depth-2 data! Just convert to GPU array
                    return cp.array(ragged_list, dtype=np.float32)

    # Detect depth using awkward
    awkward_array = ak.from_iter(ragged_list)
    min_depth, max_depth = awkward_array.layout.minmax_depth

    # Validate uniform depth
    assert min_depth == max_depth, "Tensor is of unequal depth"

    depth = max_depth

    # Use fast NumPy path for depth 1-2 (common cases)
    if depth <= 2:
        return _convert_numpy_fast(ragged_list, depth)

    # Fall back to awkward for depth 3+ (rare cases)
    else:
        return _convert_awkward(awkward_array, depth)


def _convert_numpy_fast(ragged_list: List[Any], depth: int) -> cp.array:
    """Fast NumPy-based conversion for depth 1-2."""

    # Depth 1: Simple 1D array (scalars)
    if depth == 1:
        return cp.array(ragged_list, dtype=np.float32)

    # Depth 2: 2D ragged array [[1,2], [3], [4,5,6]] or [{1,2}, {3}, {4,5,6}]
    elif depth == 2:
        # Handle empty sublists and convert sets to lists
        max_len = max((len(row) if isinstance(row, (list, tuple, set)) else 1 for row in ragged_list), default=0)

        if max_len == 0:
            return cp.full((len(ragged_list), 0), np.nan, dtype=np.float32)

        result = np.full((len(ragged_list), max_len), np.nan, dtype=np.float32)

        for i, row in enumerate(ragged_list):
            if isinstance(row, (list, tuple, set)) and len(row) > 0:
                # Convert sets to lists for indexing
                row_data = list(row) if isinstance(row, set) else row
                result[i, :len(row_data)] = row_data
            elif not isinstance(row, (list, tuple, set)):
                result[i, 0] = row

        return cp.array(result)


def _convert_awkward(awkward_array, depth: int) -> cp.array:
    """Awkward-based conversion for depth 3+."""
    i = 1
    while i < depth:
        awkward_array = ak.fill_none(awkward_array, [], axis=i - 1)
        nums_in_level = ak.fill_none(ak.ravel(ak.num(awkward_array, axis=i)), value=0)
        awkward_array = ak.pad_none(
            awkward_array, int(max(nums_in_level)), axis=i, clip=True
        )
        i += 1

    awkward_array = ak.fill_none(awkward_array, np.nan, axis=-1)

    return ak.to_cupy(awkward_array).astype(np.float32)


def _detect_depth(sample):
    """Detect nesting depth by probing first element."""
    depth = 1
    current = sample
    while isinstance(current, (list, tuple, set)):
        depth += 1
        if len(current) == 0:
            return depth
        current = next(iter(current))
    return depth


def build_csr_from_ragged(ragged_list: List[Any]):
    """
    Convert ragged list of neighbor lists to CSR (Compressed Sparse Row) format.

    CSR uses two arrays instead of padding ragged lists to rectangular:
      - offsets: array of length (num_agents + 1), where agent i's neighbors
                 are values[offsets[i] : offsets[i+1]]
      - values:  flat array of all neighbor entries concatenated

    Input:  [[5, 2], [8, 3, 1], [], [7, 4, 9, 6]]
    Output: offsets = np.array([0, 2, 5, 5, 9], dtype=np.int32)
            values  = np.array([5, 2, 8, 3, 1, 7, 4, 9, 6], dtype=np.int32)

    Handles sets (unordered), lists (ordered), numpy arrays, and empty entries.

    :param ragged_list: List of lists/sets/arrays of neighbor IDs or indices
    :return: (offsets, values) as numpy int32 arrays
    """
    if not ragged_list:
        return np.array([0], dtype=np.int32), np.array([], dtype=np.int64)

    # Compute offsets from lengths
    lengths = []
    for row in ragged_list:
        if isinstance(row, (list, tuple, set, np.ndarray)):
            lengths.append(len(row))
        else:
            lengths.append(0)

    offsets = np.empty(len(lengths) + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])

    total_entries = offsets[-1]
    values = np.empty(total_entries, dtype=np.int64)

    # Fill values array
    pos = 0
    for row in ragged_list:
        if isinstance(row, np.ndarray):
            n = len(row)
            if n > 0:
                values[pos:pos + n] = row.astype(np.int64)
            pos += n
        elif isinstance(row, set):
            for val in row:
                values[pos] = int(val)
                pos += 1
        elif isinstance(row, (list, tuple)):
            for val in row:
                values[pos] = int(val)
                pos += 1

    return offsets, values


def build_csr_values_only(ragged_list, offsets):
    """Build CSR values reusing pre-computed offsets."""
    total_entries = offsets[-1]
    values = np.empty(total_entries, dtype=np.int32)
    pos = 0
    for row in ragged_list:
        if isinstance(row, np.ndarray):
            n = len(row)
            if n > 0:
                values[pos:pos + n] = row.astype(np.int32)
            pos += n
        elif isinstance(row, set):
            for val in row:
                values[pos] = int(val)
                pos += 1
        elif isinstance(row, (list, tuple)):
            for val in row:
                values[pos] = int(val)
                pos += 1
    return values


# A padded property tensor is (capacity x width) float32. A width in the
# thousands over a capacity of tens of millions is almost always a bug — e.g. a
# shared/deduped column object mutated in place, so one row's length inflates the
# whole column. Warn before the allocation so a would-be silent OOM prints its
# shape first. Threshold ~8 GB (2e9 float32 elements).
_PADDED_TENSOR_WARN_ELEMS = 2_000_000_000


def _warn_if_huge_padded(capacity, width):
    if capacity * width > _PADDED_TENSOR_WARN_ELEMS:
        import sys
        print(
            f"[SAGESim][warn] convert_to_padded_gpu_tensor allocating a "
            f"({capacity} x {width}) float32 tensor "
            f"(~{capacity * width * 4 / 1e9:.1f} GB) — a width this large usually "
            f"means a shared/deduped property column was mutated in place.",
            file=sys.stderr, flush=True,
        )


# Set by convert_to_padded_gpu_tensor to name the branch the last call took.
# Read by the GPU buffer manager so per-property timings can be attributed to a
# code path: "depth3_awkward" is the slow one (ak.from_iter walks every row).
LAST_CONVERSION_PATH = None


# A shared/deduplicated column is built as [row_a] * n + [row_b] * m, so a strided
# sample of object ids tells us whether a full scan is worth running. It matters:
# `map(id, column)` over genuinely distinct rows chases scattered heap objects at
# ~0.66 us/row, about 8 s per column at 12.5M rows, and the record construction path
# has no shared rows at all. The sample costs microseconds at any column size.
_DEDUP_SAMPLE = 64
# Cap the table two ways: an absolute ceiling, and a compression ratio, so the gather
# is only used when it actually replaces a lot of per-row work. A Poisson-driven
# input_spikes_tensor is the motivating case: ~12.5k distinct rows among 12.5M still
# compresses 1000:1 and converts in a fraction of the per-row path.
_DEDUP_MAX_DISTINCT = 1 << 17
_DEDUP_MIN_RATIO = 8


def _identity_groups(ragged_list):
    """Distinct rows and an inverse index, when a column is built from a few shared
    row objects. Returns None when deduplicating would not pay.

    Only rows that are list/tuple are considered: depth-1 scalars may be interned,
    and sets have no stable order (the existing uniform fast path skips them too).
    """
    n = len(ragged_list)
    if n <= _DEDUP_SAMPLE or not isinstance(ragged_list[0], (list, tuple)):
        return None

    step = max(1, n // _DEDUP_SAMPLE)
    sampled = [ragged_list[i] for i in range(0, n, step)]
    if len({id(r) for r in sampled}) * 4 > len(sampled):
        return None  # mostly distinct; the full scan would not pay for itself
    if not all(isinstance(r, (list, tuple)) for r in sampled):
        return None

    ids = np.fromiter(map(id, ragged_list), np.int64, n)
    # return_index gives the FIRST occurrence of each id, and np.unique sorts by id
    # value, so the table must be built in that same order for `inverse` to index it.
    uniq, first_idx, inverse = np.unique(ids, return_index=True, return_inverse=True)
    if len(uniq) > _DEDUP_MAX_DISTINCT or len(uniq) * _DEDUP_MIN_RATIO > n:
        return None

    rows = [ragged_list[int(i)] for i in first_idx]
    if not all(isinstance(r, (list, tuple)) for r in rows):
        return None
    return rows, inverse


def convert_to_padded_gpu_tensor(ragged_list, capacity):
    """Convert ragged list directly to padded GPU tensor (single allocation)."""
    global LAST_CONVERSION_PATH
    LAST_CONVERSION_PATH = None
    if not ragged_list:
        LAST_CONVERSION_PATH = "empty"
        return cp.zeros(capacity, dtype=np.float32)

    # Fast path: the column is a handful of shared row objects repeated. Pad only the
    # distinct rows and gather on the device, replacing one Python statement per row
    # with a single take. The recursion converts the small table through the branches
    # below, so padding, dtype and depth handling stay byte-identical.
    groups = _identity_groups(ragged_list)
    if groups is not None:
        rows, inverse = groups
        table = convert_to_padded_gpu_tensor(rows, len(rows))
        LAST_CONVERSION_PATH = "identity_dedup"
        n = len(ragged_list)
        _warn_if_huge_padded(capacity, int(np.prod(table.shape[1:])) or 1)
        fill = cp.nan if table.ndim >= 2 else 0
        out = cp.full((capacity,) + table.shape[1:], fill, dtype=table.dtype)
        cp.take(table, cp.asarray(inverse.astype(np.int32)), axis=0, out=out[:n])
        return out

    # Fast path: already-padded depth-2
    if isinstance(ragged_list[0], (list, tuple)):
        row_lengths = [len(r) if isinstance(r, (list, tuple)) else 1 for r in ragged_list]
        if len(set(row_lengths)) == 1:
            first_len = len(ragged_list[0])
            if first_len > 0 and not isinstance(ragged_list[0][0], (list, tuple)):
                LAST_CONVERSION_PATH = "depth2_uniform"
                _warn_if_huge_padded(capacity, first_len)
                result = np.full((capacity, first_len), np.nan, dtype=np.float32)
                result[:len(ragged_list)] = ragged_list
                return cp.array(result)

    depth = _detect_depth(ragged_list[0])

    if depth == 1:
        LAST_CONVERSION_PATH = "depth1"
        result = np.zeros(capacity, dtype=np.float32)
        result[:len(ragged_list)] = np.array(ragged_list, dtype=np.float32)
        return cp.array(result)
    elif depth == 2:
        LAST_CONVERSION_PATH = "depth2_ragged"
        max_len = max((len(r) if isinstance(r, (list, tuple, set)) else 1
                       for r in ragged_list), default=0)
        if max_len == 0:
            return cp.full((capacity, 0), np.nan, dtype=np.float32)
        _warn_if_huge_padded(capacity, max_len)
        result = np.full((capacity, max_len), np.nan, dtype=np.float32)
        for i, row in enumerate(ragged_list):
            if isinstance(row, (list, tuple, set)) and len(row) > 0:
                row_data = list(row) if isinstance(row, set) else row
                result[i, :len(row_data)] = row_data
            elif not isinstance(row, (list, tuple, set)):
                result[i, 0] = row
        return cp.array(result)
    else:
        # Depth 3+: fall back to awkward, pad on GPU
        LAST_CONVERSION_PATH = "depth3_awkward"
        awkward_array = ak.from_iter(ragged_list)
        tensor = _convert_awkward(awkward_array, depth)
        padded_shape = (capacity,) + tensor.shape[1:]
        padded = cp.full(padded_shape, cp.nan, dtype=tensor.dtype)
        padded[:tensor.shape[0]] = tensor
        return padded


def compress_tensor(regular_tensor: cp.array, min_axis: int = 1) -> List[Any]:
    awkward_tensor = ak.from_cupy(regular_tensor)
    awkward_tensor = ak.nan_to_none(awkward_tensor)
    awkward_tensor = ak.drop_none(awkward_tensor)

    i = -1
    while awkward_tensor.layout.minmax_depth[0] + i > min_axis:
        awkward_tensor = ak.mask(awkward_tensor, ak.num(awkward_tensor, axis=i) > 0)
        awkward_tensor = ak.drop_none(awkward_tensor)
        i -= 1

    return ak.to_list(awkward_tensor)
