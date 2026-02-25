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
        return np.array([0], dtype=np.int32), np.array([], dtype=np.int32)

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
    values = np.empty(total_entries, dtype=np.int32)

    # Fill values array
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

    return offsets, values


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
