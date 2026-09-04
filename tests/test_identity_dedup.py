"""The identity fast path in convert_to_padded_gpu_tensor must be invisible.

A column built from shared row objects must convert to exactly the tensor the
per-row path produces, and the resulting device rows must be independent, so a
kernel writing one agent's state can never touch another's.
"""

import cupy as cp
import numpy as np
import pytest

from sagesim import internal_utils
from sagesim.internal_utils import convert_to_padded_gpu_tensor


def _both_ways(row_a, row_b, n_a, n_b, capacity):
    """The same column built with shared objects, and with every row copied."""
    shared = [row_a] * n_a + [row_b] * n_b
    copied = [list(row_a) for _ in range(n_a)] + [list(row_b) for _ in range(n_b)]
    return shared, copied


@pytest.mark.parametrize(
    "row_a, row_b",
    [
        ([1.0, 2.0, 3.0], [7.0, 8.0]),          # depth 2, ragged widths
        ([1.0, 2.0], [7.0, 8.0]),               # depth 2, equal widths
        ([[1.0, 2.0, 3.0]], [[7.0, 8.0, 9.0]]), # depth 3, the tracking-off buffer shape
    ],
)
def test_dedup_matches_per_row_conversion(row_a, row_b):
    n_a, n_b, capacity = 200, 400, 700
    shared, copied = _both_ways(row_a, row_b, n_a, n_b, capacity)

    fast = convert_to_padded_gpu_tensor(shared, capacity)
    assert internal_utils.LAST_CONVERSION_PATH == "identity_dedup"
    slow = convert_to_padded_gpu_tensor(copied, capacity)
    assert internal_utils.LAST_CONVERSION_PATH != "identity_dedup"

    assert fast.shape == slow.shape
    assert fast.dtype == slow.dtype
    # Byte-identical including the NaN padding inside short rows and the unused
    # capacity tail beyond n_a + n_b.
    assert cp.array_equal(fast, slow, equal_nan=True)


def test_dedup_rows_are_independent_on_device():
    """A shared host row must never alias GPU memory: that is what would break
    per-agent state such as a membrane potential or an STDP weight."""
    row = [7.0, 8.0]
    column = [row] * 200                   # above _DEDUP_SAMPLE so the fast path runs
    out = convert_to_padded_gpu_tensor(column, 256)
    assert internal_utils.LAST_CONVERSION_PATH == "identity_dedup"

    out[2, 0] = 999.0                      # what an STDP kernel does to one synapse
    assert float(out[2, 0]) == 999.0
    for other in (0, 1, 3, 4, 199):
        assert float(out[other, 0]) == 7.0
    assert row == [7.0, 8.0]               # and the host row is untouched


def test_distinct_column_skips_the_fast_path():
    """The record construction path has no shared rows; it must not pay for a scan."""
    column = [[float(i), float(i + 1)] for i in range(500)]
    out = convert_to_padded_gpu_tensor(column, 600)
    assert internal_utils.LAST_CONVERSION_PATH != "identity_dedup"
    assert out.shape == (600, 2)
    assert float(out[499, 0]) == 499.0
