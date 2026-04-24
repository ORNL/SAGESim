import unittest
import numpy as np
from numpy.testing import assert_array_equal

from sagesim.internal_utils import build_csr_from_ragged, build_csr_values_only


class TestCSR(unittest.TestCase):

    def test_build_csr_from_lists(self):
        ragged = [[5, 2], [8, 3, 1], [], [7, 4, 9, 6]]
        offsets, values = build_csr_from_ragged(ragged)
        assert_array_equal(offsets, [0, 2, 5, 5, 9])
        assert_array_equal(values, [5, 2, 8, 3, 1, 7, 4, 9, 6])

    def test_build_csr_from_sets(self):
        ragged = [{5, 2}, {8}]
        offsets, values = build_csr_from_ragged(ragged)
        assert_array_equal(offsets, [0, 2, 3])
        # Sets are unordered, so just check the right elements are present
        self.assertEqual(set(values[:2]), {5, 2})
        self.assertEqual(values[2], 8)

    def test_build_csr_empty(self):
        offsets, values = build_csr_from_ragged([])
        assert_array_equal(offsets, [0])
        self.assertEqual(len(values), 0)

    def test_build_csr_single_agent(self):
        ragged = [[1, 2, 3]]
        offsets, values = build_csr_from_ragged(ragged)
        assert_array_equal(offsets, [0, 3])
        assert_array_equal(values, [1, 2, 3])

    def test_build_csr_mixed_empty(self):
        ragged = [[1], [], [2, 3]]
        offsets, values = build_csr_from_ragged(ragged)
        assert_array_equal(offsets, [0, 1, 1, 3])
        assert_array_equal(values, [1, 2, 3])

    def test_build_csr_values_only(self):
        ragged = [[10, 20], [30]]
        offsets, _ = build_csr_from_ragged(ragged)
        # Rebuild values with updated data
        updated = [[100, 200], [300]]
        new_values = build_csr_values_only(updated, offsets)
        assert_array_equal(new_values, [100, 200, 300])


if __name__ == "__main__":
    unittest.main()
