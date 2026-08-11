"""
Regression test for the int32 agent-ID overflow.

Global agent IDs can exceed int32 max (2**31 - 1 = 2,147,483,647) at scale
(e.g. 64 workers x ~50M synapse-agents/worker -> ~3.2e9). Before the int64 fix,
convert_agent_ids_to_indices cast the ID keys to np.int32 and raised
    OverflowError: Python integer 3201162500 out of bounds for int32
and build_csr_from_ragged silently truncated large IDs. These tests use IDs above
2**31 to guard both paths. Pure numpy -- no GPU / MPI needed at run time.
"""

import unittest

import numpy as np

from sagesim.model import convert_agent_ids_to_indices, convert_agent_indices_to_ids
from sagesim.internal_utils import build_csr_from_ragged

BIG = 3_000_000_000  # > 2**31 - 1 = 2_147_483_647


class TestLargeAgentIds(unittest.TestCase):
    def test_forward_array_dense(self):
        # Dense-lookup branch (id span ~ number of ids). Pre-fix: OverflowError at model.py:48.
        id_map = {BIG: 0, BIG + 1: 1, BIG + 2: 2}
        row = np.array([BIG, BIG + 2, -1, BIG + 1], dtype=np.float64)
        out = convert_agent_ids_to_indices([row], id_map, return_arrays=True)
        self.assertEqual(out[0].tolist(), [0, 2, -1, 1])

    def test_forward_array_sparse(self):
        # Widely separated ids force the sparse searchsorted branch (use_dense == False).
        id_map = {BIG: 0, BIG + 1_000_000_000: 1}
        row = np.array([BIG, BIG + 1_000_000_000, -1], dtype=np.float64)
        out = convert_agent_ids_to_indices([row], id_map, return_arrays=True)
        self.assertEqual(out[0].tolist(), [0, 1, -1])

    def test_forward_list_and_scalar(self):
        id_map = {BIG: 0, BIG + 1: 1, BIG + 2: 2}
        out_list = convert_agent_ids_to_indices([[BIG, BIG + 1]], id_map)
        self.assertEqual(out_list[0], [0, 1])
        out_scalar = convert_agent_ids_to_indices([BIG + 2], id_map)
        self.assertEqual(out_scalar[0], 2)

    def test_reverse_roundtrip(self):
        # index -> global id restoration (used before MPI send). id_array must be int64.
        idx_to_id = [BIG, BIG + 1, BIG + 2]
        back = convert_agent_indices_to_ids(
            np.array([[0, 2, -1, 1]], dtype=np.float64), idx_to_id
        )
        self.assertEqual([int(x) for x in back[0]], [BIG, BIG + 2, -1, BIG + 1])

    def test_csr_from_ragged_preserves_large_ids(self):
        # build_csr_from_ragged builds the agent-ID CSR; values must not truncate.
        offsets, values = build_csr_from_ragged([[BIG + 1, BIG + 2], [BIG + 2], []])
        self.assertEqual(values.dtype, np.int64)
        self.assertEqual(offsets.tolist(), [0, 2, 3, 3])
        self.assertEqual(values.tolist(), [BIG + 1, BIG + 2, BIG + 2])


if __name__ == "__main__":
    unittest.main()
