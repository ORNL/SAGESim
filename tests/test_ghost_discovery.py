"""
discover_ghost_topology must (a) return nothing on a single rank without scanning, and
(b) on several ranks return exactly the ids owned elsewhere, whether the local ids are
supplied (vectorised membership, dict lookups only for the remote boundary) or not.
"""
import numpy as np

from sagesim.gpu_kernels import discover_ghost_topology, _resolve_neighbor_ranks


def _reference_ghosts(all_neighbors, agent2rank, my_rank):
    flat = np.concatenate([np.asarray(a, dtype=np.float64) for a in all_neighbors])
    ids = flat[~np.isnan(flat) & (flat >= 0)].astype(np.int64)
    ghosts = {int(i) for i in ids if agent2rank.get(int(i), -1) not in (-1, my_rank)}
    return np.array(sorted(ghosts), dtype=np.int64)


def _fixture(seed=0, n_agents=2000, n_ranks=3, my_rank=1):
    rng = np.random.default_rng(seed)
    agent2rank = {i: i % n_ranks for i in range(n_agents)}
    local_ids = np.array([i for i in range(n_agents) if agent2rank[i] == my_rank], dtype=np.int64)
    rows = []
    for _ in range(400):
        k = rng.integers(0, 8)
        row = rng.integers(0, n_agents + 50, size=k).astype(np.float64)   # some ids nobody owns
        row[rng.random(k) < 0.1] = -1.0                                    # external-input sentinel
        row = np.concatenate([row, np.full(rng.integers(0, 3), np.nan)])  # NaN padding
        rows.append(row)
    return rows, agent2rank, local_ids, my_rank


def test_single_rank_has_no_ghosts_without_scanning():
    rows, agent2rank, local_ids, my_rank = _fixture()
    # A dict that raises on any lookup proves the scan is skipped.
    class Untouchable(dict):
        def get(self, *a, **k):
            raise AssertionError("agent2rank was consulted on a single rank")
    out = discover_ghost_topology(rows, Untouchable(), 0, num_workers=1, local_ids=local_ids)
    assert out.dtype == np.int64 and out.size == 0


def test_multi_rank_matches_reference_with_and_without_local_ids():
    rows, agent2rank, local_ids, my_rank = _fixture()
    ref = _reference_ghosts(rows, agent2rank, my_rank)
    assert ref.size > 0
    a = discover_ghost_topology(rows, agent2rank, my_rank, num_workers=3, local_ids=local_ids)
    b = discover_ghost_topology(rows, agent2rank, my_rank, num_workers=3)
    c = discover_ghost_topology(rows, agent2rank, my_rank)          # legacy call shape
    for out in (a, b, c):
        np.testing.assert_array_equal(out, ref)
    # no local id may ever be reported as a ghost
    assert not np.isin(a, local_ids).any()


def test_resolve_ranks_only_consults_dict_for_remote_ids():
    rows, agent2rank, local_ids, my_rank = _fixture()
    ids = np.concatenate([np.asarray(r) for r in rows])
    ids = ids[~np.isnan(ids) & (ids >= 0)].astype(np.int64)
    consulted = []
    class Counting(dict):
        def get(self, k, d=-1):
            consulted.append(k); return super().get(k, d)
    ranks = _resolve_neighbor_ranks(ids, Counting(agent2rank), local_ids, my_rank)
    expected = np.array([agent2rank.get(int(i), -1) for i in ids], dtype=np.int32)
    np.testing.assert_array_equal(ranks, expected)
    assert not (set(consulted) & set(local_ids.tolist())), "local ids went through the dict"


def test_single_flat_csr_row_is_not_copied():
    flat = np.array([1, 2, -1, 3], dtype=np.int64)
    agent2rank = {1: 0, 2: 1, 3: 1}
    out = discover_ghost_topology([flat], agent2rank, 0, num_workers=2, local_ids=np.array([1]))
    np.testing.assert_array_equal(out, np.array([2, 3]))
