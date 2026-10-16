"""plan_exchange_layout: the field-major buffer one gather per field fills and the
peer-major buffer one contiguous message per peer needs are the same elements in two
orders, and the planned permutation maps one onto the other for any mix of widths and
per-peer counts, including peers with nothing for some field."""
import numpy as np
import pytest

from sagesim.gpu_kernels import plan_exchange_layout


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_permutation_maps_field_major_onto_peer_major(seed):
    rng = np.random.default_rng(seed)
    peers = [3, 0, 7]                                   # message order, not sorted on purpose
    widths = [2, 1, 3]
    counts = [{r: int(rng.integers(0, 5)) for r in peers} for _ in widths]
    counts[1][0] = 0                                     # a peer with nothing for one field
    fields = list(zip(widths, counts))
    total, field_off, peer_off, peer_size, perm = plan_exchange_layout(fields, peers)

    assert total == sum(w * sum(c.values()) for w, c in fields) == perm.size
    assert sorted(perm.tolist()) == list(range(total))   # a permutation

    # Field-major: field f holds its rows for every peer, peers in message order
    A = np.empty(total, dtype=np.float64)
    for f, (w, c) in enumerate(fields):
        off, n_rows = field_off[f]
        assert n_rows == sum(c.values())
        pos = off
        for r in peers:
            for k in range(c[r]):
                A[pos:pos + w] = 1000 * f + 10 * r + k + np.arange(w) / 10.0
                pos += w
    B = A[perm]
    # Peer-major: peer r's block holds fields in field order, each with that peer's rows
    for r in peers:
        pos = peer_off[r]
        for f, (w, c) in enumerate(fields):
            for k in range(c[r]):
                np.testing.assert_array_equal(B[pos:pos + w], 1000 * f + 10 * r + k + np.arange(w) / 10.0)
                pos += w
        assert pos - peer_off[r] == peer_size[r]
    inv = np.argsort(perm)
    np.testing.assert_array_equal(B[inv], A)


def test_empty_layout():
    total, field_off, peer_off, peer_size, perm = plan_exchange_layout([(2, {})], [])
    assert total == 0 and perm.size == 0 and field_off == [(0, 0)] and peer_off == {} and peer_size == {}
