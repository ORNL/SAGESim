"""
AgentIndex must answer exactly as the dict it replaces, dict(zip(ids, range(n))): the same
rows, misses, length, iteration order and items, for compact and sparse id ranges; reject
duplicate ids; survive pickling (Model.save pickles the factory); and rows_of must give the
same rows through a dict and through an AgentIndex, for arrays, lists and sets of ids.
"""
import pickle

import numpy as np
import pytest

from sagesim.agent_index import AgentIndex, rows_of


def _cases():
    rng = np.random.default_rng(0)
    compact = rng.permutation(np.arange(1000, 6000))                     # span == n
    gappy = rng.permutation(np.arange(0, 9000, 2))                       # span ~ 2n: table
    sparse = rng.choice(10**12, size=4000, replace=False)                # sorted-ids path
    return {"compact": compact, "gappy": gappy, "sparse": sparse}


@pytest.mark.parametrize("name", ["compact", "gappy", "sparse"])
def test_answers_like_the_dict(name):
    ids = _cases()[name]
    d = dict(zip(ids.tolist(), range(len(ids))))
    ai = AgentIndex(ids)
    assert len(ai) == len(d)
    assert list(ai) == list(d) and list(ai.keys()) == list(d.keys())
    assert list(ai.values()) == list(d.values()) and list(ai.items()) == list(d.items())
    assert dict(ai) == d and ai == d
    rng = np.random.default_rng(1)
    probe = np.concatenate([rng.choice(ids, 500), [-1, -7, int(ids.max()) + 1, int(ids.min()) - 1],
                            rng.integers(0, 2 * int(ids.max()) + 2, 500)])
    for k in probe.tolist():
        assert ai.get(k) == d.get(k)
        assert ai.get(k, -1) == d.get(k, -1)
        assert (k in ai) == (k in d)
        if k in d:
            assert ai[k] == d[k]
        else:
            with pytest.raises(KeyError):
                ai[k]
    expected = np.array([d.get(k, -1) for k in probe.tolist()], dtype=np.int64)
    np.testing.assert_array_equal(ai.rows(probe), expected)
    np.testing.assert_array_equal(rows_of(ai, probe), expected)
    np.testing.assert_array_equal(rows_of(d, probe), expected)
    np.testing.assert_array_equal(rows_of(ai, probe.tolist()), expected)
    as_set = set(probe.tolist())
    np.testing.assert_array_equal(np.sort(rows_of(ai, as_set)), np.sort(rows_of(d, as_set)))


@pytest.mark.parametrize("ids", [[3, 4, 3], [5, 10**12, 5]])
def test_duplicate_ids_are_rejected(ids):
    with pytest.raises(ValueError):
        AgentIndex(np.array(ids))


def test_empty_and_pickle():
    empty = AgentIndex(np.zeros(0, dtype=np.int64))
    assert len(empty) == 0 and list(empty) == [] and empty.get(0) is None and 0 not in empty
    assert empty.rows(np.array([1, 2])).tolist() == [-1, -1]
    for ids in _cases().values():
        ai = AgentIndex(ids)
        back = pickle.loads(pickle.dumps(ai))
        assert dict(back) == dict(ai)
        np.testing.assert_array_equal(back.ids, ai.ids)
