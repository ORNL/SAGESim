"""
ArrayColumn: a property column stored as one padded numpy array (sagesim/columns.py).

Part 1 pins the list protocol the framework relies on. Part 2 checks that a model
built from numpy columns produces the same device tensors, results and read-backs
as the same model built from list columns, and that a reset() round trip (which
now stores the downloaded rectangle as an ArrayColumn) keeps values intact.
"""
import pickle
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from cupyx import jit

from sagesim.breed import Breed
from sagesim.columns import ArrayColumn
from sagesim.internal_utils import build_csr_from_ragged, convert_to_padded_gpu_tensor
from sagesim.model import Model
from sagesim.space import NetworkSpace


# ---------------------------------------------------------------- Part 1: protocol

def test_rows_read_back_at_their_own_width():
    c = ArrayColumn(np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32), lengths=[3, 2])
    assert c[0] == [1, 2, 3] and c[1] == [4, 5] and len(c) == 2 and bool(c)
    assert list(c) == [[1, 2, 3], [4, 5]] and c[-1] == [4, 5]
    assert c[0] is not c[0], "reads are copies, not views"


def test_write_pads_grows_and_records_length():
    c = ArrayColumn(np.zeros((2, 3), dtype=np.float32))
    c[1] = [7, 8]
    assert c[1] == [7, 8] and c.lengths[1] == 2 and np.isnan(c.values[1, 2])
    c[0] = [1, 2, 3, 4, 5]                     # wider than the column: grows in place
    assert c.width == 5 and c[0] == [1, 2, 3, 4, 5] and c[1] == [7, 8]


def test_ragged_write_degrades_in_place_and_keeps_identity():
    c = ArrayColumn(np.zeros((2, 2), dtype=np.float32))
    ref = c
    c[0] = [[1, 2], [3]]                       # not array-like
    assert c.degraded and ref is c and c[0] == [[1, 2], [3]] and c[1] == [0.0, 0.0]
    c.append([9])
    assert len(c) == 3 and c[2] == [9]


def test_depth3_slice_assign_and_fill_rows():
    d = ArrayColumn(np.zeros((3, 1, 2), dtype=np.float32), lengths=np.zeros(3, dtype=np.int32))
    assert d[0] == []                          # length 0 reads back as the empty default
    d[:] = [[[0.0, 0.0]]] * 3
    assert d[1] == [[0.0, 0.0]] and d.max_length() == 1
    d.fill_rows([[1.0, 1.0]])
    assert d[2] == [[1.0, 1.0]]
    d[0] = [[1, 2], [3, 4]]                    # tracking-on shape: grows axis 1
    assert d.values.shape == (3, 2, 2) and d[0] == [[1, 2], [3, 4]] and d[1] == [[1.0, 1.0]]


def test_scalar_column_int_dtype_append_permute():
    b = ArrayColumn(np.array([2, 0, 1], dtype=np.int32))
    assert b[0] == 2 and isinstance(b[0], int)
    b.append(5)
    b.permute([1, 2, 0, 3])
    assert list(b) == [0, 1, 2, 5]
    assert np.array(b, dtype=np.int32).tolist() == [0, 1, 2, 5]


def test_add_ghost_rows_pad_width_pickle_and_device():
    c = ArrayColumn(np.arange(6, dtype=np.float32).reshape(3, 2))
    e = c + [[5.0, 6.0]]
    assert len(e) == 4 and e[3] == [5.0, 6.0] and len(c) == 3
    c.pad_width(4)
    assert c[0] == [0.0, 1.0] and np.asarray(c).shape == (3, 4) and c.values[0, 3] == 0.0
    f = pickle.loads(pickle.dumps(c))
    assert list(f) == list(c)
    t = c.to_device(5)
    assert t.shape == (5, 4) and bool(cp.isnan(t[3:]).all()) and float(t[1, 0]) == 2.0
    s = ArrayColumn(np.array([3, 1], dtype=np.int32)).to_device(4)
    assert s.dtype == cp.float32 and s.tolist() == [3.0, 1.0, 0.0, 0.0]


def test_converter_uses_array_path_and_matches_list_path():
    rows = [[1.0, 2.0], [3.0], [], [4.0, 5.0]]
    lst = convert_to_padded_gpu_tensor(rows, 6)
    width = max(map(len, rows))
    values = np.full((4, width), np.nan, dtype=np.float32)
    for i, r in enumerate(rows):
        values[i, : len(r)] = r
    arr = convert_to_padded_gpu_tensor(ArrayColumn(values, [len(r) for r in rows]), 6)
    from sagesim import internal_utils
    assert internal_utils.LAST_CONVERSION_PATH == "array_column"
    assert cp.array_equal(lst, arr, equal_nan=True) and arr.shape == lst.shape


# ------------------------------------------------- Part 2: through Model / GPU

@jit.rawkernel(device="cuda")
def accumulate_step_func(tick, agent_index, agent_ids, breeds, locations, state, gain):
    # state[0] += gain[0]; state[1] counts ticks (width-2 rectangular property)
    state[agent_index][0] = state[agent_index][0] + gain[agent_index][0]
    state[agent_index][1] = state[agent_index][1] + 1.0


class AccBreed(Breed):
    def __init__(self):
        super().__init__("Acc")
        self.register_property("state", [0.0, 0.0])
        self.register_property("gain", [0.0])
        self.register_step_func(accumulate_step_func, Path(__file__).resolve(), 0,
                                no_double_buffer=["state", "gain"])


class AccModel(Model):
    def __init__(self, tag):
        super().__init__(NetworkSpace(), step_function_file_path=f"step_func_code_arraycol_{tag}.py")
        self.breed = AccBreed()
        self.register_breed(self.breed)


def _build(tag, as_arrays, n=300):
    m = AccModel(tag)
    state = np.stack([np.arange(n, dtype=np.float32), np.zeros(n, dtype=np.float32)], axis=1)
    gain = (np.arange(n, dtype=np.float32) % 7).reshape(n, 1)
    off, val = build_csr_from_ragged([[] for _ in range(n)])
    if as_arrays:
        cols = {"state": state, "gain": gain}
        breeds = np.full(n, m.breed._breedidx, dtype=np.int64)
    else:
        cols = {"state": state.tolist(), "gain": gain.tolist()}
        breeds = [m.breed._breedidx] * n
    m.build_from_local_columns(agent_ids=np.arange(n), breed_indices=breeds,
                               property_columns=cols, neighbor_offsets=off,
                               neighbor_values_ids=val)
    return m


@pytest.fixture(autouse=True)
def _clear_generated():
    yield
    for k in list(sys.modules):
        if k.startswith("step_func_code_arraycol"):
            sys.modules.pop(k, None)


def test_array_columns_match_list_columns_end_to_end():
    ma = _build("a", as_arrays=True)
    ml = _build("l", as_arrays=False)
    af_a = ma._agent_factory._property_name_2_agent_data_tensor
    assert isinstance(af_a["state"], ArrayColumn) and isinstance(af_a["breed"], ArrayColumn)
    # pre-tick read-back identical (row width as stored)
    assert ma.get_agent_property_value(5, "state") == ml.get_agent_property_value(5, "state") == [5.0, 0.0]
    assert ma.get_agent_property_value(5, "gain") == [5.0]

    for m in (ma, ml):
        m.setup()
        m.simulate(3, sync_workers_every_n_ticks=3)
    ta = ma._gpu_buffers.property_tensors
    tl = ml._gpu_buffers.property_tensors
    for i, (x, y) in enumerate(zip(ta, tl)):
        if x is None:
            continue
        assert cp.array_equal(x, y, equal_nan=True), f"property {i} differs on device"
    assert ma.get_agent_property_value(10, "state") == [10.0 + 3 * (10 % 7), 3.0]


def test_reset_round_trip_keeps_values_as_array_column():
    m = _build("r", as_arrays=True)
    m.setup()
    m.simulate(2, sync_workers_every_n_ticks=2)
    m.reset()
    col = m._agent_factory._property_name_2_agent_data_tensor["state"]
    assert isinstance(col, ArrayColumn) and not col.degraded
    assert m.get_agent_property_value(4, "state") == [4.0 + 2 * (4 % 7), 2.0]
    # a per-agent write after reset goes through the array column and is uploaded
    m.set_agent_property_value(4, "state", [100.0, 0.0])
    m.simulate(1, sync_workers_every_n_ticks=1)
    assert m.get_agent_property_value(4, "state") == [104.0, 1.0]


def test_create_agent_after_reset_appends_to_array_columns():
    """reset() stores the downloaded rectangle as ArrayColumns; adding an agent
    afterwards must append to them (the create_agent path)."""
    m = AccModel("g")
    ids = [m.create_agent_of_breed(m.breed, state=[float(i), 0.0], gain=[1.0]) for i in range(8)]
    m.setup()
    m.simulate(2, sync_workers_every_n_ticks=2)
    m.reset()
    cols = m._agent_factory._property_name_2_agent_data_tensor
    assert isinstance(cols["state"], ArrayColumn)
    new_id = m.create_agent_of_breed(m.breed, state=[50.0, 0.0], gain=[3.0])
    assert len(cols["state"]) == 9 and cols["state"][8] == [50.0, 0.0] and cols["gain"][8] == [3.0]
    m.setup()
    m.simulate(1, sync_workers_every_n_ticks=1)
    assert m.get_agent_property_value(new_id, "state") == [53.0, 1.0]
    assert m.get_agent_property_value(ids[3], "state") == [3.0 + 3.0, 3.0]
