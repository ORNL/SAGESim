"""
Property interning: a property no kernel writes, not neighbor-visible, with mostly
duplicate rows is held on the device as `table + codes` and kernels read
`p_table[p_codes[i]]`. Results must equal the dense run bit for bit; a property written
inside a forwarded helper, or a neighbor-visible one, must not be interned.
"""
import sys
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from cupyx import jit

from sagesim.breed import Breed
from sagesim.columns import ArrayColumn, IndexedColumn
from sagesim.gpu_kernels import TableProperty
from sagesim.internal_utils import build_csr_from_ragged
from sagesim.model import Model
from sagesim.space import NetworkSpace


# properties: breeds=0, locations=1, params=2, state=3, mods=4
@jit.rawkernel(device="cuda")
def _scale(tick, agent_index, agent_ids, breeds, locations, params, state, mods):
    # helper receives the (possibly interned) params tensor and reads a row of it
    return params[agent_index][0] * params[agent_index][1] + mods[agent_index][0]


@jit.rawkernel(device="cuda")
def _bump_mods(tick, agent_index, agent_ids, breeds, locations, params, state, mods):
    mods[agent_index][0] = mods[agent_index][0] + 0.5


@jit.rawkernel(device="cuda")
def step_read_params(tick, agent_index, agent_ids, breeds, locations, params, state, mods):
    row = params[agent_index]
    s = _scale(tick, agent_index, agent_ids, breeds, locations, params, state, mods)
    state[agent_index][0] = state[agent_index][0] + s + row[2]
    state[agent_index][1] = state[agent_index][1] + 1.0
    _bump_mods(tick, agent_index, agent_ids, breeds, locations, params, state, mods)


class ParamBreed(Breed):
    def __init__(self, params_visible=False):
        super().__init__("Param")
        self.register_property("params", [0.0, 0.0, 0.0], neighbor_visible=params_visible)
        self.register_property("state", [0.0, 0.0], neighbor_visible=False)
        self.register_property("mods", [0.0], neighbor_visible=False)
        self.register_step_func(step_read_params, Path(__file__).resolve(), 0,
                                no_double_buffer=["state", "mods"])


class ParamModel(Model):
    def __init__(self, tag, params_visible=False):
        super().__init__(NetworkSpace(), step_function_file_path=f"step_func_code_intern_{tag}.py")
        self.breed = ParamBreed(params_visible)
        self.register_breed(self.breed)


TABLE = np.array([[1.0, 2.0, 0.25], [3.0, 4.0, 0.5], [5.0, 6.0, 0.75]], dtype=np.float32)


def _build(tag, how, n=600, params_visible=False, interning=True):
    m = ParamModel(tag, params_visible)
    m.enable_property_interning = interning
    codes = np.arange(n) % 3
    off, val = build_csr_from_ragged([[] for _ in range(n)])
    if how == "indexed":
        params = IndexedColumn(TABLE, codes.astype(np.int32))
    elif how == "array":
        params = TABLE[codes].copy()
    elif how == "list":
        rows = [TABLE[k].tolist() for k in range(3)]
        params = [rows[k] for k in codes]          # shared row objects
    else:
        raise ValueError(how)
    cols = {"params": params,
            "state": np.zeros((n, 2), dtype=np.float32),
            "mods": np.zeros((n, 1), dtype=np.float32)}
    m.build_from_local_columns(agent_ids=np.arange(n), breed_indices=np.full(n, m.breed._breedidx),
                               property_columns=cols, neighbor_offsets=off, neighbor_values_ids=val)
    return m


@pytest.fixture(autouse=True)
def _clear_generated():
    yield
    for k in list(sys.modules):
        if k.startswith("step_func_code_intern"):
            sys.modules.pop(k, None)


def _run(m, ticks=4):
    m.setup()
    m.simulate(ticks, sync_workers_every_n_ticks=ticks)
    return m


def _expected(n, ticks):
    codes = np.arange(n) % 3
    t = TABLE[codes]
    out = np.zeros((n, 2), dtype=np.float64)
    mods = np.zeros(n)
    for _ in range(ticks):
        out[:, 0] += t[:, 0] * t[:, 1] + mods + t[:, 2]
        out[:, 1] += 1.0
        mods += 0.5
    return out.astype(np.float32)


@pytest.mark.parametrize("how", ["indexed", "array", "list"])
def test_interned_kernel_matches_dense_and_reference(how):
    n, ticks = 600, 4
    dense = _run(_build("d_" + how, how, n, interning=False), ticks)
    interned = _run(_build("i_" + how, how, n), ticks)

    idx = interned._agent_factory._property_name_2_index
    assert idx["params"] in interned._interned_property_indices, interned._interned_property_indices
    assert idx["state"] not in interned._interned_property_indices     # written
    assert idx["mods"] not in interned._interned_property_indices      # written inside a helper
    assert isinstance(interned._gpu_buffers.property_tensors[idx["params"]], TableProperty)
    assert interned._gpu_buffers.property_tensors[idx["params"]].table.shape == (3, 3)
    assert not dense._interned_property_indices

    sd = dense._gpu_buffers.property_tensors[idx["state"]][:n].get()
    si = interned._gpu_buffers.property_tensors[idx["state"]][:n].get()
    np.testing.assert_array_equal(sd.view(np.uint32), si.view(np.uint32))
    np.testing.assert_array_equal(si, _expected(n, ticks))

    # host read-back goes through the table
    assert interned.get_agent_property_value(4, "params") == TABLE[1].tolist()
    assert interned.get_breed_data("Param", "params")[:3].tolist() == TABLE.tolist()
    # the generated kernel really takes table + codes for params
    src = Path(interned._generated_step_function_file_path).read_text()
    assert "a2_table" in src and "a2_codes" in src and "params_table[params_codes[" in src


def test_neighbor_visible_property_is_not_interned():
    m = _run(_build("v", "indexed", 300, params_visible=True), 2)
    assert m._agent_factory._property_name_2_index["params"] not in m._interned_property_indices


def test_reset_and_host_write_keep_working():
    n = 300
    m = _run(_build("r", "indexed", n), 3)
    m.reset()
    m.set_agent_property_value(7, "params", [10.0, 10.0, 0.0])   # new distinct row -> table grows
    col = m._agent_factory._property_name_2_agent_data_tensor["params"]
    assert isinstance(col, IndexedColumn) and col.n_distinct == 4
    m.simulate(1, sync_workers_every_n_ticks=1)
    st = m.get_agent_property_value(7, "state")
    assert st[0] == pytest.approx(_expected(n, 3)[7, 0] + 100.0 + 1.5)      # 10*10 + mods(1.5) + 0
    assert m.get_agent_property_value(7, "params") == [10.0, 10.0, 0.0]
