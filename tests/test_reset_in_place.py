"""reset(release_buffers=False): the device buffers stay and are refreshed in place.

A model that resets once per sample pays a rebuild of the device structures (index
map, CSR, interning, uploads) on every reset(). Keeping the buffers and uploading the
reset host columns over the local rows gives the same state for a few column copies.
These tests pin that it IS the same state: reads after the reset see initial values,
a rerun equals a fresh model's run to the bit, a kept column survives, a reduce inbox
starts empty and event wake state is cleared. Every getter here is collective, so the
file runs under mpirun too (tests/run_mpi_tests.sh).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_reset_sync_properties import BumpModel, N_AGENTS, TICKS   # noqa: E402
from test_reduce_bla import InboxModel, _totals                              # noqa: E402


def _device(model, name):
    ids = sorted(model._agent_factory._agent2breed)
    return [float(np.ravel(model.get_agent_property_value(i, name))[0]) for i in ids]


def test_in_place_reset_reads_initial_values():
    m = BumpModel("inplace_read")
    m.create_agents(N_AGENTS)
    m.setup()
    m.simulate(TICKS)
    assert _device(m, "alpha") == [float(TICKS)] * N_AGENTS
    m.reset(sync_properties=(), release_buffers=False)
    assert m._gpu_buffers.is_initialized, "buffers must survive the reset"
    # host columns were never written (no readback), device must now agree with them
    col = m._agent_factory._property_name_2_agent_data_tensor["alpha"]   # this rank's rows
    assert all(float(np.ravel(col[i])[0]) == 0.0 for i in range(len(col)))
    assert _device(m, "alpha") == [0.0] * N_AGENTS
    assert _device(m, "beta") == [0.0] * N_AGENTS


def test_rerun_after_in_place_reset_equals_fresh_run():
    fresh = BumpModel("inplace_fresh")
    fresh.create_agents(N_AGENTS)
    fresh.setup()
    fresh.simulate(TICKS)

    reused = BumpModel("inplace_reused")
    reused.create_agents(N_AGENTS)
    reused.setup()
    reused.simulate(2 * TICKS)                       # a different first run
    reused.reset(sync_properties=(), release_buffers=False)
    reused.simulate(TICKS)
    assert _device(reused, "alpha") == _device(fresh, "alpha")
    assert _device(reused, "beta") == _device(fresh, "beta")
    assert reused.tick == fresh.tick


def test_kept_column_survives_in_place_reset():
    """sync_properties names what comes back from the device; a kept column keeps the
    kernel's values on the device too (nothing is uploaded over it)."""
    m = BumpModel("inplace_keep")
    m.create_agents(N_AGENTS)
    m.setup()
    m.simulate(TICKS)
    m.reset(sync_properties=["alpha"], release_buffers=False)
    assert _device(m, "alpha") == [float(TICKS)] * N_AGENTS
    assert _device(m, "beta") == [0.0] * N_AGENTS
    m.simulate(1)
    assert _device(m, "alpha") == [float(TICKS + 1)] * N_AGENTS
    assert _device(m, "beta") == [10.0] * N_AGENTS


def test_release_buffers_is_the_old_path():
    m = BumpModel("inplace_release")
    m.create_agents(N_AGENTS)
    m.setup()
    m.simulate(TICKS)
    m.reset(sync_properties=(), release_buffers=True)
    assert not m._gpu_buffers.is_initialized
    m.simulate(TICKS)
    assert _device(m, "alpha") == [float(TICKS)] * N_AGENTS


def test_reduce_inbox_starts_empty_after_in_place_reset():
    """Deposits of the previous run must not leak into the next: the inbox buffers
    are zeroed, so after T ticks the total is (T-1) * sum again."""
    weights = [[1.0, 2.0, 3.5], [10.0]]
    m = InboxModel()
    rids = m.build(weights)
    m.setup()
    m.simulate(ticks=5)
    m.reset(sync_properties=(), release_buffers=False)
    T = 4
    m.simulate(ticks=T)
    np.testing.assert_allclose(_totals(m, rids), (T - 1) * np.array([sum(w) for w in weights]))
