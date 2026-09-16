"""Reading an agent's property must return the OWNER's row, never a ghost copy.

A rank's GPU buffer holds its own agents *and* ghost copies of the neighbors it
borrows from other ranks. ``Model.get_agent_property_value`` is collective and
resolves ties in rank order, so if a rank claims ownership of an id merely because
the id is present in its buffer, the lowest-numbered rank holding a ghost answers
for the true owner.

Two ways that goes wrong, both covered below:

* a **neighbor-visible** property is exchanged every tick, but only at the *top* of
  each tick, so a ghost row is one tick stale -- the final tick's value is lost;
* a property that is **not** neighbor-visible is never exchanged at all, so the
  ghost row keeps the zero placeholder forever and the reader gets zeros.

Single rank:  pytest tests/test_ghost_readback.py
Multi rank:   mpirun --oversubscribe -n 2 python -m pytest tests/test_ghost_readback.py
              (also meaningful at -n 3 and -n 4; see tests/run_mpi_tests.sh)
"""

import networkx as nx
import pytest
from cupyx import jit
from mpi4py import MPI

from sagesim.breed import Breed
from sagesim.model import Model
from sagesim.space import NetworkSpace

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()

NUM_AGENTS = 10
NUM_TICKS = 5


@jit.rawkernel(device="cuda")
def step_func(
    tick,
    agent_index,
    agent_ids,
    breeds,
    locations,
    private_state,
    shared_history,
):
    """Stamp a value only the owning rank can compute, into both properties.

    ``agent_ids[agent_index]`` is the agent's own id, so the stamp identifies which
    agent's row a reader actually got, and the tick component identifies how stale
    it is. Neither property is read from a neighbor -- this test is about readback,
    not about propagation.
    """
    stamp = (agent_ids[agent_index] + 1.0) * 100.0 + tick
    private_state[agent_index][0] = stamp
    shared_history[agent_index][tick % len(shared_history[agent_index])] = stamp


def expected_stamp(agent_id, tick):
    return (agent_id + 1.0) * 100.0 + tick


class StampBreed(Breed):
    def __init__(self) -> None:
        super().__init__("Stamp")
        # Never read by a neighbor: never exchanged, so a ghost row stays all zeros.
        self.register_property("private_state", [0.0], neighbor_visible=False)
        # Exchanged every tick, but only at the top of a tick: a ghost row lags by one.
        self.register_property(
            "shared_history", [0.0] * NUM_TICKS, neighbor_visible=True
        )
        self.register_step_func(step_func, __file__, 0)


class StampModel(Model):
    def __init__(self) -> None:
        super().__init__(NetworkSpace())
        self._breed = StampBreed()
        self.register_breed(breed=self._breed)


@pytest.fixture(scope="module")
def stamped_chain():
    """Chain 0-1-...-9 so that under round-robin ranks every agent is a ghost
    somewhere, run NUM_TICKS ticks, hand back the model."""
    model = StampModel()
    graph = nx.Graph()
    graph.add_nodes_from(range(NUM_AGENTS))
    for i in range(NUM_AGENTS - 1):
        graph.add_edge(i, i + 1)
    for _ in graph.nodes:
        model.create_agent_of_breed(model._breed)
    for a, b in graph.edges:
        model.get_space().connect_agents(a, b)
    model.set_property_neighbor_visible("breed", False)
    model.setup()
    model.simulate(NUM_TICKS, sync_workers_every_n_ticks=1)
    return model


def test_non_neighbor_visible_property_comes_from_the_owner(stamped_chain):
    """The sharpest form of the bug: a ghost row for an unexchanged property is
    all zeros, so a reader that accepts it gets 0.0 instead of the owner's value."""
    for agent_id in range(NUM_AGENTS):
        # Collective: every rank must call this for every agent, in the same order.
        value = stamped_chain.get_agent_property_value(agent_id, "private_state")
        assert value is not None, f"no rank claimed agent {agent_id}"
        assert value[0] == pytest.approx(expected_stamp(agent_id, NUM_TICKS - 1)), (
            f"agent {agent_id}: got {value[0]}, expected "
            f"{expected_stamp(agent_id, NUM_TICKS - 1)}; 0.0 means a ghost row "
            f"(never exchanged) was returned instead of the owner's"
        )


def test_neighbor_visible_history_includes_the_final_tick(stamped_chain):
    """A ghost row is refreshed at the top of each tick and never after the last
    one, so accepting it loses exactly the final tick."""
    for agent_id in range(NUM_AGENTS):
        history = stamped_chain.get_agent_property_value(agent_id, "shared_history")
        assert history is not None, f"no rank claimed agent {agent_id}"
        for tick in range(NUM_TICKS):
            assert history[tick] == pytest.approx(expected_stamp(agent_id, tick)), (
                f"agent {agent_id} slot {tick}: got {history[tick]}, expected "
                f"{expected_stamp(agent_id, tick)}"
                + (" -- the final tick is the one a stale ghost row loses"
                   if tick == NUM_TICKS - 1 else "")
            )


def test_local_getter_rejects_a_ghost_id(stamped_chain):
    """``get_local_agent_property_value`` documents a KeyError for a non-local id.

    A ghost id is in this rank's buffer, so the lookup must not silently succeed
    and hand back the borrowed row. Non-collective, so it is safe to run per rank.
    """
    if num_workers == 1:
        pytest.skip("no ghosts under a single rank")
    buf = stamped_chain._gpu_buffers
    factory = stamped_chain._agent_factory
    ghost_ids = [
        agent_id
        for agent_id in range(NUM_AGENTS)
        if agent_id in buf.agent_id_to_index and not factory._owns_locally(agent_id)
    ]
    assert ghost_ids, f"rank {worker} holds no ghosts; chain should guarantee some"
    for ghost_id in ghost_ids:
        with pytest.raises(KeyError):
            stamped_chain.get_local_agent_property_value(ghost_id, "private_state")
