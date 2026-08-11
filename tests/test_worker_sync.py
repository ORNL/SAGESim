"""Worker-sync test: infection spreading along a chain of agents.

A 10-agent chain ``0-1-2-...-9`` with agent 0 infected and
``p_infection = p_recovery = 1.0``, so every transition is deterministic: each
tick the infection advances exactly one hop and the previously infected agent
recovers. Running it under multiple ranks exercises ghost exchange, since the
chain is split across ranks and every agent's infecting neighbor may be remote.

Single rank:  pytest tests/test_worker_sync.py
Multi rank:   srun -n4 --ntasks-per-gpu=1 --gpu-bind=closest \
                  python -m pytest tests/test_worker_sync.py
              (on Frontier; use your site's launcher elsewhere)
"""

from enum import Enum
from random import random, seed as set_random_seed

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
NUM_TICKS = 10
RANDOM_SEED = 2


# Define the SIRState enumeration for agent states
class SIRState(Enum):
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3


# Define the step function to be registered for SIRBreed
@jit.rawkernel(device="cuda")
def step_func(
    tick,
    agent_index,
    p_infection,
    p_recovery,
    agent_ids,
    breeds,
    locations,
    state_tensor,
    state_history_buffer,
):
    """
    Simplified step function without preventative measures.
    At each simulation step, this function evaluates agents and determines whether
    an agent's state should change based on interactions with neighbors.
    """
    # Get the list of neighboring agent indices for the current agent based on network topology
    neighbor_indices = locations[agent_index]

    # Draw a random float in [0, 1) for stochastic decision-making
    rand = random()

    # Retrieve the global infection and recovery probabilities from global tensors
    # p_infection and p_recovery are scalars, auto-extracted by framework

    # Get the current state of the agent (e.g., susceptible, infected, recovered)
    agent_state = state_tensor[agent_index]

    # If agent is infected and the recovery condition passes, update agent's state
    if agent_state == 2 and rand < p_recovery:
        state_tensor[agent_index] = 3
    elif agent_state == 1:
        # Loop through each neighbor index
        i = 0
        while i < len(neighbor_indices) and neighbor_indices[i] != -1:
            neighbor_index = neighbor_indices[i]

            # Retrieve the state of the neighbor (e.g., susceptible, infected, recovered)
            neighbor_state = state_tensor[neighbor_index]

            # If neighbor is infected and the infection condition passes, update agent's state
            if neighbor_state == 2 and rand < p_infection:
                state_tensor[agent_index] = 2
            i += 1

    # Safe buffer indexing: use modulo to prevent out-of-bounds access
    # When tracking is disabled, buffer length is 1, so tick % 1 = 0 always
    buffer_idx = tick % len(state_history_buffer[agent_index])
    state_history_buffer[agent_index][buffer_idx] = state_tensor[agent_index]


class SIRBreed(Breed):
    """
    SIRBreed class the SIR model.
    Inherits from the Breed class in the sagesim library.
    """

    def __init__(self) -> None:
        name = "SIR"
        super().__init__(name)
        # Register properties for the breed
        # Use single-element buffer as default (circular buffer with modulo indexing)
        self.register_property("state", SIRState.SUSCEPTIBLE.value)
        self.register_property("state_history_buffer", [0.0] * NUM_TICKS)
        # Register the step function
        self.register_step_func(step_func, __file__, 0)


class SIRModel(Model):
    """
    SIRModel class for the SIR model.
    Inherits from the Model class in the sagesim library.
    """

    def __init__(self, p_infection=0.2, p_recovery=0.2) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()

        # Register the breed
        self.register_breed(breed=self._sir_breed)

        # register global properties
        self.register_global_property("p_infection", p_infection)
        self.register_global_property("p_recovery", p_recovery)

    # create_agent method takes user-defined properties, that is, the state to create an agent
    def create_agent(self, state):
        return self.create_agent_of_breed(self._sir_breed, state=state)

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)

    def get_state_history(self, agent_id: int):
        """Return the per-tick state history for a specific agent."""
        return self.get_agent_property_value(agent_id, "state_history_buffer")


def generate_chain_network(num_agents):
    """Generate a chain network 0-1-2-...-(num_agents-1)."""
    G = nx.Graph()
    G.add_nodes_from(range(num_agents))
    for i in range(num_agents - 1):
        G.add_edge(i, i + 1)
    return G


def generate_chain_of_agents(model, num_agents: int, seed=None) -> SIRModel:
    """Populate ``model`` with a chain of agents, agent 0 infected."""
    if seed is not None:
        set_random_seed(seed)

    network = generate_chain_network(num_agents)

    # Create all agents as susceptible
    for _ in network.nodes:
        model.create_agent(SIRState.SUSCEPTIBLE.value)

    # Set agent 0 as infected
    model.set_agent_property_value(0, "state", SIRState.INFECTED.value)

    # Add all edges to the model (chain: 0-1, 1-2, ..., 8-9)
    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])

    return model


@pytest.fixture(scope="module")
def simulated_chain():
    """Build the chain, run ``NUM_TICKS`` ticks, hand back the model.

    Module-scoped: setup compiles CuPy kernels, so the run is shared by the tests
    below rather than repeated per test.
    """
    model = SIRModel(p_infection=1.0, p_recovery=1.0)
    model = generate_chain_of_agents(model, NUM_AGENTS, seed=RANDOM_SEED)
    # No step func reads neighbor breeds, so skip exchanging them.
    model.set_property_neighbor_visible("breed", False)
    model.setup()
    model.simulate(NUM_TICKS, sync_workers_every_n_ticks=1)
    return model


def _local_agent_ids(model):
    """Agent ids whose data this rank owns (all of them under a single rank)."""
    return [
        agent_id
        for agent_id in range(NUM_AGENTS)
        if model.get_agent_property_value(agent_id, "state") is not None
    ]


def test_this_rank_owns_at_least_one_agent(simulated_chain):
    """Guards the tests below from vacuously passing on an empty rank."""
    owned = _local_agent_ids(simulated_chain)
    assert owned, f"rank {worker} of {num_workers} owns no agents"


def test_infection_advances_one_hop_per_tick(simulated_chain):
    """Agent ``i`` is infected exactly on tick ``i``, and recovers thereafter.

    This is the ghost-exchange assertion: agent ``i`` can only learn it should be
    infected by reading neighbor ``i-1``'s state, which under multiple ranks lives
    on another rank.
    """
    for agent_id in _local_agent_ids(simulated_chain):
        history = [int(s) for s in simulated_chain.get_state_history(agent_id)]
        for tick in range(NUM_TICKS):
            if tick < agent_id:
                expected = SIRState.SUSCEPTIBLE
            elif tick == agent_id:
                expected = SIRState.INFECTED
            else:
                expected = SIRState.RECOVERED
            assert history[tick] == expected.value, (
                f"agent {agent_id} at tick {tick}: expected {expected.name}, "
                f"got {SIRState(history[tick]).name}; full history={history}"
            )


def test_every_agent_is_infected_exactly_once(simulated_chain):
    """No agent is re-infected after recovering."""
    for agent_id in _local_agent_ids(simulated_chain):
        history = [int(s) for s in simulated_chain.get_state_history(agent_id)]
        infected_ticks = [
            t for t, s in enumerate(history) if s == SIRState.INFECTED.value
        ]
        assert infected_ticks == [agent_id], (
            f"agent {agent_id} infected on ticks {infected_ticks}, expected "
            f"exactly [{agent_id}]; full history={history}"
        )


def test_whole_chain_has_recovered_at_end(simulated_chain):
    """After the infection has walked the full chain, everyone is recovered."""
    for agent_id in _local_agent_ids(simulated_chain):
        state = int(simulated_chain.get_agent_property_value(agent_id, "state"))
        assert state == SIRState.RECOVERED.value, (
            f"agent {agent_id} final state is {SIRState(state).name}, "
            f"expected RECOVERED"
        )
