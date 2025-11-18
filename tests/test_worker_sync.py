from time import time
from mpi4py import MPI
from random import random, seed as set_random_seed
import csv
from enum import Enum

import networkx as nx
from cupyx import jit
from sagesim.breed import Breed
from sagesim.model import Model
from sagesim.space import NetworkSpace

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


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
    globals,
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

    # Retrieve the global infection and recovery probabilities defined in the model
    p_infection = globals[0]
    p_recovery = globals[1]

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
        self.register_property("state", SIRState.SUSCEPTIBLE.value)
        self.register_property("state_history_buffer", [])  # Buffer to track state at each tick
        # Register the step function
        self.register_step_func(step_func, __file__, 0)


class SIRModel(Model):
    """
    SIRModel class for the SIR model.
    Inherits from the Model class in the sagesim library.
    """

    def __init__(self, p_infection=0.2, p_recovery=0.2, enable_state_tracking=True) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()

        # Register the breed
        self.register_breed(breed=self._sir_breed)

        # register user-defined global properties
        self.register_global_property("p_infection", p_infection)
        self.register_global_property("p_recovery", p_recovery)

        # Store state tracking setting
        self.enable_state_tracking = enable_state_tracking

    # create_agent method takes user-defined properties, that is, the state to create an agent
    def create_agent(self, state):
        agent_id = self.create_agent_of_breed(
            self._sir_breed, state=state
        )
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)

    def simulate(self, ticks: int, sync_workers_every_n_ticks: int = 1) -> None:
        """
        Override simulate to allocate state_history_buffer before simulation.
        """
        # Allocate state_history_buffer for all agents
        for agent_id in range(self._agent_factory.num_agents):
            current_state = self.get_agent_property_value(agent_id, "state")

            if self.enable_state_tracking:
                # Allocate buffer for all ticks
                state_history_buffer = [current_state for _ in range(ticks)]
            else:
                # Allocate single-element buffer that will be overwritten
                state_history_buffer = [current_state]

            self.set_agent_property_value(
                id=agent_id,
                property_name="state_history_buffer",
                value=state_history_buffer
            )

        # Call parent simulate
        super().simulate(ticks, sync_workers_every_n_ticks)

    def get_state_history(self, agent_id: int):
        """
        Get the state history for a specific agent.

        Returns:
            List of states for each tick if tracking enabled, otherwise list with final state
        """
        if not self.enable_state_tracking:
            return None
        return self.get_agent_property_value(agent_id, "state_history_buffer")


def generate_chain_network(num_agents):
    """
    Generate a chain network where 0->1->2->3->...->9

    Parameters:
    - num_agents (int): The number of agents in the chain.

    Returns:
    - networkx.Graph: The generated chain network.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_agents))

    # Create chain: 0-1-2-3-4-5-6-7-8-9
    for i in range(num_agents - 1):
        G.add_edge(i, i + 1)

    return G


def generate_chain_of_agents(
    model, num_agents: int, num_infected: int = 1, seed=None
) -> SIRModel:
    # Set random seed for reproducibility
    if seed is not None:
        set_random_seed(seed)

    network = generate_chain_network(num_agents)

    # Create all agents as susceptible
    for n in network.nodes:
        model.create_agent(SIRState.SUSCEPTIBLE.value)

    # Set agent 0 as infected
    model.set_agent_property_value(0, "state", SIRState.INFECTED.value)

    # Add all edges to the model (chain: 0-1, 1-2, 2-3, ..., 8-9)
    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])

    return model


if __name__ == "__main__":
    # Hardcoded parameters
    num_agents = 10  # Chain of 10 agents: 0-1-2-3-4-5-6-7-8-9
    num_nodes = num_workers
    num_ticks = 10
    random_seed = 2  # Set seed for reproducible results

    model = SIRModel(p_infection=1.0, p_recovery=1.0, enable_state_tracking=True)
    model.setup(use_gpu=True)

    model = generate_chain_of_agents(
        model, num_agents, num_infected=1, seed=random_seed
    )

    # Print which agents are assigned (non-ghost) to each worker
    # Use the framework's internal agent-to-rank mapping for accurate results
    agent_factory = model._agent_factory
    owned_agents = list(agent_factory._rank2agentid2agentidx.get(worker, {}).keys())

    print(f"Worker {worker}: owns agents {owned_agents}")

    print(model._agent_factory.num_agents)
    comm.Barrier()

    if worker == 0:
        print(f"\nCreated chain network: 0-1-2-3-4-5-6-7-8-9")
        print(f"Agent 0 starts INFECTED, all others SUSCEPTIBLE")
        print(f"p_infection=1.0, p_recovery=1.0")
        print(f"Running with {num_workers} workers")
        print()

    model.simulate(num_ticks, sync_workers_every_n_ticks=1)

    result = [
        int(model.get_agent_property_value(agent_id, property_name="state"))
        for agent_id in range(num_agents)
        if model.get_agent_property_value(agent_id, property_name="state") is not None
    ]

    # Collect state history for all agents first (all workers participate)
    all_state_histories = {}
    for agent_id in range(num_agents):
        state_history = model.get_state_history(agent_id)
        if state_history is not None:
            all_state_histories[agent_id] = state_history

    # Synchronize all workers before writing
    comm.Barrier()

    if worker == 0:
        print(f"Collected state histories for {len(all_state_histories)} agents")

    # Save state history to CSV (only on worker 0 to avoid duplicates)
    if worker == 0:
        with open('test_worker_sync_results.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(['agent_id'] + [f'tick_{i}' for i in range(num_ticks)])

            # Write state history for each agent
            for agent_id in range(num_agents):
                if agent_id in all_state_histories:
                    state_history = all_state_histories[agent_id]
                    # Convert float states to integers
                    state_history_int = [int(state) for state in state_history]
                    writer.writerow([agent_id] + state_history_int)

        print(f"\nState history saved to test_worker_sync_results.csv")

        # Print expected vs actual results
        print("\nExpected behavior with deterministic infection/recovery:")
        print("Tick 0: Agent 0 infected")
        print("Tick 1: Agent 0 recovers, Agent 1 gets infected")
        print("Tick 2: Agent 1 recovers, Agent 2 gets infected")
        print("...")
        print("Tick 9: Agent 8 recovers, Agent 9 gets infected")
        print("\nActual final states:", result)
