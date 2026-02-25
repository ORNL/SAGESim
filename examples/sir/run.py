from time import time
from mpi4py import MPI
from random import random, seed as set_random_seed
from random import sample
import csv
from enum import Enum

import networkx as nx
from cupyx import jit
from sagesim.breed import Breed
from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.utils import get_num_neighbors, get_neighbor

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
    preventative_measures_tensor,
    state_history_buffer,
):
    """
    At each simulation step, this function evaluates a subset of agents—either all agents in a serial run or a partition assigned to
    a specific rank in parallel processing—and determines whether an agent's state should change based on interactions with its neighbors
    and their respective preventative behaviors.
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

    # Get the preventative measures vector for the current agent
    agent_preventative_measures = preventative_measures_tensor[agent_index]

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

            # Get the preventative measures vector of the neighbor
            neighbor_preventative_measures = preventative_measures_tensor[neighbor_index]

            # Initialize cumulative safety score for the interaction
            abs_safety_of_interaction = 0.0

            # Calculate total safety of interaction based on pairwise product of measures
            for n in range(len(agent_preventative_measures)):
                for m in range(len(neighbor_preventative_measures)):
                    abs_safety_of_interaction += (
                        agent_preventative_measures[n]
                        * neighbor_preventative_measures[m]
                    )

            # Normalize the safety score to be in [0, 1]
            normalized_safety_of_interaction = abs_safety_of_interaction / (
                len(agent_preventative_measures) ** 2
            )

            # If neighbor is infected and the infection condition passes, update agent's state
            if neighbor_state == 2 and rand < p_infection * (
                1 - normalized_safety_of_interaction
            ):
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
        self.register_property("preventative_measures", [random() for _ in range(100)])
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

    # create_agent method takes user-defined properties, that is, the state and preventative_measures, to create an agent
    def create_agent(self, state, preventative_measures):
        agent_id = self.create_agent_of_breed(
            self._sir_breed, state=state, preventative_measures=preventative_measures
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


def generate_small_world_network(n, k, p, seed=None):
    """
    Generate a small world network using the Watts-Strogatz model.

    Parameters:
    - n (int): The number of nodes in the network.
    - k (int): Each node is connected to its k nearest neighbors in a ring topology.
    - p (float): The probability of rewiring each edge.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - networkx.Graph: The generated small world network.
    """
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def generate_small_world_of_agents(
    model, num_agents: int, num_init_connections: int, num_infected: int, seed=None
) -> SIRModel:
    # Set random seed for reproducibility
    if seed is not None:
        set_random_seed(seed)

    network = generate_small_world_network(num_agents, num_init_connections, 0.1, seed=seed)
    for n in network.nodes:
        preventative_measures = [0.0]*100
        model.create_agent(SIRState.SUSCEPTIBLE.value, preventative_measures)

    # Set seed again before sampling to ensure deterministic selection of infected agents
    if seed is not None:
        set_random_seed(seed)
    for n in sample(sorted(network.nodes), num_infected):
        model.set_agent_property_value(n, "state", SIRState.INFECTED.value)

    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])
    return model


if __name__ == "__main__":
    # Hardcoded parameters
    num_agents = 20
    num_init_connections = 2
    num_nodes = num_workers
    num_ticks = 10
    random_seed = 2  # Set seed for reproducible results

    model = SIRModel(p_infection=1.0, p_recovery=1.0, enable_state_tracking=True)
    model.setup(use_gpu=True)

    model_creation_start = time()
    model = generate_small_world_of_agents(
        model, num_agents, num_init_connections, int(0.1 * num_agents), seed=random_seed
    )
    model_creation_end = time()
    model_creation_duration = model_creation_end - model_creation_start

    simulate_start = time()
    model.simulate(num_ticks, sync_workers_every_n_ticks=1)
    simulate_end = time()
    simulate_duration = simulate_end - simulate_start

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
        with open('state_history.csv', 'w', newline='') as csvfile:
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

        print(f"State history saved to state_history.csv")
