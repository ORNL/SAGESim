from time import time
from sir_model import SIRModel
from state import SIRState
from mpi4py import MPI
from random import random, seed as set_random_seed
from random import sample
import csv

import networkx as nx

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


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
        preventative_measures = [0.0]*100#[random() for _ in range(100)]
        model.create_agent(SIRState.SUSCEPTIBLE.value, preventative_measures)

    """print(f"Number of infected agents: {num_infected}")"""
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
    num_ticks = 1
    random_seed = 2  # Set seed for reproducible results

    model = SIRModel(p_infection=1.0, p_recovery=1.0)
    model.setup(use_gpu=True)

    model_creation_start = time()
    model = generate_small_world_of_agents(
        model, num_agents, num_init_connections, int(0.1 * num_agents), seed=random_seed
    )  # test_network()  #
    model_creation_end = time()
    model_creation_duration = model_creation_end - model_creation_start
    """print(
        [
            SIRState(model.get_agent_property_value(agent_id, property_name="state"))
            for agent_id in range(n_agents)
        ]
    )"""

    simulate_start = time()
    model.simulate(num_ticks, sync_workers_every_n_ticks=1)
    simulate_end = time()
    simulate_duration = simulate_end - simulate_start


    result = [
        int(model.get_agent_property_value(agent_id, property_name="state"))
        for agent_id in range(num_agents)
        if model.get_agent_property_value(agent_id, property_name="state") is not None
    ]

    if worker == 0:
        print(result)

    # # Save state history to CSV (only on worker 0 to avoid duplicates)
    # if worker == 0:
    #     with open('state_history.csv', 'w', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         # Write header
    #         writer.writerow(['agent_id'] + [f'tick_{i}' for i in range(num_ticks)])

    #         # Write state history for each agent
    #         for agent_id in range(model._agent_factory.num_agents):
    #             state_history = model.get_state_history(agent_id)
    #             if state_history is not None:
    #                 # Convert float states to integers
    #                 state_history_int = [int(state) for state in state_history]
    #                 writer.writerow([agent_id] + state_history_int)

    """if worker == 0:
        print(
            [
                SIRState(
                    model.get_agent_property_value(agent_id, property_name="state")
                )
                for agent_id in range(num_agents)
            ]
        )

        print(result)"""
