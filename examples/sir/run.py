from time import time
import argparse
from sir_model import SIRModel
from state import SIRState
from mpi4py import MPI
from random import random

from random import sample

import networkx as nx

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


def generate_small_world_network(n, k, p):
    """
    Generate a small world network using the Watts-Strogatz model.

    Parameters:
    - n (int): The number of nodes in the network.
    - k (int): Each node is connected to its k nearest neighbors in a ring topology.
    - p (float): The probability of rewiring each edge.

    Returns:
    - networkx.Graph: The generated small world network.
    """
    return nx.watts_strogatz_graph(n, k, p)


def generate_small_world_of_agents(
    model, num_agents: int, num_init_connections: int, num_infected: int
) -> SIRModel:
    network = generate_small_world_network(num_agents, num_init_connections, 0.2)
    for n in network.nodes:
        preventative_measures = [random() for _ in range(100)]
        model.create_agent(SIRState.SUSCEPTIBLE.value, preventative_measures)

    """print(f"Number of infected agents: {num_infected}")"""
    for n in sample(sorted(network.nodes), num_infected):
        model.set_agent_property_value(n, "state", SIRState.INFECTED.value)

    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])
    return model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_agents",
        type=int,
    )
    parser.add_argument(
        "--num_init_connections",
        type=float,
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
    )
    args = parser.parse_args()

    model = SIRModel()
    model.setup(use_gpu=True)
    num_agents = args.num_agents
    num_init_connections = int(args.num_init_connections)
    num_nodes = args.num_nodes

    model_creation_start = time()
    model = generate_small_world_of_agents(
        model, num_agents, num_init_connections, int(0.1 * num_agents)
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
    model.simulate(10, sync_workers_every_n_ticks=1)
    simulate_end = time()
    simulate_duration = simulate_end - simulate_start

    if worker == 0:
        with open("execution_times.csv", "a") as f:
            f.write(
                f"{num_agents}, {num_init_connections}, {num_nodes}, {num_workers}, {model_creation_duration}, {simulate_duration}\n"
            )

    result = [
        SIRState(model.get_agent_property_value(agent_id, property_name="state"))
        for agent_id in range(num_agents)
        if model.get_agent_property_value(agent_id, property_name="state") is not None
    ]

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
