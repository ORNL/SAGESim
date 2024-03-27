import argparse
from examples.sir.sir_model import SIRModel
from examples.sir.state import SIRState

from random import sample

import networkx as nx


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


def test_network():
    network = nx.Graph()
    network.add_nodes_from(range(8))
    network.add_edges_from(
        [(0, 1), (0, 4), (2, 3), (3, 4), (4, 5), (4, 7), (6, 5), (7, 6)]
    )

    for n in network.nodes:
        if n == 0:
            model.create_agent(SIRState.INFECTED.value)
        else:
            model.create_agent(SIRState.SUSCEPTIBLE.value)

    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])
        print(edge[0], "->", edge[1])
    return model


def generate_small_world_of_agents(
    model, n_agents: int, num_infected: float
) -> SIRModel:
    network = generate_small_world_network(n_agents, 2, 0.2)
    for n in network.nodes:
        model.create_agent(SIRState.SUSCEPTIBLE.value)

    for n in sample(network.nodes, num_infected):
        model.set_agent_property_value(n, "state", SIRState.INFECTED.value)

    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "scheduler_fpath",
        help="Optional path to scheduler path of Dask cluster",
        required=False,
    )
    args = parser.parse_args()

    model = SIRModel()
    model.setup(use_cuda=True, num_dask_worker=4, scheduler_fpath=args.scheduler_fpath)
    n_agents = 1000
    model = generate_small_world_of_agents(model, n_agents, 1)  # test_network()  #
    """print(
        [
            SIRState(model.get_agent_property_value(agent_id, property_name="state"))
            for agent_id in range(n_agents)
        ]
    )"""
    model.simulate(300, sync_workers_every_n_ticks=100)
    print(
        [
            SIRState(model.get_agent_property_value(agent_id, property_name="state"))
            for agent_id in range(n_agents)
        ]
    )
