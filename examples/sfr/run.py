from time import time
import argparse
from typing import OrderedDict
from sfr_model import SFRModel
from mpi4py import MPI
import random
from random import sample
import json
import pandas as pd

import networkx as nx

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


def generate_small_world_of_agents(model, network) -> SFRModel:
    # create agent with random popularity value and init number of vehicles
    for i, n in enumerate(network.nodes):
        model.create_agent(
            # osmnxid=network.nodes[n]["osmnxid"],
            popularity=random.random(),
            vehicle_num=random.randint(0, 10),
        )

    for edge in network.edges:
        if (
            edge[0] < model._agent_factory.num_agents
            and edge[1] < model._agent_factory.num_agents
        ):
            model.connect_agents(edge[0], edge[1])
    return model


if __name__ == "__main__":

    # load the sf drivng network
    # load json file
    with open(f"sfr.json", "r") as f:
        G_info = json.load(f)

    nodes = G_info["nodes"]
    edges = G_info["edges"]

    # create a muliiDiGraph using the node and edge lists
    G = nx.MultiDiGraph()

    # List of nodes
    G.add_nodes_from(nodes)

    # Add edges with attributes
    for u, v, k, length, maxspeed in edges:
        G.add_edge(u, v, key=k, length=length, maxspeed=maxspeed)

    # create a identical network with integer node id
    # Create mapping from original node ID to integers
    original_nodes = list(nodes)
    osmnx2idx = OrderedDict((node, idx) for idx, node in enumerate(original_nodes))
    osmnxids = list(osmnx2idx.keys())

    # Create a new MultiDiGraph with masked node IDs
    # [TO-do] Maybe a easier way to re-key the nodes
    G_masked = nx.MultiDiGraph()
    G_masked.add_nodes_from(range(len(original_nodes)))

    for osmnxid, masked_id in osmnx2idx.items():
        G_masked.add_node(masked_id, osmnxid=osmnxid)

    # Add edges using the new node IDs and preserve attributes
    for u, v, k, data in G.edges(keys=True, data=True):
        new_u = osmnx2idx[u]
        new_v = osmnx2idx[v]
        G_masked.add_edge(new_u, new_v, key=k, **data)

    model = SFRModel()
    model.setup(use_gpu=True)
    model = generate_small_world_of_agents(model, G_masked)

    simulate_start = time()
    model.simulate(10, sync_workers_every_n_ticks=1)
    simulate_end = time()
    simulate_duration = simulate_end - simulate_start

    if worker == 0:
        model_creation_start = time()
        model_creation_duration = model_creation_start - simulate_start
        print(f"Model creation took {model_creation_duration} seconds")
        print(f"Simulation took {simulate_duration} seconds")

    checked = set()
    result = []
    for agent_idx in range(len(nodes)):
        vn = model.get_agent_property_value(agent_idx, property_name="vehicle_num")
        osmnxid = osmnxids[agent_idx]
        if osmnxid in checked:
            print(f"Duplicate osmnxid found: {osmnxid}")
            exit()
        checked.add(osmnxid)
        result.append(
            {
                "osmnxid": osmnxid,
                "popularity": model.get_agent_property_value(
                    agent_idx, property_name="popularity"
                ),
                "vehicle_num": vn,
            }
        )
    df = pd.DataFrame(result)
    df.to_csv("sfr_results.csv", index=False)

    # if worker == 0:
    #     with open("execution_times.csv", "a") as f:
    #         f.write(
    #             f"{num_agents}, {num_edges}, {num_nodes}, {num_workers}, {model_creation_duration}, {simulate_duration}\n"
    #         )

    # if worker == 0:
    #     print(
    #         [
    #             SFRState(
    #                 model.get_agent_property_value(agent_id, property_name="state")
    #             )
    #             for agent_id in range(num_agents)
    #         ]
    #     )
