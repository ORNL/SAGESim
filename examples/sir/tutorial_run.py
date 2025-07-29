from time import time
from random import random, sample

import networkx as nx
from mpi4py import MPI

from sir_model import SIRModel
from state import SIRState

if __name__ == "__main__":

    num_agents = 1000
    num_init_connections = 2
    rewiring_prob = 0.1

    num_infected = 10

    # Generate the Contact Network
    network = nx.watts_strogatz_graph(
        num_agents, num_init_connections, rewiring_prob, seed=42
    )

    # Instantiate the SIR Model
    model = SIRModel(p_infection=0.6, p_recovery=0.00)

    # Create agents
    for n in network.nodes:
        preventative_measures = [random() for _ in range(100)]
        model.create_agent(SIRState.SUSCEPTIBLE.value, preventative_measures)

    # Connect agents in the network
    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])

    # Infect a random sample of agents
    for n in sample(sorted(network.nodes), num_infected):
        model.set_agent_property_value(n, "state", SIRState.INFECTED.value)

    model.setup(use_gpu=True)  # Enables GPU acceleration if available

    # # MPI environment setup
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    worker = comm.Get_rank()

    # Run the simulation with 1 rank, and measure the time taken
    simulate_start = time()
    model.simulate(ticks=10, sync_workers_every_n_ticks=1)
    simulate_end = time()
    simulate_duration = simulate_end - simulate_start

    result = [
        SIRState(model.get_agent_property_value(agent_id, property_name="state"))
        for agent_id in range(num_agents)
        if model.get_agent_property_value(agent_id, property_name="state") is not None
    ]

    # count the number of infected agents
    num_infected = sum(1 for state in result if state == SIRState.INFECTED)
    num_recovered = sum(1 for state in result if state == SIRState.RECOVERED)
    num_susceptible = sum(1 for state in result if state == SIRState.SUSCEPTIBLE)

    if worker == 0:
        print(f"Simulation took {simulate_duration:.2f} seconds.")
        print(f"Number of infected agents: {num_infected}")
        print(f"Number of recovered agents: {num_recovered}")
        print(f"Number of susceptible agents: {num_susceptible}")
