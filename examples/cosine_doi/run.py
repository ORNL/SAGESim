from time import time
import argparse
from typing import OrderedDict
from cosine_doi_model import SIRModel
from state import SIRState
from mpi4py import MPI
from random import random
from random import sample
import networkx as nx
from dataloader import CascadeNetworkDataLoader

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


def generate_network_of_agents_with_dataloader(
    model, fpath_network, fpath_embeddings, fpath_cascade
) -> SIRModel:
    """Generate network of agents using CascadeNetworkDataLoader.

    Args:
        model: The SIRModel instance
        fpath_network: Path to network GraphML file
        fpath_embeddings: Path to embeddings JSON file
        fpath_cascade: Path to cascades JSON file
        num_infected: Number of initially infected agents

    Returns:
        SIRModel: The model with agents and network structure
    """
    # Create dataloader instance
    dataloader = CascadeNetworkDataLoader(
        fpath_cascade, fpath_network, fpath_embeddings
    )

    # Load the network with embeddings
    network = dataloader.load_network()

    # Create mapping from node names to agent IDs
    node_to_agent_id = OrderedDict()

    # Create agents for each node in the network
    for node_name in network.nodes():
        # Get node data including embedding
        node_data = network.nodes[node_name]
        embedding = node_data.get(
            "embedding", []
        ).tolist()  # Convert numpy array to list
        preventative_measures = [random() for _ in range(100)]

        # Create agent with embedding
        agent_id = model.create_agent(
            state=SIRState.SUSCEPTIBLE.value,
            preventative_measures=preventative_measures,
            embedding=embedding,
        )
        node_to_agent_id[node_name] = agent_id

    # Connect agents based on network edges
    for edge in network.edges():
        agent_0 = node_to_agent_id[edge[0]]
        agent_1 = node_to_agent_id[edge[1]]
        model.connect_agents(agent_0, agent_1)

    return model, dataloader, node_to_agent_id


def run_model_for_cascade(model, node_to_agent_id, cascade_name, cascade, ticks):
    """Run the SIR model for a specific cascade.

    Args:
        model: The SIRModel instance
        cascade: The cascade events to simulate

    Returns:
        None
    """
    # Set first event's agents as initially infected based on the cascade
    infected_agent = cascade[0][1]  # Get the first event's source agent
    infected_agent_id = node_to_agent_id[infected_agent]
    model.set_agent_property_value(infected_agent_id, "state", SIRState.INFECTED.value)

    # Run the simulation for a specified number of ticks
    simulate_start = time()
    model.simulate(ticks, sync_workers_every_n_ticks=ticks)
    simulate_end = time()
    simulate_duration = simulate_end - simulate_start

    if worker == 0:
        with open("execution_times.csv", "a") as f:
            f.write(
                f"{num_agents}, {network_stats['num_edges']}, {num_agents}, {num_workers}, {model_creation_duration}, {simulate_duration}\n"
            )

    if worker == 0:
        # Save infection history to CSV file
        import csv

        # Create CSV file with infection history
        with open(f"infection_history_{cascade_name}.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write header row
            header = ["agent_id"] + [
                f"tick_{i}" for i in range(ticks)
            ]  # 10 simulation steps
            writer.writerow(header)

            # Write infection history for each agent
            for agent_id in range(num_agents):
                try:
                    # Get the infection history for this agent using the model's method
                    agent_history = model.get_infection_history(agent_id)
                    if agent_history is not None:
                        # Convert to list of integers (state values)
                        history_values = [
                            int(state) if state is not None else 1
                            for state in agent_history[:ticks]
                        ]
                        # Pad with current state if history is shorter than simulation steps
                        while len(history_values) < ticks:
                            current_state = model.get_agent_property_value(
                                agent_id, "state"
                            )
                            history_values.append(
                                int(current_state) if current_state is not None else 1
                            )

                        row = [agent_id] + history_values
                        writer.writerow(row)
                except Exception as e:
                    # If no infection history available, use current state for all ticks
                    current_state = model.get_agent_property_value(agent_id, "state")
                    state_value = int(current_state) if current_state is not None else 1
                    row = [agent_id] + [state_value] * ticks
                    writer.writerow(row)

        print(
            f"Infection history saved to infection_history.csv for {num_agents} agents"
        )

        # Get final states of agents after simulation
        result = [
            SIRState(model.get_agent_property_value(agent_id, "state"))
            for agent_id in range(num_agents)
        ]

        # Print summary of final states
        final_states = {}
        for state in SIRState:
            count = sum(1 for r in result if r == state)
            final_states[state.name] = count

        print("Final state distribution:")
        for state_name, count in final_states.items():
            print(f"  {state_name}: {count}")


def run_model_for_all_cascades(model, node_to_agent_id, dataloader, ticks=1000):
    """Run the SIR model for all cascades in the dataloader.

    Args:
        model: The SIRModel instance
        dataloader: The CascadeNetworkDataLoader instance

    Returns:
        None
    """
    # Iterate over all cascades in the dataloader
    for cascade_name, cascade_events in dataloader.cascade_iterator():
        if worker == 0:
            print(
                f"Running simulation for cascade '{cascade_name}' with {len(cascade_events)} events"
            )
        run_model_for_cascade(
            model, node_to_agent_id, cascade_name, cascade_events, ticks
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fpath_network", type=str, required=True, help="Path to network GraphML file"
    )
    parser.add_argument(
        "--fpath_embeddings",
        type=str,
        required=True,
        help="Path to embeddings JSON file",
    )
    parser.add_argument(
        "--fpath_cascade", type=str, required=True, help="Path to cascade JSON file"
    )
    args = parser.parse_args()

    model = SIRModel()
    model.setup(use_gpu=True)

    model_creation_start = time()
    model, dataloader, node_to_agent_id = generate_network_of_agents_with_dataloader(
        model, args.fpath_network, args.fpath_embeddings, args.fpath_cascade
    )
    model_creation_end = time()
    model_creation_duration = model_creation_end - model_creation_start

    # Get network statistics
    network_stats = dataloader.get_network_stats()
    num_agents = network_stats["num_nodes"]

    if worker == 0:
        print(
            f"Loaded network with {num_agents} nodes and {network_stats['num_edges']} edges"
        )
        print(f"Embedding dimension: {network_stats['embedding_dimension']}")
        print(f"Number of cascades: {network_stats['num_cascades']}")
        print(f"Cascade names: {network_stats['cascade_names']}")

    # Run the model for all cascades
    run_model_for_all_cascades(model, node_to_agent_id, dataloader)
