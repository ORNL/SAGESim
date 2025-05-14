from enum import Enum
from time import time
from random import random, sample
import networkx as nx
from statistics import mean
from mpi4py import MPI



# import the Breed class from sagesim
from sagesim.breed import Breed

# Define the SIRState enumeration for agent states
class SIRState(Enum):
    SUSCEPTIBLE = 1
    INFECTED = 2
    RECOVERED = 3

# Define the step function to be registered for SIRBreed
def step_func(agent_ids, agent_index, globals, breeds, locations, state_adt, preventative_measures_adt):
    """
    At each simulation step, this function evaluates a subset of agents—either all agents in a serial run or a partition assigned to
    a specific rank in parallel processing—and determines whether an agent's state should change based on interactions with its neighbors
    and their respective preventative behaviors.

    Parameters:
    ----------
    agent_ids : list[int]
        The adt that contains the IDs of all agents assigned to the current rank, and their neighbors.
    agent_index : int
        Index of the agent being evaluated in the agent_ids list. 
    globals : list
        Global parameters; 
        the zero-th global parameter is by default the simulation tick, 
        the first item will be our infection probability $p$.
        the second item is the susceptibility reduction strength $alpha$.
        the third item is the infectiousness reduction strength $beta$.
    breeds : list
        List of breed objects (unused here as we only have one type of breed, but must passed for interface compatibility).
    locations : list[list[int]]
        Adjacency list specifying neighbors for each agent.
    state_adt : list[int]
        List of current state of each agent.
    preventative_measures_adt : list[list[float]]
        List of vectors representing each agent’s preventative behaviors. 
    Returns:
    -------
    None
        The function updates the `states` list in-place if an agent becomes infected.
    """
    # Get the list of neighboring agent IDs for the current agent based on network topology
    neighbor_ids = locations[agent_index]

    # Draw a random float in [0, 1) for stochastic decision-making
    rand = random()  # can replace with step_func_helper_get_random_float(rng_states, id)

    # Retrieve the global infection probability defined in the model
    p_infection = globals[1]

    # Get the preventative measures vector for the current agent
    agent_preventative_measures = preventative_measures_adt[agent_index]

    # Loop through each neighbor ID
    for i in range(len(neighbor_ids)):

        # Initialize neighbor_index to invalid value
        neighbor_index = -1

        i = 0
        while i < len(agent_ids) and agent_ids[i] != neighbor_ids[0]:
            i += 1
        if i < len(agent_ids):
            neighbor_index = i

            # Retrieve the state of the neighbor (e.g., susceptible, infected, recovered)
            neighbor_state = int(state_adt[neighbor_index])

            # Get the preventative measures vector of the neighbor
            neighbor_preventative_measures = preventative_measures_adt[neighbor_index]

            # Initialize cumulative safety score for the interaction
            abs_safety_of_interaction = 0.0

            # Calculate total safety of interaction based on pairwise product of measures
            for n in range(len(agent_preventative_measures)):
                for m in range(len(neighbor_preventative_measures)):
                    abs_safety_of_interaction += (
                        agent_preventative_measures[n] * neighbor_preventative_measures[m]
                    )

            # Normalize the safety score to be in [0, 1]
            normalized_safety_of_interaction = abs_safety_of_interaction / (
                len(agent_preventative_measures) ** 2
            )

            # If neighbor is infected and the infection condition passes, update agent’s state
            if neighbor_state == 2 and rand < p_infection * (
                1 - normalized_safety_of_interaction
            ):
                state_adt[agent_index] = 2  # Agent becomes infected



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
        # Register the step function
        self.register_step_func(step_func)



from sagesim.model import Model
from sagesim.space import NetworkSpace # hopefully, we can avoid this import


class SIRModel(Model):
    """
    SIRModel class for the SIR model.
    Inherits from the Model class in the sagesim library.
    """

    def __init__(self, p_infection=1.0) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()

        # Register the breed
        self.register_breed(breed=self._sir_breed)

        # register user-defined global properties
        self.register_global_property("p_infection", p_infection)

    # create_agent method takes user-defined properties, that is, the state and preventative_measures, to create an agent
    def create_agent(self, state, preventative_measures):
        agent_id = self.create_agent_of_breed(
            self._sir_breed, state=state, preventative_measures=preventative_measures
        )
        self.get_space().add_agent(agent_id)
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)


num_agents = 1000
num_init_connections = 20
rewiring_prob = 0.1

num_infected = 10

# Generate the Contact Network
network = nx.watts_strogatz_graph(num_agents, num_init_connections, rewiring_prob)

# Instantiate the SIR Model
model = SIRModel()
model.setup(use_gpu=True)  # Enables GPU acceleration if available

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



# # MPI environment setup
comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()

# Run the simulation with 1 rank, and measure the time taken
simulate_start = time()
model.simulate(ticks = 10, sync_workers_every_n_ticks=1)
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


