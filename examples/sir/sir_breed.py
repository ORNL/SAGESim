from enum import Enum
from random import random

# import the Breed class from sagesim
from sagesim.breed import Breed
from state import SIRState

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
