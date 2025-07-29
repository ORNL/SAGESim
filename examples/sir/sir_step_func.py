from random import random

import cupy as cp
from cupyx import jit
from sagesim.utils import (
    get_this_agent_data_from_tensor,
    set_this_agent_data_from_tensor,
    get_neighbor_data_from_tensor,
)


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
):
    """
    At each simulation step, this function evaluates a subset of agents—either all agents in a serial run or a partition assigned to
    a specific rank in parallel processing—and determines whether an agent's state should change based on interactions with its neighbors
    and their respective preventative behaviors.

    Parameters:
    ----------
    tick : int
        The current simulation tick, which is the first item in the globals array.
    agent_index : int
        Index of the agent being evaluated in the agent_ids list.
    globals : cupy.array
        Global parameters;
        the first item will be our infection probability.
        the second item will be our recovery probability.
    agent_ids : cupy.array
        The tensor that contains the IDs of all agents assigned to the current rank, and their neighbors.
    breeds : cupy.array
        1D cupy.array of breed objects (unused here as we only have one type of breed, but must passed for interface compatibility).
    locations : cupy.array
        Adjacency list specifying neighbors for each agent, padded with cp.nan.
    state_tensor : cupy.array
        1D cupy array of current state of each agent.
    preventative_measures_tensor : cupy.array[]
        2D cupy array of vectors representing each agent’s preventative behaviors.
    Returns:
    -------
    None
        The function updates the `states` list in-place if an agent becomes infected.
    """
    # Get the list of neighboring agent IDs for the current agent based on network topology
    neighbor_ids = locations[agent_index]

    # Draw a random float in [0, 1) for stochastic decision-making
    rand = (
        random()
    )  # can replace with step_func_helper_get_random_float(rng_states, id)

    # Retrieve the global infection and recovery probabilities defined in the model
    p_infection = globals[0]
    p_recovery = globals[1]

    # Get the current state of the agent (e.g., susceptible, infected, recovered)
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))

    # Get the preventative measures vector for the current agent
    agent_preventative_measures = get_this_agent_data_from_tensor(
        agent_index, preventative_measures_tensor
    )

    # If agent is infected and the recovery condition passes, update agent’s state
    if agent_state == 2 and rand < p_recovery:
        set_this_agent_data_from_tensor(
            agent_index, state_tensor, 3
        )  # Agent becomes infected
    elif agent_state == 1:
        # Loop through each neighbor ID
        i = 0
        while i < len(neighbor_ids) and not cp.isnan(neighbor_ids[i]):
            neighbor_id = neighbor_ids[i]

            # Retrieve the state of the neighbor (e.g., susceptible, infected, recovered)
            neighbor_state = int(
                get_neighbor_data_from_tensor(agent_ids, neighbor_id, state_tensor)
            )

            # Get the preventative measures vector of the neighbor
            neighbor_preventative_measures = get_neighbor_data_from_tensor(
                agent_ids, neighbor_id, preventative_measures_tensor
            )

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

            # If neighbor is infected and the infection condition passes, update agent’s state
            if neighbor_state == 2 and rand < p_infection * (
                1 - normalized_safety_of_interaction
            ):
                set_this_agent_data_from_tensor(
                    agent_index, state_tensor, 2
                )  # Agent becomes infected
            i += 1
