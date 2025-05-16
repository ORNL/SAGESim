from enum import Enum
from random import random
import cupy as cp
from cupyx import jit


# import the Breed class from sagesim
from sagesim.breed import Breed
from sagesim.utils import (
    get_this_agent_data_from_tensor,
    set_this_agent_data_from_tensor,
    get_neighbor_data_from_tensor,
)

from state import SIRState


class SIRBreed(Breed):

    def __init__(self) -> None:
        name = "SIR"
        super().__init__(name)
        self.register_property("state", SIRState.SUSCEPTIBLE.value)
        self.register_property("preventative_measures", [-1 for _ in range(100)])
        self.register_step_func(step_func, __file__)


@jit.rawkernel(device="cuda")
def step_func(
    agent_index, globals, agent_ids, breeds, locations, state_adt, preventative_measures_adt
):
    """
    At each simulation step, this function evaluates a subset of agents—either all agents in a serial run or a partition assigned to
    a specific rank in parallel processing—and determines whether an agent's state should change based on interactions with its neighbors
    and their respective preventative behaviors.

    Parameters:
    ----------
    agent_index : int
        Index of the agent being evaluated in the agent_ids list.
    globals : list
        Global parameters; 
        the zero-th global parameter is by default the simulation tick, 
        the first item will be our infection probability $p$.
    agent_ids : list[int]
        The adt that contains the IDs of all agents assigned to the current rank, and their neighbors.
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
    # Get the current agent's infection state
    my_state = get_this_agent_data_from_tensor(agent_index, state_adt)

    # Get the current agent's preventative measures vector
    my_preventative_measures = get_this_agent_data_from_tensor(
        agent_index, preventative_measures_adt
    )

    # Get the list of neighboring agent IDs
    neighbor_ids = locations[agent_index]

    # Draw a random number for stochastic infection decision
    rand = random()

    # Get the global infection probability (stored at index 1 in globals)
    p_infection = globals[1]

    # Loop through each neighbor of the agent
    for i in range(len(neighbor_ids)):
        neighbor_id = neighbor_ids[i]

        # Proceed only if the neighbor ID is valid (not NaN)
        if not cp.isnan(neighbor_id):
            # Retrieve the neighbor's infection state
            neighbor_state = get_neighbor_data_from_tensor(
                agent_ids, neighbor_id, state_adt
            )

            # Retrieve the neighbor's preventative measures vector
            neighbor_preventative_measures = get_neighbor_data_from_tensor(
                agent_ids, neighbor_id, preventative_measures_adt
            )

            # Compute the interaction safety score as the dot product between both vectors
            abs_safety_of_interaction = 0.0
            for n in range(len(my_preventative_measures)):
                for m in range(len(neighbor_preventative_measures)):
                    abs_safety_of_interaction += (
                        my_preventative_measures[n] * neighbor_preventative_measures[m]
                    )

            # Normalize safety score to [0, 1] by dividing by maximum possible interaction score
            normalized_safety_of_interaction = abs_safety_of_interaction / (
                len(my_preventative_measures) ** 2
            )

            # If the neighbor is infected and a random draw passes the risk-adjusted threshold,
            # then the current agent becomes infected (state = 2)
            if neighbor_state == 2 and rand < p_infection * (
                1 - normalized_safety_of_interaction
            ):
                set_this_agent_data_from_tensor(agent_index, state_adt, 2)
