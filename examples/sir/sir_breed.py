from random import random
import cupy as cp
from cupyx import jit


from sagesim.breed import Breed
from sagesim.utils import (
    get_this_agent_data_from_tensor,
    set_this_agent_data_from_tensor,
    get_neighbor_data_from_tensor,
)

from state import SIRState

global INFECTED
INFECTED = SIRState.INFECTED.value
global SUSCEPTIBLE
SUSCEPTIBLE = SIRState.SUSCEPTIBLE.value


class SIRBreed(Breed):

    def __init__(self) -> None:
        name = "SIR"
        super().__init__(name)
        self.register_property("state", SIRState.SUSCEPTIBLE.value)
        self.register_property("preventative_measures", [-1 for _ in range(100)])
        self.register_step_func(step_func, __file__)


@jit.rawkernel(device="cuda")
def step_func(
    agent_index, globals, agent_ids, breeds, locations, states, preventative_measures
):
    my_state = get_this_agent_data_from_tensor(agent_index, states)
    my_preventative_measures = get_this_agent_data_from_tensor(
        agent_index, preventative_measures
    )
    neighbor_ids = locations[agent_index]
    rand = random()  # 0.1#step_func_helper_get_random_float(rng_states, id)

    p_infection = globals[1]

    for i in range(len(neighbor_ids)):
        neighbor_id = neighbor_ids[i]
        if not cp.isnan(neighbor_id):
            neighbor_state = get_neighbor_data_from_tensor(
                agent_ids, neighbor_id, states
            )
            neighbor_preventative_measures = get_neighbor_data_from_tensor(
                agent_ids, neighbor_id, preventative_measures
            )

            abs_safety_of_interaction = 0.0
            for n in range(len(my_preventative_measures)):
                for m in range(len(neighbor_preventative_measures)):
                    abs_safety_of_interaction += (
                        my_preventative_measures[n] * neighbor_preventative_measures[m]
                    )
            normalized_safety_of_interaction = abs_safety_of_interaction / (
                len(my_preventative_measures) ** 2
            )
            if neighbor_state == 2 and rand < p_infection * (
                1 - normalized_safety_of_interaction
            ):
                set_this_agent_data_from_tensor(agent_index, states, 2)
