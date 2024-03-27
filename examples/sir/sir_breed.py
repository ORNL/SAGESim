from math import isnan

from sagesim.breed import Breed
from examples.sir.state import SIRState
from sagesim.step_func_utils import (
    step_func_helper_get_agent_index,
    step_func_helper_get_random_float,
)
from numba import cuda

global INFECTED
INFECTED = SIRState.INFECTED.value
global SUSCEPTIBLE
SUSCEPTIBLE = SIRState.SUSCEPTIBLE.value


class SIRBreed(Breed):

    def __init__(self) -> None:
        name = "SIR"
        super().__init__(name)
        self.register_property("state", SIRState.SUSCEPTIBLE.value)
        self.register_step_func(step_func)


def step_func(id, id2index, rng_states, globals, breeds, locations, states):
    agent_index = step_func_helper_get_agent_index(id, id2index)
    if isnan(agent_index):
        return
    else:
        agent_index = int(agent_index)
    neighbors = locations[agent_index]  # network location is defined by neighbors
    rand = step_func_helper_get_random_float(rng_states, id)
    p_infection = globals[1]
    for i in range(len(neighbors)):
        neighbor_id = neighbors[i]
        if isnan(neighbor_id):
            break
        neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
        if isnan(neighbor_index):
            break
        neighbor_state = states[int(neighbor_index)]
        if neighbor_state == 2 and rand < p_infection:
            states[agent_index] = 2.0
