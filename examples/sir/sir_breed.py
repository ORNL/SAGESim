from random import random

from sagesim.breed import Breed
from state import SIRState
from cupyx import jit
import cupy as cp
import bisect

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


def step_func(agent_ids, agent_index, globals, breeds, locations, states):

    neighbor_ids = locations[agent_index]  # network location is defined by neighbors
    rand = random()  # 0.1#step_func_helper_get_random_float(rng_states, id)

    p_infection = globals[1]
    for i in range(len(neighbor_ids)):

        neighbor_index = -1
        i = 0
        while i < len(agent_ids) and agent_ids[i] != neighbor_ids[0]:
            i += 1
        if i < len(agent_ids):
            neighbor_index = i
            neighbor_state = int(states[neighbor_index])
            if neighbor_state == 2 and rand < p_infection:
                states[agent_index] = 2
