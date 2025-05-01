from random import random

from sagesim.breed import Breed
from state import SIRState
from cupyx import jit
import cupy as cp
from random import random

global INFECTED
INFECTED = SIRState.INFECTED.value
global SUSCEPTIBLE
SUSCEPTIBLE = SIRState.SUSCEPTIBLE.value


class SIRBreed(Breed):

    def __init__(self) -> None:
        name = "SIR"
        super().__init__(name)
        self.register_property("state", SIRState.SUSCEPTIBLE.value)
        self.register_property("preventative_measures", [-1 for _ in range(10000)])
        self.register_step_func(step_func)


def step_func(
    agent_ids, agent_index, globals, breeds, locations, states, preventative_measures
):

    neighbor_ids = locations[agent_index]  # network location is defined by neighbors
    rand = random()  # 0.1#step_func_helper_get_random_float(rng_states, id)

    p_infection = globals[1]

    agent_preventative_measures = preventative_measures[agent_index]

    for i in range(len(neighbor_ids)):

        neighbor_index = -1
        i = 0
        while i < len(agent_ids) and agent_ids[i] != neighbor_ids[0]:
            i += 1
        if i < len(agent_ids):
            neighbor_index = i
            neighbor_state = int(states[neighbor_index])
            neighbor_preventative_measures = preventative_measures[neighbor_index]
            abs_safety_of_interaction = 0.0
            for j in range(len(agent_preventative_measures)):
                abs_safety_of_interaction += (
                    agent_preventative_measures[j] + neighbor_preventative_measures[j]
                )
            normalized_safety_of_interaction = abs_safety_of_interaction / (
                2 * len(agent_preventative_measures)
            )
            if neighbor_state == 2 and rand < p_infection * (
                1 - normalized_safety_of_interaction
            ):
                states[agent_index] = 2
