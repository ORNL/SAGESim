from random import random

from sagesim.breed import Breed
from state import SIRState
from cupyx import jit

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


def step_func(id, id2index, globals, breeds, locations, states):
    # agent_index = step_func_helper_get_agent_index(id, id2index)
    # nan checked by inequality to self. Unfortunate limitation of cupyx
    if (id == id) and (id2index[int(id)] == id2index[int(id)]):
        agent_index = int(id2index[int(id)])

        neighbors = locations[agent_index]  # network location is defined by neighbors
        rand = random()  # 0.1#step_func_helper_get_random_float(rng_states, id)

        p_infection = globals[1]

        for i in range(len(neighbors)):
            neighbor_id = neighbors[i]
            # neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
            if (neighbor_id == neighbor_id) and (
                id2index[int(neighbor_id)] == id2index[int(neighbor_id)]
            ):
                neighbor_index = int(id2index[int(neighbor_id)])
                neighbor_state = states[neighbor_index]
                if neighbor_state == 2 and rand < p_infection:
                    states[agent_index] = 2
