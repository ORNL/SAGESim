from random import random

from sagesim.breed import Breed
from cupyx import jit


class SFRBreed(Breed):

    def __init__(self) -> None:
        name = "SFR"
        super().__init__(name)

        # register osmnxid 
        self.register_property("osmnxid")

        # popularity of the agent, real value between [0,1]
        self.register_property("popularity", 
                               default=0.5)
        
        # number of vehicles at the current agent/node/intersection
        self.register_property("vehicle_num", 
                               default=10)
        

        self.register_step_func(step_func)


def step_func(id, id2index, globals, breeds, locations, osmnxids, popularities, vehicle_nums):
    ### Must have entities: id, id2index, globals, breeds, locations
    # agent_index = step_func_helper_get_agent_index(id, id2index)
    # nan checked by inequality to self. Unfortunate limitation of cupyx
    if (id == id) and (id2index[int(id)] == id2index[int(id)]):
        agent_index = int(id2index[int(id)])

        # agent vehicle_num
        agent_vehicle_num = vehicle_nums[agent_index]
        neighbors = locations[agent_index]  # network location is defined by neighbors  

        # create a list to get eh popularity of all neighbors
        # [NOTE]: if the list and append operation does not work, create Cupy aarray
        neighbor_popularities = []
        for i in range(len(neighbors)):
            neighbor_id = neighbors[i]
            # neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
            if (neighbor_id == neighbor_id) and (
                id2index[int(neighbor_id)] == id2index[int(neighbor_id)]
            ):
                neighbor_index = int(id2index[int(neighbor_id)])
                neighbor_popularity = popularities[neighbor_index]
                neighbor_popularities.append((neighbor_index, neighbor_popularity))
        


