from random import random

from sagesim.breed import Breed
from cupyx import jit
import math


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

# import cupy as cp
# def step_func(id, id2index, globals, breeds, locations, osmnxids, popularities, vehicle_nums):
#     ### Must have entities: id, id2index, globals, breeds, locations
#     # agent_index = step_func_helper_get_agent_index(id, id2index)
#     # nan checked by inequality to self. Unfortunate limitation of cupyx
#     if (id == id) and (id2index[int(id)] == id2index[int(id)]):
#         agent_index = int(id2index[int(id)])

#         # agent vehicle_num
#         agent_vehicle_num = vehicle_nums[agent_index]
#         neighbors = locations[agent_index]  # network location is defined by neighbors  

#         # create a list to get eh popularity of all neighbors
#         # [NOTE]: if the list and append operation does not work, create Cupy aarray
#         neighbor_indexes = cp.zeros(neighbors.shape)
#         neighbor_popularities = cp.zeros(neighbors.shape)

#         for i in range(len(neighbors)):
#             neighbor_id = neighbors[i]
#             # neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
#             if (neighbor_id == neighbor_id) and (
#                 id2index[int(neighbor_id)] == id2index[int(neighbor_id)]
#             ):
#                 neighbor_index = int(id2index[int(neighbor_id)])
#                 neighbor_popularity = popularities[neighbor_index]

#                 neighbor_indexes[i]=neighbor_index
#                 neighbor_popularities[i]=neighbor_popularity

        
#         if len(neighbor_indexes) > 0 and total_popularity > 0:
#             total_popularity = sum(neighbor_popularities)
#             # Initial allocation (float), then round down to ensure non-negativity
#             raw_allocations = [
#                 (agent_vehicle_num * p / total_popularity) for p in neighbor_popularities
#             ]
#             int_allocations = [int(x) for x in raw_allocations]
#             allocated = sum(int_allocations)
#             remainder = agent_vehicle_num - allocated

#             # Distribute the remainder (due to rounding) to top contributors
#             fractional_parts = [
#                 (raw - intg, idx) for raw, intg, idx in zip(raw_allocations, int_allocations, range(len(int_allocations)))
#             ]
#             fractional_parts.sort(reverse=True)  # sort by descending fractional part

#             for i in range(remainder):
#                 _, idx = fractional_parts[i]
#                 int_allocations[idx] += 1

#             # Apply the transfer
#             for i, neighbor_index in enumerate(neighbor_indexes):
#                 vehicle_nums[neighbor_index] += int_allocations[i]

#             # Zero out agent's vehicle count
#             vehicle_nums[agent_index] = 0

import cupy as cp
def step_func(id, id2index, globals, breeds, locations, osmnxids, popularities, vehicle_nums):
    # Ensure id and id2index are valid by checking NaN or invalid values
    if (id == id) and (id2index[int(id)] == id2index[int(id)]):
        agent_index = int(id2index[int(id)])

    # Get agent's vehicle number and neighbors' locations
    agent_vehicle_num = vehicle_nums[agent_index]
    neighbors = locations[agent_index]

    # find total popularity of all neighbors
    total_popularity = cp.float64(0)
    for i in range(len(neighbors)):
        neighbor_id = neighbors[i]
        # neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
        if (neighbor_id == neighbor_id) and (
            id2index[int(neighbor_id)] == id2index[int(neighbor_id)]
        ):
            neighbor_index = int(id2index[int(neighbor_id)])
            neighbor_popularity = popularities[neighbor_index]

            total_popularity += neighbor_popularity
    
    remainder = agent_vehicle_num
    largest_alloc = 0
    if total_popularity > 0:
        for i in range(len(neighbors)):
            neighbor_id = neighbors[i]
            # neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
            if (neighbor_id == neighbor_id) and (
                id2index[int(neighbor_id)] == id2index[int(neighbor_id)]
            ):
                neighbor_index = int(id2index[int(neighbor_id)])
                neighbor_popularity = popularities[neighbor_index]

                neighbor_allocation = int(agent_vehicle_num * neighbor_popularity / total_popularity)
                vehicle_nums[neighbor_index] += neighbor_allocation

                # find the top popularity neighbor
                if neighbor_allocation > largest_alloc:
                    remainder_alloc_index = neighbor_index
                
                remainder -= neighbor_allocation

    # Distribute the remainder (due to rounding) to top contributors
    vehicle_nums[remainder_alloc_index] += remainder

    # Zero out agent's vehicle count
    vehicle_nums[agent_index] = 0

