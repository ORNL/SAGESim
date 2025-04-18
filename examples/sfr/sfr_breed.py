from random import random

from sagesim.breed import Breed
from cupyx import jit
import math
import cupy as cp


class SFRBreed(Breed):

    def __init__(self) -> None:
        name = "SFR"
        super().__init__(name)

        # register osmnxid
        self.register_property("osmnxid")

        # popularity of the agent, real value between [0,1]
        self.register_property("popularity", default=0.5)

        # number of vehicles at the current agent/node/intersection
        self.register_property("vehicle_num", default=10)

        self.register_step_func(step_func)


def step_func(id, globals, breeds, locations, osmnxids, popularities, vehicle_nums):
    # Ensure id and id2index are valid by checking NaN or invalid values
    if id == id:
        agent_index = int(id)

        # Get agent's vehicle number and neighbors' locations
        agent_vehicle_num = vehicle_nums[agent_index]
        neighbors = locations[agent_index]

        # find total popularity of all neighbors
        total_popularity = cp.float64(0)
        for i in range(len(neighbors)):
            neighbor_index = neighbors[i]
            # neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
            if neighbor_index == neighbor_index:
                neighbor_popularity = popularities[int(neighbor_index)]

                total_popularity += neighbor_popularity

        remainder = agent_vehicle_num
        largest_alloc = 0
        if total_popularity > 0:
            for i in range(len(neighbors)):
                neighbor_index = neighbors[i]
                # neighbor_index = step_func_helper_get_agent_index(neighbor_id, id2index)
                if neighbor_index == neighbor_index:
                    neighbor_popularity = popularities[int(neighbor_index)]

                    neighbor_allocation = int(
                        agent_vehicle_num * neighbor_popularity / total_popularity
                    )
                    # vehicle_nums[int(neighbor_index)] += neighbor_allocation

                    # find the top popularity neighbor
                    if neighbor_allocation > largest_alloc:
                        remainder_alloc_index = neighbor_index

                    remainder -= neighbor_allocation

        # Distribute the remainder (due to rounding) to top contributors
        # vehicle_nums[int(remainder_alloc_index)] += remainder

        # Zero out agent's vehicle count
        vehicle_nums[agent_index] = agent_vehicle_num
