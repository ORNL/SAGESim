from sagesim.breed import Breed
import cupy as cp
from cupyx import jit
from pathlib import Path


class SFRBreed(Breed):

    def __init__(self) -> None:
        name = "SFR"
        super().__init__(name)

        # register osmnxid
        # self.register_property("osmnxid")

        # popularity of the agent, real value between [0,1]
        self.register_property("popularity", default=0.5)

        # number of vehicles at the current agent/node/intersection
        self.register_property("vehicle_num", default=10)

        # register step function
        curr_fpath = Path(__file__).resolve()
        self.register_step_func(step_func, curr_fpath)


# Define the step function to be registered for SIRBreed
@jit.rawkernel(device="cuda")
def step_func(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    popularities,
    vehicle_nums,
):

    # Get agent's vehicle number and neighbors' locations
    agent_vehicle_num = vehicle_nums[agent_index]
    # Zero out agent's vehicle count
    vehicle_nums[agent_index] = agent_vehicle_num
    neighbor_ids = locations[agent_index]

    # find total popularity of all neighbors
    total_popularity = cp.float64(0)
    neighbor_i = 0
    while not cp.isnan(neighbor_ids[neighbor_i]) and neighbor_i < len(neighbor_ids):
        neighbor_id = neighbor_ids[neighbor_i]
        # Find the index of the neighbor_id in agent_ids
        neighbor_index = -1
        i = 0
        while i < len(agent_ids) and agent_ids[i] != neighbor_id:
            i += 1
        if i < len(agent_ids):
            neighbor_index = i
            neighbor_popularity = popularities[int(neighbor_index)]
            total_popularity += neighbor_popularity
        neighbor_i += 1

    remainder = agent_vehicle_num
    largest_alloc = 0

    if total_popularity > 0:
        remainder_alloc_index = -1
        neighbor_i = 0
        while not cp.isnan(neighbor_ids[neighbor_i]) and neighbor_i < len(neighbor_ids):
            neighbor_id = neighbor_ids[neighbor_i]
            # Find the index of the neighbor_id in agent_ids
            neighbor_index = 0
            while (
                neighbor_index < len(agent_ids)
                and agent_ids[neighbor_index] != neighbor_id
            ):
                neighbor_index += 1
            if (neighbor_index < len(agent_ids)) and (agent_ids[i] != neighbor_id):
                neighbor_popularity = popularities[int(neighbor_index)]

                neighbor_allocation = int(
                    agent_vehicle_num * neighbor_popularity / total_popularity
                )
                # find the top popularity neighbor
                if neighbor_allocation > largest_alloc:
                    remainder_alloc_index = neighbor_index
                    largest_alloc = neighbor_allocation

                remainder -= neighbor_allocation
            neighbor_i += 1

        # Distribute the remainder (due to rounding) to top contributors
        if remainder > 0 and remainder_alloc_index >= 0:
            vehicle_nums[int(remainder_alloc_index)] += remainder
