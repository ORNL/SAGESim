from cupyx import jit


@jit.rawkernel(device="cuda")
def get_this_agent_data_from_tensor(agent_index, property_tensor):
    return property_tensor[agent_index]


@jit.rawkernel(device="cuda")
def set_this_agent_data_from_tensor(agent_index, property_tensor, value):
    property_tensor[agent_index] = value


@jit.rawkernel(device="cuda")
def get_neighbor_data_from_tensor(agent_ids, neighbor_id, property_tensor):
    neighbor_index = -1
    i = 0
    while i < len(agent_ids) and int(agent_ids[i]) != int(neighbor_id):
        i += 1
    if i < len(agent_ids):
        neighbor_index = i
    return property_tensor[neighbor_index]


@jit.rawkernel(device="cuda")
def set_neighbor_ids_for_network_space(agent_ids, neighbor_id, property_tensor, value):
    neighbor_index = -1
    i = 0
    while i < len(agent_ids) and int(agent_ids[i]) != int(neighbor_id):
        i += 1
    if i < len(agent_ids):
        neighbor_index = i
    property_tensor[neighbor_index] = value
