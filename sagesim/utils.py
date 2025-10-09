"""
SAGESim Utility Functions

DEPRECATION NOTICE:
==================
The linear-search-based utility functions in this module (get_agent_idx,
get_neighbor_data_from_tensor, set_neighbor_data_from_tensor, etc.) are
now OBSOLETE due to SAGESim's agent ID-to-index conversion optimization.

As of the latest SAGESim version, the 'locations' property (and optionally
user-defined properties) are automatically converted from agent IDs to
local indices before being passed to GPU kernels.

OLD WAY (SLOW - DON'T USE):
    neighbor_id = locations[agent_index][0]  # Contains agent ID
    data = get_neighbor_data_from_tensor(agent_ids, neighbor_id, property_tensor)
    # ^ Performs linear search O(n) - VERY SLOW!

NEW WAY (FAST - USE THIS):
    neighbor_index = locations[agent_index][0]  # Already converted to index!
    data = property_tensor[neighbor_index]  # Direct access O(1) - FAST!

The conversion happens once per tick on CPU using hash maps (O(1) lookups),
eliminating the need for thousands of linear searches in GPU kernels.

Users should now access data directly using the pre-converted indices.
"""

from cupyx import jit


@jit.rawkernel(device="cuda")
def get_this_agent_data_from_tensor(agent_index, property_tensor):
    """
    Get data for the current agent. This is useful for readability.

    Usage: data = get_this_agent_data_from_tensor(agent_index, property_tensor)
    Or simply: data = property_tensor[agent_index]
    """
    return property_tensor[agent_index]


@jit.rawkernel(device="cuda")
def set_this_agent_data_from_tensor(agent_index, property_tensor, value):
    """
    Set data for the current agent. This is  useful for readability.

    Usage: set_this_agent_data_from_tensor(agent_index, property_tensor, value)
    Or simply: property_tensor[agent_index] = value
    """
    property_tensor[agent_index] = value


@jit.rawkernel(device="cuda")
def get_agent_idx(agent_ids, agent_id):
    """
    DEPRECATED: Linear search to find agent index from agent ID.

    This function performs O(n) linear search and is VERY SLOW in GPU kernels.

    MIGRATION: Use pre-converted indices from 'locations' property instead.
    SAGESim now converts agent IDs to local indices before passing to kernels.

    If you have an agent ID that needs conversion, it should already be
    converted in the locations array. Access it directly without search.

    """
    agent_index = -1
    i = 0
    while i < len(agent_ids) and int(agent_ids[i]) != int(agent_index):
        i += 1
    if i < len(agent_ids):
        agent_index = i
    return int(agent_index)


@jit.rawkernel(device="cuda")
def get_neighbor_data_from_tensor(agent_ids, neighbor_id, property_tensor):
    """
    DEPRECATED: Get neighbor data using linear search.

    This function performs O(n) linear search and is VERY SLOW in GPU kernels.

    OLD (SLOW):
        neighbor_id = locations[agent_index][0]
        data = get_neighbor_data_from_tensor(agent_ids, neighbor_id, property_tensor)

    NEW (FAST):
        neighbor_index = locations[agent_index][0]  # Already an index!
        data = property_tensor[neighbor_index]
    """
    neighbor_index = -1
    i = 0
    while i < len(agent_ids) and int(agent_ids[i]) != int(neighbor_id):
        i += 1
    if i < len(agent_ids):
        neighbor_index = i
    return property_tensor[neighbor_index]


@jit.rawkernel(device="cuda")
def set_neighbor_data_from_tensor(agent_ids, neighbor_id, property_tensor, value):
    """
    DEPRECATED: Set neighbor data using linear search.

    This function performs O(n) linear search and is VERY SLOW in GPU kernels.

    OLD (SLOW):
        neighbor_id = locations[agent_index][0]
        set_neighbor_data_from_tensor(agent_ids, neighbor_id, property_tensor, value)

    NEW (FAST):
        neighbor_index = locations[agent_index][0]  # Already an index!
        property_tensor[neighbor_index] = value
    """
    neighbor_index = -1
    i = 0
    while i < len(agent_ids) and int(agent_ids[i]) != int(neighbor_id):
        i += 1
    if i < len(agent_ids):
        neighbor_index = i
    property_tensor[neighbor_index] = value


@jit.rawkernel(device="cuda")
def set_neighbor_ids_for_network_space(agent_ids, neighbor_id, property_tensor, value):
    """
    DEPRECATED: Set neighbor data using linear search.

    This function performs O(n) linear search and is VERY SLOW in GPU kernels.

    MIGRATION: Use pre-converted indices from 'locations' property instead.
    """
    neighbor_index = -1
    i = 0
    while i < len(agent_ids) and int(agent_ids[i]) != int(neighbor_id):
        i += 1
    if i < len(agent_ids):
        neighbor_index = i
    property_tensor[neighbor_index] = value
