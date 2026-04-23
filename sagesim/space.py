"""
This file describes the different spaces that SAGESim agents
may exist in. Space is important in calculating proximity.
Promixity is important in approximating interactions.
Intractions are important when binning agents into
workers, which is the crux of load balancing...

"""

from typing import List, Any, Union


class Space:
    """
    Abstract class for space

    """

    def __init__(
        self, neighbor_compute_func, locations_max_dims, locations_defaults
    ) -> None:
        """
        self_agent_adj_list hold
        """
        self._locations: List[Any] = []
        self._neighbor_compute_func = neighbor_compute_func
        self._locations_max_dims = locations_max_dims
        self._locations_defaults = locations_defaults
        self._agent_factory = None

    def get_location(self, agent_id: int) -> set:
        """Returns agents location"""
        return self._locations[agent_id]

    def add_agent(self, agent: int) -> None:
        """Adds agent to space"""
        self._locations = None

    def get_neighbors(self, agent_id: int) -> set:
        """Returns agents neighbors"""
        return self._neighbor_compute_func(self._locations, agent_id)

    ###NOTE: we need a function for load balancing


def _network_space_compute_neighbors(agent_locations):
    agents_current_neighbors = agent_locations
    return agents_current_neighbors


class NetworkSpace(Space):
    """Defines a NetworkSpace"""

    def __init__(self, ordered=False) -> None:
        """
        Uses super()._neighbors to hold adj list of the network

        Args:
            ordered (bool): If True, neighbors are stored as lists with insertion order preserved.
                           If False (default), neighbors are stored as sets (unordered, no duplicates).
                           Use ordered=True when neighbor order matters (e.g., directional relationships).
        """
        locations_max_dims = [0]
        locations_defaults = []
        self._ordered = ordered
        self._sparse = False
        # Parallel set for O(1) duplicate checks when ordered=True
        if ordered:
            self._locations_set = {}
        super().__init__(
            _network_space_compute_neighbors, locations_max_dims, locations_defaults
        )

    def add_local_agents(self, agent_ids) -> None:
        """Create location containers for specific local agents only.

        In this mode, _locations is a dict {agent_id: [neighbors]} instead of a
        list indexed by global agent ID. Only local agents get entries — remote
        agents referenced in edges don't need their own neighbor lists.

        :param agent_ids: Iterable of local agent IDs to create containers for.
        """
        self._sparse = True
        if self._ordered:
            self._locations = {int(aid): [] for aid in agent_ids}
            self._locations_set = {int(aid): set() for aid in agent_ids}
        else:
            self._locations = {int(aid): set() for aid in agent_ids}

    def add_agent(self, agent: int) -> None:
        if self._sparse:
            # In sparse mode, container already exists from add_local_agents().
            # Just link the locations property to the agent's property tensor.
            self._agent_factory.set_agent_property_value(
                "locations",
                agent,
                self._locations[agent],
            )
        else:
            # Original behavior
            neighbor_container = [] if self._ordered else set()
            self._locations.append(neighbor_container)
            if self._ordered:
                self._locations_set[agent] = set()
            self._agent_factory.set_agent_property_value(
                "locations",
                agent,
                self._locations[agent],
            )

    def get_location(self, agent_id: int) -> Union[List[int], set]:
        """Returns agent's location (neighbors as list if ordered=True, set if ordered=False)"""
        return self._locations[agent_id]

    def get_neighbors(self, agent_id: int) -> Union[List[int], set]:
        """Returns agent's neighbors (list if ordered=True, set if ordered=False)"""
        return self._neighbor_compute_func(self._locations, agent_id)

    def connect_agents(
        self, agent_0: int, agent_1: int, directed: bool = False
    ) -> None:
        agent_0 = int(agent_0)
        agent_1 = int(agent_1)

        if self._ordered:
            # For ordered neighbors, use parallel set for O(1) duplicate check
            if agent_1 not in self._locations_set[agent_0]:
                self._locations_set[agent_0].add(agent_1)
                self._locations[agent_0].append(agent_1)
            if not directed and agent_0 not in self._locations_set[agent_1]:
                self._locations_set[agent_1].add(agent_0)
                self._locations[agent_1].append(agent_0)
        else:
            # For unordered neighbors, add to set (automatically prevents duplicates)
            self._locations[agent_0].add(agent_1)
            if not directed:
                self._locations[agent_1].add(agent_0)

    def disconnect_agents(
        self, agent_0: int, agent_1: int, directed: bool = False
    ) -> None:
        agent_0 = int(agent_0)
        agent_1 = int(agent_1)

        if self._ordered:
            # For ordered neighbors (list), remove by value; keep set in sync
            if agent_1 in self._locations_set[agent_0]:
                self._locations_set[agent_0].discard(agent_1)
                self._locations[agent_0].remove(agent_1)
            if not directed and agent_0 in self._locations_set[agent_1]:
                self._locations_set[agent_1].discard(agent_0)
                self._locations[agent_1].remove(agent_0)
        else:
            # For unordered neighbors (set), remove directly
            self._locations[agent_0].discard(agent_1)  # discard won't raise error if not present
            if not directed:
                self._locations[agent_1].discard(agent_0)

    ###NOTE: we need a function for load balancing


if __name__ == "__main__":
    ns = NetworkSpace()
    for i in range(6):
        ns.add_agent(i)

    for i in [1, 3, 4]:
        ns.connect_agents(0, i)
