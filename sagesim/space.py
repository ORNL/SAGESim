"""
This file describes the different spaces that SAGESim agents
may exist in. Space is important in calculating proximity.
Promixity is important in approximating interactions.
Intractions are important when binning agents into
workers, which is the crux of load balancing...

"""

from typing import List, Any


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

    def _compute_partitioning(self):
        """Compute partitioning of the space for load balancing"""
        # Placeholder for partitioning logic
        pass


def _network_space_compute_neighbors(agent_locations):
    agents_current_neighbors = agent_locations
    return agents_current_neighbors


class NetworkSpace(Space):
    """Defines a NetworkSpace"""

    def __init__(self) -> None:
        """
        Uses super()._neighbors to hold adj list of the network
        """
        locations_max_dims = [0]
        locations_defaults = []
        super().__init__(
            _network_space_compute_neighbors, locations_max_dims, locations_defaults
        )

    def add_agent(self, agent: int) -> None:
        self._locations.append(set())

    def connect_agents(
        self, agent_0: int, agent_1: int, directed: bool = False
    ) -> None:
        agent_0 = int(agent_0)
        agent_1 = int(agent_1)
        self._locations[agent_0].add(agent_1)
        self._agent_factory.set_agent_property_value(
            "locations",
            agent_0,
            self._locations[agent_0],
        )
        if not directed:
            self._locations[agent_1].add(agent_0)
            self._agent_factory.set_agent_property_value(
                "locations",
                agent_1,
                self._locations[agent_1],
            )

    def disconnect_agents(
        self, agent_0: int, agent_1: int, directed: bool = False
    ) -> None:
        self._locations[agent_0].remove(agent_1)
        if not directed:
            self._locations[agent_1].remove(agent_0)

    def _compute_partitioning(self, k: int = 2) -> List[set]:
        """Compute partitioning of the network space for load balancing"""
        all_agent_ids = [int(i) for i in range(len(self._locations))]
        chunk_size = len(all_agent_ids) // k
        agent_ids_chunks = [
            all_agent_ids[i * chunk_size : (i + 1) * chunk_size] for i in range(k - 1)
        ] + [all_agent_ids[(k - 1) * chunk_size :]]
        return agent_ids_chunks


if __name__ == "__main__":
    ns = NetworkSpace()
    for i in range(6):
        ns.add_agent(i)

    for i in range(6):
        print(f"Neighbors of {i}: {ns.get_neighbors(i)}")

    for i in [1, 3, 4]:
        ns.connect_agents(0, i)

    for i in range(6):
        print(f"Neighbors of {i}: {ns.get_neighbors(i)}")
