import unittest
from unittest.mock import MagicMock
from sagesim.space import NetworkSpace
from sagesim.agent import AgentFactory


class TestNetworkSpace(unittest.TestCase):
    def setUp(self):
        # Initialize NetworkSpace
        self.network_space = NetworkSpace()
        # Create a dummy AgentFactory and assign it to the NetworkSpace
        self.agent_factory = AgentFactory(self.network_space)
        self.network_space._agent_factory = self.agent_factory

        # Mock the set_agent_property_value to avoid unnecessary logic
        self.agent_factory.set_agent_property_value = MagicMock()

        # Add 3 agents to the network
        for _ in range(3):
            self.network_space.add_agent(_)

    def test_add_agent(self):
        # After adding 3 agents, there should be 3 locations
        self.assertEqual(len(self.network_space._locations), 3)
        for location in self.network_space._locations:
            self.assertEqual(location, set())

    def test_connect_agents_undirected_network(self):
        # Connect agent 0 and agent 1
        self.network_space.connect_agents(0, 1)

        # Verify connections in both directions
        self.assertIn(1, self.network_space._locations[0])
        self.assertIn(0, self.network_space._locations[1])

    def test_connect_agents_directed_network(self):
        # Reset the mock call count
        self.agent_factory.set_agent_property_value.reset_mock()
        # Connect agent 0 and agent 1 in a directed manner
        self.network_space.connect_agents(0, 1, directed=True)

        # Verify connection from 0 to 1
        self.assertIn(1, self.network_space._locations[0])
        # Verify that agent 1 does not have a connection back to agent 0
        self.assertNotIn(0, self.network_space._locations[1])

    # check if agent_factory.set_agent_property_value call is needed
    def test_disconnect_agents(self):
        self.network_space.connect_agents(0, 1)
        self.network_space.disconnect_agents(0, 1)

        # After disconnect, they should not be neighbors anymore
        self.assertNotIn(1, self.network_space._locations[0])
        self.assertNotIn(0, self.network_space._locations[1])

    def test_disconnect_agents_directed(self):
        self.network_space.connect_agents(0, 1)
        self.network_space.disconnect_agents(0, 1, directed=True)

        # After disconnect, agent 0 should not have a connection to agent 1
        self.assertNotIn(1, self.network_space._locations[0])
        # Agent 1 should have a connection back to agent 0, as the disconnect is directed
        self.assertIn(0, self.network_space._locations[1])


class TestBulkConnect(unittest.TestCase):
    """bulk_connect must produce the same neighbor lists as per-connection
    connect_agents(directed=True), in one pass, under sparse mode."""

    def _sparse_space(self, ordered, agent_ids):
        ns = NetworkSpace(ordered=ordered)
        af = AgentFactory(ns)
        ns._agent_factory = af
        af.set_agent_property_value = MagicMock()
        ns.add_local_agents(agent_ids)
        return ns

    def test_requires_sparse_mode(self):
        ns = NetworkSpace()
        with self.assertRaises(RuntimeError):
            ns.bulk_connect({0: [1]})

    def test_parity_with_connect_agents_unordered(self):
        pairs = [(0, 1), (0, 2), (1, 2), (2, 0)]
        ids = [0, 1, 2]

        ref = self._sparse_space(False, ids)
        for a, b in pairs:
            ref.connect_agents(a, b, directed=True)

        bulk = self._sparse_space(False, ids)
        adj = {}
        for a, b in pairs:
            adj.setdefault(a, []).append(b)
        bulk.bulk_connect(adj)

        for aid in ids:
            self.assertEqual(ref._locations[aid], bulk._locations[aid])

    def test_parity_with_connect_agents_ordered(self):
        # ordered=True: lists, insertion order preserved, no duplicates.
        pairs = [(0, 2), (0, 1), (0, 2), (1, 0)]  # note duplicate (0, 2)
        ids = [0, 1, 2]

        ref = self._sparse_space(True, ids)
        for a, b in pairs:
            ref.connect_agents(a, b, directed=True)

        bulk = self._sparse_space(True, ids)
        adj = {}
        for a, b in pairs:
            adj.setdefault(a, []).append(b)
        bulk.bulk_connect(adj)

        for aid in ids:
            # exact order AND membership must match
            self.assertEqual(ref._locations[aid], bulk._locations[aid])
            self.assertEqual(ref._locations_set[aid], bulk._locations_set[aid])
        # sanity: duplicate (0, 2) collapsed, agent 0's neighbors stay [2, 1]
        self.assertEqual(bulk._locations[0], [2, 1])

    def test_dict_input_round_trips(self):
        adj = {0: [1, 2], 1: [2], 2: [0]}
        ns = self._sparse_space(True, [0, 1, 2])
        ns.bulk_connect(adj)
        self.assertEqual(ns._locations[0], [1, 2])
        self.assertEqual(ns._locations[1], [2])
        self.assertEqual(ns._locations[2], [0])

    def test_ordered_preserves_given_neighbor_order(self):
        # In ordered mode the neighbor slot index is meaningful (a consumer may
        # read neighbor[0], neighbor[1], ... positionally). bulk_connect must
        # keep neighbors in exactly the order given, NOT sorted, so that the
        # i-th neighbor stays at slot i. Use a non-sorted, non-trivial order.
        ns = self._sparse_space(True, [0, 1, 2, 3, 4, 5])
        ns.bulk_connect({0: [5, 2, 8, 2, 4]})  # note out-of-order + duplicate 2
        # 8 is a non-local neighbor (allowed); duplicate 2 collapses to first pos
        self.assertEqual(ns._locations[0], [5, 2, 8, 4])
        # parity: the same sequence via per-connection connect_agents
        ref = self._sparse_space(True, [0, 1, 2, 3, 4, 5])
        for b in [5, 2, 8, 2, 4]:
            ref.connect_agents(0, b, directed=True)
        self.assertEqual(ref._locations[0], ns._locations[0])

    def test_remote_id_as_value_is_allowed(self):
        # A non-local id (99) may appear as a neighbor (value) without having
        # its own container: it's an agent on another partition that this rank
        # does not own.
        ns = self._sparse_space(True, [0, 1])
        ns.bulk_connect({0: [99, 1]})
        self.assertEqual(ns._locations[0], [99, 1])
        self.assertNotIn(99, ns._locations)

    def test_nonlocal_id_as_key_raises(self):
        ns = self._sparse_space(True, [0, 1])
        with self.assertRaises(KeyError):
            ns.bulk_connect({99: [0]})


if __name__ == "__main__":
    unittest.main()
