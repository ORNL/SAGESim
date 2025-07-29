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


if __name__ == "__main__":
    unittest.main()
