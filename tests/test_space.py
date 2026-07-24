from unittest.mock import MagicMock

import pytest

from sagesim.space import NetworkSpace
from sagesim.agent import AgentFactory


@pytest.fixture
def network_space():
    """A NetworkSpace with a stub AgentFactory and 3 agents already added."""
    space = NetworkSpace()
    agent_factory = AgentFactory(space)
    space._agent_factory = agent_factory

    # Stub out set_agent_property_value to avoid unnecessary logic
    agent_factory.set_agent_property_value = MagicMock()

    for agent_id in range(3):
        space.add_agent(agent_id)
    return space


class TestNetworkSpace:
    def test_add_agent(self, network_space):
        # After adding 3 agents, there should be 3 locations
        assert len(network_space._locations) == 3
        for location in network_space._locations:
            assert location == set()

    def test_connect_agents_undirected_network(self, network_space):
        network_space.connect_agents(0, 1)

        # Verify connections in both directions
        assert 1 in network_space._locations[0]
        assert 0 in network_space._locations[1]

    def test_connect_agents_directed_network(self, network_space):
        network_space.connect_agents(0, 1, directed=True)

        # Verify connection from 0 to 1
        assert 1 in network_space._locations[0]
        # Verify that agent 1 does not have a connection back to agent 0
        assert 0 not in network_space._locations[1]

    def test_disconnect_agents(self, network_space):
        network_space.connect_agents(0, 1)
        network_space.disconnect_agents(0, 1)

        # After disconnect, they should not be neighbors anymore
        assert 1 not in network_space._locations[0]
        assert 0 not in network_space._locations[1]

    def test_disconnect_agents_directed(self, network_space):
        network_space.connect_agents(0, 1)
        network_space.disconnect_agents(0, 1, directed=True)

        # After disconnect, agent 0 should not have a connection to agent 1
        assert 1 not in network_space._locations[0]
        # Agent 1 should have a connection back to agent 0, as the disconnect is directed
        assert 0 in network_space._locations[1]


def _sparse_space(ordered, agent_ids):
    ns = NetworkSpace(ordered=ordered)
    af = AgentFactory(ns)
    ns._agent_factory = af
    af.set_agent_property_value = MagicMock()
    ns.add_local_agents(agent_ids)
    return ns


class TestBulkConnect:
    """bulk_connect must produce the same neighbor lists as per-connection
    connect_agents(directed=True), in one pass, under sparse mode."""

    def test_requires_sparse_mode(self):
        ns = NetworkSpace()
        with pytest.raises(RuntimeError):
            ns.bulk_connect({0: [1]})

    def test_parity_with_connect_agents_unordered(self):
        pairs = [(0, 1), (0, 2), (1, 2), (2, 0)]
        ids = [0, 1, 2]

        ref = _sparse_space(False, ids)
        for a, b in pairs:
            ref.connect_agents(a, b, directed=True)

        bulk = _sparse_space(False, ids)
        adj = {}
        for a, b in pairs:
            adj.setdefault(a, []).append(b)
        bulk.bulk_connect(adj)

        for aid in ids:
            assert ref._locations[aid] == bulk._locations[aid]

    def test_parity_with_connect_agents_ordered(self):
        # ordered=True: lists, insertion order preserved, no duplicates.
        pairs = [(0, 2), (0, 1), (0, 2), (1, 0)]  # note duplicate (0, 2)
        ids = [0, 1, 2]

        ref = _sparse_space(True, ids)
        for a, b in pairs:
            ref.connect_agents(a, b, directed=True)

        bulk = _sparse_space(True, ids)
        adj = {}
        for a, b in pairs:
            adj.setdefault(a, []).append(b)
        bulk.bulk_connect(adj)

        for aid in ids:
            # exact order AND membership must match
            assert ref._locations[aid] == bulk._locations[aid]
            assert ref._locations_set[aid] == bulk._locations_set[aid]
        # sanity: duplicate (0, 2) collapsed, agent 0's neighbors stay [2, 1]
        assert bulk._locations[0] == [2, 1]

    def test_dict_input_round_trips(self):
        adj = {0: [1, 2], 1: [2], 2: [0]}
        ns = _sparse_space(True, [0, 1, 2])
        ns.bulk_connect(adj)
        assert ns._locations[0] == [1, 2]
        assert ns._locations[1] == [2]
        assert ns._locations[2] == [0]

    def test_ordered_preserves_given_neighbor_order(self):
        # In ordered mode the neighbor slot index is meaningful (a consumer may
        # read neighbor[0], neighbor[1], ... positionally). bulk_connect must
        # keep neighbors in exactly the order given, NOT sorted, so that the
        # i-th neighbor stays at slot i. Use a non-sorted, non-trivial order.
        ns = _sparse_space(True, [0, 1, 2, 3, 4, 5])
        ns.bulk_connect({0: [5, 2, 8, 2, 4]})  # note out-of-order + duplicate 2
        # 8 is a non-local neighbor (allowed); duplicate 2 collapses to first pos
        assert ns._locations[0] == [5, 2, 8, 4]
        # parity: the same sequence via per-connection connect_agents
        ref = _sparse_space(True, [0, 1, 2, 3, 4, 5])
        for b in [5, 2, 8, 2, 4]:
            ref.connect_agents(0, b, directed=True)
        assert ref._locations[0] == ns._locations[0]

    def test_remote_id_as_value_is_allowed(self):
        # A non-local id (99) may appear as a neighbor (value) without having
        # its own container: it's an agent on another partition that this rank
        # does not own.
        ns = _sparse_space(True, [0, 1])
        ns.bulk_connect({0: [99, 1]})
        assert ns._locations[0] == [99, 1]
        assert 99 not in ns._locations

    def test_nonlocal_id_as_key_raises(self):
        ns = _sparse_space(True, [0, 1])
        with pytest.raises(KeyError):
            ns.bulk_connect({99: [0]})
