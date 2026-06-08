import sys
import unittest
import random
import cupy as cp
from cupyx import jit
from pathlib import Path

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed
from sagesim.utils import (
    get_this_agent_data_from_tensor,
    set_this_agent_data_from_tensor,
)


# --- Step function (same as test_double_buffer.py) ---

@jit.rawkernel(device="cuda")
def infection_step_func(
    tick,
    agent_index,
    p_infection,
    agent_ids,
    breeds,
    locations,
    state_tensor,
):
    neighbor_indices = locations[agent_index]
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))

    if agent_state == 1:  # SUSCEPTIBLE
        i = 0
        infected = False
        while i < len(neighbor_indices) and not cp.isnan(neighbor_indices[i]) and not infected:
            neighbor_index = int(neighbor_indices[i])
            neighbor_state = int(state_tensor[neighbor_index])
            if neighbor_state == 2:  # INFECTED
                rand = random.random()
                if rand < p_infection:
                    set_this_agent_data_from_tensor(agent_index, state_tensor, 2)
                    infected = True
            i += 1


# --- Breed / Model ---

class SIBreed(Breed):
    def __init__(self):
        super().__init__("Infection")
        self.register_property("state", 1)
        self.register_step_func(infection_step_func, Path(__file__).resolve(), 0)


class SIModel(Model):
    def __init__(self, p_infection=1.0):
        space = NetworkSpace()
        super().__init__(space)
        self._infection_breed = SIBreed()
        self.register_breed(breed=self._infection_breed)
        self.register_global_property("p_infection", p_infection)

    def create_agent(self, state):
        return self.create_agent_of_breed(self._infection_breed, state=state)

    def connect_agents(self, a, b):
        self.get_space().connect_agents(a, b)


# --- Helpers ---

def generate_hierarchical_network(total_agents=111):
    """Generate 1->10->100 directed network. Same as test_double_buffer.py."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_nodes_from(range(total_agents))

    for middle in range(1, 11):
        G.add_edge(0, middle)

    random.seed(46)
    end_agents = list(range(11, 111))

    for mi, middle in enumerate(range(1, 11)):
        start = mi * 10
        assigned = end_agents[start:start + 10]
        for ea in assigned:
            G.add_edge(middle, ea)
        remaining = [a for a in end_agents if a not in assigned]
        extra = random.randint(1, 10)
        for ea in random.sample(remaining, min(extra, len(remaining))):
            G.add_edge(middle, ea)

    return G


# --- Tests ---

class TestBuildFromLocalData(unittest.TestCase):

    def tearDown(self):
        if 'step_func_code' in sys.modules:
            del sys.modules['step_func_code']

    def test_bulk_agents_property_values(self):
        """Bulk-created agents have correct property values and mappings."""
        model = SIModel()
        breed = model._infection_breed
        af = model._agent_factory

        states = [10, 20, 30, 40, 50]
        agents = [
            {'id': i, 'breed': breed, 'properties': {'state': s}}
            for i, s in enumerate(states)
        ]
        connections = [(0, 1), (1, 2)]

        model.build_from_local_data(agents, connections)

        # Property tensor values
        state_tensor = af._property_name_2_agent_data_tensor['state']
        self.assertEqual(state_tensor, states)

        # Agent-to-rank mapping
        for i in range(5):
            self.assertEqual(af._agent2rank[i], 0)

        # Agent-to-breed mapping
        for i in range(5):
            self.assertEqual(af._agent2breed[i], breed._breedidx)

        # Index mapping
        rank_map = af._rank2agentid2agentidx[0]
        for i in range(5):
            self.assertEqual(rank_map[i], i)

    def test_connections_undirected(self):
        """Default undirected connections create bidirectional edges."""
        model = SIModel()
        breed = model._infection_breed

        agents = [
            {'id': i, 'breed': breed, 'properties': {'state': 1}}
            for i in range(4)
        ]
        connections = [(0, 1), (0, 2), (1, 3), (2, 3)]

        model.build_from_local_data(agents, connections)

        space = model.get_space()
        # 0 <-> 1, 0 <-> 2
        self.assertIn(1, space._locations[0])
        self.assertIn(2, space._locations[0])
        self.assertIn(0, space._locations[1])
        self.assertIn(0, space._locations[2])
        # 1 <-> 3, 2 <-> 3
        self.assertIn(3, space._locations[1])
        self.assertIn(3, space._locations[2])
        self.assertIn(1, space._locations[3])
        self.assertIn(2, space._locations[3])

    def test_connections_directed(self):
        """directed=True creates one-way edges only."""
        model = SIModel()
        breed = model._infection_breed

        agents = [
            {'id': i, 'breed': breed, 'properties': {'state': 1}}
            for i in range(4)
        ]
        connections = [(0, 1), (0, 2), (1, 3), (2, 3)]

        model.build_from_local_data(agents, connections, directed=True)

        space = model.get_space()
        # 0 -> 1, 0 -> 2 (one-way)
        self.assertIn(1, space._locations[0])
        self.assertIn(2, space._locations[0])
        self.assertNotIn(0, space._locations[1])
        self.assertNotIn(0, space._locations[2])
        # 1 -> 3, 2 -> 3 (one-way)
        self.assertIn(3, space._locations[1])
        self.assertIn(3, space._locations[2])
        self.assertNotIn(1, space._locations[3])
        self.assertNotIn(2, space._locations[3])

    def test_remote_agent_ranks_registered(self):
        """Remote agent ranks are recorded in agent factory."""
        model = SIModel()
        breed = model._infection_breed
        af = model._agent_factory

        agents = [
            {'id': i, 'breed': breed, 'properties': {'state': 1}}
            for i in range(3)
        ]
        remote_ranks = {100: 1, 200: 2}

        model.build_from_local_data(agents, [], remote_ranks)

        self.assertEqual(af._agent2rank[100], 1)
        self.assertEqual(af._agent2rank[200], 2)
        # Local agents still on rank 0
        for i in range(3):
            self.assertEqual(af._agent2rank[i], 0)

    def test_bulk_vs_sequential_1tick_spread(self):
        """Bulk and sequential paths produce identical simulation results."""
        network = generate_hierarchical_network(111)

        # --- Model A: sequential path ---
        model_a = SIModel(p_infection=1.0)
        for node in network.nodes:
            model_a.create_agent(1)  # SUSCEPTIBLE
        model_a.set_agent_property_value(0, "state", 2)  # INFECTED
        for src, dst in network.edges:
            model_a.connect_agents(src, dst)
        model_a.setup(use_gpu=True)
        model_a.simulate(1, sync_workers_every_n_ticks=1)

        # --- Model B: bulk path ---
        model_b = SIModel(p_infection=1.0)
        breed = model_b._infection_breed

        agents = []
        for node in network.nodes:
            state = 2 if node == 0 else 1
            agents.append({
                'id': node,
                'breed': breed,
                'properties': {'state': state},
            })
        connections = list(network.edges)

        model_b.build_from_local_data(agents, connections)
        model_b.setup(use_gpu=True)
        model_b.simulate(1, sync_workers_every_n_ticks=1)

        # Compare all agent states
        for agent_id in range(111):
            state_a = model_a.get_agent_property_value(agent_id, "state")
            state_b = model_b.get_agent_property_value(agent_id, "state")
            self.assertEqual(
                state_a, state_b,
                f"Agent {agent_id}: sequential={state_a}, bulk={state_b}",
            )

    def test_empty_agents(self):
        """Empty agent list does not crash."""
        model = SIModel()
        model.build_from_local_data([], [], {})
        af = model._agent_factory
        self.assertEqual(af._num_agents, 0)


if __name__ == "__main__":
    unittest.main()
