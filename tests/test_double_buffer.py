import unittest
import networkx as nx
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
    get_neighbor_data_from_tensor,
)


# Define step function for infection spread
@jit.rawkernel(device="cuda")
def infection_step_func(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    state_tensor,
    dummy_tensor
):
    """
    Step function for infection spread with probability p (default p=1 for testing)
    """
    # Get the list of neighboring agent IDs
    neighbor_ids = locations[agent_index]
    
    # Get infection probability from globals (default p=1 for testing)
    p_infection = globals[0]
    
    # Get current agent state
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))
    
    # Only susceptible agents can be infected
    if agent_state == 1:  # SUSCEPTIBLE
        # Check all neighbors
        i = 0
        infected = False
        while i < len(neighbor_ids) and not cp.isnan(neighbor_ids[i]) and not infected:
            neighbor_id = neighbor_ids[i]
            
            # Get neighbor state
            neighbor_state = int(
                get_neighbor_data_from_tensor(agent_ids, neighbor_id, state_tensor)
            )
            
            # If neighbor is infected and random chance passes, infect this agent
            if neighbor_state == 2:  # INFECTED
                rand = random.random()
                if rand < p_infection:
                    set_this_agent_data_from_tensor(agent_index,state_tensor,2)
                    # state_tensor[agent_index]=  2  # INFECTED
                    infected = True  # Once infected, no need to check more neighbors
            i += 1

# Define step function for infection spread
@jit.rawkernel(device="cuda")
def recovery_step_func(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    state_tensor,
    dummy_tensor
):
    """
    Step function for infection spread with probability p (default p=1 for testing)
    """
    dummy_tensor[agent_index] = dummy_tensor[agent_index] + 1  # test if write_dummy_tensor will be created
    # Get infection probability from globals (default p=1 for testing)
    p_recovery = globals[1]
    
    # Get current agent state
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))
    
    # Only infected agents can be recovered
    if agent_state == 2:  # SUSCEPTIBLE
       
        rand = random.random()
        if rand < p_recovery:
            # set_this_agent_data_from_tensor(agent_index,state_tensor,3)
            state_tensor[agent_index] = 3  # RECOVERED



class SIBreed(Breed):
    """Breed for infection spreading test"""
    
    def __init__(self) -> None:
        name = "Infection"
        super().__init__(name)
        # Register state property
        self.register_property("state", 1)
        # Register the step function
        curr_fpath = Path(__file__).resolve()
        self.register_step_func(infection_step_func, curr_fpath, 0)


class SIModel(Model):
    """Model for infection spreading test"""
    
    def __init__(self, p_infection=1.0) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._infection_breed = SIBreed()
        
        # Register the breed
        self.register_breed(breed=self._infection_breed)
        
        # Register infection probability (default p=1 for testing)
        self.register_global_property("p_infection", p_infection)
    
    def create_agent(self, state):
        agent_id = self.create_agent_of_breed(
            self._infection_breed, state=state
        )
        return agent_id
    
    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)


class SIRBreed(Breed):
    """Breed for infection and recovery spreading test"""
    
    def __init__(self) -> None:
        name = "Infection"
        super().__init__(name)
        # Register state property
        self.register_property("state", 1)
        self.register_property("dummy", 0)  # Dummy property for double buffering

        # Register the step function
        curr_fpath = Path(__file__).resolve()
        self.register_step_func(infection_step_func, curr_fpath, 0)
        self.register_step_func(recovery_step_func, curr_fpath, 1)


class SIRModel(Model):
    """Model for infection spreading and recovery test"""
    
    def __init__(self, p_infection=1.0, p_recovery=1.0) -> None:
        space = NetworkSpace()
        super().__init__(space)
        self._sir_breed = SIRBreed()
        
        # Register the breed
        self.register_breed(breed=self._sir_breed)
        
        # Register infection probability (default p=1 for testing)
        self.register_global_property("p_infection", p_infection)
        # Register recovery probability (default p=1 for testing)
        self.register_global_property("p_recovery", p_recovery)
    
    def create_agent(self, state):
        agent_id = self.create_agent_of_breed(
            self._sir_breed, state=state
        )
        return agent_id
    
    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)

def generate_hierarchical_network(total_agents=111):
    """Generate 1->10->100 network using NetworkX"""
    # Create empty graph
    G = nx.DiGraph()
    
    # Add all nodes
    G.add_nodes_from(range(total_agents))
    
    # Root agent (0) connects to middle agents (1-10)
    for middle_agent in range(1, 11):
        G.add_edge(0, middle_agent)
    
    # Middle agents (1-10) connect to end agents (11-110)
    # First: each middle agent connects to exactly 10 end agents to ensure all 100 are connected
    random.seed(46)  # For reproducible tests
    end_agents = list(range(11, 111))
    
    # Distribute the 100 end agents evenly: each of 10 middle agents gets 10 end agents
    for middle_agent_idx, middle_agent in enumerate(range(1, 11)):
        # Each middle agent gets exactly 10 end agents (10 * 10 = 100 total)
        start_idx = middle_agent_idx * 10
        end_idx = start_idx + 10
        assigned_end_agents = end_agents[start_idx:end_idx]
        
        for end_agent in assigned_end_agents:
            G.add_edge(middle_agent, end_agent)
        
        # Then: randomly connect to 1-10 additional end agents from the remaining 90
        remaining_end_agents = [agent for agent in end_agents if agent not in assigned_end_agents]
        num_additional = random.randint(1, 10)
        additional_connections = random.sample(remaining_end_agents, min(num_additional, len(remaining_end_agents)))
        
        for end_agent in additional_connections:
            G.add_edge(middle_agent, end_agent)

    return G

def create_model_from_network(model, network):
    """Create SAGESim model from NetworkX network"""
    # Create all agents as susceptible
    for node in network.nodes:
        model.create_agent(1)  # 1 = SUSCEPTIBLE
    
    # Set root agent as infected
    model.set_agent_property_value(0, "state", 2)  # 2 = INFECTED
    
    # Add all edges to the model
    for edge in network.edges:
        model.connect_agents(edge[0], edge[1])
    
    return model


class TestDoubleBuffer(unittest.TestCase):
    def setUp(self):
        """Set up the network"""
        self.total_agents = 111
        self.network = generate_hierarchical_network(self.total_agents)
        
    def test_1_tick_spread_with_SIModel(self):
        """Test internal ticks with infection spreading on hierarchical network"""
        # Generate network and create model
        self.model = create_model_from_network(SIModel(p_infection=1.0), self.network)
        
        # Setup model
        self.model.setup(use_gpu=True)
        
        # Verify initial state: only root agent is infected
        infected_count = 0
        for agent_id in range(self.total_agents):
            state = self.model.get_agent_property_value(agent_id, "state")
            if state == 2:
                infected_count += 1
        self.assertEqual(infected_count, 1)  # Only root agent
        
        # Run simulation
        # test with 1 tick to make sure infection spreads only to the middle agents
        print(f"Running simulation with 1 tick, sync_workers_every_n_ticks=1")
        self.model.simulate(1, sync_workers_every_n_ticks=1)

        # Verify infection spread (with p=1, all connected susceptible agents should be infected)
        susceptible_agents = []
        infected_agents = []
        
        for agent_id in range(self.total_agents):
            state = self.model.get_agent_property_value(agent_id, "state")
            if state == 1:  # SUSCEPTIBLE
                susceptible_agents.append(agent_id)
            elif state == 2:  # INFECTED
                infected_agents.append(agent_id)
        
        print(f"Susceptible agents: {susceptible_agents}")
        print(f"Infected agents: {infected_agents}")
        print(f"Total susceptible: {len(susceptible_agents)}")
        print(f"Total infected: {len(infected_agents)}")
        
        # Check that root agent (0) is still infected
        root_state = self.model.get_agent_property_value(0, "state")
        self.assertEqual(root_state, 2, "Root agent should remain infected")
        
        # Check that all middle agents (1-10) are infected
        for agent_id in range(1, 11):
            state = self.model.get_agent_property_value(agent_id, "state")
            self.assertEqual(state, 2, f"Middle agent {agent_id} should be infected after 1 tick")
        
        # Check that all second layer agents (11-110) are still susceptible
        for agent_id in range(11, 111):
            state = self.model.get_agent_property_value(agent_id, "state")
            self.assertEqual(state, 1, f"Second layer agent {agent_id} should be susceptible after 1 tick")

    def test_2_tick_spread_with_SIRModel(self):
        """Test 2 ticks with SIR model: middle layer recovers, second layer gets infected"""
        # Generate network and create model with both infection and recovery probabilities = 1
        self.model = create_model_from_network(SIRModel(p_infection=1.0, p_recovery=1.0), self.network)
        
        # Setup model
        self.model.setup(use_gpu=True) 
        
        # Run simulation for 2 ticks
        print(f"Running SIR simulation with 2 ticks, sync_workers_every_n_ticks=1")
        self.model.simulate(2, sync_workers_every_n_ticks=1)
        
        # After 2 ticks with SIR model (infection and recovery prob = 1):
        # Tick 1: Root (0) infects middle layer (1-10), then root recovers
        # Tick 2: Middle layer (1-10) infects second layer (11-110), then middle layer recovers
   
        # Check that root agent (0) is recovered
        root_state = self.model.get_agent_property_value(0, "state")
        self.assertEqual(root_state, 3, "Root agent should be recovered after 2 ticks")
        
        # Check that all middle agents (1-10) are recovered
        for agent_id in range(1, 11):
            state = self.model.get_agent_property_value(agent_id, "state")
            self.assertEqual(state, 3, f"Middle agent {agent_id} should be recovered after 2 ticks")
        
        # Check that all second layer agents (11-110) are infected
        for agent_id in range(11, 111):
            state = self.model.get_agent_property_value(agent_id, "state")
            self.assertEqual(state, 2, f"Second layer agent {agent_id} should be infected after 2 ticks")

    
if __name__ == "__main__":
    unittest.main()