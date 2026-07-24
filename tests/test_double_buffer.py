import sys

import networkx as nx
import random
import numpy as np
import cupy as cp
from cupyx import jit
from pathlib import Path

import pytest

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
    p_infection,
    agent_ids,
    breeds,
    locations,
    state_tensor,
):
    """
    Step function for infection spread with probability p (default p=1 for testing)
    """
    # Get the list of neighboring indices (pre-converted from agent IDs by SAGESim)
    neighbor_indices = locations[agent_index]

    # Get infection probability from global tensor
    p_inf = p_infection  # scalar auto-extracted by framework

    # Get current agent state
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))

    # Only susceptible agents can be infected
    if agent_state == 1:  # SUSCEPTIBLE
        # Check all neighbors
        i = 0
        infected = False
        while i < len(neighbor_indices) and not cp.isnan(neighbor_indices[i]) and not infected:
            neighbor_index = int(neighbor_indices[i])

            # Get neighbor state using pre-converted index (no search needed!)
            neighbor_state = int(state_tensor[neighbor_index])

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
def infection_step_func_with_dummy(
    tick,
    agent_index,
    p_infection,
    p_recovery,
    agent_ids,
    breeds,
    locations,
    state_tensor,
    dummy_tensor
):
    """
    Step function for infection spread with probability p (default p=1 for testing)
    """
    # Get the list of neighboring indices (pre-converted from agent IDs by SAGESim)
    neighbor_indices = locations[agent_index]

    # Get infection probability from global tensor
    p_inf = p_infection  # scalar auto-extracted by framework

    # Get current agent state
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))

    # Only susceptible agents can be infected
    if agent_state == 1:  # SUSCEPTIBLE
        # Check all neighbors
        i = 0
        infected = False
        while i < len(neighbor_indices) and not cp.isnan(neighbor_indices[i]) and not infected:
            neighbor_index = int(neighbor_indices[i])

            # Get neighbor state using pre-converted index (no search needed!)
            neighbor_state = int(state_tensor[neighbor_index])

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
    p_infection,
    p_recovery,
    agent_ids,
    breeds,
    locations,
    state_tensor,
    dummy_tensor
):
    """
    Step function for recovery with probability p (default p=1 for testing)
    """
    dummy_tensor[agent_index] = dummy_tensor[agent_index] + 1  # test if write_dummy_tensor will be created
    # Get recovery probability from global tensor
    p_rec = p_recovery  # scalar auto-extracted by framework
    
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
        self.register_step_func(infection_step_func_with_dummy, curr_fpath, 0)
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


@pytest.fixture(scope="module")
def network():
    """Built once for the module: both tests run on the same 111-agent network."""
    return generate_hierarchical_network(111)


@pytest.fixture(autouse=True)
def clear_step_func_cache():
    """Drop the generated step-function module so each test regenerates it."""
    yield
    sys.modules.pop("step_func_code", None)


def test_1_tick_spread_with_SIModel(network):
    """One tick on a hierarchical network infects only the middle layer."""
    model = create_model_from_network(SIModel(p_infection=1.0), network)
    model.setup()

    # 1 tick, so infection must spread only to the middle agents
    model.simulate(1, sync_workers_every_n_ticks=1)

    assert model.get_agent_property_value(0, "state") == 2, (
        "root agent should remain infected"
    )
    for agent_id in range(1, 11):
        assert model.get_agent_property_value(agent_id, "state") == 2, (
            f"middle agent {agent_id} should be infected after 1 tick"
        )
    for agent_id in range(11, 111):
        assert model.get_agent_property_value(agent_id, "state") == 1, (
            f"second layer agent {agent_id} should be susceptible after 1 tick"
        )


def test_2_tick_spread_with_SIRModel(network):
    """Two ticks with SIR: middle layer recovers, second layer gets infected.

    Tick 1: root (0) infects the middle layer (1-10), then root recovers.
    Tick 2: middle layer infects the second layer (11-110), then it recovers.
    """
    model = create_model_from_network(
        SIRModel(p_infection=1.0, p_recovery=1.0), network
    )
    model.setup()
    model.simulate(2, sync_workers_every_n_ticks=1)

    assert model.get_agent_property_value(0, "state") == 3, (
        "root agent should be recovered after 2 ticks"
    )
    for agent_id in range(1, 11):
        assert model.get_agent_property_value(agent_id, "state") == 3, (
            f"middle agent {agent_id} should be recovered after 2 ticks"
        )
    for agent_id in range(11, 111):
        assert model.get_agent_property_value(agent_id, "state") == 2, (
            f"second layer agent {agent_id} should be infected after 2 ticks"
        )
