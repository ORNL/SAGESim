"""
Tests for SAGESim's automatic double-buffer AST rewriting.

Tests two aspects:
1. SIR infection spread on a hierarchical network — verifies that double
   buffering correctly isolates read/write within a tick so infection
   propagates exactly one hop per tick.
2. Double-buffer vs no-double-buffer priority semantics — verifies that
   the no_double_buffer option allows same-tick visibility between
   priority groups.
"""

import sys
import unittest
import random as stdlib_random
from pathlib import Path

import networkx as nx
import cupy as cp
from cupyx import jit

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed
from sagesim.utils import (
    get_this_agent_data_from_tensor,
    set_this_agent_data_from_tensor,
)


# ── Step functions for SIR spread tests ──────────────────────────────

@jit.rawkernel(device="cuda")
def infection_step_func(
    tick, agent_index, p_infection, agent_ids, breeds, locations, state_tensor,
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
                rand = stdlib_random.random()
                if rand < p_infection:
                    set_this_agent_data_from_tensor(agent_index, state_tensor, 2)
                    infected = True
            i += 1


@jit.rawkernel(device="cuda")
def infection_step_func_with_dummy(
    tick, agent_index, p_infection, p_recovery,
    agent_ids, breeds, locations, state_tensor, dummy_tensor,
):
    neighbor_indices = locations[agent_index]
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))
    if agent_state == 1:
        i = 0
        infected = False
        while i < len(neighbor_indices) and not cp.isnan(neighbor_indices[i]) and not infected:
            neighbor_index = int(neighbor_indices[i])
            neighbor_state = int(state_tensor[neighbor_index])
            if neighbor_state == 2:
                rand = stdlib_random.random()
                if rand < p_infection:
                    set_this_agent_data_from_tensor(agent_index, state_tensor, 2)
                    infected = True
            i += 1


@jit.rawkernel(device="cuda")
def recovery_step_func(
    tick, agent_index, p_infection, p_recovery,
    agent_ids, breeds, locations, state_tensor, dummy_tensor,
):
    dummy_tensor[agent_index] = dummy_tensor[agent_index] + 1
    agent_state = int(get_this_agent_data_from_tensor(agent_index, state_tensor))
    if agent_state == 2:
        rand = stdlib_random.random()
        if rand < p_recovery:
            state_tensor[agent_index] = 3  # RECOVERED


# ── Step functions for double-buffer semantics tests ───────────────���─

@jit.rawkernel(device="cuda")
def write_counter_step_func(
    tick, agent_index, agent_ids, breeds, locations, counter, result,
):
    counter[agent_index] = 10


@jit.rawkernel(device="cuda")
def read_and_multiply_step_func(
    tick, agent_index, agent_ids, breeds, locations, counter, result,
):
    result[agent_index] = counter[agent_index] * 2


# ── Breed / Model definitions for SIR spread ────────────────────────

class SIBreed(Breed):
    def __init__(self):
        super().__init__("Infection")
        self.register_property("state", 1)
        self.register_step_func(infection_step_func, Path(__file__).resolve(), 0)


class SIModel(Model):
    def __init__(self, p_infection=1.0):
        super().__init__(NetworkSpace())
        self._breed = SIBreed()
        self.register_breed(self._breed)
        self.register_global_property("p_infection", p_infection)

    def create_agent(self, state):
        return self.create_agent_of_breed(self._breed, state=state)

    def connect_agents(self, a0, a1):
        self.get_space().connect_agents(a0, a1)


class SIRBreed(Breed):
    def __init__(self):
        super().__init__("Infection")
        self.register_property("state", 1)
        self.register_property("dummy", 0)
        self.register_step_func(infection_step_func_with_dummy, Path(__file__).resolve(), 0)
        self.register_step_func(recovery_step_func, Path(__file__).resolve(), 1)


class SIRModel(Model):
    def __init__(self, p_infection=1.0, p_recovery=1.0):
        super().__init__(NetworkSpace())
        self._breed = SIRBreed()
        self.register_breed(self._breed)
        self.register_global_property("p_infection", p_infection)
        self.register_global_property("p_recovery", p_recovery)

    def create_agent(self, state):
        return self.create_agent_of_breed(self._breed, state=state)

    def connect_agents(self, a0, a1):
        self.get_space().connect_agents(a0, a1)


# ── Breed / Model definitions for double-buffer semantics ───────────

class _DBBreed(Breed):
    """Double-buffered (default)."""
    def __init__(self):
        super().__init__("DBBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)
        fpath = Path(__file__).resolve()
        self.register_step_func(write_counter_step_func, fpath, priority=0)
        self.register_step_func(read_and_multiply_step_func, fpath, priority=1)


class _NoDBBreed(Breed):
    """No double buffering for counter."""
    def __init__(self):
        super().__init__("NoDBBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)
        fpath = Path(__file__).resolve()
        self.register_step_func(write_counter_step_func, fpath, priority=0, no_double_buffer=["counter"])
        self.register_step_func(read_and_multiply_step_func, fpath, priority=1)


class _RBWDBBreed(Breed):
    """Reader-before-writer, double-buffered."""
    def __init__(self):
        super().__init__("RBWDBBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)
        fpath = Path(__file__).resolve()
        self.register_step_func(read_and_multiply_step_func, fpath, priority=0)
        self.register_step_func(write_counter_step_func, fpath, priority=1)


class _RBWNoDBBreed(Breed):
    """Reader-before-writer, no double buffer."""
    def __init__(self):
        super().__init__("RBWNoDBBreed")
        self.register_property("counter", 0)
        self.register_property("result", 0)
        fpath = Path(__file__).resolve()
        self.register_step_func(read_and_multiply_step_func, fpath, priority=0)
        self.register_step_func(write_counter_step_func, fpath, priority=1, no_double_buffer=["counter"])


def _make_db_model(breed_cls, codefile):
    model = Model(NetworkSpace(), step_function_file_path=codefile)
    breed = breed_cls()
    model.register_breed(breed)
    model.create_agent_of_breed(breed, counter=0, result=0)
    return model


# ── Helpers ──────────────────────────────────────────────────────────

def _generate_hierarchical_network(total_agents=111):
    G = nx.DiGraph()
    G.add_nodes_from(range(total_agents))
    for mid in range(1, 11):
        G.add_edge(0, mid)
    stdlib_random.seed(46)
    end_agents = list(range(11, 111))
    for idx, mid in enumerate(range(1, 11)):
        start = idx * 10
        assigned = end_agents[start:start + 10]
        for ea in assigned:
            G.add_edge(mid, ea)
        remaining = [a for a in end_agents if a not in assigned]
        for ea in stdlib_random.sample(remaining, min(stdlib_random.randint(1, 10), len(remaining))):
            G.add_edge(mid, ea)
    return G


def _create_model_from_network(model, network):
    for _ in network.nodes:
        model.create_agent(1)
    model.set_agent_property_value(0, "state", 2)
    for e in network.edges:
        model.connect_agents(e[0], e[1])
    return model


# ── Test classes ─────────────────────────────────────────────────────

class TestSIRSpread(unittest.TestCase):
    """Tests that double buffering correctly isolates per-tick reads/writes
    so infection propagates exactly one hop per tick."""

    @classmethod
    def setUpClass(cls):
        cls.network = _generate_hierarchical_network(111)

    def tearDown(self):
        for k in [k for k in sys.modules if k.startswith("step_func_code")]:
            del sys.modules[k]

    def test_si_one_tick_spread(self):
        """Infection spreads from root to middle layer in exactly 1 tick."""
        model = _create_model_from_network(SIModel(p_infection=1.0), self.network)
        model.setup(use_gpu=True)
        model.simulate(1, sync_workers_every_n_ticks=1)

        self.assertEqual(model.get_agent_property_value(0, "state"), 2)
        for aid in range(1, 11):
            self.assertEqual(model.get_agent_property_value(aid, "state"), 2,
                             f"Middle agent {aid} should be infected")
        for aid in range(11, 111):
            self.assertEqual(model.get_agent_property_value(aid, "state"), 1,
                             f"Leaf agent {aid} should still be susceptible")

    def test_sir_two_tick_spread(self):
        """After 2 ticks: root+middle recovered, leaf layer infected."""
        model = _create_model_from_network(SIRModel(p_infection=1.0, p_recovery=1.0), self.network)
        model.setup(use_gpu=True)
        model.simulate(2, sync_workers_every_n_ticks=1)

        self.assertEqual(model.get_agent_property_value(0, "state"), 3)
        for aid in range(1, 11):
            self.assertEqual(model.get_agent_property_value(aid, "state"), 3)
        for aid in range(11, 111):
            self.assertEqual(model.get_agent_property_value(aid, "state"), 2)


class TestDoubleBufferSemantics(unittest.TestCase):
    """Tests the AST-rewritten double-buffer behavior for priority-ordered
    step functions."""

    def tearDown(self):
        for k in [k for k in sys.modules if k.startswith("step_func_code")]:
            del sys.modules[k]

    def test_double_buffer_isolates_reads(self):
        """With double buffering: priority 1 reader sees old counter value (0)."""
        model = _make_db_model(_DBBreed, "step_func_code_db.py")
        model.setup(use_gpu=True)
        model.simulate(1, sync_workers_every_n_ticks=1)
        self.assertEqual(model.get_agent_property_value(0, "counter"), 10)
        self.assertEqual(model.get_agent_property_value(0, "result"), 0)

    def test_no_double_buffer_same_tick_visibility(self):
        """Without double buffering: priority 1 reader sees new counter value (10)."""
        model = _make_db_model(_NoDBBreed, "step_func_code_nodb.py")
        model.setup(use_gpu=True)
        model.simulate(1, sync_workers_every_n_ticks=1)
        self.assertEqual(model.get_agent_property_value(0, "counter"), 10)
        self.assertEqual(model.get_agent_property_value(0, "result"), 20)

    def test_reader_before_writer_with_double_buffer(self):
        """Reader runs first (priority 0), writer second — result = 0 regardless."""
        model = _make_db_model(_RBWDBBreed, "step_func_code_rbw_db.py")
        model.setup(use_gpu=True)
        model.simulate(1, sync_workers_every_n_ticks=1)
        self.assertEqual(model.get_agent_property_value(0, "counter"), 10)
        self.assertEqual(model.get_agent_property_value(0, "result"), 0)

    def test_reader_before_writer_without_double_buffer(self):
        """Same as above but without double buffer — same result since reader runs first."""
        model = _make_db_model(_RBWNoDBBreed, "step_func_code_rbw_nodb.py")
        model.setup(use_gpu=True)
        model.simulate(1, sync_workers_every_n_ticks=1)
        self.assertEqual(model.get_agent_property_value(0, "counter"), 10)
        self.assertEqual(model.get_agent_property_value(0, "result"), 0)


if __name__ == "__main__":
    unittest.main()
