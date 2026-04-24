"""
End-to-end simulation correctness tests.

Tests deterministic SIR wave propagation and seed-based reproducibility
using SAGESim's counter-based RNG (Philox-2x32-10).
"""

import sys
import unittest
from pathlib import Path

import networkx as nx
import cupy as cp
from cupyx import jit

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed
from sagesim.math_utils import rand_uniform_philox


# ── Step function using deterministic Philox RNG ─────────────────────

@jit.rawkernel(device="cuda")
def sir_step_func(
    tick, agent_index, p_infection, p_recovery,
    agent_ids, breeds, locations, state_tensor,
):
    """SIR step: infection + recovery using counter-based RNG."""
    neighbor_indices = locations[agent_index]
    agent_state = int(state_tensor[agent_index])
    rand = rand_uniform_philox(tick, agent_index, 1)

    if agent_state == 2 and rand < p_recovery:
        state_tensor[agent_index] = 3  # RECOVERED
    elif agent_state == 1:
        i = 0
        while i < len(neighbor_indices) and not cp.isnan(neighbor_indices[i]):
            neighbor_index = int(neighbor_indices[i])
            neighbor_state = int(state_tensor[neighbor_index])
            if neighbor_state == 2 and rand < p_infection:
                state_tensor[agent_index] = 2  # INFECTED
            i += 1


# ── Breed / Model ────────────────────────────────────────────────────

class _SIRBreed(Breed):
    def __init__(self):
        super().__init__("SIR")
        self.register_property("state", 1)
        self.register_step_func(sir_step_func, Path(__file__).resolve(), priority=0)


class _SIRModel(Model):
    def __init__(self, p_infection=1.0, p_recovery=1.0, codefile="step_func_code_sim_test.py"):
        super().__init__(NetworkSpace(), step_function_file_path=codefile)
        self._breed = _SIRBreed()
        self.register_breed(self._breed)
        self.register_global_property("p_infection", p_infection)
        self.register_global_property("p_recovery", p_recovery)

    def create_agent(self, state=1):
        return self.create_agent_of_breed(self._breed, state=state)

    def connect_agents(self, a0, a1):
        self.get_space().connect_agents(a0, a1)


def _build_chain_model(num_agents, p_infection=1.0, p_recovery=1.0, seed=42,
                        codefile="step_func_code_sim_test.py"):
    """Build a chain: 0-1-2-...-N, agent 0 infected."""
    model = _SIRModel(p_infection=p_infection, p_recovery=p_recovery, codefile=codefile)
    model.set_seed(seed)
    for _ in range(num_agents):
        model.create_agent(state=1)
    model.set_agent_property_value(0, "state", 2)
    for i in range(num_agents - 1):
        model.connect_agents(i, i + 1)
    return model


def _get_states(model, n):
    return [int(model.get_agent_property_value(i, "state")) for i in range(n)]


# ── Tests ────────────────────────────────────────────────────────────

class TestSimulation(unittest.TestCase):

    def tearDown(self):
        for k in [k for k in sys.modules if k.startswith("step_func_code")]:
            del sys.modules[k]

    def test_deterministic_sir_chain(self):
        """With p=1.0, infection wave propagates exactly one hop per tick on a chain.

        The SIR model with p_infection=1 and p_recovery=1:
        - Tick 1: agent 0 recovers, agent 1 gets infected
        - Tick 2: agent 1 recovers, agent 2 gets infected
        - ...
        After 5 ticks on a 10-agent chain, agents 0-4 recovered, agent 5 infected, 6-9 susceptible.
        """
        n = 10
        model = _build_chain_model(n, p_infection=1.0, p_recovery=1.0)
        model.setup(use_gpu=True)
        model.simulate(5, sync_workers_every_n_ticks=1)

        states = _get_states(model, n)
        # Agents 0-4 should be recovered (3), agent 5 infected (2), agents 6-9 susceptible (1)
        for i in range(5):
            self.assertEqual(states[i], 3, f"Agent {i} should be recovered")
        self.assertEqual(states[5], 2, "Agent 5 should be infected")
        for i in range(6, 10):
            self.assertEqual(states[i], 1, f"Agent {i} should be susceptible")

    def test_seed_reproducibility(self):
        """Same seed produces identical results across two runs."""
        n = 20

        model1 = _build_chain_model(n, p_infection=0.5, p_recovery=0.3, seed=999,
                                     codefile="step_func_code_sim_test_r1.py")
        model1.setup(use_gpu=True)
        model1.simulate(10, sync_workers_every_n_ticks=1)
        states1 = _get_states(model1, n)

        # Clear cached modules between runs
        for k in [k for k in sys.modules if k.startswith("step_func_code")]:
            del sys.modules[k]

        model2 = _build_chain_model(n, p_infection=0.5, p_recovery=0.3, seed=999,
                                     codefile="step_func_code_sim_test_r2.py")
        model2.setup(use_gpu=True)
        model2.simulate(10, sync_workers_every_n_ticks=1)
        states2 = _get_states(model2, n)

        self.assertEqual(states1, states2, "Same seed should produce identical states")

    def test_different_seed_different_result(self):
        """Different seeds produce different outcomes (stochastic model)."""
        n = 50

        model1 = _build_chain_model(n, p_infection=0.3, p_recovery=0.2, seed=111,
                                     codefile="step_func_code_sim_test_d1.py")
        model1.setup(use_gpu=True)
        model1.simulate(20, sync_workers_every_n_ticks=1)
        states1 = _get_states(model1, n)

        for k in [k for k in sys.modules if k.startswith("step_func_code")]:
            del sys.modules[k]

        model2 = _build_chain_model(n, p_infection=0.3, p_recovery=0.2, seed=222,
                                     codefile="step_func_code_sim_test_d2.py")
        model2.setup(use_gpu=True)
        model2.simulate(20, sync_workers_every_n_ticks=1)
        states2 = _get_states(model2, n)

        self.assertNotEqual(states1, states2,
                            "Different seeds should (almost certainly) produce different states")


if __name__ == "__main__":
    unittest.main()
