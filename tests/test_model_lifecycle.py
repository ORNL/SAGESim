"""Tests for Model and AgentFactory core lifecycle API."""

import sys
import unittest
from pathlib import Path

import cupy as cp
from cupyx import jit

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed


@jit.rawkernel(device="cuda")
def _increment_step(tick, agent_index, agent_ids, breeds, locations, counter):
    """Minimal step function: increment counter each tick."""
    counter[agent_index] = counter[agent_index] + 1


class MinimalBreed(Breed):
    def __init__(self):
        super().__init__("Minimal")
        self.register_property("counter", 0)
        self.register_step_func(_increment_step, __file__, priority=0)


class MinimalModel(Model):
    def __init__(self):
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_lifecycle_test.py")
        self._breed = MinimalBreed()
        self.register_breed(self._breed)

    def create_agent(self, counter=0):
        return self.create_agent_of_breed(self._breed, counter=counter)


class TestModelLifecycle(unittest.TestCase):

    def tearDown(self):
        modules_to_remove = [k for k in sys.modules if k.startswith("step_func_code")]
        for m in modules_to_remove:
            del sys.modules[m]

    def test_register_breed_and_create_agent(self):
        model = MinimalModel()
        agent_id = model.create_agent()
        self.assertEqual(agent_id, 0)
        self.assertEqual(model._agent_factory.num_agents, 1)

    def test_create_multiple_agents(self):
        model = MinimalModel()
        ids = [model.create_agent() for _ in range(5)]
        self.assertEqual(ids, [0, 1, 2, 3, 4])
        self.assertEqual(model._agent_factory.num_agents, 5)

    def test_set_get_agent_property(self):
        model = MinimalModel()
        agent_id = model.create_agent(counter=0)
        model.set_agent_property_value(agent_id, "counter", 42)
        val = model.get_agent_property_value(agent_id, "counter")
        self.assertEqual(val, 42)

    def test_register_global_property(self):
        model = MinimalModel()
        model.register_global_property("rate", 0.5)
        val = model.get_global_property_value("rate")
        self.assertAlmostEqual(val, 0.5)

    def test_set_global_property(self):
        model = MinimalModel()
        model.register_global_property("rate", 0.5)
        model.set_global_property_value("rate", 0.9)
        val = model.get_global_property_value("rate")
        self.assertAlmostEqual(val, 0.9)

    def test_model_setup(self):
        model = MinimalModel()
        model.create_agent()
        model.setup(use_gpu=True)
        self.assertTrue(model._is_setup)

    def test_set_seed(self):
        model = MinimalModel()
        model.set_seed(12345)
        self.assertEqual(model._seed, 12345)


if __name__ == "__main__":
    unittest.main()
