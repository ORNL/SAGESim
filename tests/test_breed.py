import unittest
from math import nan
from pathlib import Path

from cupyx import jit

from sagesim.breed import Breed


@jit.rawkernel(device="cuda")
def _dummy_step(tick, agent_index, agent_ids, breeds, locations, prop_a):
    pass


class TestBreed(unittest.TestCase):

    def test_register_property_scalar(self):
        breed = Breed("TestBreed")
        breed.register_property("health", 100)
        self.assertIn("health", breed.properties)
        self.assertEqual(breed.properties["health"], 100)

    def test_register_property_list_default(self):
        breed = Breed("TestBreed")
        breed.register_property("history", [0.0] * 10)
        self.assertEqual(breed.properties["history"], [0.0] * 10)
        self.assertEqual(len(breed.properties["history"]), 10)

    def test_register_step_func(self):
        breed = Breed("TestBreed")
        breed.register_property("prop_a", 0)
        breed.register_step_func(_dummy_step, __file__, priority=0)
        self.assertIn(0, breed.step_funcs)
        func, path = breed.step_funcs[0]
        self.assertEqual(func, _dummy_step)
        self.assertEqual(path, str(Path(__file__).resolve()))

    def test_register_step_func_no_double_buffer(self):
        breed = Breed("TestBreed")
        breed.register_property("counter", 0)
        breed.register_property("result", 0)
        breed.register_step_func(
            _dummy_step, __file__, priority=0,
            no_double_buffer=["counter"]
        )
        self.assertIn("counter", breed.no_double_buffer_props)
        self.assertNotIn("result", breed.no_double_buffer_props)

    def test_property_neighbor_visible(self):
        breed = Breed("TestBreed")
        breed.register_property("state", 1, neighbor_visible=True)
        breed.register_property("internal", 0, neighbor_visible=False)
        self.assertTrue(breed.prop2neighbor_visible["state"])
        self.assertFalse(breed.prop2neighbor_visible["internal"])


if __name__ == "__main__":
    unittest.main()
