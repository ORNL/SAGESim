"""A rank with no agents must still take part in every collective.

Two agents on N ranks: for N > 2 some ranks own nothing. Before 2026-10-14 such a rank
raised KeyError in worker_coroutine (no entry in _rank2agentid2agentidx), launched zero
blocks, and allocated 1-D tensors where the shared kernel expected 2-D/3-D ones; the
other ranks then hung in a collective. Run under tests/run_mpi_tests.sh at 1, 2, 4 ranks.
"""

from pathlib import Path

import numpy as np
from cupyx import jit

from sagesim.breed import Breed
from sagesim.model import Model
from sagesim.space import NetworkSpace

HERE = Path(__file__).resolve()


@jit.rawkernel(device="cuda")
def count_step(tick, agent_index, agent_ids, breeds, locations, value, pair, history,
               scratch, scratch_idx):
    # a scalar, a 2-D and a 3-D property plus a breed-local array: every tensor
    # shape an empty rank has to allocate without any row to infer it from
    value[agent_index] = value[agent_index] + 1.0
    pair[agent_index][1] = pair[agent_index][1] + 2.0
    history[agent_index][0][2] = history[agent_index][0][2] + 3.0
    row = scratch_idx[agent_index]
    if row >= 0:
        scratch[row][0] = scratch[row][0] + 4.0


class CountBreed(Breed):
    def __init__(self):
        super().__init__("Count")
        self.register_property("value", 0.0)
        self.register_property("pair", [0.0, 0.0])
        self.register_property("history", [[0.0, 0.0, 0.0]])
        self.register_step_func(count_step, HERE, priority=0,
                                no_double_buffer=["value", "pair", "history", "scratch"])


class CountModel(Model):
    def __init__(self):
        super().__init__(NetworkSpace(), step_function_file_path="step_func_code_empty_rank.py")
        self.breed = CountBreed()
        self.register_breed(self.breed)
        self.register_breed_local_array("scratch", self.breed, shape_per_agent=(1,))


def test_two_agents_any_rank_count():
    m = CountModel()
    ids = [m.create_agent_of_breed(m.breed, value=0.0, pair=[0.0, 0.0],
                                   history=[[0.0, 0.0, 0.0]]) for _ in range(2)]
    m.setup()
    m.simulate(ticks=5)
    for aid in ids:                                   # collective getters
        np.testing.assert_allclose(m.get_agent_property_value(aid, "value"), 5.0)
        np.testing.assert_allclose(m.get_agent_property_value(aid, "pair"), [0.0, 10.0])
        np.testing.assert_allclose(m.get_agent_property_value(aid, "history"), [[0.0, 0.0, 15.0]])
