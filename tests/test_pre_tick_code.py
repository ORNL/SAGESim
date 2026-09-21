"""
Test for the `pre_tick_code` kernel hook.

A subclass can return `'pre_tick_code': [lines]` from `_get_extra_kernel_config()`.
The lines are emitted at the top of every tick, before priority 0, with
`thread_id`, `total_threads` and `thread_local_tick` in scope; a line that is
exactly `__GRID_BARRIER__` becomes a software grid barrier at that indentation.

Scenario: an extra kernel array `schedule` holds one value per tick. The pre-tick
code (thread 0 only, then a barrier) writes schedule[tick] into every agent's
`inbox` row, and the only step function (priority 0) accumulates inbox into `total`.
If the hook runs before priority 0 and the barrier makes the write visible,
total == sum(schedule) after the run. SuperNeuroABM uses exactly this shape to
deliver external input spikes.
"""
from pathlib import Path

import cupy as cp
import numpy as np
from cupyx import jit

from sagesim.breed import Breed
from sagesim.model import Model
from sagesim.space import NetworkSpace

SCHEDULE = np.array([1.0, 2.0, 4.0, 8.0, 16.0], dtype=np.float32)


@jit.rawkernel(device="cuda")
def accumulate_step_func(tick, agent_index, agent_ids, breeds, locations, inbox, total):
    total[agent_index] = total[agent_index] + inbox[agent_index]


class InboxBreed(Breed):
    def __init__(self):
        super().__init__("InboxBreed")
        self.register_property("inbox", 0.0)
        self.register_property("total", 0.0)
        self.register_step_func(accumulate_step_func, Path(__file__).resolve(), priority=0,
                                no_double_buffer=["inbox", "total"])


class PreTickModel(Model):
    def __init__(self):
        super().__init__(NetworkSpace(), step_function_file_path="step_func_code_pretick.py")
        self._breed = InboxBreed()
        self.register_breed(self._breed)
        self._schedule_gpu = None

    def create_agent(self):
        return self.create_agent_of_breed(self._breed, inbox=0.0, total=0.0)

    def _get_extra_kernel_config(self):
        inbox_idx = self._agent_factory._property_name_2_index["inbox"]
        return {
            "extra_kernel_params": ["schedule", "n_schedule"],
            "pre_tick_code": [
                "if thread_local_tick < int(n_schedule):",
                "\t_k = int(thread_id)",
                "\twhile _k < num_rank_local_agents:",
                f"\t\ta{inbox_idx}[_k] = schedule[thread_local_tick]",
                "\t\t_k = _k + int(total_threads)",
                "\t__GRID_BARRIER__",
            ],
        }

    def _prepare_kernel_extras(self, num_local_agents, sync_ticks):
        if self._schedule_gpu is None:
            self._schedule_gpu = cp.asarray(SCHEDULE)
        return (self._schedule_gpu, cp.int32(self._schedule_gpu.size))


def test_pre_tick_code_runs_before_priority_zero_every_tick():
    model = PreTickModel()
    ids = [model.create_agent() for _ in range(300)]   # more agents than one block
    model.setup()
    model.simulate(len(SCHEDULE), sync_workers_every_n_ticks=len(SCHEDULE))
    totals = [model.get_agent_property_value(a, "total") for a in ids]
    assert all(t == float(SCHEDULE.sum()) for t in totals), totals[:5]
    # the last tick's delivery is what remains in the inbox
    assert model.get_agent_property_value(ids[0], "inbox") == float(SCHEDULE[-1])


def test_pre_tick_code_ticks_past_the_schedule_deliver_nothing():
    model = PreTickModel()
    a = model.create_agent()
    model.setup()
    model.simulate(len(SCHEDULE) + 3, sync_workers_every_n_ticks=len(SCHEDULE) + 3)
    # inbox keeps the last delivered value; total counts it once per remaining tick
    assert model.get_agent_property_value(a, "inbox") == float(SCHEDULE[-1])
    assert model.get_agent_property_value(a, "total") == float(SCHEDULE.sum() + 3 * SCHEDULE[-1])


def test_no_hook_generates_no_pre_tick_code():
    """A model without the hook produces a kernel with no code before the first priority loop."""
    from sagesim.model import _gen_barrier_code  # noqa: F401  (sanity: symbol still exists)
    m = PreTickModel()
    m._get_extra_kernel_config = lambda: {}
    m._prepare_kernel_extras = lambda n, s: ()
    m.create_agent()
    m.setup()
    src = Path(m._generated_step_function_file_path).read_text()
    body = src.split("thread_local_tick = int(global_tick) + tick", 1)[1]
    first_stmt = next(l for l in body.splitlines() if l.strip())
    assert first_stmt.strip() == "agent_index = thread_id", first_stmt
