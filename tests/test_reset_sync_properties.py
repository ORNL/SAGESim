"""reset(sync_properties=...) controls which columns come back off the device.

The readback exists so a caller can see what the kernel produced. A caller that restores
a column from its own source -- initial conditions, a checkpoint, a generator -- pays a
full device->host copy per column for data it immediately overwrites, and at large agent
counts that dominates reset(). These tests pin the three cases:

  * None (default)  -- every written column comes back, the historical behaviour
  * a name list     -- only those columns come back; the rest keep their host values
  * ()              -- nothing comes back; the host columns are untouched

The last one also pins the invariant the option relies on: nothing but this readback
writes the host property columns, so skipping it leaves them exactly as the last build
produced them.
"""

from pathlib import Path

import numpy as np
from cupyx import jit

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed


@jit.rawkernel(device="cuda")
def bump_step(tick, agent_index, agent_ids, breeds, locations, alpha, beta):
    """Move both properties away from whatever the host set them to."""
    alpha[agent_index] = alpha[agent_index] + 1.0
    beta[agent_index] = beta[agent_index] + 10.0


class BumpBreed(Breed):
    def __init__(self):
        super().__init__("BumpBreed")
        self.register_property("alpha", 0.0)
        self.register_property("beta", 0.0)
        self.register_step_func(
            bump_step,
            Path(__file__).resolve(),
            priority=0,
            no_double_buffer=["alpha", "beta"],
        )


class BumpModel(Model):
    def __init__(self, tag):
        super().__init__(NetworkSpace(),
                         step_function_file_path=f"step_func_code_sync_{tag}.py")
        self._breed = BumpBreed()
        self.register_breed(breed=self._breed)

    def create_agents(self, n):
        for _ in range(n):
            self.create_agent_of_breed(self._breed, alpha=0.0, beta=0.0)


N_AGENTS, TICKS = 8, 3


def _host(model, name):
    col = model._agent_factory._property_name_2_agent_data_tensor[name]
    return [float(np.ravel(col[i])[0]) for i in range(N_AGENTS)]


def _run(tag, **reset_kwargs):
    m = BumpModel(tag)
    m.create_agents(N_AGENTS)
    m.setup()
    m.simulate(TICKS)
    m.reset(**reset_kwargs)
    return m


def test_default_syncs_everything():
    """None keeps the historical behaviour: both columns reflect the kernel."""
    m = _run("all")
    assert _host(m, "alpha") == [float(TICKS)] * N_AGENTS
    assert _host(m, "beta") == [float(10 * TICKS)] * N_AGENTS


def test_named_subset_syncs_only_those():
    m = _run("subset", sync_properties=["alpha"])
    assert _host(m, "alpha") == [float(TICKS)] * N_AGENTS, "alpha should have come back"
    assert _host(m, "beta") == [0.0] * N_AGENTS, "beta was not requested but came back"


def test_empty_skips_the_readback_entirely():
    """The host columns are left exactly as the build produced them."""
    m = _run("none", sync_properties=())
    assert _host(m, "alpha") == [0.0] * N_AGENTS
    assert _host(m, "beta") == [0.0] * N_AGENTS


def test_skipped_readback_still_rebuilds_and_runs():
    """Whatever the host holds after the skip is what the next simulate() uploads."""
    m = _run("rerun", sync_properties=())
    m.simulate(TICKS)
    m.reset(sync_properties=["alpha"])
    # Started from the host's 0.0 again, so alpha is TICKS, not 2 * TICKS.
    assert _host(m, "alpha") == [float(TICKS)] * N_AGENTS
