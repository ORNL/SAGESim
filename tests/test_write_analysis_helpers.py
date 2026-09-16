"""
analyze_step_function_for_writes must see writes made inside device helpers that a
step function forwards its property tensors to (dispatcher pattern), including
generated dispatchers that already pass the framework-injected `_seed`/`logical_ids`.
"""
from cupyx import jit

from sagesim.model import analyze_step_function_for_writes


# properties (num_properties = 4): breeds, locations, params, state
@jit.rawkernel(device="cuda")
def _helper_writes_state(tick, agent_index, agent_ids, breeds, locations, params, state):
    state[agent_index][0] = params[agent_index][0] * 2.0


@jit.rawkernel(device="cuda")
def _helper_reads_only(tick, agent_index, agent_ids, breeds, locations, params, state):
    return params[agent_index][0] + state[agent_index][0]


@jit.rawkernel(device="cuda")
def _helper_writes_params(tick, agent_index, agent_ids, breeds, locations, params, state):
    params[agent_index][0] = 0.0


@jit.rawkernel(device="cuda")
def step_direct_write(tick, agent_index, agent_ids, breeds, locations, params, state):
    state[agent_index][1] = 1.0


@jit.rawkernel(device="cuda")
def step_forwards_to_writer(tick, agent_index, agent_ids, breeds, locations, params, state):
    _helper_writes_state(tick, agent_index, agent_ids, breeds, locations, params, state)


@jit.rawkernel(device="cuda")
def step_forwards_to_reader(tick, agent_index, agent_ids, breeds, locations, params, state):
    v = _helper_reads_only(tick, agent_index, agent_ids, breeds, locations, params, state)
    return v


@jit.rawkernel(device="cuda")
def step_dispatcher_with_injected_args(tick, agent_index, agent_ids, breeds, locations, params, state):
    # shaped like superneuroabm's generated learning_rule_selector: `_seed` and
    # `logical_ids` are already in the call although the callee's source has 7 params
    if params[agent_index][1] > 0.0:
        _helper_writes_params(tick, agent_index, _seed, agent_ids, logical_ids, breeds, locations, params, state)  # noqa: F821


@jit.rawkernel(device="cuda")
def step_swapped_forward(tick, agent_index, agent_ids, breeds, locations, params, state):
    # passes `state` where the helper expects `params`: the write lands on `state` (idx 3)
    _helper_writes_params(tick, agent_index, agent_ids, breeds, locations, state, params)


N = 4  # breeds=0, locations=1, params=2, state=3


def test_direct_write_detected():
    assert analyze_step_function_for_writes(step_direct_write, N) == {3}


def test_write_inside_forwarded_helper_detected():
    assert analyze_step_function_for_writes(step_forwards_to_writer, N) == {3}


def test_read_only_helper_adds_nothing():
    assert analyze_step_function_for_writes(step_forwards_to_reader, N) == set()


def test_dispatcher_with_injected_names_maps_positionally():
    assert analyze_step_function_for_writes(step_dispatcher_with_injected_args, N) == {2}


def test_forwarding_maps_by_position_not_name():
    assert analyze_step_function_for_writes(step_swapped_forward, N) == {3}
