"""Event-woken step functions: Model.configure_wake + register_step_func(wake_on_roles).

Emitters (priority 0, clocked) raise an event on ticks where (tick + id) % period == 0
by writing slot tick % 2 of the width-2 `flag` property. Followers (priority 1) list an
emitter as neighbour 0 and are stepped only on the tick after that emitter raised; each
step increments `count`. A follower listing two emitters is woken through role 0 only.
Runs under mpirun too (tests/run_mpi_tests.sh): a follower's emitter may be a ghost row,
whose flag arrives through the property exchange.
"""

from pathlib import Path

import numpy as np
import pytest
from cupyx import jit

from sagesim.breed import Breed
from sagesim.model import Model
from sagesim.space import NetworkSpace

HERE = Path(__file__).resolve()
PROPS = ("flag", "period", "count", "last")


@jit.rawkernel(device="cuda")
def emitter_step(tick, agent_index, agent_ids, breeds, locations, flag, period, count, last):
    t = int(tick)
    p = int(period[agent_index])
    gid = int(agent_ids[agent_index])
    raised = 0.0
    if p > 0 and (t + gid) % p == 0:
        raised = 1.0
    flag[agent_index][t % 2] = raised


@jit.rawkernel(device="cuda")
def follower_step(tick, agent_index, agent_ids, breeds, locations, flag, period, count, last):
    count[agent_index] = count[agent_index] + 1.0
    last[agent_index] = 1.0 * int(tick)


@jit.rawkernel(device="cuda")
def clocked_step(tick, agent_index, agent_ids, breeds, locations, flag, period, count, last):
    count[agent_index] = count[agent_index] + 1.0


def _register_props(breed):
    breed.register_property("flag", [0.0, 0.0])
    breed.register_property("period", 0.0)
    breed.register_property("count", 0.0)
    breed.register_property("last", -1.0)


class EmitterBreed(Breed):
    def __init__(self):
        super().__init__("Emitter")
        _register_props(self)
        self.register_step_func(emitter_step, HERE, priority=0, no_double_buffer=list(PROPS))


class FollowerBreed(Breed):
    def __init__(self, roles=(0,)):
        super().__init__("Follower")
        _register_props(self)
        self.register_step_func(follower_step, HERE, priority=1, no_double_buffer=list(PROPS),
                                wake_on_roles=list(roles))


@jit.rawkernel(device="cuda")
def follower_second_step(tick, agent_index, agent_ids, breeds, locations, flag, period, count, last):
    # a second woken priority on the same breed: counts in `period` (unused by followers)
    period[agent_index] = period[agent_index] + 1.0


class TwoPriorityFollowerBreed(Breed):
    """Woken at two priorities by the same emitter: each priority must run (its own stamp)."""
    def __init__(self):
        super().__init__("Follower")
        _register_props(self)
        self.register_step_func(follower_step, HERE, priority=1, no_double_buffer=list(PROPS),
                                wake_on_roles=[0])
        self.register_step_func(follower_second_step, HERE, priority=2,
                                no_double_buffer=list(PROPS), wake_on_roles=[0])


class ClockedBreed(Breed):
    def __init__(self):
        super().__init__("Clocked")
        _register_props(self)
        self.register_step_func(clocked_step, HERE, priority=1, no_double_buffer=list(PROPS))


class WakeModel(Model):
    def __init__(self, configure=True, with_clocked=False, roles=(0,), two_priorities=False):
        super().__init__(NetworkSpace(), step_function_file_path="step_func_code_wake.py")
        self.emitter = EmitterBreed()
        self.follower = TwoPriorityFollowerBreed() if two_priorities else FollowerBreed(roles)
        self.register_breed(self.emitter)
        self.register_breed(self.follower)
        if with_clocked:
            self.register_breed(ClockedBreed())
        if configure:
            self.configure_wake(event_property="flag")

    def emitter_agent(self, period):
        return self.create_agent_of_breed(self.emitter, flag=[0.0, 0.0], period=float(period),
                                          count=0.0, last=-1.0)

    def follower_of(self, *emitters):
        f = self.create_agent_of_breed(self.follower, flag=[0.0, 0.0], period=0.0, count=0.0, last=-1.0)
        for e in emitters:
            self.get_space().connect_agents(f, e, directed=True)
        return f


def _expected(period, gid, ticks):
    """Follower steps at tick t iff its emitter raised at t-1: (t-1+gid) % period == 0."""
    if period <= 0:
        return 0, -1
    hits = [t for t in range(1, ticks) if (t - 1 + gid) % period == 0]
    return len(hits), (hits[-1] if hits else -1)


def _get(m, aid, prop):
    return m.get_agent_property_value(aid, prop)   # collective


class TestWakeList:

    def test_followers_step_only_after_events(self):
        m = WakeModel()
        periods = [3, 5, 1, 0]                     # 0 = never raises
        emitters = [m.emitter_agent(p) for p in periods]
        followers = [m.follower_of(e) for e in emitters]
        twins = [m.follower_of(emitters[0]), m.follower_of(emitters[0])]   # two followers, one emitter
        m.setup()
        T = 40
        m.simulate(ticks=T)
        for p, e, f in zip(periods, emitters, followers):
            n, last = _expected(p, e, T)
            assert _get(m, f, "count") == n, (p, e)
            assert _get(m, f, "last") == last, (p, e)
        for f in twins:
            n, last = _expected(3, emitters[0], T)
            assert _get(m, f, "count") == n
            assert _get(m, f, "last") == last

    def test_role_selection(self):
        """A follower listing two emitters is woken through role 0 only."""
        m = WakeModel()
        a, b = m.emitter_agent(7), m.emitter_agent(2)
        f = m.follower_of(a, b)
        m.setup()
        m.simulate(ticks=30)
        assert _get(m, f, "count") == _expected(7, a, 30)[0]

    def test_both_roles(self):
        """With roles [0, 1] both emitters wake it, but a tick counts once."""
        m = WakeModel(roles=(0, 1))
        a, b = m.emitter_agent(4), m.emitter_agent(6)
        f = m.follower_of(a, b)
        m.setup()
        T = 50
        m.simulate(ticks=T)
        hits = {t for t in range(1, T) if (t - 1 + a) % 4 == 0} | {t for t in range(1, T) if (t - 1 + b) % 6 == 0}
        assert _get(m, f, "count") == len(hits)

    def test_same_row_woken_at_two_priorities(self):
        """A row dequeued at two woken priorities in one tick runs at both (the dedupe
        stamp is per priority, not per row)."""
        m = WakeModel(two_priorities=True)
        e = m.emitter_agent(3)
        f = m.follower_of(e)
        m.setup()
        m.simulate(ticks=30)
        n, _ = _expected(3, e, 30)
        assert _get(m, f, "count") == n          # priority 1
        assert _get(m, f, "period") == n         # priority 2

    def test_resume_simulate(self):
        m1, m2 = WakeModel(), WakeModel()
        for m in (m1, m2):
            e = m.emitter_agent(3)
            m.follower_of(e)
            m.setup()
        m1.simulate(ticks=24)
        for _ in range(6):
            m2.simulate(ticks=4)
        f1 = [a for a in range(2) if True]
        np.testing.assert_array_equal(
            [_get(m1, 1, "count"), _get(m1, 1, "last")], [_get(m2, 1, "count"), _get(m2, 1, "last")])


class TestWakeValidation:

    def test_needs_event_property(self):
        m = WakeModel(configure=False)
        m.follower_of(m.emitter_agent(2))
        with pytest.raises(ValueError):
            m.setup()

    def test_priority_all_or_nothing(self):
        m = WakeModel(with_clocked=True)
        m.follower_of(m.emitter_agent(2))
        with pytest.raises(ValueError):
            m.setup()
