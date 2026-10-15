"""Reduce inbox: a breed-local array other agents deposit into.

`register_breed_local_array(..., reduce="sum")` makes an inbox. A sender's step
function calls `deposit_to_agent(inbox, inbox_idx, target, slot, value)`, an atomic
add into the write buffer; the end-of-tick write-back copies the write buffer to the
read buffer and clears it, so the owner reads the tick's total next tick.

Senders have one directed edge to their receiver; receiver step functions run at
priority 0, senders at priority 1. A deposit made in tick t is therefore added to the
receiver's `total` in tick t+1, and after T ticks `total == (T-1) * sum(weights)`.
"""

from pathlib import Path

import numpy as np
import pytest
from cupyx import jit

from sagesim.breed import Breed
from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.utils import deposit_to_agent

HERE = Path(__file__).resolve()


@jit.rawkernel(device="cuda")
def receiver_step(
    tick, agent_index, agent_ids, breeds, locations, total, weight, inbox, inbox_idx,
):
    row = inbox_idx[agent_index]
    if row >= 0:
        total[agent_index] = total[agent_index] + inbox[row][0]


@jit.rawkernel(device="cuda")
def sender_step(
    tick, agent_index, agent_ids, breeds, locations, total, weight, inbox, inbox_idx,
):
    # neighbour 0 is the receiver this sender feeds
    target = int(locations[agent_index][0])
    if target >= 0:
        deposit_to_agent(inbox, inbox_idx, target, 0, weight[agent_index])


@jit.rawkernel(device="cuda")
def blind_sender_step(
    tick, agent_index, agent_ids, breeds, locations, total, weight, inbox, inbox_idx,
):
    # deposits to the "no neighbour" sentinel: must be a no-op, never an index of -1
    deposit_to_agent(inbox, inbox_idx, -1, 0, weight[agent_index])


@jit.rawkernel(device="cuda")
def bystander_step(
    tick, agent_index, agent_ids, breeds, locations, total, weight,
):
    # declares no breed-local arrays: its signature must stay as it was
    total[agent_index] = total[agent_index] + 1.0


class ReceiverBreed(Breed):
    def __init__(self, name="Receiver"):
        super().__init__(name)
        self.register_property("total", 0.0)
        self.register_step_func(receiver_step, HERE, priority=0,
                                no_double_buffer=["total"],
                                breed_local_arrays=["inbox"])


class SenderBreed(Breed):
    def __init__(self):
        super().__init__("Sender")
        self.register_property("weight", 0.0)
        self.register_step_func(sender_step, HERE, priority=1,
                                breed_local_arrays=["inbox"])


class BlindSenderBreed(Breed):
    def __init__(self):
        super().__init__("BlindSender")
        self.register_property("weight", 0.0)
        self.register_step_func(blind_sender_step, HERE, priority=1,
                                breed_local_arrays=["inbox"])


class BystanderBreed(Breed):
    def __init__(self):
        super().__init__("Bystander")
        self.register_step_func(bystander_step, HERE, priority=2,
                                no_double_buffer=["total"],
                                breed_local_arrays=[])


class InboxModel(Model):
    def __init__(self, n_receiver_breeds=1, with_bystander=False, with_blind=False, **bla_kwargs):
        super().__init__(NetworkSpace(), step_function_file_path="step_func_code_reduce_bla.py")
        self.receivers = [ReceiverBreed(f"Receiver{i}") for i in range(n_receiver_breeds)]
        self.sender = SenderBreed()
        for b in self.receivers:
            self.register_breed(b)
        self.register_breed(self.sender)
        self.bystander = None
        if with_bystander:
            self.bystander = BystanderBreed()
            self.register_breed(self.bystander)
        self.blind = None
        if with_blind:
            self.blind = BlindSenderBreed()
            self.register_breed(self.blind)
        kwargs = dict(shape_per_agent=(1,), reduce="sum")
        kwargs.update(bla_kwargs)
        self.register_breed_local_array(
            "inbox", self.receivers if n_receiver_breeds > 1 else self.receivers[0], **kwargs)

    def build(self, weights_per_receiver, bystanders=0, cross_rank=False):
        """weights_per_receiver: list (one per receiver) of lists of sender weights.

        Senders are created on their receiver's rank (cross_rank=False): a deposit into
        a LOCAL row needs no exchange. With cross_rank=True every sender lives on the
        next rank, so each deposit lands in a ghost row and reaches the receiver only
        through the reduce-back-to-owner exchange (neighbor_visible=True). Receivers
        themselves are round-robin.
        """
        receiver_ids = []
        for r, weights in enumerate(weights_per_receiver):
            breed = self.receivers[r % len(self.receivers)]
            rid = self.create_agent_of_breed(breed, total=0.0)
            receiver_ids.append(rid)
            home = self._agent_factory._agent2rank[rid]
            where = (home + 1) % _size() if cross_rank else home
            for w in weights:
                sid = self.create_agent_of_breed(self.sender, rank=where, weight=float(w))
                self.get_space().connect_agents(sid, rid, directed=True)
        self.bystander_ids = [self.create_agent_of_breed(self.bystander, total=0.0)
                              for _ in range(bystanders)]
        return receiver_ids


def _totals(model, receiver_ids):
    # collective getter: every rank calls it for every id
    return np.array([model.get_agent_property_value(r, "total") for r in receiver_ids])


def _size():
    try:
        from mpi4py import MPI
        return MPI.COMM_WORLD.Get_size()
    except ImportError:
        return 1


def _single_rank():
    return _size() == 1


class TestReduceInbox:

    def test_sum_arrives_next_tick(self):
        weights = [[1.0, 2.0, 3.5], [10.0], [0.25] * 40]
        m = InboxModel()
        rids = m.build(weights)
        m.setup()
        T = 7
        m.simulate(ticks=T)
        expected = (T - 1) * np.array([sum(w) for w in weights])
        np.testing.assert_allclose(_totals(m, rids), expected, rtol=1e-6)
        if _single_rank():
            # the read buffer holds the last tick's deposits (rank-local array)
            inbox = m.get_breed_local_array("inbox")
            np.testing.assert_allclose(inbox[:, 0], [sum(w) for w in weights], rtol=1e-6)

    def test_receiver_without_senders_reads_zero(self):
        m = InboxModel()
        rids = m.build([[2.0], []])
        m.setup()
        m.simulate(ticks=4)
        np.testing.assert_allclose(_totals(m, rids), [3 * 2.0, 0.0])

    def test_resume_matches_single_launch(self):
        """simulate(T) once vs simulate(1) x T: the drained write buffer must not be
        copied over the read buffer between launches."""
        weights = [[1.0, 2.0], [4.0, 0.5, 0.5]]
        m1 = InboxModel(); r1 = m1.build(weights); m1.setup()
        m2 = InboxModel(); r2 = m2.build(weights); m2.setup()
        m1.simulate(ticks=6)
        for _ in range(6):
            m2.simulate(ticks=1)
        np.testing.assert_allclose(_totals(m1, r1), _totals(m2, r2))
        np.testing.assert_allclose(_totals(m1, r1), 5 * np.array([3.0, 5.0]))

    def test_bystander_breed_keeps_its_signature(self):
        m = InboxModel(with_bystander=True)
        rids = m.build([[1.0, 1.0]], bystanders=3)
        m.setup()
        m.simulate(ticks=3)
        np.testing.assert_allclose(_totals(m, rids), [2 * 2.0])
        np.testing.assert_allclose(_totals(m, m.bystander_ids), [3.0, 3.0, 3.0])

    def test_inbox_shared_by_two_receiver_breeds(self):
        weights = [[1.0, 2.0], [5.0], [0.5, 0.5, 0.5]]  # receivers alternate breeds
        m = InboxModel(n_receiver_breeds=2)
        rids = m.build(weights)
        m.setup()
        m.simulate(ticks=5)
        np.testing.assert_allclose(_totals(m, rids), 4 * np.array([3.0, 5.0, 1.5]))

    def test_deposit_to_sentinel_is_a_noop(self):
        """A target of -1 (no neighbour) deposits nowhere and must not touch idx_map[-1]."""
        m = InboxModel(with_blind=True)
        rids = m.build([[1.0, 2.0]])
        for _ in range(5):
            m.create_agent_of_breed(m.blind, weight=100.0)
        m.setup()
        m.simulate(ticks=4)
        np.testing.assert_allclose(_totals(m, rids), [3 * 3.0])

    def test_many_senders_one_target(self):
        n = 4096
        m = InboxModel()
        rids = m.build([[1.0] * n])
        m.setup()
        m.simulate(ticks=3)
        np.testing.assert_allclose(_totals(m, rids), [2 * n])


class TestCrossRankDeposits:
    """Depositors on another rank than their receiver: the deposits land in a ghost row
    and the reduce-back-to-owner exchange sums them into the owner's row. At one rank
    the "next rank" is the same rank, so these degrade to the local cases."""

    def test_remote_senders_reach_owner(self):
        weights = [[1.0, 2.0, 3.5], [10.0], [0.25] * 40]
        m = InboxModel(neighbor_visible=True)
        rids = m.build(weights, cross_rank=True)
        m.setup()
        T = 7
        m.simulate(ticks=T)
        np.testing.assert_allclose(_totals(m, rids), (T - 1) * np.array([sum(w) for w in weights]),
                                   rtol=1e-6)

    def test_local_and_remote_senders_add_up(self):
        """Half the senders beside the receiver, half on the next rank: one total."""
        m = InboxModel(neighbor_visible=True)
        rid = m.create_agent_of_breed(m.receivers[0], total=0.0)
        home = m._agent_factory._agent2rank[rid]
        for w, where in ((1.0, home), (2.0, (home + 1) % _size()), (4.0, home), (8.0, (home + 1) % _size())):
            sid = m.create_agent_of_breed(m.sender, rank=where, weight=w)
            m.get_space().connect_agents(sid, rid, directed=True)
        m.setup()
        m.simulate(ticks=5)
        np.testing.assert_allclose(_totals(m, [rid]), [4 * 15.0])

    def test_two_receiver_breeds_visible(self):
        """A multi-breed array may be neighbor_visible; ghosts of either breed get rows."""
        weights = [[1.0, 2.0], [5.0], [0.5, 0.5, 0.5], [3.0]]
        m = InboxModel(n_receiver_breeds=2, neighbor_visible=True)
        rids = m.build(weights, cross_rank=True)
        m.setup()
        m.simulate(ticks=5)
        np.testing.assert_allclose(_totals(m, rids), 4 * np.array([3.0, 5.0, 1.5, 3.0]))

    def test_chunked_launches_match(self):
        weights = [[1.0, 2.0], [4.0, 0.5, 0.5]]
        m1 = InboxModel(neighbor_visible=True); r1 = m1.build(weights, cross_rank=True); m1.setup()
        m2 = InboxModel(neighbor_visible=True); r2 = m2.build(weights, cross_rank=True); m2.setup()
        m1.simulate(ticks=6)
        for _ in range(6):
            m2.simulate(ticks=1)
        np.testing.assert_allclose(_totals(m1, r1), _totals(m2, r2))
        np.testing.assert_allclose(_totals(m1, r1), 5 * np.array([3.0, 5.0]))


class TestReduceValidation:

    def test_neighbor_visible_accepted(self):
        m = InboxModel(neighbor_visible=True)
        rids = m.build([[1.0, 2.0]])
        m.setup()
        m.simulate(ticks=3)
        np.testing.assert_allclose(_totals(m, rids), [2 * 3.0])

    def test_bad_reduce_op(self):
        with pytest.raises(ValueError):
            InboxModel(reduce="mean")

    def test_unknown_declared_array(self):
        m = InboxModel()
        m.build([[1.0]])
        m.sender.register_step_func(sender_step, HERE, priority=1,
                                    breed_local_arrays=["inbox", "nope"])
        with pytest.raises(ValueError):
            m.setup()

    def test_no_double_buffer_rejected(self):
        m = InboxModel()
        m.build([[1.0]])
        m.sender.register_step_func(sender_step, HERE, priority=1,
                                    no_double_buffer=["inbox"],
                                    breed_local_arrays=["inbox"])
        with pytest.raises(ValueError):
            m.setup()
