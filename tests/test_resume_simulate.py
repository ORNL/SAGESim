"""
Test that simulate() can be called multiple times and produce the same
results as a single call with the total number of ticks.

simulate(50) should equal simulate(10) x 5.

This tests that the fused kernel correctly handles state between
worker_coroutine calls - no state corruption from post-kernel
write-back, GPU synchronization, or buffer management.
"""

import unittest
from pathlib import Path

import numpy as np
import cupy as cp
from cupyx import jit

from sagesim.model import Model
from sagesim.space import NetworkSpace
from sagesim.breed import Breed
from sagesim.math_utils import rand_uniform_philox


# A step function that accumulates state each tick.
# Uses RNG so results are sensitive to any state corruption.
@jit.rawkernel(device="cuda")
def accumulate_step(
    tick,
    agent_index,
    agent_ids,
    breeds,
    locations,
    value,
    accum,
):
    """Each tick: value += rand, accum += value."""
    v = value[agent_index]
    r = rand_uniform_philox(tick, agent_index, 1)
    v = v + r
    value[agent_index] = v
    accum[agent_index] = accum[agent_index] + v


class AccumBreed(Breed):
    def __init__(self):
        super().__init__("AccumBreed")
        self.register_property("value", 0.0)
        self.register_property("accum", 0.0)
        self.register_step_func(
            accumulate_step,
            Path(__file__).resolve(),
            priority=0,
            no_double_buffer=["value", "accum"],
        )


class AccumModel(Model):
    def __init__(self):
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_resume.py")
        self._breed = AccumBreed()
        self.register_breed(breed=self._breed)

    def create_agents(self, n):
        for _ in range(n):
            self.create_agent_of_breed(self._breed, value=0.0, accum=0.0)


# --- Step function that reads/writes a double-buffered breed-local array ---
@jit.rawkernel(device="cuda")
def bla_step(
    tick,
    agent_index,
    agent_ids,
    breeds,
    locations,
    counter,
    bla_sum, bla_sum_idx,
):
    """Each tick: read prev from bla_sum, increment, store prev in counter."""
    idx = bla_sum_idx[agent_index]
    if idx >= 0:
        prev = bla_sum[idx][0]
        bla_sum[idx][0] = prev + 1.0
        counter[agent_index] = prev


class BLABreed(Breed):
    def __init__(self):
        super().__init__("BLABreed")
        self.register_property("counter", 0.0)
        self.register_step_func(
            bla_step,
            Path(__file__).resolve(),
            priority=0,
            no_double_buffer=["counter"],  # bla_sum is NOT listed → double-buffered
        )


class BLAModel(Model):
    def __init__(self):
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_bla_resume.py")
        self._breed = BLABreed()
        self.register_breed(breed=self._breed)

    def create_agents(self, n):
        for _ in range(n):
            self.create_agent_of_breed(self._breed, counter=0.0)

    def register_bla(self):
        self.register_breed_local_array(
            "bla_sum", breed=self._breed,
            shape_per_agent=(2,), neighbor_visible=False)


class TestResumeSimulate(unittest.TestCase):
    """simulate(N) once must equal simulate(k) repeated N/k times."""

    def test_single_vs_multi_batch_small(self):
        """simulate(10) once vs simulate(1) x 10, 4 agents."""
        m1 = AccumModel(); m1.create_agents(4); m1.set_seed(42); m1.setup()
        m2 = AccumModel(); m2.create_agents(4); m2.set_seed(42); m2.setup()

        m1.simulate(ticks=10, sync_workers_every_n_ticks=1)
        for _ in range(10):
            m2.simulate(ticks=1, sync_workers_every_n_ticks=1)

        np.testing.assert_allclose(
            m1.get_breed_data("AccumBreed", "value"),
            m2.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="value differs: simulate(10) vs simulate(1)x10")
        np.testing.assert_allclose(
            m1.get_breed_data("AccumBreed", "accum"),
            m2.get_breed_data("AccumBreed", "accum"),
            atol=1e-6, err_msg="accum differs: simulate(10) vs simulate(1)x10")

    def test_single_vs_multi_batch_large(self):
        """simulate(50) once vs simulate(10) x 5, 1000 agents."""
        m1 = AccumModel(); m1.create_agents(1000); m1.set_seed(42); m1.setup()
        m2 = AccumModel(); m2.create_agents(1000); m2.set_seed(42); m2.setup()

        m1.simulate(ticks=50, sync_workers_every_n_ticks=1)
        for _ in range(5):
            m2.simulate(ticks=10, sync_workers_every_n_ticks=1)

        np.testing.assert_allclose(
            m1.get_breed_data("AccumBreed", "value"),
            m2.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="value differs: simulate(50) vs simulate(10)x5")
        np.testing.assert_allclose(
            m1.get_breed_data("AccumBreed", "accum"),
            m2.get_breed_data("AccumBreed", "accum"),
            atol=1e-6, err_msg="accum differs: simulate(50) vs simulate(10)x5")

    def test_various_batch_sizes(self):
        """simulate(60) vs simulate(20)x3 vs simulate(12)x5, 500 agents."""
        m1 = AccumModel(); m1.create_agents(500); m1.set_seed(42); m1.setup()
        m2 = AccumModel(); m2.create_agents(500); m2.set_seed(42); m2.setup()
        m3 = AccumModel(); m3.create_agents(500); m3.set_seed(42); m3.setup()

        m1.simulate(ticks=60, sync_workers_every_n_ticks=1)
        for _ in range(3):
            m2.simulate(ticks=20, sync_workers_every_n_ticks=1)
        for _ in range(5):
            m3.simulate(ticks=12, sync_workers_every_n_ticks=1)

        v1 = m1.get_breed_data("AccumBreed", "value")
        np.testing.assert_allclose(v1,
            m2.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="60 vs 20x3")
        np.testing.assert_allclose(v1,
            m3.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="60 vs 12x5")


class TestDisconnectedClusters(unittest.TestCase):
    """Two disconnected clusters in one model must produce the same
    results as running each cluster in its own separate model."""

    SEED = 42
    TICKS = 20

    def _build_model(self, agent_counts, logical_id_offsets=None):
        """Build a model with one or more disconnected clusters (ring topology).

        agent_counts: number of agents per cluster
        logical_id_offsets: logical ID start per cluster (default: packed)
        """
        if logical_id_offsets is None:
            logical_id_offsets = []
            offset = 0
            for c in agent_counts:
                logical_id_offsets.append(offset)
                offset += c

        m = AccumModel()
        space = m.get_space()
        for cluster_idx, n in enumerate(agent_counts):
            ids = []
            for _ in range(n):
                ids.append(m.create_agent_of_breed(m._breed, value=0.0, accum=0.0))
            # Ring within cluster
            if n > 1:
                for i in range(n):
                    space.connect_agents(ids[i], ids[(i + 1) % n])
            # Assign logical IDs so RNG matches across models
            for i, aid in enumerate(ids):
                m.set_agent_logical_id(aid, logical_id_offsets[cluster_idx] + i)

        m.set_seed(self.SEED)
        m.setup()
        return m

    def test_two_clusters_small(self):
        """8+12 agents combined vs two solo models."""
        n_a, n_b = 8, 12

        combined = self._build_model([n_a, n_b])
        combined.simulate(ticks=self.TICKS, sync_workers_every_n_ticks=1)

        solo_a = self._build_model([n_a], logical_id_offsets=[0])
        solo_a.simulate(ticks=self.TICKS, sync_workers_every_n_ticks=1)

        solo_b = self._build_model([n_b], logical_id_offsets=[n_a])
        solo_b.simulate(ticks=self.TICKS, sync_workers_every_n_ticks=1)

        cv = combined.get_breed_data("AccumBreed", "value")
        sv_a = solo_a.get_breed_data("AccumBreed", "value")
        sv_b = solo_b.get_breed_data("AccumBreed", "value")
        np.testing.assert_allclose(cv[:n_a], sv_a, atol=1e-6,
            err_msg="cluster A value differs")
        np.testing.assert_allclose(cv[n_a:], sv_b, atol=1e-6,
            err_msg="cluster B value differs")

        ca = combined.get_breed_data("AccumBreed", "accum")
        sa_a = solo_a.get_breed_data("AccumBreed", "accum")
        sa_b = solo_b.get_breed_data("AccumBreed", "accum")
        np.testing.assert_allclose(ca[:n_a], sa_a, atol=1e-6,
            err_msg="cluster A accum differs")
        np.testing.assert_allclose(ca[n_a:], sa_b, atol=1e-6,
            err_msg="cluster B accum differs")

    def test_two_clusters_large(self):
        """500+500 agents combined vs two solo models, 50 ticks."""
        n_a, n_b = 500, 500

        combined = self._build_model([n_a, n_b])
        combined.simulate(ticks=50, sync_workers_every_n_ticks=1)

        solo_a = self._build_model([n_a], logical_id_offsets=[0])
        solo_a.simulate(ticks=50, sync_workers_every_n_ticks=1)

        solo_b = self._build_model([n_b], logical_id_offsets=[n_a])
        solo_b.simulate(ticks=50, sync_workers_every_n_ticks=1)

        cv = combined.get_breed_data("AccumBreed", "value")
        np.testing.assert_allclose(cv[:n_a],
            solo_a.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="cluster A value differs (large)")
        np.testing.assert_allclose(cv[n_a:],
            solo_b.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="cluster B value differs (large)")

        ca = combined.get_breed_data("AccumBreed", "accum")
        np.testing.assert_allclose(ca[:n_a],
            solo_a.get_breed_data("AccumBreed", "accum"),
            atol=1e-6, err_msg="cluster A accum differs (large)")
        np.testing.assert_allclose(ca[n_a:],
            solo_b.get_breed_data("AccumBreed", "accum"),
            atol=1e-6, err_msg="cluster B accum differs (large)")

    def test_asymmetric_clusters(self):
        """10+200 agents — asymmetric sizes."""
        n_a, n_b = 10, 200

        combined = self._build_model([n_a, n_b])
        combined.simulate(ticks=30, sync_workers_every_n_ticks=1)

        solo_a = self._build_model([n_a], logical_id_offsets=[0])
        solo_a.simulate(ticks=30, sync_workers_every_n_ticks=1)

        solo_b = self._build_model([n_b], logical_id_offsets=[n_a])
        solo_b.simulate(ticks=30, sync_workers_every_n_ticks=1)

        cv = combined.get_breed_data("AccumBreed", "value")
        np.testing.assert_allclose(cv[:n_a],
            solo_a.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="cluster A value differs (asymmetric)")
        np.testing.assert_allclose(cv[n_a:],
            solo_b.get_breed_data("AccumBreed", "value"),
            atol=1e-6, err_msg="cluster B value differs (asymmetric)")

        ca = combined.get_breed_data("AccumBreed", "accum")
        np.testing.assert_allclose(ca[:n_a],
            solo_a.get_breed_data("AccumBreed", "accum"),
            atol=1e-6, err_msg="cluster A accum differs (asymmetric)")
        np.testing.assert_allclose(ca[n_a:],
            solo_b.get_breed_data("AccumBreed", "accum"),
            atol=1e-6, err_msg="cluster B accum differs (asymmetric)")


class TestBLAResume(unittest.TestCase):
    """Double-buffered breed-local array must produce same results
    whether ticks run in one kernel launch or multiple."""

    def _build_bla_model(self, n_agents):
        m = BLAModel()
        m.create_agents(n_agents)
        m.register_bla()
        m.set_seed(42)
        m.setup()
        return m

    def test_bla_single_vs_multi(self):
        """simulate(10) once vs simulate(1) x 10 with double-buffered BLA."""
        m1 = self._build_bla_model(20)
        m2 = self._build_bla_model(20)

        m1.simulate(ticks=10, sync_workers_every_n_ticks=1)
        for _ in range(10):
            m2.simulate(ticks=1, sync_workers_every_n_ticks=1)

        np.testing.assert_allclose(
            m1.get_breed_data("BLABreed", "counter"),
            m2.get_breed_data("BLABreed", "counter"),
            atol=1e-6, err_msg="counter differs: simulate(10) vs simulate(1)x10")
        np.testing.assert_allclose(
            m1.get_breed_local_array("bla_sum"),
            m2.get_breed_local_array("bla_sum"),
            atol=1e-6, err_msg="bla_sum differs: simulate(10) vs simulate(1)x10")

    def test_bla_correctness(self):
        """After N ticks, bla_sum[0] should be N (incremented each tick)."""
        m = self._build_bla_model(4)
        m.simulate(ticks=20, sync_workers_every_n_ticks=1)

        bla = m.get_breed_local_array("bla_sum")
        # Each tick adds 1.0 to bla_sum[idx][0], so after 20 ticks it should be 20.0
        np.testing.assert_allclose(bla[:, 0], 20.0, atol=1e-6,
            err_msg="bla_sum[0] should be 20.0 after 20 ticks")
        # counter stores prev value from last tick = 19.0
        np.testing.assert_allclose(
            m.get_breed_data("BLABreed", "counter"), 19.0, atol=1e-6,
            err_msg="counter should be 19.0 (prev from last tick)")

    def test_bla_batch_sizes(self):
        """simulate(30) vs simulate(10)x3 vs simulate(5)x6 with BLA."""
        m1 = self._build_bla_model(100)
        m2 = self._build_bla_model(100)
        m3 = self._build_bla_model(100)

        m1.simulate(ticks=30, sync_workers_every_n_ticks=1)
        for _ in range(3):
            m2.simulate(ticks=10, sync_workers_every_n_ticks=1)
        for _ in range(6):
            m3.simulate(ticks=5, sync_workers_every_n_ticks=1)

        bla1 = m1.get_breed_local_array("bla_sum")
        bla2 = m2.get_breed_local_array("bla_sum")
        bla3 = m3.get_breed_local_array("bla_sum")
        np.testing.assert_allclose(bla1, bla2, atol=1e-6,
            err_msg="bla_sum: 30 vs 10x3")
        np.testing.assert_allclose(bla1, bla3, atol=1e-6,
            err_msg="bla_sum: 30 vs 5x6")


# --- Step function that reads from neighbors (like GGap P7) ---
@jit.rawkernel(device="cuda")
def neighbor_read_step(
    tick,
    agent_index,
    agent_ids,
    breeds,
    locations,
    value,
    accum,
):
    """Each tick: sum neighbor values, add RNG, write to own value and accum."""
    neighbor_sum = 0.0
    neighbor_indices = locations[agent_index]
    i = 0
    while i < len(neighbor_indices) and neighbor_indices[i] != -1:
        neighbor_idx = int(neighbor_indices[i])
        neighbor_sum = neighbor_sum + value[neighbor_idx]
        i = i + 1

    r = rand_uniform_philox(tick, agent_index, 1)
    value[agent_index] = value[agent_index] + r + neighbor_sum * 0.01
    accum[agent_index] = accum[agent_index] + value[agent_index]


class NeighborBreed(Breed):
    def __init__(self):
        super().__init__("NeighborBreed")
        self.register_property("value", 0.0)
        self.register_property("accum", 0.0)
        self.register_step_func(
            neighbor_read_step,
            Path(__file__).resolve(),
            priority=0,
            no_double_buffer=["value", "accum"],
        )


class NeighborModel(Model):
    def __init__(self):
        space = NetworkSpace()
        super().__init__(space, step_function_file_path="step_func_code_neighbor.py")
        self._breed = NeighborBreed()
        self.register_breed(breed=self._breed)

    def create_cluster(self, n, logical_id_offset=0):
        """Create n agents in a ring, return agent IDs."""
        ids = []
        for _ in range(n):
            ids.append(self.create_agent_of_breed(self._breed, value=0.0, accum=0.0))
        for i in range(n):
            self.get_space().connect_agents(ids[i], ids[(i + 1) % n])
        for i, aid in enumerate(ids):
            self.set_agent_logical_id(aid, logical_id_offset + i)
        return ids


class TestDisconnectedClustersBitExact(unittest.TestCase):
    """Bit-exact: disconnected clusters must produce identical results
    whether run alone or together. Uses neighbor reads to match GGap pattern."""

    SEED = 42
    TICKS = 50

    def _run_alone(self, n, logical_offset=0):
        m = NeighborModel()
        m.create_cluster(n, logical_id_offset=logical_offset)
        m.set_seed(self.SEED)
        m.setup()
        for _ in range(self.TICKS):
            m.simulate(ticks=1, sync_workers_every_n_ticks=1)
        return m

    def _run_together(self, n_a, n_b):
        m = NeighborModel()
        m.create_cluster(n_a, logical_id_offset=0)
        m.create_cluster(n_b, logical_id_offset=n_a)
        m.set_seed(self.SEED)
        m.setup()
        for _ in range(self.TICKS):
            m.simulate(ticks=1, sync_workers_every_n_ticks=1)
        return m

    def _get_breed_data_raw(self, model, prop, start, count):
        pidx = model._agent_factory._property_name_2_index[prop]
        return model._gpu_buffers.property_tensors[pidx][start:start+count].get()

    def test_small_clusters_bit_exact(self):
        """20+30 agents: cluster A alone must bit-match cluster A in combined."""
        n_a, n_b = 20, 30
        solo_a = self._run_alone(n_a, logical_offset=0)
        combined = self._run_together(n_a, n_b)

        for prop in ["value", "accum"]:
            s_a, c_a = solo_a._gpu_buffers.breed_ranges[solo_a._breed._breedidx]
            s_c, c_c = combined._gpu_buffers.breed_ranges[combined._breed._breedidx]
            data_solo = self._get_breed_data_raw(solo_a, prop, s_a, c_a)
            data_comb = self._get_breed_data_raw(combined, prop, s_c, n_a)
            np.testing.assert_array_equal(data_solo.view(np.uint32), data_comb.view(np.uint32),
                err_msg=f"{prop}: cluster A bit-differs alone vs combined (20+30)")

    def test_large_clusters_bit_exact(self):
        """500+500 agents: both clusters bit-match."""
        n_a, n_b = 500, 500
        solo_a = self._run_alone(n_a, logical_offset=0)
        solo_b = self._run_alone(n_b, logical_offset=n_a)
        combined = self._run_together(n_a, n_b)

        for prop in ["value", "accum"]:
            s_a, _ = solo_a._gpu_buffers.breed_ranges[solo_a._breed._breedidx]
            s_c, c_c = combined._gpu_buffers.breed_ranges[combined._breed._breedidx]

            data_solo_a = self._get_breed_data_raw(solo_a, prop, s_a, n_a)
            data_comb_a = self._get_breed_data_raw(combined, prop, s_c, n_a)
            np.testing.assert_array_equal(data_solo_a.view(np.uint32), data_comb_a.view(np.uint32),
                err_msg=f"{prop}: cluster A bit-differs (500+500)")

            s_b, _ = solo_b._gpu_buffers.breed_ranges[solo_b._breed._breedidx]
            data_solo_b = self._get_breed_data_raw(solo_b, prop, s_b, n_b)
            data_comb_b = self._get_breed_data_raw(combined, prop, s_c + n_a, n_b)
            np.testing.assert_array_equal(data_solo_b.view(np.uint32), data_comb_b.view(np.uint32),
                err_msg=f"{prop}: cluster B bit-differs (500+500)")

    def test_asymmetric_clusters_bit_exact(self):
        """10+1000 agents: small cluster must bit-match."""
        n_a, n_b = 10, 1000
        solo_a = self._run_alone(n_a, logical_offset=0)
        combined = self._run_together(n_a, n_b)

        for prop in ["value", "accum"]:
            s_a, _ = solo_a._gpu_buffers.breed_ranges[solo_a._breed._breedidx]
            s_c, _ = combined._gpu_buffers.breed_ranges[combined._breed._breedidx]
            data_solo = self._get_breed_data_raw(solo_a, prop, s_a, n_a)
            data_comb = self._get_breed_data_raw(combined, prop, s_c, n_a)
            np.testing.assert_array_equal(data_solo.view(np.uint32), data_comb.view(np.uint32),
                err_msg=f"{prop}: cluster A bit-differs (10+1000)")


if __name__ == "__main__":
    unittest.main()
