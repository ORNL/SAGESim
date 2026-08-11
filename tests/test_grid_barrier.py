"""Tests for the software grid barrier used in SAGESim's fused kernel.

The barrier lets a single kernel launch run multiple dependent phases: every
block arrives at a counter, spins until all blocks have arrived, and two
threadfences bracket the atomic so writes from the previous phase are visible
device-wide afterwards. ``sagesim/model.py`` emits this pattern into generated
step functions; the kernels below are hand-written replicas of it.

The ``jit.threadfence`` builtin these kernels need is installed by the
session-scoped fixture in conftest.py.
"""

import time

import numpy as np
import cupy as cp
import pytest
from cupyx import jit


def test_threadfence_builtin_is_installed():
    """``jit.threadfence`` compiles and executes inside a rawkernel."""
    assert hasattr(jit, "threadfence"), "jit.threadfence not installed"

    @jit.rawkernel(device="cuda")
    def fence_test(out):
        tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        out[tid] = 1
        jit.threadfence()
        out[tid] = 2

    out = cp.zeros(32, dtype=cp.int32)
    fence_test[1, 32](out)
    cp.cuda.Stream.null.synchronize()

    assert (out == 2).all(), f"expected all 2s, got {out}"


def test_barrier_makes_writes_visible_across_blocks():
    """Two-phase kernel, one barrier.

    Phase 1: each block writes its own index. Barrier. Phase 2: each block reads
    a *different* block's slot. Without a working barrier the phase-2 read races
    the phase-1 write.
    """

    @jit.rawkernel(device="cuda")
    def barrier_2phase(data, barrier_counter, num_blocks_param):
        bid = jit.blockIdx.x
        tid = jit.threadIdx.x

        barrier_id = 0

        # Phase 1: write block index
        if tid == 0:
            data[bid] = bid + 1

        # --- barrier ---
        jit.syncthreads()
        if tid == 0:
            jit.threadfence()
            jit.atomic_add(barrier_counter, 0, 1)
            _barrier_target = (barrier_id + 1) * num_blocks_param
            while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                pass
            jit.threadfence()
        jit.syncthreads()
        barrier_id = barrier_id + 1

        # Phase 2: read from neighbor block (circular)
        # Use jit.gridDim.x (uint32) for ops with uint32 CUDA builtins
        if tid == 0:
            neighbor = (bid + 1) % jit.gridDim.x
            data[jit.gridDim.x + bid] = data[neighbor]

    num_blocks = 8
    data = cp.zeros(num_blocks * 2, dtype=cp.int32)
    counter = cp.zeros(1, dtype=cp.int32)

    barrier_2phase[num_blocks, 32](data, counter, cp.int32(num_blocks))
    cp.cuda.Stream.null.synchronize()

    result = data.get()
    # Phase 1 writes: data[0..7] = [1, 2, ..., 8]
    # Phase 2 reads: data[8+i] = data[(i+1) % 8] = (i+1) % 8 + 1
    for i in range(num_blocks):
        expected = (i + 1) % num_blocks + 1
        assert result[num_blocks + i] == expected, (
            f"block {i}: expected {expected}, got {result[num_blocks + i]}"
        )


def test_two_barriers_chain_three_phases():
    """Three phases separated by two barriers.

    Phase 1 writes V1, phase 2 reads a neighbor's V1 to build V2, phase 3 reads a
    neighbor's V2. Catches a barrier that works once but not repeatedly.
    """

    @jit.rawkernel(device="cuda")
    def barrier_3phase(v1, v2, result, barrier_counter, num_blocks_param):
        bid = jit.blockIdx.x
        tid = jit.threadIdx.x

        barrier_id = 0
        neighbor = (bid + 1) % jit.gridDim.x

        # Phase 1: write V1 (use float to avoid int32/uint32 mixing)
        if tid == 0:
            v1[bid] = float(bid) * 10.0

        # --- barrier 1 ---
        jit.syncthreads()
        if tid == 0:
            jit.threadfence()
            jit.atomic_add(barrier_counter, 0, 1)
            _barrier_target = (barrier_id + 1) * num_blocks_param
            while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                pass
            jit.threadfence()
        jit.syncthreads()
        barrier_id = barrier_id + 1

        # Phase 2: read neighbor's V1, write V2
        if tid == 0:
            v2[bid] = v1[neighbor] + float(bid)

        # --- barrier 2 ---
        jit.syncthreads()
        if tid == 0:
            jit.threadfence()
            jit.atomic_add(barrier_counter, 0, 1)
            _barrier_target = (barrier_id + 1) * num_blocks_param
            while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                pass
            jit.threadfence()
        jit.syncthreads()
        barrier_id = barrier_id + 1

        # Phase 3: read neighbor's V2, write result
        if tid == 0:
            result[bid] = v2[neighbor]

    num_blocks = 16
    v1 = cp.zeros(num_blocks, dtype=cp.float32)
    v2 = cp.zeros(num_blocks, dtype=cp.float32)
    result = cp.zeros(num_blocks, dtype=cp.float32)
    counter = cp.zeros(1, dtype=cp.int32)

    barrier_3phase[num_blocks, 32](v1, v2, result, counter, cp.int32(num_blocks))
    cp.cuda.Stream.null.synchronize()

    result_cpu = result.get()
    for i in range(num_blocks):
        neighbor = (i + 1) % num_blocks
        neighbor2 = (neighbor + 1) % num_blocks
        expected = float(neighbor2 * 10 + neighbor)
        assert result_cpu[i] == expected, (
            f"block {i}: expected {expected}, got {result_cpu[i]}"
        )


def test_fused_tick_loop_matches_sequential_reference():
    """50 ticks fused into one launch must match a sequential CPU reference.

    Each tick, per block ``i``:
      A. ``data[i] += 0.5 * (i + 1)``   — block-dependent, so blocks diverge
      B. ``out[i] = data[i]*0.9 + data[i+1]*0.1``  — reads a *neighbor's* slot
      C. ``data[i] = out[i]``

    with a barrier after each phase. The per-block increment in A matters: with a
    uniform increment every slot holds the same value, the blend in B collapses to
    ``data[i] * 1.0``, and the neighbor read is never actually exercised. B writes
    to a separate buffer so it has no intra-phase race of its own — the only thing
    under test is whether the barrier makes the previous phase's writes visible
    device-wide. float32 throughout to match SAGESim.
    """
    num_ticks = 50

    @jit.rawkernel(device="cuda")
    def fused_kernel(data, out, barrier_counter, num_blocks_param, num_ticks_param):
        bid = jit.blockIdx.x
        tid = jit.threadIdx.x

        barrier_id = 0
        neighbor = (bid + 1) % jit.gridDim.x

        for tick in range(num_ticks_param):
            # Phase A: block-dependent increment
            if tid == 0:
                data[bid] = data[bid] + 0.5 * (float(bid) + 1.0)

            # --- barrier ---
            jit.syncthreads()
            if tid == 0:
                jit.threadfence()
                jit.atomic_add(barrier_counter, 0, 1)
                _barrier_target = (barrier_id + 1) * num_blocks_param
                while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                    pass
                jit.threadfence()
            jit.syncthreads()
            barrier_id = barrier_id + 1

            # Phase B: blend with neighbor, into a separate buffer
            if tid == 0:
                out[bid] = data[bid] * 0.9 + data[neighbor] * 0.1

            # --- barrier ---
            jit.syncthreads()
            if tid == 0:
                jit.threadfence()
                jit.atomic_add(barrier_counter, 0, 1)
                _barrier_target = (barrier_id + 1) * num_blocks_param
                while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                    pass
                jit.threadfence()
            jit.syncthreads()
            barrier_id = barrier_id + 1

            # Phase C: copy back
            if tid == 0:
                data[bid] = out[bid]

            # --- barrier ---
            jit.syncthreads()
            if tid == 0:
                jit.threadfence()
                jit.atomic_add(barrier_counter, 0, 1)
                _barrier_target = (barrier_id + 1) * num_blocks_param
                while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                    pass
                jit.threadfence()
            jit.syncthreads()
            barrier_id = barrier_id + 1

    num_blocks = 8
    data_gpu = cp.zeros(num_blocks, dtype=cp.float32)
    out_gpu = cp.zeros(num_blocks, dtype=cp.float32)
    counter = cp.zeros(1, dtype=cp.int32)

    fused_kernel[num_blocks, 32](
        data_gpu, out_gpu, counter, cp.int32(num_blocks), num_ticks
    )
    cp.cuda.Stream.null.synchronize()
    result_gpu = data_gpu.get()

    # Sequential reference on CPU (float32 to match GPU precision)
    data_ref = np.zeros(num_blocks, dtype=np.float32)
    increment = np.float32(0.5) * (np.arange(num_blocks, dtype=np.float32) + 1)
    for _ in range(num_ticks):
        data_ref = data_ref + increment
        new_data = np.empty_like(data_ref)
        for i in range(num_blocks):
            new_data[i] = np.float32(
                data_ref[i] * 0.9 + data_ref[(i + 1) % num_blocks] * 0.1
            )
        data_ref = new_data

    max_diff = float(np.max(np.abs(result_gpu - data_ref)))
    assert np.allclose(result_gpu, data_ref, rtol=1e-5), (
        f"fused tick-loop diverged from sequential reference over {num_ticks} "
        f"ticks: max diff {max_diff:.2e}\n  gpu={result_gpu}\n  cpu={data_ref}"
    )


def test_persistent_threads_with_barriers():
    """Persistent-thread pattern: far fewer threads than work items.

    10k agents over 128 threads, striding by total_threads, 10 ticks with a
    barrier between the update and the write-back.
    """

    @jit.rawkernel(device="cuda")
    def persistent_kernel(
        data, read_buf, write_buf, barrier_counter, num_blocks_param, num_agents
    ):
        thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        total_threads = jit.gridDim.x * jit.blockDim.x
        barrier_id = 0

        for tick in range(10):
            # Phase 1: each agent increments its write buffer
            agent_index = thread_id
            while agent_index < num_agents:
                write_buf[agent_index] = read_buf[agent_index] + 1
                agent_index = agent_index + total_threads

            # --- barrier ---
            jit.syncthreads()
            if jit.threadIdx.x == 0:
                jit.threadfence()
                jit.atomic_add(barrier_counter, 0, 1)
                _barrier_target = (barrier_id + 1) * num_blocks_param
                while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                    pass
                jit.threadfence()
            jit.syncthreads()
            barrier_id = barrier_id + 1

            # Write-back: copy write → read
            agent_index = thread_id
            while agent_index < num_agents:
                read_buf[agent_index] = write_buf[agent_index]
                agent_index = agent_index + total_threads

            # --- barrier ---
            jit.syncthreads()
            if jit.threadIdx.x == 0:
                jit.threadfence()
                jit.atomic_add(barrier_counter, 0, 1)
                _barrier_target = (barrier_id + 1) * num_blocks_param
                while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                    pass
                jit.threadfence()
            jit.syncthreads()
            barrier_id = barrier_id + 1

    num_agents = 10000
    num_blocks = 4  # Far fewer threads than agents — tests persistent pattern
    threads_per_block = 32

    read_buf = cp.zeros(num_agents, dtype=cp.float32)
    write_buf = cp.zeros(num_agents, dtype=cp.float32)
    data = cp.zeros(num_agents, dtype=cp.float32)
    counter = cp.zeros(1, dtype=cp.int32)

    persistent_kernel[num_blocks, threads_per_block](
        data, read_buf, write_buf, counter,
        cp.int32(num_blocks), cp.float32(num_agents),
    )
    cp.cuda.Stream.null.synchronize()

    result = read_buf.get()
    expected = 10.0  # 10 ticks, each adds 1
    assert np.allclose(result, expected), (
        f"expected all {expected}, got min={result.min()} max={result.max()}"
    )


@pytest.mark.benchmark
def test_fused_launch_beats_separate_launches(capsys):
    """Timing: 1000 separate launches vs one fused launch with barriers.

    Deselected by default (``-m 'not benchmark'`` in pyproject.toml) because
    timings on a shared login node are noise. Run with ``pytest -m benchmark -s``.
    """

    @jit.rawkernel(device="cuda")
    def single_phase(data):
        tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        data[tid] = data[tid] + 1

    @jit.rawkernel(device="cuda")
    def fused_phases(data, barrier_counter, num_blocks_param, num_phases):
        tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        barrier_id = 0

        for phase in range(num_phases):
            data[tid] = data[tid] + 1

            jit.syncthreads()
            if jit.threadIdx.x == 0:
                jit.threadfence()
                jit.atomic_add(barrier_counter, 0, 1)
                _barrier_target = (barrier_id + 1) * num_blocks_param
                while jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:
                    pass
                jit.threadfence()
            jit.syncthreads()
            barrier_id = barrier_id + 1

    num_blocks = 32
    threads = 32
    num_phases = 1000
    data = cp.zeros(num_blocks * threads, dtype=cp.float32)

    # Warmup
    single_phase[num_blocks, threads](data)
    counter = cp.zeros(1, dtype=cp.int32)
    fused_phases[num_blocks, threads](data, counter, cp.int32(num_blocks), 10)
    cp.cuda.Stream.null.synchronize()

    # Benchmark: N separate launches
    data[:] = 0
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(num_phases):
        single_phase[num_blocks, threads](data)
        cp.cuda.Stream.null.synchronize()
    t_separate = time.perf_counter() - t0

    # Benchmark: 1 fused launch
    data[:] = 0
    counter[:] = 0
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    fused_phases[num_blocks, threads](data, counter, cp.int32(num_blocks), num_phases)
    cp.cuda.Stream.null.synchronize()
    t_fused = time.perf_counter() - t0

    speedup = t_separate / t_fused if t_fused > 0 else float("inf")
    with capsys.disabled():
        print(
            f"\n  {num_phases} phases:"
            f"\n    separate launches: {t_separate * 1000:.1f} ms "
            f"({t_separate / num_phases * 1e6:.1f} us/launch)"
            f"\n    fused (barriers):  {t_fused * 1000:.1f} ms "
            f"({t_fused / num_phases * 1e6:.1f} us/barrier)"
            f"\n    speedup: {speedup:.1f}x"
        )
