#!/usr/bin/env python3
"""
Standalone tests for the software grid barrier used in SAGESim's fused kernel.

Run: conda activate superneuroabm && python tests/test_grid_barrier.py

Tests:
1. GPU detection (NVIDIA vs AMD)
2. __threadfence() availability via CuPy JIT monkeypatch
3. Software grid barrier correctness (2-phase kernel)
4. Multi-barrier correctness (3 phases, 2 barriers)
5. Fused tick-loop (100 ticks with barriers)
6. Performance: N separate launches vs 1 fused launch
"""

import sys
import time
import numpy as np

# Ensure sagesim is importable
sys.path.insert(0, ".")

import cupy as cp
from cupyx import jit


def test_gpu_detection():
    """Test 1: Detect GPU type and print info."""
    print("=" * 60)
    print("Test 1: GPU Detection")
    print("=" * 60)

    dev = cp.cuda.Device()
    attrs = dev.attributes
    name = cp.cuda.runtime.getDeviceProperties(dev.id)['name']

    is_hip = hasattr(cp.cuda.runtime, 'is_hip') and cp.cuda.runtime.is_hip
    gpu_type = "AMD (HIP/ROCm)" if is_hip else "NVIDIA (CUDA)"

    print(f"  GPU: {name}")
    print(f"  Type: {gpu_type}")
    print(f"  SMs: {attrs['MultiProcessorCount']}")
    print(f"  Max threads/SM: {attrs.get('MaxThreadsPerMultiProcessor', 'N/A')}")
    print(f"  Max blocks/SM: {attrs.get('MaxBlocksPerMultiprocessor', 'N/A')}")
    print("  PASSED")
    return True


def test_threadfence():
    """Test 2: Verify __threadfence() monkeypatch works in CuPy JIT."""
    print("\n" + "=" * 60)
    print("Test 2: __threadfence() Monkeypatch")
    print("=" * 60)

    from sagesim.jit_extensions import install_jit_extensions
    install_jit_extensions()

    # Verify the attribute exists
    assert hasattr(jit, 'threadfence'), "jit.threadfence not installed"

    # Test it compiles in a kernel
    @jit.rawkernel(device='cuda')
    def fence_test(out):
        tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        out[tid] = 1
        jit.threadfence()
        out[tid] = 2

    out = cp.zeros(32, dtype=cp.int32)
    fence_test[1, 32](out)
    cp.cuda.Stream.null.synchronize()

    assert (out == 2).all(), f"Expected all 2s, got {out}"
    print("  threadfence() compiles and executes correctly")
    print("  PASSED")
    return True


def test_barrier_2phase():
    """Test 3: Two-phase kernel with one grid barrier.

    Phase 1: Each block writes its block index to its slot.
    Barrier.
    Phase 2: Each block reads from a different block's slot.
    Verifies cross-block visibility after barrier.
    """
    print("\n" + "=" * 60)
    print("Test 3: Software Grid Barrier (2-phase)")
    print("=" * 60)

    from sagesim.jit_extensions import install_jit_extensions
    install_jit_extensions()

    @jit.rawkernel(device='cuda')
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
        actual = result[num_blocks + i]
        assert actual == expected, \
            f"Block {i}: expected {expected}, got {actual}"

    print(f"  {num_blocks} blocks, cross-block reads all correct after barrier")
    print("  PASSED")
    return True


def test_barrier_3phase():
    """Test 4: Three-phase kernel with 2 barriers (mimics Soma→Synapse→STDP).

    Phase 1: Each block writes value V1 = bid * 10
    Barrier 1
    Phase 2: Each block reads neighbor's V1, writes V2 = V1_neighbor + bid
    Barrier 2
    Phase 3: Each block reads neighbor's V2, writes final result
    """
    print("\n" + "=" * 60)
    print("Test 4: Multi-Barrier (3 phases, 2 barriers)")
    print("=" * 60)

    from sagesim.jit_extensions import install_jit_extensions
    install_jit_extensions()

    @jit.rawkernel(device='cuda')
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
        expected_v2_neighbor = float(neighbor2 * 10 + neighbor)
        assert result_cpu[i] == expected_v2_neighbor, \
            f"Block {i}: expected {expected_v2_neighbor}, got {result_cpu[i]}"

    print(f"  {num_blocks} blocks, 3 phases with 2 barriers all correct")
    print("  PASSED")
    return True


def test_fused_tick_loop():
    """Test 5: Fused tick-loop with barriers, verify against sequential.

    Simulates 50 ticks where each tick:
      Phase A: data[i] += 0.5 (all blocks)
      Barrier
      Phase B: data[i] = data[i] * 0.9 + data[(i+1)%N] * 0.1 (neighbor blend)
      Barrier

    Uses float32 (matching SAGESim) and a stable operation (no overflow).
    """
    print("\n" + "=" * 60)
    print("Test 5: Fused Tick-Loop (50 ticks)")
    print("=" * 60)

    from sagesim.jit_extensions import install_jit_extensions
    install_jit_extensions()

    num_ticks = 50

    @jit.rawkernel(device='cuda')
    def fused_kernel(data, barrier_counter, num_blocks_param, num_ticks_param):
        bid = jit.blockIdx.x
        tid = jit.threadIdx.x

        barrier_id = 0
        neighbor = (bid + 1) % jit.gridDim.x

        for tick in range(num_ticks_param):
            # Phase A: increment own value
            if tid == 0:
                data[bid] = data[bid] + 0.5

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

            # Phase B: blend with neighbor
            if tid == 0:
                data[bid] = data[bid] * 0.9 + data[neighbor] * 0.1

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
    counter = cp.zeros(1, dtype=cp.int32)

    fused_kernel[num_blocks, 32](data_gpu, counter, cp.int32(num_blocks), num_ticks)
    cp.cuda.Stream.null.synchronize()
    result_gpu = data_gpu.get()

    # Sequential reference on CPU (float32 to match GPU precision)
    data_ref = np.zeros(num_blocks, dtype=np.float32)
    for tick in range(num_ticks):
        data_ref += np.float32(0.5)
        new_data = np.empty_like(data_ref)
        for i in range(num_blocks):
            new_data[i] = np.float32(data_ref[i] * 0.9 + data_ref[(i + 1) % num_blocks] * 0.1)
        data_ref = new_data

    match = np.allclose(result_gpu, data_ref, rtol=1e-5)
    print(f"  GPU result: {result_gpu[:4]}...")
    print(f"  CPU result: {data_ref[:4]}...")
    print(f"  Max diff: {np.max(np.abs(result_gpu - data_ref)):.2e}")
    print(f"  Match: {match}")
    if not match:
        print("  FAILED")
        return False
    print("  PASSED")
    return True


def test_persistent_threads():
    """Test 5b: Persistent thread pattern — fewer blocks than work items.

    Tests that persistent threads (while agent_index < N, stride by total_threads)
    produce correct results with barriers.
    """
    print("\n" + "=" * 60)
    print("Test 5b: Persistent Threads with Barriers")
    print("=" * 60)

    from sagesim.jit_extensions import install_jit_extensions
    install_jit_extensions()

    @jit.rawkernel(device='cuda')
    def persistent_kernel(data, read_buf, write_buf, barrier_counter,
                          num_blocks_param, num_agents):
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
    assert np.allclose(result, expected), \
        f"Expected all {expected}, got min={result.min()} max={result.max()}"

    print(f"  {num_agents} agents with {num_blocks * threads_per_block} threads")
    print(f"  All values = {result[0]} (expected {expected})")
    print("  PASSED")
    return True


def test_performance():
    """Test 6: Benchmark N separate launches vs 1 fused launch."""
    print("\n" + "=" * 60)
    print("Test 6: Performance Benchmark")
    print("=" * 60)

    from sagesim.jit_extensions import install_jit_extensions
    install_jit_extensions()

    @jit.rawkernel(device='cuda')
    def single_phase(data):
        tid = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        data[tid] = data[tid] + 1

    @jit.rawkernel(device='cuda')
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

    speedup = t_separate / t_fused if t_fused > 0 else float('inf')
    print(f"  {num_phases} phases:")
    print(f"    Separate launches: {t_separate*1000:.1f} ms ({t_separate/num_phases*1e6:.1f} us/launch)")
    print(f"    Fused (barriers):  {t_fused*1000:.1f} ms ({t_fused/num_phases*1e6:.1f} us/barrier)")
    print(f"    Speedup: {speedup:.1f}x")
    print("  PASSED")
    return True


def main():
    print("SAGESim Fused Kernel - Grid Barrier Validation Tests")
    print("=" * 60)

    tests = [
        ("GPU Detection", test_gpu_detection),
        ("ThreadFence Monkeypatch", test_threadfence),
        ("2-Phase Barrier", test_barrier_2phase),
        ("3-Phase Multi-Barrier", test_barrier_3phase),
        ("Fused Tick-Loop", test_fused_tick_loop),
        ("Persistent Threads", test_persistent_threads),
        ("Performance Benchmark", test_performance),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
