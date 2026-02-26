# Overhead Analysis: Neighbor Info Reading in SAGESim

## Context

SAGESim's per-tick communication pattern is **GPU → CPU → MPI → CPU → GPU**. For HPC multi-node/multi-GPU runs, the actual GPU kernel execution is only ~10% of tick time. The remaining ~90% is overhead from data preparation, transfers, and serialization — making the simulator CPU-bound despite running on GPUs.

**Target platform:** Frontier HPC (AMD MI250X GPUs, Cray Slingshot interconnect, ROCm). Development and testing on NVIDIA GPUs; CuPy-ROCm is experimental but works on Frontier.

---

## Overhead Breakdown (by severity)

### 1. Data Preparation — ~40% of tick time

**What happens:** Every tick, `worker_coroutine()` (model.py:626-724):
- Rebuilds combined local+neighbor lists from scratch (line 656-681)
- Builds a new `agent_id_to_index` hash map (line 631)
- Calls `convert_to_equal_side_tensor()` to pad ragged neighbor lists → rectangular arrays (line 690)
- Uploads entire arrays to GPU via `cp.array()` (line 690)
- Creates new write buffers on GPU (lines 708-712)
- Frees all GPU memory blocks afterward (line 766)

**Why it's slow:** All of this is pure Python on CPU, and it happens on **every single tick**. For large networks (100K+ agents with 1000+ neighbors each), padding ragged arrays and building hash maps dominate.

### 2. GPU↔CPU Transfers — ~30% of tick time

**What happens:** Every tick:
- **Upload** (model.py:690): Full property tensors transferred CPU → GPU via `cp.array()`
- **Download** (model.py:779): Results transferred GPU → CPU via `.get()`
- **Index↔ID conversion** (model.py:679, 784): Done on CPU before upload and after download

**Why it's slow:** PCIe bandwidth is finite (~32 GB/s). For 1M agents × 10 properties × 1000 neighbors, this is GBs of data transferred per tick.

### 3. MPI Contextualization — ~20% of tick time

**What happens:** `contextualize_agent_data_tensors()` (agent.py:479-753):

| Sub-step | Location | Issue |
|----------|----------|-------|
| Python loop over all agents × neighbors | Lines 518-606 | O(N × avg_neighbors) in pure Python |
| Change detection via `np.array_equal` | Lines 550-575 | Expensive per-agent comparison every tick |
| Pickle-based serialization (`comm.isend`) | Lines 640-656, 676 | **mpi4py lowercase API uses pickle** — orders of magnitude slower than buffer protocol |
| 128-byte chunking with individual MPI messages | Lines 617-708 | Each MPI message has fixed latency (~1-10μs); thousands of tiny messages = huge overhead |
| Reconstruction from visible subset | Lines 710-735 | Python loop rebuilding full property lists |

**The pickle serialization is the single biggest MPI bottleneck.** `comm.isend()` (lowercase) pickles Python objects. For HPC, you need `comm.Isend()` (uppercase) with pre-allocated numpy/cupy buffers.

### 4. Synchronization — variable

- `comm.barrier()` (model.py:796): Slowest worker determines tick time
- `comm.allreduce()` with `.tolist()` (model.py:800-802): Python-level reduction, not GPU-native

---

## Root Causes Summary

| Root Cause | Impact | Where |
|------------|--------|-------|
| **Data not GPU-resident between ticks** | Causes full rebuild + transfer every tick | model.py:626-724, 769-787 |
| **Pickle-based MPI (lowercase API)** | ~100x slower than buffer-protocol MPI | agent.py:640-708 |
| **Pure Python iteration over agents/neighbors** | Cannot scale to 100K+ agents | agent.py:518-606 |
| **Per-tick memory allocation/deallocation** | GPU memory fragmentation, allocation overhead | model.py:690, 711, 766 |
| **Rectangular padding for ragged neighbor lists** | Wastes memory and bandwidth, must re-pad when max changes | model.py:690 |
| **No overlap of computation and communication** | GPU sits idle during MPI | Sequential in worker_coroutine |

---

## Key Design Constraints for Large-Scale HPC

Before discussing solutions, these constraints must be understood:

1. **No global-sized buffers.** At 100B agents, no GPU can hold a buffer indexed by agent_id. Each GPU only holds its local agents + immediate ghost neighbors.

2. **Dynamic agent populations.** Agents can be born or die during simulation, so local agent counts change between ticks. The local portion of the array is not stable.

3. **Dynamic topology.** Agents can connect/disconnect during simulation, so neighbor lists change in both content and length.

4. **Ragged neighbor counts.** Agent A may have 3 neighbors, Agent B may have 50,000. Padding to max wastes enormous memory and bandwidth, and the max can change every tick.

These constraints mean:
- ID↔index conversion **cannot be eliminated** — it's needed because each GPU works with a small local subset of a huge global agent pool
- Rectangular padding **should be eliminated** — it's the wrong data structure for variable-length neighbor lists
- The goal is to make these operations **GPU-parallel and incremental** rather than CPU-sequential and full-rebuild

---

## Suggested Optimization Strategy

### Phase 0: CSR Format for Neighbor Lists (Foundational — eliminates padding entirely)

**The Problem:**
Current rectangular format pads all neighbor lists to the same max length:
```
Rectangular (current):                      Memory for 3 agents, max_neighbors=5000:
agent 0: [5, 2, NaN, NaN, ..., NaN]        3 × 5000 × 4 bytes = 60KB
agent 1: [8, 3, 1, NaN, ..., NaN]          (but only 9 actual neighbor values)
agent 2: [7, 4, 9, 6, NaN, ..., NaN]       99.94% wasted on NaN padding
```

If any agent gains one neighbor beyond the current max, ALL agents must be re-padded.

**The Solution: CSR (Compressed Sparse Row)**

This is the standard GPU data structure for variable-length lists — used by cuGraph, PyTorch Geometric, DGL, and all serious graph GPU frameworks:

```
CSR format:
values:  [5, 2, 8, 3, 1, 7, 4, 9, 6]      ← flat array, all neighbors concatenated
offsets: [0, 2, 5, 9]                       ← where each agent's neighbors start

Agent 0's neighbors = values[offsets[0] : offsets[1]] = values[0:2] = [5, 2]
Agent 1's neighbors = values[offsets[1] : offsets[2]] = values[2:5] = [8, 3, 1]
Agent 2's neighbors = values[offsets[2] : offsets[3]] = values[5:9] = [7, 4, 9, 6]

Memory: 9 × 4 + 4 × 4 = 52 bytes (vs 60KB rectangular)
```

**Properties:**
- **No padding, no wasted memory** — memory is proportional to actual edges, not max_neighbors × N
- **No max-neighbor dependency** — adding a neighbor to one agent doesn't affect any other agent
- **GPU-friendly** — each thread reads `start = offsets[my_idx]`, `end = offsets[my_idx + 1]`, loops `values[start:end]`
- **Incremental update** — append new neighbors to `values[]`, update affected entries in `offsets[]`

**How the GPU kernel changes:**

Current kernel reads neighbors like:
```cuda
// Current: rectangular, NaN-terminated
for (int n = 0; n < max_neighbors; n++) {
    float neighbor_idx = locations[agent_idx * max_neighbors + n];
    if (isnan(neighbor_idx) || neighbor_idx < 0) break;
    // read neighbor data...
}
```

CSR kernel reads neighbors like:
```cuda
// CSR: offset-based, exact bounds
int start = offsets[agent_idx];
int end = offsets[agent_idx + 1];
for (int n = start; n < end; n++) {
    int neighbor_idx = values[n];
    // read neighbor data...
}
```

The CSR version is actually **faster** — no NaN checks, no wasted iterations over padding, better memory access patterns (sequential reads of `values[]`).

**Handling dynamic topology:**

When neighbors change between ticks:
- **Append-only strategy:** New edges are appended to `values[]`. Keep a pre-allocated `values[]` buffer with slack space (e.g., 2x current edges). Rebuild `offsets[]` only when the buffer fills up.
- **Per-tick delta:** Only send changed edges via MPI, apply them to the CSR structure on GPU.
- **Worst case (full rebuild):** Even rebuilding CSR from scratch on GPU is fast — it's just a parallel prefix sum on neighbor counts to compute `offsets[]`, then a scatter to fill `values[]`. Both are O(N) parallel ops.

**Handling agent birth/death:**

- **Death:** Mark agent's offset range as invalid (set `offsets[dead_agent] = offsets[dead_agent + 1]` — zero-length range). Optionally compact periodically.
- **Birth:** Append new agent to the end. `offsets[]` grows by one entry, `values[]` grows by the new agent's neighbor count.
- **Compaction:** Periodically rebuild to reclaim dead agent slots. This is a parallel stream compaction — well-studied GPU primitive.

**Files to modify:**
- `sagesim/model.py` — Replace rectangular tensor handling with CSR arrays
- `sagesim/internal_utils.py` — Replace `convert_to_equal_side_tensor()` with CSR construction
- `sagesim/model.py:825-1135` — Update GPU kernel generation to emit CSR-style neighbor loops
- Step function API — Users write `for n in range(start, end)` instead of `for n in range(max_neighbors)` with NaN checks

**Estimated improvement:** Eliminates padding overhead entirely. For networks where max_neighbors >> avg_neighbors, memory savings can be 10-1000x.

---

### Phase 1: GPU-Resident Data with Incremental Updates

Keep agent data on GPU between ticks. Instead of rebuilding all arrays from scratch, update only what changed.

**Key changes:**
- Pre-allocate GPU buffers with slack space at `setup()` time (e.g., `max_local_agents * 1.5`)
- Maintain a **GPU-side hash map** (open-addressing) for `agent_id → local_buffer_index`
- When agents are born/die/migrate: update the hash map and the affected buffer slots — not the entire array
- When neighbor data arrives via MPI: write directly into the ghost zone of the GPU buffer
- Eliminate per-tick `cp.array()`, `.get()`, `free_all_blocks()`

**GPU-side hash map for ID↔Index:**

Since we can't use agent_id as array index (100B agents won't fit), and the local agent set changes (birth/death), we need a dynamic mapping. A GPU hash map handles this:

```python
# Open-addressing hash map on GPU (CuPy RawKernel)
# Pre-allocate to 2x expected max local+ghost agents
hash_table_size = next_power_of_2(max_agents * 2)
hash_keys = cp.full(hash_table_size, -1, dtype=cp.int64)    # agent_id or -1 (empty)
hash_values = cp.full(hash_table_size, -1, dtype=cp.int32)   # local buffer index

# Insert kernel: each thread inserts one (agent_id, buffer_idx) pair
# Lookup kernel: each thread looks up one agent_id → buffer_idx
# Both are O(1) average on GPU with thousands of threads in parallel
```

When the agent set changes:
1. Remove dead agents from hash map (mark as deleted)
2. Insert new agents into hash map
3. Run ID→index conversion kernel on the CSR `values[]` array to convert neighbor IDs to local indices

All three steps are GPU-parallel kernels — no CPU involvement.

**Handling growth beyond pre-allocated size:**
- When buffer fills past a threshold (e.g., 80%), reallocate at 2x size (standard amortized growth)
- This is rare — only happens when agent population grows significantly
- The cost of one reallocation is amortized over many ticks

**Estimated improvement:** 3-5x tick time reduction

---

### Phase 2: Buffer-Protocol MPI (High Impact — eliminates pickle overhead)

Replace `comm.isend()`/`comm.irecv()` (pickle) with `comm.Isend()`/`comm.Irecv()` (buffer protocol) using pre-allocated contiguous numpy or cupy arrays.

**Key changes:**
- `agent.py`: Pre-allocate send/recv buffers as contiguous numpy arrays
- Pack agent data into flat buffers with a GPU kernel or vectorized numpy ops
- Use `comm.Isend(buf, dest=rank)` instead of `comm.isend(python_obj, dest=rank)`
- Eliminate the 128-byte chunking scheme entirely — send one contiguous buffer per rank

**Send buffer layout (flat, contiguous):**
```
Per-rank send buffer:
[num_agents | agent_0_id, prop_0_val, prop_1_val, ... | agent_1_id, prop_0_val, ... | ...]
```

**Estimated improvement:** 5-20x for MPI communication specifically

---

### Phase 3: GPU-Aware MPI / GPU-Direct RDMA (For HPC at scale)

On Frontier, Cray MPICH supports ROCm-aware MPI with `MPICH_GPU_SUPPORT_ENABLED=1`, allowing MPI to read/write GPU buffers directly via GPU-Direct RDMA over Slingshot — no CPU staging needed. This is the primary target for this phase.

**Key changes:**
- `gpu_kernels.py` (`CommunicationManager`): Send CuPy arrays directly via `comm.Isend(cupy_buf, dest=rank)`
- Pack/unpack using GPU kernels instead of CPU
- Completely eliminates GPU↔CPU transfer for communication

With CSR + GPU-resident data + GPU-aware MPI, the per-tick flow becomes:
```
Before Phase 2: GPU → CPU → pickle → MPI → unpickle → CPU → pad → GPU
After Phase 2:  GPU → CPU (.get()) → MPI Isend/Irecv (buffer protocol) → CPU → GPU (cp.array())
After Phase 3:  GPU → (GPU pack) → MPI RDMA → (GPU unpack) → GPU
```

**Estimated improvement:** 2-10x for communication on GPU-aware MPI systems

---

### Phase 4: RCCL for Collectives

Replace `comm.allreduce()` with GPU-native collective operations via RCCL (AMD's equivalent of NCCL).

**Estimated improvement:** 1.2-2x for collective operations

---

### Phase 5: Computation-Communication Overlap (HIP/CUDA Streams)

Use HIP/CUDA streams to overlap GPU kernel execution with MPI communication for the next tick's neighbor data.

**What are GPU streams?**

A GPU stream (HIP stream on AMD, CUDA stream on NVIDIA) is a sequence of GPU operations that execute in order with respect to each other, but can execute **concurrently** with operations in other streams. By default, all GPU operations go into the "default stream" (stream 0) and execute sequentially. By creating multiple streams, you can overlap independent operations:

```
Default (current SAGESim) — everything sequential:
|-- GPU kernel --|-- GPU→CPU --|-- MPI send/recv --|-- CPU→GPU --|-- GPU kernel --|

With HIP/CUDA streams — overlap communication with computation:
Stream 1: |-- GPU kernel (tick N) --|                          |-- GPU kernel (tick N+1) --|
Stream 2:                           |-- pack --|-- MPI --|-- unpack --|
                                    ← these overlap with kernel on stream 1 →
```

In CuPy, streams are created with `cp.cuda.Stream()` and used as context managers (CuPy-ROCm uses the same `cp.cuda.Stream` API despite targeting HIP):

```python
compute_stream = cp.cuda.Stream()
comm_stream = cp.cuda.Stream()

# Launch kernel on compute stream
with compute_stream:
    kernel[blocks, threads](...)

# Concurrently, pack and send neighbor data on comm stream
with comm_stream:
    pack_kernel[blocks, threads](data, send_buf)
    # comm_stream.synchronize()  # Wait for pack to finish
    # MPI.Isend(send_buf, ...)   # Send while kernel still runs on compute_stream

# Synchronize both before next tick
compute_stream.synchronize()
comm_stream.synchronize()
```

**Key requirement:** The operations on different streams must be **independent** — they cannot read/write the same memory. This means:
- The GPU kernel reads/writes local agent data
- The communication stream reads ghost cell data from the *previous* tick and packs it for sending
- These don't conflict because they touch different memory regions

**Estimated improvement:** Up to 2x for the overlapped portion

---

## Recommended Implementation Order

```
Phase 0 (CSR Format)           ✓ DONE
    ↓
Phase 1 (GPU-Resident)         ✓ DONE
    ↓
Phase 2 (Buffer-Protocol MPI)  ✓ DONE
    ↓
Lazy CPU Download              ✓ DONE
    ↓
Phase 3 (GPU-Aware MPI)        ← Next: ROCm-aware cray-mpich on Frontier
    ↓
Phase 4 (RCCL Collectives)    ← GPU-native collectives
    ↓
Phase 5 (Stream Overlap)      ← Advanced, requires careful design
```

**Files modified (Phases 0-2):**
- `sagesim/model.py` — CSR data structures, GPU-resident buffers (`GPUBufferManager`), kernel generation (`_CSRBodyTransformer`), lazy CPU download
- `sagesim/gpu_kernels.py` — `GPUHashMap` (agent_id→buffer_index), `GPUBufferManager` (persistent GPU buffers), `CommunicationManager` (buffer-protocol MPI with pre-computed topology)
- `sagesim/internal_utils.py` — CSR construction replacing `convert_to_equal_side_tensor()`

**Files to modify (remaining phases):**
- `sagesim/gpu_kernels.py` (`CommunicationManager`) — GPU-aware MPI / GPU-Direct RDMA (Phase 3)
- `sagesim/model.py` — RCCL collectives (Phase 4), HIP/CUDA stream overlap (Phase 5)

---

## Verification

- Run with `verbose_timing=True` before/after each phase to measure improvement
- Profile with `nsys` (NVIDIA) or `rocprof` (AMD) to verify GPU utilization increases
- Test weak scaling: fixed agents-per-GPU, increase GPU count — should show near-linear scaling after Phase 2+3
- Memory usage: verify CSR uses proportional-to-edges memory (not max_neighbors × N)
