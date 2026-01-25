# SAGESim GPU-CPU Data Flow

This document explains how data flows between CPU and GPU in SAGESim's distributed simulation architecture.

## Overview

SAGESim runs agent-based simulations across multiple MPI workers, each with its own GPU. The simulation loop follows this pattern for each tick:

```
CPU: Prepare Data → GPU: Execute Kernel → CPU: Collect Results → MPI: Synchronize
```

The fundamental challenge is that **GPUs require rectangular arrays** for efficient parallel processing, but **agent-based simulations produce ragged/irregular data** (agents have variable numbers of neighbors).

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SIMULATION TICK                                    │
└─────────────────────────────────────────────────────────────────────────────┘

     Worker 0                                          Worker 1
┌─────────────────┐                              ┌─────────────────┐
│  Local Agents   │                              │  Local Agents   │
│  (Python lists) │                              │  (Python lists) │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐      MPI isend/irecv         ┌─────────────────┐
│ Contextualize   │◄────────────────────────────►│ Contextualize   │
│ (exchange       │                              │ (exchange       │
│  neighbor data) │                              │  neighbor data) │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│ Data Prep       │                              │ Data Prep       │
│ - ID → Index    │                              │ - ID → Index    │
│ - Pad arrays    │                              │ - Pad arrays    │
│ - Upload to GPU │                              │ - Upload to GPU │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│ GPU Kernel      │                              │ GPU Kernel      │
│ (step function) │                              │ (step function) │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐                              ┌─────────────────┐
│ Collect Results │                              │ Collect Results │
│ - Download      │                              │ - Download      │
│ - Index → ID    │                              │ - Index → ID    │
└────────┬────────┘                              └────────┬────────┘
         │                                                │
         ▼                                                ▼
┌─────────────────┐       MPI barrier            ┌─────────────────┐
│ Synchronize     │◄────────────────────────────►│ Synchronize     │
│ (global reduce) │                              │ (global reduce) │
└─────────────────┘                              └─────────────────┘
```

---

## Phase 1: MPI Contextualization (CPU ↔ CPU)

**Location:** `sagesim/agent.py:493-767`

### What Happens

Each worker exchanges neighbor agent data with other workers via MPI.

```python
# Simplified flow
for each local agent:
    for each neighbor of this agent:
        if neighbor is on another worker:
            send this agent's data to that worker

receive neighbor data from all other workers
```

### Why This Is Needed

Agents need to read their neighbors' properties during the step function. If a neighbor lives on another worker, that data must be transferred before GPU execution.

**Example:** Agent 5 on Worker 0 has neighbor Agent 100 on Worker 1. Worker 0 must receive Agent 100's properties before the GPU kernel can execute Agent 5's step function.

### Optimization: Selective Property Sync

Only properties marked `neighbor_visible=True` are sent via MPI:

```python
breed.register_property("internal_counter", default=0, neighbor_visible=False)  # Not sent
breed.register_property("health", default=100, neighbor_visible=True)           # Sent
```

This reduces MPI bandwidth when agents have properties that neighbors never read.

---

## Phase 2: Data Preparation (CPU → GPU)

**Location:** `sagesim/model.py:595-696`

This phase transforms Python data structures into GPU-compatible arrays.

### Step 2.1: Combine Local + Neighbor Data

```python
all_agent_ids = local_agent_ids + received_neighbor_ids
combined_data = local_data + received_neighbor_data
```

**Why:** The GPU needs all relevant data in contiguous arrays. Local agents are at the beginning (indices 0 to N-1), neighbors follow (indices N to M).

```
GPU Array Layout:
┌───────────────────────────────────────┐
│ Index 0-99:   Local agents            │ ← Owned by this worker, will be modified
│ Index 100-150: Neighbor agents        │ ← From other workers, read-only
└───────────────────────────────────────┘
```

### Step 2.2: Convert Agent IDs to Array Indices

**Location:** `sagesim/model.py:29-108`

```
Before: Agent 5's neighbors = [12, 87, 103]    # Global agent IDs
After:  Agent 5's neighbors = [3, 45, 112]     # Local array indices
```

**Why:** GPU kernels access data via array indexing:

```python
# GPU kernel code
neighbor_index = locations[my_index][i]
neighbor_health = health[neighbor_index]  # Direct array access - fast!
```

Using global agent IDs would require hash map lookups on the GPU, which is extremely slow. Array indices allow direct memory access with simple arithmetic: `address = base + index * element_size`.

### Step 2.3: Pad Ragged Arrays to Rectangular

**Location:** `sagesim/internal_utils.py:8-96`

```
Before (ragged - variable length rows):
Agent 0: [5, 12, 8]           # 3 neighbors
Agent 1: [2]                   # 1 neighbor
Agent 2: [1, 4, 7, 9, 11]     # 5 neighbors

After (rectangular - all rows same length):
Agent 0: [5, 12, 8, NaN, NaN]  # Padded to max length (5)
Agent 1: [2, NaN, NaN, NaN, NaN]
Agent 2: [1, 4, 7, 9, 11]
```

**Why:** GPUs require rectangular arrays because:

1. **Memory Layout:** GPU calculates addresses as `base + row * stride + col`. Variable row lengths make this impossible.

2. **Parallel Execution:** All GPU threads execute the same instructions. With rectangular arrays, all threads can use the same loop bounds and memory access patterns.

3. **Coalesced Access:** Adjacent threads reading adjacent memory locations is fast. Ragged arrays break this pattern.

The NaN values serve as "no neighbor here" markers. The GPU kernel checks for NaN/-1 and skips those positions.

### Step 2.4: Upload to GPU

```python
gpu_array = cp.array(padded_numpy_array)  # CPU → GPU transfer
```

**Why:** The data must physically reside in GPU memory (VRAM) for the GPU kernel to access it. This transfer happens over PCIe and has significant latency, which is why minimizing CPU↔GPU transfers is important.

### Step 2.5: Create Write Buffers (Double Buffering)

```python
write_buffers = []
for prop_idx in write_property_indices:
    write_buffer = cp.copy(read_tensor[prop_idx])
    write_buffers.append(write_buffer)
```

**Why:** Prevents race conditions during parallel execution.

**The Problem:** If Agent A reads Agent B's health while Agent B simultaneously writes to its own health, the result is undefined (race condition).

**The Solution:** Double buffering separates read and write:
- All threads **read** from `read_tensors` (original state)
- All threads **write** to `write_buffers` (new state)
- After all threads complete, copy `write_buffers` → `read_tensors`

This ensures all agents see a consistent snapshot of the simulation state.

---

## Phase 3: GPU Kernel Execution

**Location:** `sagesim/model.py:706-734`

### What Happens

The auto-generated CUDA kernel executes the user-defined step function for each agent in parallel.

```python
# Kernel launch
self._step_func[blocks_per_grid, threads_per_block](
    current_tick,
    global_data,
    *read_tensors,      # Read from these
    *write_buffers,     # Write to these
    num_local_agents,
    agent_ids,
    priority_idx,
)
```

### Thread Execution Model

```
GPU Grid:
┌─────────────────────────────────────────────────────┐
│ Thread 0 → Agent 0    Thread 1 → Agent 1    ...     │
│ Thread 100 → Agent 100 (neighbor, no step function) │
└─────────────────────────────────────────────────────┘
```

Each thread:
1. Checks if it's a local agent (index < num_local_agents)
2. Reads its breed to determine which step function to run
3. Executes the step function, reading from neighbors as needed
4. Writes results to write buffer

### Cross-Breed Synchronization

If multiple breeds exist with different priorities, they execute sequentially:

```python
for priority_idx in range(num_priority_groups):
    # Launch kernel for this priority group only
    kernel(priority_idx)

    # Wait for all threads to complete
    cp.cuda.Stream.null.synchronize()

# After all priorities complete, copy writes to reads
for prop_idx in write_indices:
    read_tensors[prop_idx] = write_buffers[prop_idx]
```

**Why:** Ensures deterministic execution order when breeds interact. Priority 0 agents complete before priority 1 agents begin.

---

## Phase 4: Data Collection (GPU → CPU)

**Location:** `sagesim/model.py:742-765`

### What Happens

Download modified agent data from GPU back to CPU.

```python
for i in range(num_properties):
    # Only get local agents (first N rows), not neighbors
    gpu_data = gpu_tensors[i][:num_local_agents]

    # GPU → CPU transfer
    cpu_array = gpu_data.get()  # CuPy → NumPy

    # Convert to Python list
    python_list = cpu_array.tolist()
```

### Convert Indices Back to Agent IDs

**Location:** `sagesim/model.py:111-217`

```
Before: Agent 5's neighbors = [3, 45, 112]     # Local array indices
After:  Agent 5's neighbors = [12, 87, 103]    # Global agent IDs
```

**Why:** The next tick's MPI exchange uses global agent IDs to route data to the correct workers. The index→ID conversion restores the global namespace.

### Why Only Local Agents?

Neighbor data (rows N to M) is read-only and came from other workers. Those workers have the authoritative copy and will send updated data next tick. Downloading neighbor rows would be wasted bandwidth.

---

## Phase 5: MPI Synchronization

**Location:** `sagesim/model.py:767-779`

### Barrier

```python
comm.barrier()  # All workers wait here
```

**Why:** Ensures all workers finished GPU execution before any worker starts the next tick. Without this, a fast worker might begin MPI send while a slow worker is still computing, leading to stale data.

### Global Data Reduction

```python
global_data = comm.allreduce(global_data, op=max)
```

**Why:** Synchronizes global simulation state (e.g., total infection count, simulation-wide flags). Each worker contributes its local view; `allreduce` combines them.

---

## Summary: Why Each Transformation Exists

| Transformation | Location | Why Needed |
|----------------|----------|------------|
| MPI exchange | agent.py | Agents need neighbor data that lives on other workers |
| Combine local + neighbor | model.py:622-653 | GPU needs all data in contiguous arrays |
| Agent ID → Index | model.py:29-108 | GPU uses array indexing, not hash lookups |
| Ragged → Rectangular padding | internal_utils.py:8-96 | GPU requires fixed-dimension arrays for parallel execution |
| CPU → GPU upload | model.py:662 | Data must be in GPU memory for kernel access |
| Double buffering | model.py:678-684 | Prevents race conditions during parallel writes |
| GPU → CPU download | model.py:742-765 | Results must return to CPU for MPI exchange |
| Index → Agent ID | model.py:111-217 | MPI uses global agent IDs for routing |
| MPI barrier | model.py:767 | Ensures consistent state before next tick |

---

## Performance Implications

The CPU↔GPU data transfer happens **every tick** (by default). This creates overhead because:

1. **PCIe Bandwidth:** CPU↔GPU transfers are limited by PCIe speed (~32 GB/s for PCIe 4.0 x16)
2. **Padding Overhead:** Ragged→rectangular conversion is CPU-bound Python code
3. **Synchronization:** MPI barriers and GPU synchronization add latency

### Mitigation Strategies

1. **`sync_workers_every_n_ticks`:** Run multiple ticks on GPU before synchronizing. Reduces MPI frequency but uses stale neighbor data.

2. **Selective Property Sync:** Mark internal properties as `neighbor_visible=False` to reduce MPI message size.

3. **Good Partitioning:** Use METIS or similar to minimize cross-worker neighbors, reducing MPI traffic.

4. **Batch Processing:** Larger agent counts amortize the fixed overhead of kernel launches and transfers.
