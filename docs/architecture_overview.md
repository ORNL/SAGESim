# SAGESim Architecture Overview

## Introduction

SAGESim (Scalable Agent-based GPU-accelerated Simulation) is a distributed GPU-accelerated agent-based simulation framework designed for high-performance computing environments. It enables simulations with millions of agents by combining two levels of parallelism:

1. **MPI-level parallelism**: Multiple worker processes (MPI ranks), each with its own GPU
2. **GPU-level parallelism**: Thousands of threads per GPU, with each thread processing one agent

> **Recommendation: One Worker = One GPU**
>
> For best performance, use one MPI worker per physical GPU. While multiple workers can share a single GPU, this adds MPI overhead without performance benefit. If your simulation fits in one GPU, use a single worker.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SAGESim Architecture                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐       │
│    │   MPI Worker 0   │    │   MPI Worker 1   │    │   MPI Worker N   │       │
│    │   (Rank 0)       │    │   (Rank 1)       │    │   (Rank N)       │       │
│    │                 │    │                 │    │                 │       │
│    │  ┌───────────┐  │    │  ┌───────────┐  │    │  ┌───────────┐  │       │
│    │  │   GPU 0   │  │    │  │   GPU 1   │  │    │  │   GPU N   │  │       │
│    │  │           │  │    │  │           │  │    │  │           │  │       │
│    │  │ Thread 0  │  │    │  │ Thread 0  │  │    │  │ Thread 0  │  │       │
│    │  │ Thread 1  │  │    │  │ Thread 1  │  │    │  │ Thread 1  │  │       │
│    │  │ Thread 2  │  │    │  │ Thread 2  │  │    │  │ Thread 2  │  │       │
│    │  │   ...     │  │    │  │   ...     │  │    │  │   ...     │  │       │
│    │  │ Thread K  │  │    │  │ Thread K  │  │    │  │ Thread K  │  │       │
│    │  └───────────┘  │    │  └───────────┘  │    │  └───────────┘  │       │
│    │                 │    │                 │    │                 │       │
│    │  Agents:        │    │  Agents:        │    │  Agents:        │       │
│    │  [0,1,2,...,M]  │    │  [M+1,...,2M]   │    │  [...,N*M]      │       │
│    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘       │
│             │                      │                      │                │
│             └──────────────────────┼──────────────────────┘                │
│                                    │                                       │
│                          MPI Communication                                 │
│                     (Neighbor data exchange)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Distribution Strategy: MPI Workers

### Agent-to-Worker Assignment

SAGESim distributes agents across MPI workers using a **single-owner model**: each agent is owned by exactly one worker. The owning worker is responsible for:
- Storing the agent's data
- Executing the agent's step function
- Sending the agent's data to other workers that need it (neighbor visibility)

**Key file:** `sagesim/model.py:220-222`

```python
comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()
```

### Partition Strategies

SAGESim supports two agent assignment strategies:

#### 1. Round-Robin Assignment (Default)

When no partition is loaded, agents are assigned to workers in round-robin fashion:

**Key file:** `sagesim/agent.py:343-352`

```python
# Fall back to round-robin assignment
assigned_rank = self._current_rank
self._current_rank += 1
if self._current_rank >= num_workers:
    self._current_rank = 0
```

**Example with 4 workers:**
```
Agent 0 → Worker 0
Agent 1 → Worker 1
Agent 2 → Worker 2
Agent 3 → Worker 3
Agent 4 → Worker 0  (wraps around)
Agent 5 → Worker 1
...
```

#### 2. Graph Partitioning (METIS)

For networked simulations, loading a pre-computed partition minimizes cross-worker communication:

**Key file:** `sagesim/agent.py:329-346`

```python
# Assign agent to rank: use partition if loaded, otherwise round-robin
if self._partition_loaded and agent_id in self._partition_mapping:
    # Use pre-loaded partition
    assigned_rank = self._partition_mapping[agent_id]
else:
    # Fall back to round-robin assignment
    ...
```

**Benefits of graph partitioning:**
- Agents that communicate frequently are placed on the same worker
- Reduces MPI data transfer by minimizing cross-worker edges
- Typical edge-cut ratios: 10-15% with good partitions vs 50%+ with round-robin

**Usage:**
```python
model = Model(space)
model.load_partition("partition.pkl")  # BEFORE creating agents
# ... create agents ...
model.setup()
```

See `docs/network_partition.md` for detailed partition generation instructions.

---

## GPU Thread Parallelization

### Thread Organization

Each local agent on a worker maps to exactly one GPU thread:

**Key file:** `sagesim/model.py:597-600`

```python
threadsperblock = 32
blockspergrid = int(
    math.ceil(len(self.__rank_local_agent_ids) / threadsperblock)
)
```

**Thread configuration:**
- **Threads per block:** 32 (warp size, configurable via `threads_per_block` parameter)
- **Blocks per grid:** Dynamically calculated based on number of local agents
- **Total threads:** `blockspergrid × threadsperblock`

### Thread ID to Agent Mapping

Inside the GPU kernel, each thread calculates its agent index:

**Key file:** `sagesim/model.py:1124` (generated kernel code)

```python
thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
agent_index = thread_id
if agent_index < num_rank_local_agents:
    # Process agent at index agent_index
    ...
```

**Example with 100 agents:**
```
Block 0: Threads 0-31   → Agents 0-31
Block 1: Threads 32-63  → Agents 32-63
Block 2: Threads 64-95  → Agents 64-95
Block 3: Threads 96-127 → Agents 96-99 (threads 100-127 idle)
```

### Kernel Generation and JIT Compilation

SAGESim dynamically generates GPU kernels based on user-defined step functions:

**Key file:** `sagesim/model.py:825-1135` (`generate_gpu_func()`)

The process:
1. Analyze step functions to determine which properties are written
2. Generate modified step function code with double-buffering support
3. Create the main GPU kernel that dispatches to appropriate step functions
4. Compile using CuPy's JIT compiler with Numba backend

**Generated kernel structure:**
```python
@jit.rawkernel(device='cuda')
def stepfunc(global_tick, device_global_data_vector,
             a0, a1, a2, ...,           # Read buffers (properties)
             write_a0, write_a2, ...,   # Write buffers (written properties)
             sync_workers_every_n_ticks,
             num_rank_local_agents,
             agent_ids,
             current_priority_index):
    thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
    agent_index = thread_id
    if agent_index < num_rank_local_agents:
        breed_id = a0[agent_index]
        # Execute step function based on breed and priority
        ...
```

---

## Synchronization Mechanisms

SAGESim implements synchronization at two levels to ensure correct parallel execution.

### MPI-Level Synchronization

#### 1. Contextualization: Neighbor Data Exchange

Before each simulation chunk, workers exchange data for agents that are neighbors of agents on other workers:

**Key file:** `sagesim/agent.py:479-753` (`contextualize_agent_data_tensors()`)

**Process:**
1. Each worker identifies which of its local agents are neighbors of agents on other workers
2. Non-blocking sends (`isend`) transmit agent data in chunks
3. Non-blocking receives (`irecv`) collect neighbor data from other workers
4. Data is transferred using chunked transfers to avoid message size limits

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MPI Contextualization                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Worker 0                              Worker 1                     │
│  ┌─────────────────┐                   ┌─────────────────┐          │
│  │ Local Agents:   │                   │ Local Agents:   │          │
│  │   Agent 0       │                   │   Agent 2       │          │
│  │   Agent 1 ──────┼───────────────────┼─► Neighbor of 2 │          │
│  └─────────────────┘                   └─────────────────┘          │
│           │                                     │                   │
│           │ isend(Agent 1 data)                 │                   │
│           └─────────────────────────────────────┘                   │
│                                                                     │
│  Result: Worker 1 has Agent 1's data for GPU kernel                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### 2. Barrier Synchronization

Workers synchronize at key points:

**Before simulation start:**
```python
comm.barrier()  # All workers start together
```

**After GPU kernel completion:**

**Key file:** `sagesim/model.py:794-797`

```python
# CRITICAL: Ensure all workers have finished processing before syncing neighbor data
if num_workers > 1:
    comm.barrier()
```

#### 3. Global Reduction

Simulation-wide properties are synchronized using MPI allreduce:

**Key file:** `sagesim/model.py:799-803`

```python
self._global_data_vector = comm.allreduce(
    self._global_data_vector.tolist(), op=reduce_global_data_vector
)
```

### GPU-Level Synchronization

#### 1. Double Buffering

To prevent race conditions where threads read data being written by other threads, SAGESim uses double buffering:

**Key file:** `sagesim/model.py:706-723`

```python
# Create write buffers for properties that need them
write_buffers = []
for prop_idx in self._write_property_indices:
    write_buffer = cp.array(rank_local_agent_and_neighbor_adts[prop_idx])
    write_buffers.append(write_buffer)
```

**Buffer roles:**
- **Read buffer:** Frozen state at tick start - all threads read from here
- **Write buffer:** Threads write their updates here

```
┌─────────────────────────────────────────────────────────────────────┐
│                      DOUBLE BUFFERING                                │
├───────────────────────────┬─────────────────────────────────────────┤
│     READ BUFFER           │        WRITE BUFFER                     │
│     (tick-start state)    │        (thread updates)                 │
├───────────────────────────┼─────────────────────────────────────────┤
│ Agent 0: health = 100     │  Agent 0: health = ?                    │
│ Agent 1: health = 80      │  Agent 1: health = ?                    │
│ Agent 2: health = 100     │  Agent 2: health = ?                    │
└───────────────────────────┴─────────────────────────────────────────┘
         ↑                              ↑
    All threads READ here         All threads WRITE here
```

#### 2. Priority-Based Execution

Step functions can be assigned priorities. Functions with the same priority run in parallel; different priorities run sequentially with GPU synchronization between them:

**Key file:** `sagesim/model.py:729-757`

```python
for tick_offset in range(sync_workers_every_n_ticks):
    current_tick = self.tick + tick_offset

    # Execute each priority group separately with synchronization
    for priority_idx, priority_group in enumerate(self._breed_idx_2_step_func_by_priority):
        self._step_func[blockspergrid, threadsperblock](
            current_tick,
            self._global_data_vector,
            *all_args,
            1,
            cp.float32(len(self.__rank_local_agent_ids)),
            rank_local_agent_and_non_local_neighbor_ids,
            priority_idx,
        )

        # GPU sync: Wait for all threads to complete this priority
        cp.cuda.Stream.null.synchronize()

    # Copy write buffers back to read buffers after ALL priorities complete
    for i, prop_idx in enumerate(self._write_property_indices):
        rank_local_agent_and_neighbor_adts[prop_idx][:len(self.__rank_local_agent_ids)] = \
            write_buffers[i][:len(self.__rank_local_agent_ids)]

    # Final sync before next tick
    cp.cuda.Stream.null.synchronize()
```

**Key design decision:** Buffer copy happens **after all priorities complete**, ensuring all priorities within a tick see the same tick-start state.

---

## Data Flow Per Tick

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA FLOW PER TICK                                  │
└─────────────────────────────────────────────────────────────────────────────┘

1. MPI CONTEXTUALIZATION
   ┌─────────────────────────────────────────────────────────────────┐
   │  Worker 0           Worker 1           Worker 2                 │
   │     │                  │                  │                     │
   │     ├──────────────────┼──────────────────┤                     │
   │     │    Exchange neighbor agent data     │                     │
   │     ├──────────────────┼──────────────────┤                     │
   │     ▼                  ▼                  ▼                     │
   │  [Local agents]     [Local agents]     [Local agents]           │
   │  [+ neighbors]      [+ neighbors]      [+ neighbors]            │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
2. DATA PREPARATION (CPU → GPU)
   ┌─────────────────────────────────────────────────────────────────┐
   │  • Combine local + neighbor agent data                          │
   │  • Convert agent IDs → array indices                            │
   │  • Pad ragged arrays → rectangular arrays (GPU compatible)      │
   │  • Upload to GPU memory                                         │
   │  • Create write buffer copies                                   │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
3. GPU KERNEL EXECUTION
   ┌─────────────────────────────────────────────────────────────────┐
   │  For each priority group:                                       │
   │    • Launch kernel (1 thread per local agent)                   │
   │    • Threads read from read buffer                              │
   │    • Threads write to write buffer                              │
   │    • GPU synchronize                                            │
   │                                                                 │
   │  After all priorities:                                          │
   │    • Copy write buffer → read buffer                            │
   │    • GPU synchronize                                            │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
4. RESULT COLLECTION (GPU → CPU)
   ┌─────────────────────────────────────────────────────────────────┐
   │  • Download local agent data from GPU                           │
   │  • Convert array indices → agent IDs                            │
   │  • Update local agent data tensors                              │
   └─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
5. MPI SYNCHRONIZATION
   ┌─────────────────────────────────────────────────────────────────┐
   │  • Barrier: Wait for all workers to finish GPU execution        │
   │  • Allreduce: Merge global properties across workers            │
   └─────────────────────────────────────────────────────────────────┘
```

---

## Key Implementation Details

### Agent ID to Array Index Conversion

GPU kernels work with array indices, not agent IDs. SAGESim converts between them:

**Key file:** `sagesim/model.py:29-108` (`convert_agent_ids_to_indices()`)

```python
# Build lookup: agent_id → local_index
agent_id_to_index = {int(agent_id): idx for idx, agent_id in enumerate(all_agent_ids_list)}
```

The GPU array contains:
```
Index 0 to N-1:    Local agents (this worker owns these)
Index N to M:      Neighbor agents (received via MPI, read-only)
```

### Ragged-to-Rectangular Array Padding

Agent properties like neighbor lists have varying lengths. SAGESim pads them to rectangular arrays for GPU compatibility:

**Key file:** `sagesim/internal_utils.py` (`convert_to_equal_side_tensor()`)

```
Before (ragged):                 After (rectangular):
Agent 0: [1, 2, 3]              Agent 0: [1,   2,   3,   NaN]
Agent 1: [4]                    Agent 1: [4,   NaN, NaN, NaN]
Agent 2: [5, 6]                 Agent 2: [5,   6,   NaN, NaN]
Agent 3: [7, 8, 9, 10]          Agent 3: [7,   8,   9,   10 ]
```

### Ghost Cell Pattern

The GPU array includes both local and non-local (ghost) agents:

```
┌────────────────────────────────────────────────────────────────────┐
│                     GPU ARRAY LAYOUT                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Index:  0    1    2    ...   N-1  │  N    N+1   ...   M          │
│         ├───────────────────────────┼───────────────────────────┤  │
│         │     LOCAL AGENTS          │     GHOST AGENTS           │  │
│         │     (owned by this        │     (neighbors from        │  │
│         │      worker)              │      other workers)        │  │
│         │                           │                            │  │
│         │  Read/Write allowed       │  Read-only                 │  │
│         │                           │  (data from MPI)           │  │
│         ├───────────────────────────┼───────────────────────────┤  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

**Why ghost agents?**
- Local agents may have neighbors owned by other workers
- Ghost agents provide read access to neighbor properties
- Only local agents (indices 0 to N-1) are written back after kernel execution

---

## Quick Reference

### Key Files and Their Roles

| File | Role |
|------|------|
| `sagesim/model.py` | Core simulation loop, GPU kernel generation, data flow orchestration |
| `sagesim/agent.py` | Agent factory, MPI contextualization, partition loading |
| `sagesim/breed.py` | Breed definition, step function registration, property declaration |
| `sagesim/space.py` | Spatial topology (NetworkSpace, GridSpace), neighbor computation |
| `sagesim/internal_utils.py` | Array conversion utilities (ragged → rectangular) |

### Synchronization Points

| Point | Level | Location | Purpose |
|-------|-------|----------|---------|
| `comm.barrier()` | MPI | `model.py:513` | Workers start simulation together |
| `contextualize_agent_data_tensors()` | MPI | `agent.py:479-753` | Exchange neighbor data between workers |
| `cp.cuda.Stream.null.synchronize()` | GPU | `model.py:748` | Wait for all GPU threads to finish priority group |
| `cp.cuda.Stream.null.synchronize()` | GPU | `model.py:757` | Ensure buffer copy completes before next tick |
| `comm.barrier()` | MPI | `model.py:796` | All workers finish GPU execution |
| `comm.allreduce()` | MPI | `model.py:800` | Merge global properties across workers |

### Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threads_per_block` | 32 | GPU threads per block (warp size) |
| `sync_workers_every_n_ticks` | 1 | Ticks between MPI synchronization |
| `verbose_timing` | False | Print timing breakdown per tick |
| `verbose_mpi_transfer` | False | Track MPI bytes sent/received |

---

## See Also

- `docs/synchronization_and_double_buffering.md` - Detailed explanation of race condition prevention
- `docs/network_partition.md` - Guide to generating and using network partitions
- `docs/selective_property_synchronization.md` - Reducing MPI overhead with neighbor_visible
- `docs/gpu_cpu_data_flow.md` - Detailed data flow diagrams
