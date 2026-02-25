# GPU Communication Redesign

## Current Per-Tick Data Flow

```
GPU → CPU download → CPU dict build → CPU pad ragged→rectangular → CPU→GPU upload → GPU kernel → GPU→CPU download → CPU index→ID → MPI pickle → ...
```

Every tick, data leaves the GPU, gets processed on CPU in Python, and goes back to GPU. The GPU kernel (the actual simulation work) is a small fraction of tick time. The rest is overhead.

### Current overhead sources

| Component | What it does | Location |
|-----------|-------------|----------|
| Data preparation | Python dict build for ID→index mapping, combine local+neighbor lists, pad ragged neighbor lists to rectangular arrays via `convert_to_equal_side_tensor()`, create write buffers | model.py:626-724 |
| GPU↔CPU transfers | Upload all arrays CPU→GPU via `cp.array()`, download results GPU→CPU via `.get()`, convert indices back to IDs on CPU | model.py:690, 779, 784 |
| MPI communication | Pickle-based `comm.isend()`, 128-byte chunking, Python loops over agents to identify cross-worker neighbors | agent.py:518-708 |
| GPU kernel | Actual simulation computation | model.py:731-757 |

---

## The Three Changes

### Change 1: CSR Format for Neighbor Lists

**Problem:** SAGESim pads all neighbor lists to the same max length to form rectangular GPU arrays.

```
Current rectangular (1M agents, max 50K neighbors but avg 50):
agent 0:      [5, 2, NaN, NaN, ..., NaN]         ← 50,000 entries, 2 real
agent 1:      [8, 3, 1, NaN, ..., NaN]            ← 50,000 entries, 3 real
agent 999999: [0, 1, 2, ..., 49999]               ← 50,000 entries, 50,000 real

Memory: 1M × 50,000 × 4 bytes = 200 GB  ← won't fit on any GPU
```

One agent with many neighbors forces every agent to be padded to that length. If any agent gains a neighbor beyond the current max, everything must be re-padded.

**Solution:** CSR (Compressed Sparse Row) — two flat arrays, no padding.

```
values:  [5, 2, 8, 3, 1, ..., 0, 1, 2, ..., 49999]    ← all neighbors concatenated
offsets: [0, 2, 5, ..., 10049999, 10099999]              ← where each agent's neighbors start

Agent i's neighbors = values[offsets[i] : offsets[i+1]]

Memory: (total_edges) × 4 + (1M + 1) × 4 ≈ 40 MB
```

**What changes in user step functions:**

Sugar functions hide the CSR details. Minimal rewrite:

```python
# Before (rectangular, NaN-terminated):
neighbor_indices = locations[agent_index]
i = 0
while i < len(neighbor_indices) and neighbor_indices[i] != -1:
    neighbor_state = state_tensor[neighbor_indices[i]]
    i += 1

# After (CSR with sugar functions):
for i in range(get_num_neighbors(agent_index, neighbor_offsets)):
    neighbor_idx = get_neighbor(agent_index, i, neighbor_offsets, neighbor_values)
    neighbor_state = state_tensor[neighbor_idx]
```

`get_num_neighbors` and `get_neighbor` are trivial inlines that CuPy JIT optimizes away:

```python
def get_num_neighbors(agent_index, offsets):
    return offsets[agent_index + 1] - offsets[agent_index]

def get_neighbor(agent_index, i, offsets, values):
    return values[offsets[agent_index] + i]
```

**Dynamic topology handling:**
- Neighbor added: append to `values[]`, update `offsets[]` entries after that agent
- Neighbor removed: mark as invalid or compact
- Agent born: append new entry to `offsets[]`, append neighbors to `values[]`
- Agent dies: set `offsets[dead] = offsets[dead+1]` (zero-length range)
- Pre-allocate `values[]` with 2x slack. Rebuild only when full (amortized O(1)).

**Scaling property:**
```
Current overhead:  O(N × max_neighbors)    ← worst single agent dictates memory for all
CSR overhead:      O(N + E)                ← proportional to actual edges
```

---

### Change 2: GPU-Resident Data with GPU Hash Map and Dual CSR Arrays

#### ID→Index Conversion with CSR

Each GPU holds a local subset of agents. Agent IDs are global (0 to billions), but GPU property arrays are indexed locally (0 to `num_local + num_ghost - 1`). The CSR `values[]` array stores neighbor agent IDs, but the kernel needs local buffer indices to look up property arrays.

**Solution:** Keep two CSR value arrays on GPU — one with IDs (source of truth), one with local indices (derived cache for the kernel). Plus one shared offsets array.

```
CSR offsets:           [0, 2, 5, 9, ...]                  ← shared, one copy
CSR values (IDs):      [agent_502, agent_17, agent_8834, ...]   ← permanent, used for MPI
CSR values (indices):  [local_3,   local_0,   local_42,  ...]   ← derived, used by kernel
```

MPI always speaks in agent IDs. The GPU kernel always reads from the indices array. The conversion is a one-way derivation (IDs → indices via GPU hash map), never reversed.

**Memory cost of dual arrays** (1M agents, 50M edges):
```
CSR offsets:            (1M + 1) × 4 bytes  =   4 MB
CSR values (IDs):       50M × 4 bytes       = 200 MB
CSR values (indices):   50M × 4 bytes       = 200 MB
                                        Total = 404 MB
```

Compare to current rectangular: 1M × 50K × 4 bytes = 200 GB. The extra array is negligible compared to the savings from eliminating padding.

**When topology doesn't change (common case):**
- Both CSR arrays are valid from last tick. Zero conversion cost. The indices array just sits there ready.

**When only property values change (ghost data updated via MPI):**
- MPI receives updated property values for ghost agents
- GPU unpack kernel writes values directly into property arrays at existing buffer indices
- CSR untouched, hash map untouched. Zero conversion cost.

**When topology changes (edges added/removed):**
1. Update CSR values (IDs) — insert/remove neighbor IDs
2. If new ghost agents appeared: allocate ghost buffer slot, insert `(agent_id → slot)` into GPU hash map
3. Rebuild CSR values (indices) from CSR values (IDs) — GPU kernel does parallel hash map lookups on changed entries

```cuda
// GPU kernel: convert agent IDs to local indices in CSR values array
__global__ void id_to_index(
    int* csr_values_ids,       // source: agent IDs
    int* csr_values_indices,   // destination: local indices
    int num_entries,
    long* hash_keys,           // hash table keys (agent IDs)
    int* hash_values,          // hash table values (local indices)
    int hash_table_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_entries) return;

    int agent_id = csr_values_ids[tid];
    // Open-addressing lookup
    int slot = agent_id % hash_table_size;
    while (hash_keys[slot] != agent_id) {
        slot = (slot + 1) % hash_table_size;
    }
    csr_values_indices[tid] = hash_values[slot];
}
```

**When agents are born/die:**
- Born: allocate buffer slot, insert into hash map, append to CSR offsets and both value arrays
- Die: mark slot as free, remove from hash map, set CSR range to zero-length
- Periodic compaction to reclaim dead slots

**Conversion cost summary:**

| Scenario | CSR values (IDs) | CSR values (indices) | Hash map | Cost |
|----------|-----------------|---------------------|----------|------|
| Nothing changed | untouched | untouched | untouched | zero |
| Only property values changed | untouched | untouched | untouched | zero |
| Topology changed | update changed entries | rebuild from IDs via hash map | update if new ghosts | O(changed_entries) on GPU |
| Agent born/died | append/mark | append/mark | insert/remove | O(1) amortized |

**Problem:** Every tick, all data is downloaded from GPU, processed on CPU, and re-uploaded. The ID→index mapping is built as a Python dict on CPU (model.py:631).

**Solution:** Data stays on GPU between ticks. A GPU-side hash map handles ID↔index conversion.

**What stays on GPU permanently (allocated once at `setup()`):**
- Property arrays (state, preventative_measures, etc.) — CuPy arrays
- CSR arrays (offsets, values) — CuPy arrays
- Write buffers — CuPy arrays, reused each tick
- Hash map (agent_id → buffer_index) — CuPy arrays

**GPU hash map** (open-addressing, pre-allocated):
```
hash_keys:   cp.full(table_size, -1, dtype=cp.int64)   # agent_id or -1 (empty)
hash_values: cp.full(table_size, -1, dtype=cp.int32)   # buffer index

Table size = 2 × expected (local + ghost) agents
Memory for 1M agents: 2 × 1M × 12 bytes = 24 MB
```

When the agent set changes (birth, death, MPI ghost update):
1. A GPU kernel updates the hash map (insert/remove entries) — parallel, microseconds
2. A GPU kernel converts neighbor IDs to local indices using the hash map — parallel, microseconds
3. No CPU involvement

**Pre-allocation strategy:**
- Allocate 1.5x initial agent count at `setup()`
- If population grows beyond allocation, reallocate at 2x (amortized doubling)
- Optional hint: `model.setup(max_agents_hint=1_000_000)`
- Slack memory cost is far less than current rectangular padding waste

**What this eliminates per tick:**
- `cp.array()` upload — gone (data already on GPU)
- `.get()` download — gone (data stays on GPU)
- `free_all_blocks()` — gone (persistent buffers)
- Python dict comprehension for ID→index — gone (GPU hash map)
- `convert_to_equal_side_tensor()` — gone (CSR, already on GPU)

Data only moves to CPU when the user explicitly requests it (e.g., `model.get_agent_property_value()` after simulation ends).

---

### Change 3: Buffer-Protocol MPI

**Problem:** `comm.isend()` (lowercase) pickles Python objects — orders of magnitude slower than raw buffer transfers. Data is broken into 128-byte chunks with individual MPI messages, each adding latency.

**Solution:** `comm.Isend()` (uppercase) with pre-allocated contiguous buffers. One message per rank.

```python
# Before (agent.py:640-676):
# Python objects → pickle → many small MPI messages
for chunk in data_chunks:
    comm.isend(chunk, dest=rank, tag=i)    # pickle serialization per chunk

# After:
# Flat buffer → single MPI message per rank
pack_kernel[blocks, threads](gpu_data, send_indices, send_buf)  # GPU packs data
send_buf_cpu = send_buf.get()  # one transfer
comm.Isend(send_buf_cpu, dest=rank)  # buffer protocol, no pickle

# Or with GPU-aware MPI (Frontier, etc.):
comm.Isend(send_buf_gpu, dest=rank)  # GPU buffer directly, no CPU at all
```

**Send buffer layout (flat, contiguous):**
```
[num_agents | agent_0_id, prop_0, prop_1, ... | agent_1_id, prop_0, prop_1, ... ]
```

One contiguous buffer per destination rank. No chunking, no pickle, no Python object overhead.

---

## Per-Tick Flow After All Changes

```
Before (CPU in the loop):
GPU → CPU → Python dict → Python pad → CPU→GPU → kernel → GPU→CPU → Python convert → MPI pickle

After (CPU only issues MPI calls):
GPU hash map update → GPU kernel → GPU pack kernel → MPI send/recv → GPU unpack kernel → repeat
```

The CPU's only role is issuing MPI calls (`Isend`/`Irecv`/`Waitall`). All data stays on GPU.

---

## Memory Impact

| Component | Current | After | Change |
|-----------|---------|-------|--------|
| Neighbor storage (1M agents, max 50K, avg 50) | 200 GB (rectangular) | ~404 MB (CSR offsets + dual value arrays) | Orders of magnitude less |
| Property arrays | Rebuilt every tick | Persistent + 1.5x slack | Slight increase from slack |
| GPU hash map (1M agents) | N/A (Python dict on CPU) | ~24 MB | New cost |
| Write buffers | Rebuilt every tick | Persistent, pre-allocated | Same size, no alloc overhead |
| MPI send/recv buffers | Pickle objects | Pre-allocated contiguous | Slight increase from pre-allocation |

Net effect: total GPU memory goes **down** for any model where max_neighbors >> avg_neighbors, which is most real-world networks.

---

## Drawbacks

| Drawback | Severity | Mitigation |
|----------|----------|------------|
| Step function API change | High — existing models must update neighbor iteration | Sugar functions (`get_num_neighbors`, `get_neighbor`) minimize rewrite |
| Pre-allocation slack | Low — ~1.5x initial size | Far less than rectangular padding waste |
| GPU hash map implementation | Medium — must write custom CuPy kernel | Well-documented GPU primitive (open-addressing) |
| Debugging difficulty | Medium — data lives on GPU, can't print easily | Add verbose mode that downloads and prints on demand |
| Reallocation on extreme growth | Low — amortized O(1) | Auto-handled, rare event |

---

## Files to Modify

| File | Changes |
|------|---------|
| `sagesim/model.py` | Replace per-tick array rebuild with GPU-resident buffers. Update `worker_coroutine()` to skip CPU data prep. Update `generate_gpu_func()` to emit CSR-style neighbor access with sugar functions. Pre-allocate buffers in `setup()`. |
| `sagesim/agent.py` | Replace `contextualize_agent_data_tensors()` with buffer-protocol MPI using pre-allocated send/recv buffers. Replace pickle `comm.isend()` with `comm.Isend()`. |
| `sagesim/internal_utils.py` | Replace `convert_to_equal_side_tensor()` with CSR construction. Add GPU hash map utilities. |
| `sagesim/space.py` | Update `NetworkSpace` to build CSR format instead of Python lists/sets. |
| New: `sagesim/gpu_kernels.py` | GPU hash map (insert/lookup/delete), pack/unpack kernels for MPI buffers, ID→index conversion kernel. |
