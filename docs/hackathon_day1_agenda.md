# Frontier Hackathon — Day 1 Agenda

**SAGESim: Scalable Agent-Based GPU-Enabled Simulator**

Day 1 focus: environment setup, I/O strategy discussion, MPI code walkthrough, profiling tools.

---

## 1. Environment Setup (for new participants)

### Frontier Module Loads

From `examples/sir/sir_frontier.sh:17-20`:

```bash
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.2.4
module load craype-accel-amd-gfx90a
```

### Conda Environment

```bash
# Activate shared env (or create your own)
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/sagesim_env
```

Currently using **CuPy 13.6** with ROCm backend.

### Building the Conda Environment (from scratch)

**MPI installation** (Cray compiler wrappers — must link against Cray MPICH):

```bash
MPICC="cc -shared" pip install --no-cache-dir --no-binary=mpi4py mpi4py
```

**CuPy 13.6.0 installation** (ROCm/HIP backend, source build):

```bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=${ROCM_PATH}
export HCC_AMDGPU_TARGET=gfx90a
CC=gcc CXX=g++ pip install --no-cache-dir --no-binary=cupy cupy==13.6.0
```

### CuPy Version Discussion: Upgrading to 14.0.1

CuPy 14 (released Feb 2026) is a major update. Key changes relevant to Frontier:

**What changed in CuPy 14:**

| Change | Impact |
|---|---|
| **NumPy 2 semantics** | Type promotion rules updated — mixed-type arithmetic may produce different dtypes (e.g., `float32 + float64` → `float64`) |
| **Dropped CUDA 11 / Python 3.9** | No impact (Frontier uses ROCm, Python 3.10+) |
| **Removed `cupy.sparse`, `cupy.prof`** | No impact (SAGESim doesn't use these) |
| **C++17 default for RawKernel** | Kernel compilation may behave differently; `-std=c++17` is now default |
| **Jitify deprecated** | We use `cupyx.jit.rawkernel`, not `jitify=True` — no impact |
| **ROCm improvements** | Feature parity with AMD's v13 fork; ROCm 7 support added; ROCm 6.2.x (our version) supported |

**Build command for CuPy 14.0.1** — same approach, just change the version:

```bash
export CUPY_INSTALL_USE_HIP=1
export ROCM_HOME=${ROCM_PATH}
export HCC_AMDGPU_TARGET=gfx90a
CC=gcc CXX=g++ pip install --no-cache-dir --no-binary=cupy cupy==14.0.1
```

**SAGESim compatibility risks to verify:**

1. **JIT private API monkeypatch** — `jit_extensions.py` imports `cupyx.jit._internal_types.BuiltinFunc`, `cupyx.jit._cuda_types`, `cupyx.jit._builtin_funcs` to add `__threadfence()` support. These are private (underscore-prefixed) APIs with no stability guarantee. Must verify they still exist and have the same signatures in 14.0.1.

2. **`cupy.disable_experimental_feature_warning`** — `sagesim/__init__.py` sets this flag. May be removed or renamed in 14.x. Check if the attribute still exists.

3. **NumPy 2 type promotion** — SAGESim uses `cp.float32`, `cp.int32` throughout GPU kernels and buffer operations. Need to verify that mixed-type operations in kernel argument passing and buffer packing/unpacking produce identical results.

4. **Peer access default** — CuPy 14 enables GPU peer access by default on multi-GPU systems. Reportedly causes issues on AMD GPUs. Each MPI rank uses one GPU so likely unaffected, but worth verifying on multi-GCD MI250X nodes.

**Verification checklist after upgrading:**

- [ ] `import sagesim` succeeds (no import errors from JIT extensions or experimental warning flag)
- [ ] Single-rank SIR model runs and produces correct results
- [ ] Multi-rank MPI run with ghost exchange works (test both CPU-staged and GPU-aware paths)
- [ ] Grid barrier kernel executes without hangs
- [ ] Numerical results match CuPy 13.6 output (compare agent state after N ticks)

**Discussion questions:**

1. Should we upgrade to 14.0.1 now, or wait for broader ROCm 6.2 testing by the community?
2. If the JIT private APIs changed, should we upstream a `threadfence()` PR to CuPy instead of monkeypatching?
3. Does the NumPy 2 type promotion affect our kernel's numerical behavior (all our buffers are explicitly `float32`/`int32`, so likely safe)?

### GPU-Aware MPI Activation

```bash
export MPICH_GPU_SUPPORT_ENABLED=1   # Enables Cray MPICH GPU-Direct RDMA
```

Detection code at `gpu_kernels.py:14-32` — also checks `OMPI_MCA_opal_cuda_support` and `SAGESIM_GPU_AWARE_MPI` override.

---

## 2. I/O — Snapshots & Checkpointing

### 2.1 Current State (SAGESim Framework)

**Save/Load:** `model.py:1289-1310` — Python pickle, single rank, not scalable
```python
def save(self, app, fpath):
    with open(fpath, "wb") as fout:
        pickle.dump(app, fout)
```

**Per-tick GPU→CPU download:** `model.py:1276-1288`
```python
def _download_local_data_to_cpu(self, num_local_agents):
    for prop_idx in buf.sorted_write_indices:
        self.__rank_local_agent_data_tensors[prop_idx] = \
            buf.property_tensors[prop_idx][:num_local_agents].get().tolist()
```

**Breed data collection:** `model.py:465-487` — `get_breed_data()` gathers to rank 0 via MPI `Gatherv`

### 2.2 Two Applications, Two I/O Patterns

GGap and SuperNeuroABM collect fundamentally different kinds of data, which drives different MPI strategies:

| | GGap (Forest Gap Model) | SuperNeuroABM (Spiking Neural Networks) |
|---|---|---|
| **What's collected** | Fixed agent properties — every tree, every snapshot | Sparse events — only neurons that fire |
| **Data size per collection** | Deterministic: N_agents × width | Variable: depends on firing activity |
| **Collection frequency** | Periodic — every N years (e.g., 10) | Every tick |
| **GPU recording mechanism** | None — reads from existing `property_tensors` | Atomic append to shared `_spike_record_gpu` buffer |
| **MPI collective** | `Gatherv` to rank 0 | `allgather` to all ranks |
| **Scaling bottleneck** | Rank 0 memory explosion | O(ranks) communication every tick |

**Why the difference?**

GGap agents write to pre-allocated, per-agent buffer slots via double-buffering — thread 0 writes `property[0]`, thread 1 writes `property[1]`, no conflicts. The data is already sitting in `property_tensors` after the kernel runs; `get_breed_data()` just downloads it with `Gatherv` to rank 0.

SuperNeuroABM has a **variable-count event** problem: only neurons that fire produce output, and you don't know how many will fire until the kernel finishes. Multiple threads race to append `[soma_id, tick]` to a shared buffer, so an **atomic counter** coordinates which position each thread writes to. Without atomics, concurrent appends corrupt or overwrite each other.

Note: the actual spike state used for STDP and network dynamics is **not** in this recording buffer. Each soma has a size-2 `output_spikes_tensor` double buffer (`value[tick % 2]`) in `property_tensors`, which is synchronized across ranks via normal ghost exchange. The `_spike_record_gpu` buffer and `allgather` are purely for **data recording and post-hoc analysis** — they are not fed back into computation.

```
GGap (no atomics needed):              SuperNeuroABM (atomics required):
  Thread 0: prop[0] = val0               Thread 0: if fires → atomic_idx = atomicAdd(counter, 1)
  Thread 1: prop[1] = val1                                     spike_buf[atomic_idx] = [0, tick]
  Thread 2: prop[2] = val2               Thread 5: if fires → atomic_idx = atomicAdd(counter, 1)
  (each thread owns its index)                                  spike_buf[atomic_idx] = [5, tick]
                                          Thread 23: no fire → nothing written
```

### 2.3 GGap (Forest Gap Model) — Periodic Snapshot

**I/O pattern** (`GGap/gap/run_one_site.py:238-309`):
```python
for year_batch in range(0, years, report_interval):
    model.simulate(ticks=years_to_run, sync_workers_every_n_ticks=1)
    # Download ALL tree data to rank 0
    tree_data = model.collect_tree_data()     # get_breed_data("Tree", "params/states")
    writer.write_tree_data(tree_data)         # CSV output
```

**Data collection path:** `get_breed_data()` → `_gatherv_breed_gpu()` (`model.py:464-583`)

```
model.py:535   Allgather(local_count) → all_counts[]   ← each rank announces its agent count
model.py:573   Gatherv(send_buf, recv_buf, root=0)     ← actual data goes to rank 0 only
```

- Downloads every N years (default 10) via `--report_interval`
- Bulk download: `get_breed_data("Tree", "params")` + `get_breed_data("Tree", "states")` (`GGap/gap/run_multi_site.py:57-60`)
- ~200K tree agents × 25 floats = ~20 MB state per download
- **All data gathered to rank 0** — bottleneck at scale
- **5 CSV files per simulation:** site_data, soil_data, genus_data, species_data, tree_data (`GGap/gap/output_utils.py:64-369`)
- **No checkpointing** — job restart = start from year 0

**Scaling concern (512 GPUs):**
- 5,120 sites × ~200K trees/site = ~1B agents
- Rank 0 gathering all tree data = memory explosion
- CSV write serialized on rank 0

### 2.4 SuperNeuroABM (Spiking Neural Networks) — Per-Tick Recording

**Spike recording** (`superneuroabm/model.py:1094-1150`):

SuperNeuroABM extends SAGESim's kernel via subclass hooks (`sagesim/model.py:892-921`):
1. `_get_extra_kernel_config()` — injects `spike_record`, `spike_record_count`, and `spike_mask` into kernel signature; adds post-step code that checks `output_spikes_tensor[_real_idx][tick % 2]` and atomically appends `[agent_id, tick]` if the neuron fired
2. `_prepare_kernel_extras()` — lazy-allocates the spike buffer on first call, resets atomic counter to 0, builds per-soma recording mask
3. `_process_kernel_extras()` — downloads `spike_buf[:count * 2]` to CPU, appends to `_recorded_spikes` list

**Spike buffer allocation** (`superneuroabm/model.py:1115-1118`):
```python
if self._spike_record_gpu is None:
    max_slots = max(10000, num_local_agents * sync_ticks // 100)
    self._spike_record_gpu = cp.full(max_slots * 2, cp.nan, dtype=cp.float32)
    self._spike_record_count_gpu = cp.zeros(1, dtype=cp.int32)
```

- **Size estimate:** `max(10000, num_local_agents * sync_ticks // 100)` slots — assumes ≤1% of neurons fire per tick. Each slot = 2 floats `[agent_id, tick]`.
- **Lazy allocation:** Buffer created once on first `_prepare_kernel_extras()` call, reused across ticks.
- **Atomic counter:** Single `int32`, reset to 0 before each kernel launch. Kernel increments via `jit.atomic_add(spike_record_count, 0, 1)`.
- **No overflow handling** — if >1% fire per tick (e.g., synchronous bursting), the buffer silently overflows.

Per-tick flow:
```
GPU kernel:   output_spikes_tensor[idx][tick % 2] > 0 && spike_mask[idx] > 0
              → slot = atomicAdd(spike_record_count, 0, 1)
              → spike_record[slot * 2] = agent_id
              → spike_record[slot * 2 + 1] = tick
Post-kernel:  spike_buf[:count * 2].get().tolist()  → CPU download  (superneuroabm/model.py:1134-1139)
MPI:          comm.allgather(self._recorded_spikes)                 (superneuroabm/model.py:1141-1150)
```

**Why `allgather` instead of `Gatherv`?** This is purely for **data recording convenience** — user scripts call `get_all_spike_times()` on any rank and expect global results. The `allgather` is NOT needed for computation. The actual spike state for STDP and network dynamics lives in `output_spikes_tensor` (size-2 double buffer per soma in `property_tensors`), which is synchronized across ranks via normal ghost exchange — completely independent of the recording buffer.

**Why not `Gatherv` to rank 0?** SuperNeuroABM's API exposes `get_spike_times(soma_id)` and `get_all_spike_times()` on all ranks, so every rank needs the full spike list. GGap only needs data on rank 0 for CSV output, so `Gatherv` suffices.

**State history buffers** (`superneuroabm/model.py:151-152`):
- O(agents × ticks × state_size) — can be 12+ GB for moderate simulations
- All kept in memory, no streaming to disk

**Manual saves in user scripts** (`examples/masquelier_2008/run_experiment_hg.py:230-235`):
```python
np.save(OUTPUT_DIR / "spike_times.npy", all_spike_times)
np.savez(OUTPUT_DIR / "weight_snapshots.npz", ...)
```

**Poor-man's checkpoint** (`superneuroabm/model.py:552-575`):
```python
def reset(self, retain_parameters=True):  # Keeps learned weights, resets state
```

**Frontier note:** `cray-hdf5-parallel` loaded in scaling scripts (`submit_weak_scaling_parallel.sh:133`) but not used in code yet.

### 2.5 Discussion Points for Experts

**Checkpointing (neither app has this):**
- Long Frontier jobs hit wall-time limits — need checkpoint/resume
- Checkpoint = save full GPU state (property_tensors, CSR, write_buffers, hash_map) + model metadata
- Resume = reload state, rebuild GPU buffers, reconstruct communication topology
- What is the recommended approach on Frontier?

**Parallel I/O (both apps gather to rank 0 or allgather):**
- `Gatherv`-to-rank-0 (GGap) doesn't scale beyond ~100 GPUs
- `allgather` (SuperNeuroABM) every tick is O(ranks) communication
- Options:
  - **ADIOS2** — Frontier-native, designed for HPC I/O, supports GPU-direct write?
  - **Parallel HDF5** — Already module-loaded for SuperNeuroABM, standard format
  - **Per-rank files** — Simple but many small files (bad for Lustre metadata)
- What works best on Lustre filesystem?

**SuperNeuroABM allgather bottleneck:**
- `comm.allgather()` for spikes every tick — O(ranks) communication
- The `allgather` is NOT needed for computation — spike state for STDP/dynamics is already in `output_spikes_tensor` (synced via ghost exchange). The `allgather` only exists because the current API (`get_spike_times()`) is callable on any rank.
- **This is really an I/O problem, not a communication problem.** If we have a proper parallel I/O strategy (ADIOS2/HDF5), each rank can write its local spikes directly to disk — no collective needed at all.
- Fallback: even without parallel I/O, `Gatherv` to rank 0 (like GGap) would halve the communication cost vs `allgather`.

**GPU-direct I/O:**
- Can GPU buffers be written directly to Lustre without CPU staging?
- GPUDirect Storage / GDS on Frontier?

---

## 3. MPI Communication Code Walkthrough

### 3.1 Architecture Overview

Each MPI rank owns one GPU and a subset of agents. Agents may have **neighbors on remote ranks** (ghost agents). Every tick:

1. Ghost exchange → synchronize boundary data between ranks
2. GPU kernel → execute all agent logic in parallel
3. Write-back → copy results for next exchange

**Orchestration code:** `model.py:1314-1588` (worker_coroutine)

### 3.2 First Tick: One-Time Setup (`model.py:1364-1408`)

The first call to `worker_coroutine` builds all persistent state. This only runs once — subsequent ticks skip to ghost exchange.

```
model.py:1364  if not buf.is_initialized:
  │
  ├─ model.py:1367-1369  Compute neighbor lists (CPU)
  │   Space._neighbor_compute_func() → ragged list of neighbor IDs per local agent
  │
  ├─ model.py:1373-1377  discover_ghost_topology() ──► gpu_kernels.py:35-88
  │   Scan neighbor lists → find agent IDs owned by other ranks → ghost IDs
  │   (vectorized CPU scan, no MPI)
  │
  ├─ model.py:1386        _build_gpu_buffers(ghost_ids, num_local_agents)
  │   Allocate ALL persistent GPU memory:
  │   - property_tensors (local agents + ghost placeholders)
  │   - CSR arrays (neighbor_offsets, neighbor_values, neighbor_values_ids)
  │   - write_buffers (double buffering)
  │   - GPUHashMap (agent_id → buffer_index)
  │   - agent_ids_gpu, logical_ids_gpu
  │
  ├─ model.py:1398-1401  CommunicationManager.build_communication_maps()
  │   ──► gpu_kernels.py:521-850
  │   The MPI handshake (detailed in Section 3.5):
  │   - Scan CSR for cross-rank neighbors
  │   - Alltoall to exchange request counts
  │   - Isend/Irecv to exchange agent ID lists
  │   - Build send/recv GPU index maps
  │   - Pre-allocate MPI buffers
  │
  └─ model.py:1402        First exchange_ghost_data()
      Fill ghost slots with actual data from remote ranks.
      Without this, ghost agent properties would be zeros.
```

After this, `buf.is_initialized = True` (`gpu_kernels.py:850`) and all subsequent calls take the fast path (`model.py:1410-1417`).

### 3.3 Per-Tick Flow (Subsequent Ticks)

```
model.py:961-998  simulate() loop
  │
  ├─ model.py:973         comm.barrier()  — ensures all ranks finished setup
  │                        before any rank starts the first tick
  │
  ├─ model.py:1414-1417  exchange_ghost_data() ──► gpu_kernels.py:1068-1123
  │   ├─ gpu_kernels.py:852-909   _batched_gpu_pack()
  │   ├─ gpu_kernels.py:1084-1087 stream.synchronize()
  │   ├─ gpu_kernels.py:911-982   mpi_exchange()
  │   └─ gpu_kernels.py:984-1066  _batched_gpu_unpack()
  │
  ├─ model.py:1514-1528  GPU kernel launch (fused)
  ├─ model.py:1535        stream.synchronize()
  └─ model.py:1563-1570  write_buffers → property_tensors (GPU→GPU)
```

**Chunking:** `simulate(ticks=100, sync_workers_every_n_ticks=10)` runs the kernel for 10 fused ticks, then does MPI exchange, repeats 10 times. Within each chunk no MPI occurs — all ticks run on GPU with grid barriers only. (`model.py:982-998`)

### 3.4 GPU Memory Layout

```
Buffer index: [0 ... N_local-1]  [N_local ... N_total-1]  [N_total ... capacity-1]
               └── local agents ─┘ └── ghost agents ──────┘ └── slack (unused) ───┘
```

- `property_tensors[i]`: CuPy float32, shape `(capacity, width_i)` — `gpu_kernels.py:277`
- `property_tensors[1]` = `None` (replaced by CSR) — `model.py:1440`
- Slack factor 1.5x — `gpu_kernels.py:270`

**CSR neighbor structure (dual representation):** `model.py:1146-1150`

```
neighbor_offsets    — CuPy int32, (N_total+1,)  — CSR row pointers
neighbor_values     — CuPy int32, (total_edges,) — LOCAL BUFFER INDICES (for kernel)
neighbor_values_ids — CuPy int32, (total_edges,) — GLOBAL AGENT IDs (for MPI)
```

Why dual? Kernel needs O(1) array indexing. MPI needs globally unique IDs to identify agents across ranks.

### 3.5 Communication Topology Setup (One-Time, Tick 1)

**Entry:** `model.py:1397-1402`  
**Full method:** `gpu_kernels.py:521-850` (`build_communication_maps`)

#### Phase 1: Discover cross-rank dependencies (`gpu_kernels.py:562-623`)

```python
# gpu_kernels.py:568-570 — Download CSR from GPU
cpu_offsets = buf.neighbor_offsets[:num_local + 1].get()
cpu_values_ids = buf.neighbor_values_ids[:total_edges].get()

# gpu_kernels.py:580-583 — Expand to (agent_idx, neighbor_id) pairs
counts = np.diff(cpu_offsets)
local_agent_indices = np.repeat(np.arange(num_local, dtype=np.int32), counts)

# gpu_kernels.py:593-607 — Dense rank lookup: agent_id → rank
rank_lookup = np.full(max_agent_id + 1, -1, dtype=np.int32)
neighbor_ranks = rank_lookup[neighbor_ids]

# gpu_kernels.py:610-612 — Filter to cross-rank only
cross_mask = (neighbor_ranks != self.my_rank) & (neighbor_ranks >= 0)

# gpu_kernels.py:617-623 — Group by source rank
for src_rank in np.unique(cross_neighbor_ranks):
    need_from_rank[src_rank] = np.unique(cross_neighbor_ids[mask])
```

#### Phase 2: Exchange request counts — MPI Alltoall (`gpu_kernels.py:625-630`)

```python
request_counts = np.zeros(num_workers, dtype=np.int32)  # how many I need FROM each
supply_counts = np.zeros(num_workers, dtype=np.int32)    # how many each needs FROM me
self.comm.Alltoall(request_counts, supply_counts)         # COLLECTIVE
```

**Question:** Is `Alltoall` the best collective for this? Would `Alltoallv` or neighbor collectives be more efficient for sparse topologies?

#### Phase 3: Exchange agent ID lists — Isend/Irecv (`gpu_kernels.py:632-651`)

```python
# gpu_kernels.py:634-637 — Tell each peer which agent IDs I need
for src_rank, ids in need_from_rank.items():
    self.comm.Isend([ids.astype(np.int32), MPI.INT], dest=src_rank, tag=100)

# gpu_kernels.py:640-649 — Learn which of MY agents each peer needs
for dest_rank where supply_counts[dest_rank] > 0:
    recv_buf = np.empty(supply_counts[dest_rank], dtype=np.int32)
    self.comm.Irecv([recv_buf, MPI.INT], source=dest_rank, tag=100)

# gpu_kernels.py:651
MPI.Request.Waitall(send_req_requests + recv_req_requests)
```

**Question:** Should we use `MPI_Neighbor_alltoallv` instead of manual Isend/Irecv for this pattern?

#### Phase 4: Build send/recv GPU index maps (`gpu_kernels.py:653-679`)

```python
# gpu_kernels.py:660-668 — Send: local buffer positions to pack for each peer
for dest_rank, agent_ids_np in requested_by_rank.items():
    local_indices = [buf.agent_id_to_index[int(aid)] for aid in agent_ids_np]
    self.send_indices_gpu[dest_rank] = cp.array(local_indices, dtype=cp.int32)

# gpu_kernels.py:671-679 — Recv: ghost buffer positions to scatter into
for src_rank, remote_ids in need_from_rank.items():
    ghost_indices = [buf.agent_id_to_index[int(aid)] for aid in remote_ids]
    self.recv_indices_gpu[src_rank] = cp.array(ghost_indices, dtype=cp.int32)
```

#### Phase 5: Batch indices for efficient GPU ops (`gpu_kernels.py:766-793`)

```python
# Concatenate all per-peer indices into single GPU arrays
_batched_send_indices_gpu = cp.concatenate([send_indices_gpu[r] for r in sorted_peers])
_batched_recv_indices_gpu = cp.concatenate([recv_indices_gpu[r] for r in sorted_peers])
```

Enables ONE GPU gather per property across ALL peers (V gathers total, not V × P).

#### Phase 6: Pre-allocate MPI buffers (`gpu_kernels.py:747-764`)

```python
# Send buffers — ALWAYS on GPU
send_bufs_gpu[dest_rank] = cp.empty(count * total_stride + bla_extra, dtype=float32)
# CPU staging (non-GPU-aware path only)
if not gpu_aware_mpi:
    send_bufs_cpu[dest_rank] = np.empty(same_size, dtype=float32)

# Recv buffers — GPU if GPU-aware, else CPU
if gpu_aware_mpi:
    recv_bufs_gpu[src_rank] = cp.empty(size, dtype=float32)
else:
    recv_bufs_cpu[src_rank] = np.empty(size, dtype=float32)
```

**Question:** Should we pin/register these buffers (`hipHostMalloc`, `MPI_Mem_attach`) for better RDMA throughput? Currently using plain `cp.empty()` → `hipMalloc`.

**Question:** Our topology is static. Would persistent communication (`MPI_Send_init`/`MPI_Recv_init`) be faster than `Isend`/`Irecv` per tick?

### 3.6 Per-Tick Ghost Exchange

**Orchestrator:** `gpu_kernels.py:1068-1123` (`exchange_ghost_data`)

#### GPU Pack (`gpu_kernels.py:852-909`)

```python
# gpu_kernels.py:869-881 — For each visible property, ONE fancy-index gather
for prop_idx in visible_prop_indices:
    width = property_widths[prop_idx]
    # gpu_kernels.py:872 — ONE gather across ALL peers
    gathered = buf.property_tensors[prop_idx][all_indices].ravel()

    # gpu_kernels.py:874-881 — Split per-peer into contiguous send buffers
    for dest_rank in _send_peer_order:
        n = send_counts[dest_rank]
        send_bufs_gpu[dest_rank][off:off + n*width] = chunk
```

Per-peer buffer layout: `[prop0_agents, prop1_agents, ..., BLA_data]`

CPU staging (non-GPU-aware): `gpu_kernels.py:906-909`
```python
send_bufs_cpu[dest_rank][:] = send_bufs_gpu[dest_rank].get()  # GPU→CPU per peer
```

#### Stream Sync Before MPI (`gpu_kernels.py:1084-1087`)

```python
if self._gpu_aware_mpi and self._send_peer_order:
    cp.cuda.get_current_stream().synchronize()
```

GPU-aware MPI's RDMA engine reads GPU memory directly. Pack ops must complete first.

**Question:** Is `hipStreamSynchronize` the right call here? Should we use HIP events for finer-grained ordering?

**Question:** After `MPI_Waitall` with GPU-aware recv, can we immediately read `recv_bufs_gpu`, or is an explicit sync needed?

#### MPI Exchange (`gpu_kernels.py:911-982`)

**GPU-Aware Path** (`gpu_kernels.py:921-930`):
```python
for dest_rank in _send_peer_order:
    sbuf = send_bufs_gpu[dest_rank]  # CuPy array (HIP device memory)
    requests.append(comm.Isend([sbuf, MPI.FLOAT], dest=dest_rank, tag=1))

for src_rank in _recv_peer_order:
    rbuf = recv_bufs_gpu[src_rank]   # CuPy array (HIP device memory)
    requests.append(comm.Irecv([rbuf, MPI.FLOAT], source=src_rank, tag=1))

MPI.Request.Waitall(requests)  # gpu_kernels.py:941
```

Data path: GPU VRAM → Slingshot NIC (GPU-Direct RDMA) → GPU VRAM

CuPy→MPI bridge: mpi4py reads `__cuda_array_interface__` → gets device pointer + size.

**CPU Staging Path** (`gpu_kernels.py:932-940`):
```python
# Same structure but with numpy CPU buffers
comm.Isend([send_bufs_cpu[dest_rank], MPI.FLOAT], ...)
comm.Irecv([recv_bufs_cpu[src_rank], MPI.FLOAT], ...)
```

Data path: GPU → CPU `.get()` → NIC → CPU → GPU `cp.array()`

**Question:** Does Cray MPICH handle CuPy-ROCm device pointers via `__cuda_array_interface__` correctly? Or do we need a different interface for HIP?

**Question:** Are we getting actual GPU-Direct RDMA, or is Cray MPICH silently staging through CPU? How do we verify?

#### GPU Unpack (`gpu_kernels.py:984-1066`)

**GPU-aware path** (`gpu_kernels.py:1000-1001`): data already on GPU
```python
peer_recv_gpu = {r: recv_bufs_gpu[r] for r in _recv_peer_order}
```

**CPU-staged path** (`gpu_kernels.py:1003-1014`): single bulk upload
```python
combined = np.concatenate([recv_bufs_cpu[r] for r in _recv_peer_order])
all_recv_gpu = cp.array(combined)  # ONE CPU→GPU transfer
```

**Property scatter** (`gpu_kernels.py:1016-1030`):
```python
for prop_idx in visible_prop_indices:
    # Extract this property's chunk from each peer
    chunks = [peer_recv_gpu[r][prop_start:prop_start + n*width] for r in peers]
    prop_data = cp.concatenate(chunks)
    # gpu_kernels.py:1030 — ONE scatter into ghost slots
    buf.property_tensors[prop_idx][all_ghost_indices] = prop_data
```

**Write-buffer ghost sync** (`gpu_kernels.py:1032-1036`):
```python
for i in _write_prop_mask:
    buf.write_buffers[i][ghost_indices] = buf.property_tensors[prop_idx][ghost_indices]
```

### 3.7 Fused GPU Kernel with Grid Barriers

**Kernel generation:** `model.py:2010-2457` (`generate_gpu_func`)
**Kernel launch:** `model.py:1514-1528`

```python
@jit.rawkernel(device='cuda')   # model.py:2430 — CuPy-ROCm maps transparently
def stepfunc(global_tick, _seed, *globals, *properties, *write_buffers,
             sync_workers_every_n_ticks, num_local_agents,
             *priority_ranges, agent_ids, logical_ids,
             barrier_counter, num_blocks_param, *bla_args):

    thread_id = blockIdx.x * blockDim.x + threadIdx.x    # model.py:2446
    total_threads = gridDim.x * blockDim.x                # model.py:2447

    for tick in range(sync_workers_every_n_ticks):         # model.py:2450 — fused ticks
        # Persistent thread loop per priority
        agent_index = thread_id
        while agent_index < priority_0_count:              # model.py:2313
            _real_idx = int(agent_index) + int(priority_0_start)
            breed_id = a0[_real_idx]
            if breed_id == 0:
                step_func_0_double_buffer(tick, _real_idx, ...)
            agent_index += total_threads

        # SOFTWARE GRID BARRIER                            # model.py:1956-1974
        syncthreads()
        if threadIdx.x == 0:
            threadfence()                                  # jit_extensions.py:24
            atomic_add(barrier_counter, 0, 1)
            _target = (barrier_id + 1) * num_blocks_param
            while atomic_add(barrier_counter, 0, 0) < _target:
                pass
            threadfence()
        syncthreads()

        # Write-back at end of each tick                   # model.py:2374-2409
        while agent_index < num_local_agents:
            a2[agent_index] = write_a2[agent_index]
            agent_index += total_threads
```

**Occupancy constraint:** Must launch ≤ `max_blocks_per_sm × num_sms` blocks to avoid grid barrier deadlock. (`model.py:1448-1489`)

**Question:** Does `__threadfence()` provide full device-scope memory ordering on MI250X?

**Question:** On MI250X (1 GCD = 110 CUs), what's the correct max co-resident workgroups? Does CuPy's `multiProcessorCount` map correctly to CUs on HIP?

### 3.8 Communication Pattern Summary

**One-time setup (tick 1):**

| MPI Call | Location | Purpose |
|---|---|---|
| `Alltoall(int32[])` | `gpu_kernels.py:630` | Exchange request counts |
| `Isend/Irecv(int32[])` + `Waitall` | `gpu_kernels.py:632-651` | Exchange agent ID lists |

**Per-tick exchange:**

| Step | GPU-Aware | CPU Staging | Code |
|---|---|---|---|
| Pack | gather → `send_bufs_gpu` | same | `gpu_kernels.py:869-881` |
| Sync | `stream.synchronize()` | N/A | `gpu_kernels.py:1087` |
| Stage | N/A | `.get()` per peer | `gpu_kernels.py:906-909` |
| Send | `Isend(gpu_buf)` | `Isend(cpu_buf)` | `gpu_kernels.py:923-926` / `933-936` |
| Recv | `Irecv(gpu_buf)` | `Irecv(cpu_buf)` | `gpu_kernels.py:927-930` / `937-940` |
| Wait | `Waitall` | `Waitall` | `gpu_kernels.py:941` |
| Upload | N/A | `cp.array(concat)` | `gpu_kernels.py:1005-1006` |
| Unpack | scatter to ghosts | same | `gpu_kernels.py:1018-1030` |

Message format: one contiguous buffer per peer, buffer protocol (no pickle), `[buf, MPI.FLOAT]`.

### 3.9 MPI Best-Practice Questions

1. **Alltoall vs neighbor collectives:** Is `Alltoall` appropriate for sparse topologies, or should we use `MPI_Dist_graph_create` + `MPI_Neighbor_alltoallv`?
2. **Persistent communication:** Static topology → can we use `MPI_Send_init`/`MPI_Recv_init` for faster per-tick exchange?
3. **Buffer registration:** Should we pin/register GPU buffers for RDMA? Currently plain `hipMalloc` via `cp.empty()`.
4. **GPU-Direct RDMA verification:** How do we confirm actual GPU-Direct RDMA vs silent CPU staging by Cray MPICH?
5. **CuPy buffer protocol:** Does Cray MPICH correctly handle `__cuda_array_interface__` from CuPy-ROCm?
6. **Stream ordering:** Is `hipStreamSynchronize` before MPI the right sync? Should we use HIP events?
7. **Post-Waitall sync:** After GPU-aware `MPI_Waitall`, do we need explicit sync before reading recv buffers?
8. **Overlap opportunity:** Can we overlap GPU pack with MPI send using multiple HIP streams?
9. **Intra-node vs inter-node:** Does GPU-Direct RDMA work the same between GCDs on the same MI250X vs across Slingshot?
10. **threadfence scope:** Does `__threadfence()` on MI250X provide device-scope memory ordering sufficient for our grid barrier?
11. **CU occupancy:** Correct max co-resident workgroups on MI250X? Does CuPy `multiProcessorCount` = CUs on HIP?

---

## 4. Profiling Tools

### 4.1 Built-In Timing

Enable with `verbose_timing=True` on model constructor. Prints per-tick breakdown:

| Metric | What It Measures | Code |
|---|---|---|
| `data_prep` | Buffer init + ghost exchange | `model.py:1424-1427` |
| `gpu_compute` | Kernel execution (minus sync) | `model.py:1540` |
| `gpu_sync` | `hipStreamSynchronize` wait | `model.py:1534-1539` |
| `write_back` | GPU→GPU write-buffer copy | `model.py:1562-1572` |
| `mpi_gpu_pack` | Fancy-index gather time | `gpu_kernels.py:1076-1081` |
| `mpi_gpu_sync_pack` | Stream sync before MPI | `gpu_kernels.py:1085-1089` |
| `mpi_exchange` | Total MPI time | `gpu_kernels.py:1092-1097` |
| `mpi_isend_overhead` | Isend queueing time | `gpu_kernels.py:978-979` |
| `mpi_irecv_overhead` | Irecv queueing time | `gpu_kernels.py:980` |
| `mpi_wait_time` | Waitall (actual transfer) | `gpu_kernels.py:981` |
| `mpi_gpu_unpack` | Scatter into ghost slots | `gpu_kernels.py:1103-1107` |
| `mpi_send_bytes` | Total bytes sent | `gpu_kernels.py:1110-1113` |
| `mpi_recv_bytes` | Total bytes received | `gpu_kernels.py:1115-1118` |

### 4.2 Frontier Profiling Tools — Discussion

**What should we use?**

| Tool | Purpose | Question |
|---|---|---|
| **rocprof** | GPU kernel profiling, hardware counters | How to profile CuPy JIT kernels? |
| **omniperf** | MI250X-specific performance analysis | Best for identifying CU occupancy, memory bandwidth issues? |
| **omnitrace** | Timeline tracing (GPU + CPU + MPI) | Can it capture MPI + HIP in a single timeline? |
| **Cray PAT** | MPI profiling, load balancing | Best for MPI communication analysis? |
| **MPICH stats** | `MPICH_RANK_REORDER_METHOD`, env-based MPI stats | What env vars give us MPI timing without code changes? |

**Questions for experts:**
1. Which profiler gives us an integrated MPI + GPU timeline?
2. How do we profile CuPy JIT-compiled rawkernels with rocprof? (kernels are compiled at runtime via hipRTC)
3. Can omniperf identify whether our grid barrier is wasting cycles spinning?
4. How to verify GPU-Direct RDMA is active (not CPU-staged) from profiler output?
5. Best way to measure actual Slingshot bandwidth utilization?

---

## 5. Summary: Day 1 Outcomes

By end of Day 1, we aim to have:

- [ ] Shared understanding of our current Frontier environment (ROCm 6.2, Cray MPICH, CuPy 13.6) and a plan for whether/when to upgrade ROCm and CuPy to newer versions
- [ ] Plan for I/O: which format/tool for snapshots, checkpointing, and replacing the `allgather`/`Gatherv` bottlenecks
- [ ] Expert review of our MPI communication code — are we following best practices?
- [ ] Profiler chosen and first profiles collected
