# SAGESim GPU Optimization Roadmap

This document outlines the technical implementation plan for optimizing SAGESim's performance on HPC systems, with a focus on GPU-resident data structures and GPU-direct MPI communication.

## Current Architecture Bottlenecks

```
Current Flow (every tick):
GPU → CPU → MPI → CPU → GPU
     ↑         ↑       ↑
   Download   Network  Upload
   + unpad    + pack   + repad
```

**Time breakdown (typical):**
- Data preparation (padding, ID conversion): ~40% of tick time
- GPU↔CPU transfers: ~30% of tick time
- MPI communication: ~20% of tick time
- GPU kernel execution: ~10% of tick time

The majority of time is spent on CPU-side data manipulation, not computation.

---

## Target Architecture

```
Proposed Flow:
GPU ──────────────────────► GPU
         GPU-Direct RDMA
         (no CPU involvement)
```

**Key principles:**
1. Data stays on GPU between ticks
2. GPU-aware MPI reads/writes GPU buffers directly
3. Pre-allocated buffers eliminate per-tick memory allocation
4. GPU kernels handle packing, unpacking, and index conversion

---

## Implementation Phases

### Phase 1: GPU-Resident Data Structures

**Goal:** Eliminate per-tick array rebuilding and padding overhead.

**Changes:**

1. Pre-allocate maximum-sized GPU buffers at simulation initialization:
```python
class GPUResidentAgentData:
    def __init__(self, max_agents, max_neighbors, num_properties):
        # Pre-allocate once, reuse every tick
        self.properties = [
            cp.full((max_agents, max_neighbors), cp.nan, dtype=cp.float32)
            for _ in range(num_properties)
        ]
        self.num_local_agents = 0
        self.num_total_agents = 0  # local + neighbors
```

2. Update data in-place rather than rebuilding:
```python
def update_agent_property(self, agent_idx, prop_idx, values):
    """Update single agent's property in-place on GPU."""
    n = len(values)
    self.properties[prop_idx][agent_idx, :n] = values
    self.properties[prop_idx][agent_idx, n:] = cp.nan
```

3. Track which agents changed to minimize updates:
```python
self.dirty_agents = set()  # Agents modified this tick
```

**Files to modify:**
- `sagesim/model.py`: Replace per-tick array creation with persistent buffers
- `sagesim/agent.py`: Update contextualization to work with GPU-resident data

**Estimated improvement:** 2-5x (eliminates padding overhead)

---

### Phase 2: GPU-Side Kernels for Data Transformation

**Goal:** Move ID↔index conversion and data packing from CPU to GPU.

**New kernels to implement:**

1. **ID to Index Conversion Kernel:**
```cuda
extern "C" __global__
void id_to_index(
    const int* lookup_table,     // ID → Index mapping (dense array)
    const int lookup_offset,     // Minimum agent ID
    const int lookup_size,
    float* locations,            // In/out: neighbor IDs → indices
    const int num_agents,
    const int max_neighbors
) {
    int agent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (agent_idx >= num_agents) return;

    for (int n = 0; n < max_neighbors; n++) {
        float val = locations[agent_idx * max_neighbors + n];
        if (isnan(val)) break;

        int neighbor_id = (int)val;
        int table_idx = neighbor_id - lookup_offset;

        int local_idx = -1;
        if (table_idx >= 0 && table_idx < lookup_size) {
            local_idx = lookup_table[table_idx];
        }
        locations[agent_idx * max_neighbors + n] = (float)local_idx;
    }
}
```

2. **Index to ID Conversion Kernel** (reverse operation for post-processing)

3. **Pack Send Buffer Kernel:**
```cuda
extern "C" __global__
void pack_send_buffer(
    const float* property_data,
    const int* agent_indices,      // Which agents to send
    const int num_agents_to_send,
    const int max_neighbors,
    float* send_buffer
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_agents_to_send) return;

    int src_agent = agent_indices[i];
    for (int n = 0; n < max_neighbors; n++) {
        send_buffer[i * max_neighbors + n] =
            property_data[src_agent * max_neighbors + n];
    }
}
```

4. **Unpack Receive Buffer Kernel** (insert received data into main arrays)

**Files to create:**
- `sagesim/gpu_kernels.py`: CuPy RawKernel implementations

**Files to modify:**
- `sagesim/model.py`: Replace `convert_agent_ids_to_indices()` with GPU kernel
- `sagesim/internal_utils.py`: Add GPU-accelerated versions of utility functions

**Estimated improvement:** 1.5-2x (eliminates CPU conversion loops)

---

### Phase 3: GPU-Aware MPI Communication

**Goal:** Enable direct GPU-to-GPU data transfer without CPU staging.

**Implementation:**

1. **GPU-Aware MPI Detection:**
```python
def detect_gpu_aware_mpi():
    """Check if MPI implementation supports GPU buffers."""
    # Method 1: Environment variable (Cray MPICH on Frontier)
    if os.environ.get('MPICH_GPU_SUPPORT_ENABLED') == '1':
        return True

    # Method 2: OpenMPI query
    try:
        from mpi4py import MPI
        info = MPI.Info.Create()
        # OpenMPI sets this attribute
        return MPI.Query_thread() >= 0  # Simplified check
    except:
        return False

    return False
```

2. **GPU-Direct Send/Receive:**
```python
class GPUDirectMPIExchange:
    def __init__(self, comm, gpu_aware=True):
        self.comm = comm
        self.gpu_aware = gpu_aware
        self.send_buffers = {}  # Pre-allocated GPU buffers per rank
        self.recv_buffers = {}

    def exchange(self, local_data_gpu, agents_to_send):
        if self.gpu_aware:
            # Direct GPU buffer to MPI (no CPU copy)
            return self._exchange_gpu_direct(local_data_gpu, agents_to_send)
        else:
            # Fallback: copy to CPU, send, copy back
            return self._exchange_cpu_staging(local_data_gpu, agents_to_send)

    def _exchange_gpu_direct(self, data, agents_to_send):
        requests = []
        for dest_rank, agent_indices in agents_to_send.items():
            send_buf = self.send_buffers[dest_rank]
            # Pack on GPU (kernel call)
            pack_kernel(data, agent_indices, send_buf)
            # MPI reads directly from GPU memory
            req = self.comm.Isend(send_buf, dest=dest_rank)
            requests.append(req)

        for src_rank in self.recv_buffers:
            recv_buf = self.recv_buffers[src_rank]
            # MPI writes directly to GPU memory
            req = self.comm.Irecv(recv_buf, source=src_rank)
            requests.append(req)

        MPI.Request.Waitall(requests)
```

3. **Pre-allocated Communication Buffers:**
```python
def setup_communication_buffers(self, neighbor_ranks, max_agents_per_rank):
    """Pre-allocate GPU buffers for each neighbor rank."""
    for rank in neighbor_ranks:
        self.send_buffers[rank] = cp.empty(
            (max_agents_per_rank, self.max_neighbors, self.num_properties),
            dtype=cp.float32
        )
        self.recv_buffers[rank] = cp.empty_like(self.send_buffers[rank])
```

**Files to create:**
- `sagesim/gpu_mpi.py`: GPU-aware MPI wrapper class

**Files to modify:**
- `sagesim/agent.py`: Replace `contextualize_agent_data_tensors()` with GPU-direct version
- `sagesim/model.py`: Initialize GPU-MPI infrastructure

**Estimated improvement:** 2-10x for communication (depends on network topology)

---

### Phase 4: RCCL/NCCL Integration for Collectives

**Goal:** Use GPU-native collective operations for global reductions.

**Implementation:**

1. **Collective Communicator Wrapper:**
```python
class GPUCollectives:
    def __init__(self, mpi_comm):
        self.mpi_comm = mpi_comm
        rank = mpi_comm.Get_rank()
        size = mpi_comm.Get_size()

        # Initialize RCCL (AMD) or NCCL (NVIDIA)
        if self._is_amd_gpu():
            import rccl
            self.backend = 'rccl'
            uid = rccl.get_unique_id() if rank == 0 else None
            uid = mpi_comm.bcast(uid, root=0)
            self.comm = rccl.Communicator(size, uid, rank)
        else:
            import nccl
            self.backend = 'nccl'
            uid = nccl.get_unique_id() if rank == 0 else None
            uid = mpi_comm.bcast(uid, root=0)
            self.comm = nccl.Communicator(size, uid, rank)

    def allreduce(self, sendbuf, recvbuf, op='max'):
        """GPU-native allreduce."""
        if self.backend == 'rccl':
            import rccl
            self.comm.allReduce(sendbuf.data.ptr, recvbuf.data.ptr,
                               sendbuf.size, rccl.FLOAT, rccl.MAX)
        else:
            import nccl
            self.comm.allReduce(sendbuf.data.ptr, recvbuf.data.ptr,
                               sendbuf.size, nccl.FLOAT, nccl.MAX)
        cp.cuda.Stream.null.synchronize()
```

2. **Replace MPI allreduce for global data:**
```python
# Current (CPU-based)
self._global_data_vector = comm.allreduce(global_data.tolist(), op=reduce_func)

# Proposed (GPU-native)
self.gpu_collectives.allreduce(
    self._global_data_vector_gpu,
    self._global_data_vector_gpu,
    op='max'
)
```

**Files to create:**
- `sagesim/gpu_collectives.py`: RCCL/NCCL wrapper

**Files to modify:**
- `sagesim/model.py`: Replace `comm.allreduce()` with GPU collective

**Estimated improvement:** 1.2-2x for collective operations

---

## Portable GPU Backend

**Goal:** Support both NVIDIA and AMD GPUs with single codebase.

```python
# sagesim/gpu_backend.py
import os

class GPUBackend:
    def __init__(self):
        self.backend = self._detect_backend()
        self._configure()

    def _detect_backend(self):
        backend = os.environ.get('SAGESIM_GPU_BACKEND', 'auto').lower()

        if backend == 'auto':
            try:
                import cupy as cp
                device_name = cp.cuda.runtime.getDeviceProperties(0)['name']
                if b'MI' in device_name or b'AMD' in device_name:
                    return 'rocm'
            except:
                pass
            return 'cuda'

        return backend

    def _configure(self):
        if self.backend == 'rocm':
            os.environ.setdefault('HIP_VISIBLE_DEVICES',
                                  os.environ.get('ROCR_VISIBLE_DEVICES', '0'))

        # Import cupy (works for both CUDA and ROCm)
        import cupy as cp
        self.cp = cp

    @property
    def is_rocm(self):
        return self.backend == 'rocm'

    @property
    def is_cuda(self):
        return self.backend == 'cuda'
```

---

## Testing Strategy

### Unit Tests
- `test_gpu_kernels.py`: Test ID conversion and pack/unpack kernels
- `test_gpu_mpi.py`: Test GPU-aware MPI with mock communicator
- `test_gpu_collectives.py`: Test RCCL/NCCL wrapper

### Integration Tests
- `test_gpu_resident_simulation.py`: Full simulation with GPU-resident data
- `test_weak_scaling.py`: Measure scaling efficiency across node counts

### Platform Tests
- Frontier (AMD MI250X): Full test suite
- Summit/Perlmutter (NVIDIA): Compatibility tests

---

## Frontier-Specific Configuration

```bash
#!/bin/bash
#SBATCH -A <project>
#SBATCH -J sagesim_test
#SBATCH -o %x-%j.out
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# Enable GPU-aware MPI on Frontier
export MPICH_GPU_SUPPORT_ENABLED=1

# Bind each rank to one GCD
export HIP_VISIBLE_DEVICES=$SLURM_LOCALID

# Optional: Enable GPU-aware MPI debugging
export MPICH_GPU_DEBUG=1

# Run with 16 ranks (8 per node, 1 per GCD)
srun -n 16 --gpus-per-task=1 python -m sagesim.run simulation_config.yaml
```

---

## Expected Performance Improvements

| Component | Current | After Phase 1 | After Phase 2 | After Phase 3 | After Phase 4 |
|-----------|---------|---------------|---------------|---------------|---------------|
| Data prep (padding) | 40% | 5% | 2% | 2% | 2% |
| GPU↔CPU transfer | 30% | 10% | 5% | 0% | 0% |
| MPI communication | 20% | 20% | 20% | 15% | 12% |
| GPU kernel | 10% | 65% | 73% | 83% | 86% |

**Overall tick time reduction:** 3-10x depending on agent count and network topology.

---

## Dependencies

### Required
- CuPy >= 12.0 (supports both CUDA and ROCm)
- mpi4py >= 3.1
- NumPy >= 1.20

### Optional (for full optimization)
- ROCm >= 5.0 (Frontier) or CUDA >= 11.0 (NVIDIA systems)
- RCCL (AMD) or NCCL (NVIDIA) for GPU collectives
- GPU-aware MPI (Cray MPICH, OpenMPI with CUDA/ROCm support, MVAPICH2-GDR)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| GPU-aware MPI not available | Automatic fallback to CPU staging |
| RCCL/NCCL not installed | Use MPI allreduce with GPU→CPU→GPU path |
| Memory fragmentation | Pre-allocate all buffers at startup |
| Variable agent counts | Size buffers for maximum expected agents |
| Platform differences | Abstraction layer with runtime detection |
