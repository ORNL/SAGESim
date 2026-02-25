"""
GPU-resident data structures for SAGESim Phase 1.

Provides persistent GPU buffer management to eliminate per-tick CPU rebuild cycles.
"""

import numpy as np
import cupy as cp
from mpi4py import MPI


class GPUHashMap:
    """Open-addressing hash map stored on GPU.

    Maps agent_id (int64) -> buffer_index (int32).
    Uses linear probing with a -1 sentinel for empty slots.
    """

    EMPTY_KEY = np.int64(-1)
    DELETED_KEY = np.int64(-2)

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.keys = cp.full(capacity, self.EMPTY_KEY, dtype=cp.int64)
        self.values = cp.full(capacity, -1, dtype=cp.int32)
        self.size = 0
        # CPU mirror for fast CPU-side lookups (avoids GPU→CPU transfers)
        self._cpu_keys = np.full(capacity, self.EMPTY_KEY, dtype=np.int64)
        self._cpu_values = np.full(capacity, -1, dtype=np.int32)

    def build_from_arrays(self, agent_ids: np.ndarray, buffer_indices: np.ndarray):
        """Bulk-insert from CPU arrays. Resets existing contents.

        :param agent_ids: numpy int64 array of agent IDs
        :param buffer_indices: numpy int32 array of corresponding buffer indices
        """
        n = len(agent_ids)
        self._cpu_keys[:] = self.EMPTY_KEY
        self._cpu_values[:] = -1
        self.size = n

        for i in range(n):
            aid = np.int64(agent_ids[i])
            slot = int(aid % self.capacity)
            while self._cpu_keys[slot] != self.EMPTY_KEY and self._cpu_keys[slot] != self.DELETED_KEY:
                slot = (slot + 1) % self.capacity
            self._cpu_keys[slot] = aid
            self._cpu_values[slot] = np.int32(buffer_indices[i])

        self.keys = cp.array(self._cpu_keys)
        self.values = cp.array(self._cpu_values)

    def insert(self, agent_id: int, buffer_index: int):
        """Insert a single entry. Updates both CPU mirror and GPU arrays."""
        aid = np.int64(agent_id)
        slot = int(aid % self.capacity)
        while (self._cpu_keys[slot] != self.EMPTY_KEY
               and self._cpu_keys[slot] != self.DELETED_KEY
               and self._cpu_keys[slot] != aid):
            slot = (slot + 1) % self.capacity

        if self._cpu_keys[slot] != aid:
            self.size += 1
        self._cpu_keys[slot] = aid
        self._cpu_values[slot] = np.int32(buffer_index)
        # Single-element GPU update
        self.keys[slot] = aid
        self.values[slot] = np.int32(buffer_index)

    def remove(self, agent_id: int):
        """Mark entry as deleted."""
        aid = np.int64(agent_id)
        slot = int(aid % self.capacity)
        while self._cpu_keys[slot] != self.EMPTY_KEY:
            if self._cpu_keys[slot] == aid:
                self._cpu_keys[slot] = self.DELETED_KEY
                self._cpu_values[slot] = np.int32(-1)
                self.keys[slot] = self.DELETED_KEY
                self.values[slot] = np.int32(-1)
                self.size -= 1
                return
            slot = (slot + 1) % self.capacity

    def lookup_cpu(self, agent_id: int) -> int:
        """CPU-side lookup. Returns buffer_index or -1 if not found."""
        aid = np.int64(agent_id)
        slot = int(aid % self.capacity)
        while self._cpu_keys[slot] != self.EMPTY_KEY:
            if self._cpu_keys[slot] == aid:
                return int(self._cpu_values[slot])
            slot = (slot + 1) % self.capacity
        return -1

    def needs_resize(self) -> bool:
        """True when load factor exceeds 0.7."""
        return self.size > int(self.capacity * 0.7)

    def resize(self, new_capacity: int):
        """Reallocate and rehash at larger capacity."""
        old_keys = self._cpu_keys
        old_values = self._cpu_values
        old_capacity = self.capacity

        self.capacity = new_capacity
        self._cpu_keys = np.full(new_capacity, self.EMPTY_KEY, dtype=np.int64)
        self._cpu_values = np.full(new_capacity, -1, dtype=np.int32)
        self.size = 0

        for i in range(old_capacity):
            if old_keys[i] != self.EMPTY_KEY and old_keys[i] != self.DELETED_KEY:
                self._raw_insert(old_keys[i], old_values[i])

        # Upload to GPU
        self.keys = cp.array(self._cpu_keys)
        self.values = cp.array(self._cpu_values)

    def _raw_insert(self, aid, buf_idx):
        """Insert into CPU mirror only (used during resize)."""
        slot = int(np.int64(aid) % self.capacity)
        while (self._cpu_keys[slot] != self.EMPTY_KEY
               and self._cpu_keys[slot] != self.DELETED_KEY):
            slot = (slot + 1) % self.capacity
        self._cpu_keys[slot] = aid
        self._cpu_values[slot] = buf_idx
        self.size += 1

    def free(self):
        """Release GPU memory."""
        del self.keys
        del self.values
        self.keys = None
        self.values = None
        self._cpu_keys = None
        self._cpu_values = None


class GPUBufferManager:
    """Manages persistent GPU buffers for GPU-resident data.

    Allocated once, reused every tick. Eliminates per-tick cp.array() and .get() calls.
    Pre-allocates with slack to accommodate growth without reallocation.
    """

    AGENT_SLACK_FACTOR = 1.5
    CSR_SLACK_FACTOR = 2.0
    MIN_CAPACITY = 64

    def __init__(self):
        self.is_initialized = False

        # Persistent GPU arrays
        self.property_tensors = []      # List of CuPy arrays, one per property
        self.write_buffers = []         # List of CuPy arrays for double-buffered properties
        self.neighbor_offsets = None    # CuPy int32 (CSR offsets)
        self.neighbor_values = None     # CuPy int32 (CSR values, local indices for kernel)
        self.neighbor_values_ids = None # CuPy int32 (CSR values, agent IDs for MPI)
        self.agent_ids_gpu = None       # CuPy array of all agent IDs (local + ghost)
        self.global_data_vector = None  # CuPy array
        self.hash_map = None            # GPUHashMap instance

        # Capacity tracking
        self.agent_capacity = 0         # Max agents property tensors can hold
        self.csr_values_capacity = 0    # Max edges CSR values can hold
        self.num_local_agents = 0
        self.num_total_agents = 0       # local + ghost

        # CPU-side metadata
        self.agent_id_to_index = {}
        self.all_agent_ids_list = []
        self.prev_ghost_ids_set = set()
        self.sorted_write_indices = []

    def allocate_property_tensors(self, num_properties, combined_lists, agent_capacity,
                                  convert_to_equal_side_tensor_func):
        """Allocate property tensor GPU arrays with slack capacity.

        :param num_properties: Number of agent properties
        :param combined_lists: List of combined (local+ghost) property data
        :param agent_capacity: Pre-allocated capacity (>= num_total_agents)
        :param convert_to_equal_side_tensor_func: Function to convert ragged lists to GPU tensors
        """
        self.agent_capacity = agent_capacity
        self.property_tensors = []

        for i in range(num_properties):
            if i == 1:
                # Property 1 uses CSR, not a rectangular tensor
                self.property_tensors.append(None)
            else:
                tensor = convert_to_equal_side_tensor_func(combined_lists[i])
                # Pre-allocate with slack
                if tensor.ndim == 1:
                    padded = cp.zeros(agent_capacity, dtype=tensor.dtype)
                    padded[:len(tensor)] = tensor
                elif tensor.ndim == 2:
                    padded = cp.full((agent_capacity, tensor.shape[1]), cp.nan, dtype=tensor.dtype)
                    padded[:tensor.shape[0]] = tensor
                else:
                    # Higher dimensions: allocate with slack on first axis
                    padded_shape = (agent_capacity,) + tensor.shape[1:]
                    padded = cp.full(padded_shape, cp.nan, dtype=tensor.dtype)
                    padded[:tensor.shape[0]] = tensor
                self.property_tensors.append(padded)

    def allocate_write_buffers(self, sorted_write_indices):
        """Create write buffers as copies of property tensors."""
        self.sorted_write_indices = sorted_write_indices
        self.write_buffers = []
        for prop_idx in sorted_write_indices:
            write_buffer = self.property_tensors[prop_idx].copy()
            self.write_buffers.append(write_buffer)

    def allocate_csr(self, offsets_np, values_np, values_ids_np, num_total_agents):
        """Allocate CSR arrays on GPU with slack for values.

        :param offsets_np: numpy int32 offsets array (length num_total_agents + 1)
        :param values_np: numpy int32 values array (local indices)
        :param values_ids_np: numpy int32 values array (agent IDs)
        :param num_total_agents: Current total agent count
        """
        total_edges = len(values_np)
        self.csr_values_capacity = max(self.MIN_CAPACITY, int(total_edges * self.CSR_SLACK_FACTOR))

        # Offsets: size is num_agents + 1, pre-allocate with slack
        offsets_capacity = max(self.MIN_CAPACITY, int(num_total_agents * self.AGENT_SLACK_FACTOR)) + 1
        padded_offsets = np.zeros(offsets_capacity, dtype=np.int32)
        padded_offsets[:len(offsets_np)] = offsets_np
        self.neighbor_offsets = cp.array(padded_offsets)

        # Values (local indices): pre-allocate with slack
        padded_values = np.full(self.csr_values_capacity, -1, dtype=np.int32)
        padded_values[:total_edges] = values_np
        self.neighbor_values = cp.array(padded_values)

        # Values (agent IDs): pre-allocate with slack
        padded_values_ids = np.full(self.csr_values_capacity, -1, dtype=np.int32)
        padded_values_ids[:total_edges] = values_ids_np
        self.neighbor_values_ids = cp.array(padded_values_ids)

    def ensure_agent_capacity(self, needed):
        """Grow property tensors if needed > current capacity. Uses 2x doubling."""
        if needed <= self.agent_capacity:
            return

        new_capacity = max(needed, self.agent_capacity * 2)
        for i, tensor in enumerate(self.property_tensors):
            if tensor is None:
                continue
            if tensor.ndim == 1:
                new_tensor = cp.zeros(new_capacity, dtype=tensor.dtype)
                new_tensor[:self.agent_capacity] = tensor
            elif tensor.ndim == 2:
                new_tensor = cp.full((new_capacity, tensor.shape[1]), cp.nan, dtype=tensor.dtype)
                new_tensor[:self.agent_capacity] = tensor
            else:
                new_shape = (new_capacity,) + tensor.shape[1:]
                new_tensor = cp.full(new_shape, cp.nan, dtype=tensor.dtype)
                new_tensor[:self.agent_capacity] = tensor
            self.property_tensors[i] = new_tensor

        # Grow write buffers
        for j, prop_idx in enumerate(self.sorted_write_indices):
            old_buf = self.write_buffers[j]
            new_tensor = self.property_tensors[prop_idx].copy()
            new_tensor[:self.agent_capacity] = old_buf[:self.agent_capacity]
            self.write_buffers[j] = new_tensor

        # Grow agent_ids_gpu
        if self.agent_ids_gpu is not None:
            new_ids = cp.full(new_capacity, -1, dtype=self.agent_ids_gpu.dtype)
            new_ids[:self.agent_capacity] = self.agent_ids_gpu
            self.agent_ids_gpu = new_ids

        # Grow CSR offsets
        if self.neighbor_offsets is not None:
            old_len = len(self.neighbor_offsets)
            new_offsets = cp.zeros(new_capacity + 1, dtype=cp.int32)
            new_offsets[:old_len] = self.neighbor_offsets
            self.neighbor_offsets = new_offsets

        self.agent_capacity = new_capacity

    def ensure_csr_capacity(self, needed):
        """Grow CSR values arrays if needed > current capacity. Uses 2x doubling."""
        if needed <= self.csr_values_capacity:
            return

        new_capacity = max(needed, self.csr_values_capacity * 2)

        if self.neighbor_values is not None:
            new_vals = cp.full(new_capacity, -1, dtype=cp.int32)
            new_vals[:self.csr_values_capacity] = self.neighbor_values
            self.neighbor_values = new_vals

        if self.neighbor_values_ids is not None:
            new_vals_ids = cp.full(new_capacity, -1, dtype=cp.int32)
            new_vals_ids[:self.csr_values_capacity] = self.neighbor_values_ids
            self.neighbor_values_ids = new_vals_ids

        self.csr_values_capacity = new_capacity

    def free(self):
        """Release all GPU memory."""
        for tensor in self.property_tensors:
            if tensor is not None:
                del tensor
        self.property_tensors = []

        for buf in self.write_buffers:
            del buf
        self.write_buffers = []

        for attr in ('neighbor_offsets', 'neighbor_values', 'neighbor_values_ids',
                      'agent_ids_gpu', 'global_data_vector'):
            arr = getattr(self, attr, None)
            if arr is not None:
                del arr
                setattr(self, attr, None)

        if self.hash_map is not None:
            self.hash_map.free()
            self.hash_map = None

        self.is_initialized = False
        self.agent_capacity = 0
        self.csr_values_capacity = 0
        self.num_local_agents = 0
        self.num_total_agents = 0
        self.agent_id_to_index = {}
        self.all_agent_ids_list = []
        self.prev_ghost_ids_set = set()
        self.sorted_write_indices = []


class CommunicationManager:
    """Buffer-protocol MPI communication with GPU pack/unpack.

    Replaces the Python-loop-based contextualize_agent_data_tensors() for
    subsequent ticks. Pre-computes communication topology once, then each
    tick does: GPU pack -> buffer-protocol MPI Isend/Irecv -> GPU unpack.
    One contiguous message per peer rank, no pickle.
    """

    def __init__(self, buf, agent_factory, my_rank, num_workers, comm):
        self.buf = buf
        self.agent_factory = agent_factory
        self.my_rank = my_rank
        self.num_workers = num_workers
        self.comm = comm

        # Per-dest-rank: CuPy int32 arrays of local buffer indices to pack
        self.send_indices_gpu = {}
        self.send_counts = {}

        # Per-src-rank: CuPy int32 arrays of ghost buffer indices to scatter into
        self.recv_indices_gpu = {}
        self.recv_counts = {}

        # Property layout
        self.visible_prop_indices = []
        self.property_widths = {}
        self.total_stride = 0

        # Pre-allocated buffers (one per peer rank)
        self.send_bufs_cpu = {}
        self.recv_bufs_cpu = {}
        self.send_bufs_gpu = {}

        self._is_initialized = False

    @property
    def is_initialized(self):
        return self._is_initialized

    def build_communication_maps(self):
        """Build pre-computed communication topology from CSR on GPU.

        Runs once after _build_gpu_buffers() or when topology changes.
        Replaces the per-agent Python loop in contextualize_agent_data_tensors().
        """
        buf = self.buf
        num_local = buf.num_local_agents

        # 1. Build property layout (visible props, skip prop 1 which is CSR)
        if not self.agent_factory._neighbor_visible_indices:
            self.agent_factory._build_neighbor_visible_indices()

        self.visible_prop_indices = [
            idx for idx in self.agent_factory._neighbor_visible_indices
            if idx != 1
        ]
        self.property_widths = {}
        self.total_stride = 0
        for prop_idx in self.visible_prop_indices:
            tensor = buf.property_tensors[prop_idx]
            if tensor is None:
                continue
            width = 1 if tensor.ndim == 1 else tensor.shape[1]
            self.property_widths[prop_idx] = width
            self.total_stride += width

        if self.total_stride == 0 or num_local == 0:
            self._is_initialized = True
            return

        # 2. Download CSR for local agents from GPU
        cpu_offsets = buf.neighbor_offsets[:num_local + 1].get()
        cpu_offsets = np.asarray(cpu_offsets, dtype=np.int32)
        total_edges_local = int(cpu_offsets[num_local])

        if total_edges_local == 0:
            self._is_initialized = True
            return

        cpu_values_ids = buf.neighbor_values_ids[:total_edges_local].get()
        cpu_values_ids = np.asarray(cpu_values_ids, dtype=np.int32)

        # 3. Vectorized: expand (local_agent_idx, neighbor_id) pairs from CSR
        counts = np.diff(cpu_offsets)
        local_agent_indices = np.repeat(np.arange(num_local, dtype=np.int32), counts)

        # Filter invalid neighbor IDs (padding: -1)
        valid_mask = cpu_values_ids >= 0
        local_agent_indices = local_agent_indices[valid_mask]
        neighbor_ids = cpu_values_ids[valid_mask]

        if len(neighbor_ids) == 0:
            self._is_initialized = True
            return

        # Build dense rank lookup: rank_lookup[agent_id] = rank
        agent2rank = self.agent_factory._agent2rank
        max_agent_id = max(agent2rank.keys())
        rank_lookup = np.full(max_agent_id + 1, -1, dtype=np.int32)
        for aid, r in agent2rank.items():
            rank_lookup[aid] = r

        # Vectorized rank lookup for all neighbors
        in_range = neighbor_ids <= max_agent_id
        neighbor_ranks = np.full(len(neighbor_ids), -1, dtype=np.int32)
        neighbor_ranks[in_range] = rank_lookup[neighbor_ids[in_range]]

        # Filter cross-rank edges
        cross_mask = (neighbor_ranks != self.my_rank) & (neighbor_ranks >= 0)
        cross_local_indices = local_agent_indices[cross_mask]
        cross_neighbor_ranks = neighbor_ranks[cross_mask]

        if len(cross_local_indices) == 0:
            self._is_initialized = True
            return

        # Group by dest rank, unique local agent indices per rank
        dest_ranks_unique = np.unique(cross_neighbor_ranks)
        send_agent_ids_per_rank = {}

        self.send_indices_gpu = {}
        self.send_counts = {}

        for dest_rank in dest_ranks_unique:
            dest_rank = int(dest_rank)
            mask = cross_neighbor_ranks == dest_rank
            unique_local_indices = np.unique(cross_local_indices[mask])
            self.send_indices_gpu[dest_rank] = cp.array(unique_local_indices, dtype=cp.int32)
            self.send_counts[dest_rank] = len(unique_local_indices)
            send_agent_ids_per_rank[dest_rank] = np.array(
                [int(buf.all_agent_ids_list[idx]) for idx in unique_local_indices],
                dtype=np.int32,
            )

        # 4. Exchange counts via Alltoall
        send_counts_array = np.zeros(self.num_workers, dtype=np.int32)
        for rank, count in self.send_counts.items():
            send_counts_array[rank] = count
        recv_counts_array = np.zeros(self.num_workers, dtype=np.int32)
        self.comm.Alltoall(send_counts_array, recv_counts_array)

        self.recv_counts = {}
        for rank in range(self.num_workers):
            if rank != self.my_rank and recv_counts_array[rank] > 0:
                self.recv_counts[rank] = int(recv_counts_array[rank])

        # 5. Exchange agent IDs so each rank knows which agents it will receive
        send_id_requests = []
        for dest_rank, agent_ids_np in send_agent_ids_per_rank.items():
            req = self.comm.Isend([agent_ids_np, MPI.INT], dest=dest_rank, tag=100)
            send_id_requests.append(req)

        recv_agent_ids_per_rank = {}
        recv_id_requests = []
        for src_rank, count in self.recv_counts.items():
            recv_buf = np.empty(count, dtype=np.int32)
            recv_agent_ids_per_rank[src_rank] = recv_buf
            req = self.comm.Irecv([recv_buf, MPI.INT], source=src_rank, tag=100)
            recv_id_requests.append(req)

        MPI.Request.Waitall(send_id_requests + recv_id_requests)

        # 6. Map received agent IDs to ghost buffer indices
        self.recv_indices_gpu = {}
        for src_rank, agent_ids_np in recv_agent_ids_per_rank.items():
            ghost_indices = np.array(
                [buf.agent_id_to_index[int(aid)] for aid in agent_ids_np],
                dtype=np.int32,
            )
            self.recv_indices_gpu[src_rank] = cp.array(ghost_indices, dtype=cp.int32)

        # 7. Pre-allocate send/recv CPU and GPU buffers
        self.send_bufs_cpu = {}
        self.send_bufs_gpu = {}
        self.recv_bufs_cpu = {}

        for dest_rank, count in self.send_counts.items():
            size = count * self.total_stride
            self.send_bufs_cpu[dest_rank] = np.empty(size, dtype=np.float32)
            self.send_bufs_gpu[dest_rank] = cp.empty(size, dtype=cp.float32)

        for src_rank, count in self.recv_counts.items():
            size = count * self.total_stride
            self.recv_bufs_cpu[src_rank] = np.empty(size, dtype=np.float32)

        self._is_initialized = True

    def gpu_pack(self, dest_rank):
        """Pack local agent data into contiguous send buffer on GPU."""
        indices = self.send_indices_gpu[dest_rank]
        n = len(indices)
        buf_gpu = self.send_bufs_gpu[dest_rank]
        offset = 0
        for prop_idx in self.visible_prop_indices:
            width = self.property_widths[prop_idx]
            data = self.buf.property_tensors[prop_idx][indices]  # GPU fancy indexing
            if width == 1:
                buf_gpu[offset:offset + n] = data
            else:
                buf_gpu[offset:offset + n * width] = data.ravel()
            offset += n * width
        # ONE GPU->CPU transfer
        self.send_bufs_cpu[dest_rank][:] = buf_gpu.get()

    def mpi_exchange(self):
        """Non-blocking buffer-protocol MPI send/recv (no pickle)."""
        requests = []
        for dest_rank, sbuf in self.send_bufs_cpu.items():
            requests.append(self.comm.Isend([sbuf, MPI.FLOAT], dest=dest_rank, tag=1))
        for src_rank, rbuf in self.recv_bufs_cpu.items():
            requests.append(self.comm.Irecv([rbuf, MPI.FLOAT], source=src_rank, tag=1))
        MPI.Request.Waitall(requests)

    def gpu_unpack(self, src_rank):
        """Unpack received CPU buffer into ghost region on GPU."""
        recv_gpu = cp.array(self.recv_bufs_cpu[src_rank])  # ONE CPU->GPU
        ghost_indices = self.recv_indices_gpu[src_rank]
        n = len(ghost_indices)
        offset = 0
        for prop_idx in self.visible_prop_indices:
            width = self.property_widths[prop_idx]
            if width == 1:
                data = recv_gpu[offset:offset + n]
            else:
                data = recv_gpu[offset:offset + n * width].reshape(n, width)
            self.buf.property_tensors[prop_idx][ghost_indices] = data  # GPU scatter
            offset += n * width

    def exchange_ghost_data(self):
        """Full ghost exchange: GPU pack -> MPI -> GPU unpack -> sync write buffers."""
        if not self._is_initialized:
            return

        # Pack all destinations
        for dest_rank in self.send_counts:
            self.gpu_pack(dest_rank)

        # MPI exchange
        if self.send_bufs_cpu or self.recv_bufs_cpu:
            self.mpi_exchange()

        # Unpack all sources
        for src_rank in self.recv_counts:
            self.gpu_unpack(src_rank)

        # Update write buffers for ghost region
        for i, prop_idx in enumerate(self.buf.sorted_write_indices):
            if prop_idx in self.property_widths:
                for src_rank in self.recv_indices_gpu:
                    ghost_indices = self.recv_indices_gpu[src_rank]
                    self.buf.write_buffers[i][ghost_indices] = \
                        self.buf.property_tensors[prop_idx][ghost_indices]
