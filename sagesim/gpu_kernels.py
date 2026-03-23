"""
GPU-resident data structures for SAGESim Phase 1.

Provides persistent GPU buffer management to eliminate per-tick CPU rebuild cycles.
"""

import os

import numpy as np
import cupy as cp
from mpi4py import MPI


def is_gpu_aware_mpi():
    """Detect if MPI library supports GPU-aware communication.

    Checks environment variables set by GPU-aware MPI implementations:
    - MPICH_GPU_SUPPORT_ENABLED=1 (Cray MPICH on Frontier / AMD MI250X)
    - OMPI_MCA_opal_cuda_support=true (Open MPI with CUDA/ROCm support)
    - SAGESIM_GPU_AWARE_MPI=1 (manual override for testing)
    """
    if os.environ.get('MPICH_GPU_SUPPORT_ENABLED', '0') == '1':
        return True
    if os.environ.get('OMPI_MCA_opal_cuda_support', '') == 'true':
        return True
    if os.environ.get('SAGESIM_GPU_AWARE_MPI', '0') == '1':
        return True
    return False


def discover_ghost_topology(all_neighbors, agent2rank, my_rank):
    """Discover ghost agent IDs from local neighbor lists (CPU-only, no MPI).

    Vectorized scan of neighbor arrays to identify agents belonging to other
    ranks.  Returns the unique set of ghost IDs.  Actual data for these ghosts
    is filled later by CommunicationManager.exchange_ghost_data().

    :param all_neighbors: Ragged list of neighbor arrays (one per local agent)
    :param agent2rank: Dict mapping agent_id -> rank
    :param my_rank: This rank's ID
    :return: Sorted list of unique ghost agent IDs
    """
    if not all_neighbors:
        return np.array([], dtype=np.int64)

    # Flatten ragged neighbor lists into one array
    if isinstance(all_neighbors[0], np.ndarray):
        flat = np.concatenate(all_neighbors)
    else:
        flat = np.array(
            [nid for sublist in all_neighbors for nid in sublist],
            dtype=np.float64,
        )

    if len(flat) == 0:
        return np.array([], dtype=np.int64)

    # Filter invalid entries (NaN padding, negative sentinels)
    valid_mask = ~np.isnan(flat) & (flat >= 0)
    neighbor_ids = flat[valid_mask].astype(np.int64)

    if len(neighbor_ids) == 0:
        return np.array([], dtype=np.int64)

    # Build dense rank lookup array for vectorized rank resolution
    if isinstance(agent2rank, np.ndarray):
        rank_lookup = agent2rank.astype(np.int32, copy=False)
        max_agent_id = len(rank_lookup) - 1
    else:
        max_agent_id = max(agent2rank.keys())
        rank_lookup = np.full(max_agent_id + 1, -1, dtype=np.int32)
        for aid, r in agent2rank.items():
            rank_lookup[aid] = r

    # Vectorized rank lookup
    in_range = neighbor_ids <= max_agent_id
    neighbor_ranks = np.full(len(neighbor_ids), -1, dtype=np.int32)
    neighbor_ranks[in_range] = rank_lookup[neighbor_ids[in_range]]

    # Keep only cross-rank neighbors
    cross_mask = (neighbor_ranks != my_rank) & (neighbor_ranks >= 0)
    ghost_ids = np.unique(neighbor_ids[cross_mask])

    return ghost_ids


class GPUHashMap:
    """Open-addressing hash map stored on GPU.

    Maps agent_id (int64) -> buffer_index (int32).
    Uses linear probing with a -1 sentinel for empty slots.
    """

    EMPTY_KEY = np.int64(-1)
    DELETED_KEY = np.int64(-2)
    _HASH_PRIME = np.intp(2654435761)  # Knuth's multiplicative constant

    def __init__(self, capacity: int):
        self.capacity = capacity
        # Allocate GPU arrays immediately (prevents device context issues)
        self.keys = cp.full(capacity, self.EMPTY_KEY, dtype=cp.int64)
        self.values = cp.full(capacity, -1, dtype=cp.int32)
        self.size = 0
        # CPU mirror for fast CPU-side lookups (avoids GPU→CPU transfers)
        self._cpu_keys = np.full(capacity, self.EMPTY_KEY, dtype=np.int64)
        self._cpu_values = np.full(capacity, -1, dtype=np.int32)

    def _hash_slot(self, aid):
        """Multiplicative hash for better slot distribution."""
        return int(np.intp(aid) * self._HASH_PRIME % self.capacity)

    def _ensure_gpu(self):
        """Upload CPU mirror to GPU if not yet uploaded."""
        if self.keys is None:
            self.keys = cp.array(self._cpu_keys)
            self.values = cp.array(self._cpu_values)

    def build_from_arrays(self, agent_ids: np.ndarray, buffer_indices: np.ndarray):
        """Bulk-insert from CPU arrays. Resets existing contents.

        :param agent_ids: numpy int64 array of agent IDs
        :param buffer_indices: numpy int32 array of corresponding buffer indices
        """
        n = len(agent_ids)
        capacity = self.capacity
        EMPTY = self.EMPTY_KEY

        self._cpu_keys[:] = EMPTY
        self._cpu_values[:] = -1
        self.size = n

        if n == 0:
            self.keys = cp.full(capacity, EMPTY, dtype=cp.int64)
            self.values = cp.full(capacity, -1, dtype=np.int32)
            return

        aids = agent_ids.astype(np.int64, copy=False)
        bidx = buffer_indices.astype(np.int32, copy=False)

        # Track remaining (unplaced) agent indices and their current probe slots
        rem_idx = np.arange(n, dtype=np.intp)
        rem_slots = ((aids.astype(np.intp) * GPUHashMap._HASH_PRIME) % capacity).astype(np.intp)

        while len(rem_idx) > 0:
            # Find first (lowest original index) agent per unique target slot
            unique_slots, first_occ = np.unique(rem_slots, return_index=True)

            # Which of those slots are empty?
            empty_mask = self._cpu_keys[unique_slots] == EMPTY

            # Place winners at empty slots
            place_pos = first_occ[empty_mask]
            place_slots = unique_slots[empty_mask]
            self._cpu_keys[place_slots] = aids[rem_idx[place_pos]]
            self._cpu_values[place_slots] = bidx[rem_idx[place_pos]]

            # Remove placed agents from remaining
            keep_mask = np.ones(len(rem_idx), dtype=np.bool_)
            keep_mask[place_pos] = False
            rem_idx = rem_idx[keep_mask]
            rem_slots = rem_slots[keep_mask]

            # Advance probe slot for all still-remaining agents
            if len(rem_idx) > 0:
                rem_slots = (rem_slots + 1) % capacity

        # Upload to GPU (create new arrays with correct size)
        self.keys = cp.array(self._cpu_keys)
        self.values = cp.array(self._cpu_values)

    def insert(self, agent_id: int, buffer_index: int):
        """Insert a single entry. Updates both CPU mirror and GPU arrays."""
        self._ensure_gpu()
        aid = np.int64(agent_id)
        slot = self._hash_slot(aid)
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
        self._ensure_gpu()
        aid = np.int64(agent_id)
        slot = self._hash_slot(aid)
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
        slot = self._hash_slot(aid)
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
        slot = self._hash_slot(np.int64(aid))
        while (self._cpu_keys[slot] != self.EMPTY_KEY
               and self._cpu_keys[slot] != self.DELETED_KEY):
            slot = (slot + 1) % self.capacity
        self._cpu_keys[slot] = aid
        self._cpu_values[slot] = buf_idx
        self.size += 1

    def free(self):
        """Release GPU memory."""
        if self.keys is not None:
            del self.keys
        if self.values is not None:
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
        self.device_globals = []  # list of CuPy arrays, one per registered global
        self.device_breed_locals = []   # list of CuPy arrays, one per breed-local array
        self.device_breed_local_idxs = []  # list of CuPy int32 index maps
        self.seed_gpu = None            # CuPy int32 scalar for framework-managed seed
        self.hash_map = None            # GPUHashMap instance
        self.barrier_counter = None     # CuPy uint32[1] for fused-kernel grid barrier

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
        self.priority_ranges = {}  # priority_idx -> (start, count)
        self.breed_ranges = {}     # breed_id -> (start, count)

    def allocate_property_tensors(self, num_properties, combined_lists, agent_capacity,
                                  convert_to_padded_func):
        """Allocate property tensor GPU arrays with slack capacity.

        :param num_properties: Number of agent properties
        :param combined_lists: List of combined (local+ghost) property data
        :param agent_capacity: Pre-allocated capacity (>= num_total_agents)
        :param convert_to_padded_func: Function to convert ragged lists directly to padded GPU tensors
        """
        self.agent_capacity = agent_capacity
        self.property_tensors = []

        for i in range(num_properties):
            if i == 1:
                # Property 1 uses CSR, not a rectangular tensor
                self.property_tensors.append(None)
            else:
                padded = convert_to_padded_func(combined_lists[i], agent_capacity)
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

        for buf in self.device_globals:
            del buf
        self.device_globals = []

        for attr in ('neighbor_offsets', 'neighbor_values', 'neighbor_values_ids',
                      'agent_ids_gpu', 'barrier_counter'):
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
        self.priority_ranges = {}
        self.breed_ranges = {}


class CommunicationManager:
    """Buffer-protocol MPI communication with GPU pack/unpack.

    Pre-computes communication topology once, then each tick does:
    GPU pack -> buffer-protocol MPI Isend/Irecv -> GPU unpack.
    One contiguous message per peer rank, no pickle.
    """

    def __init__(self, buf, agent_factory, my_rank, num_workers, comm, verbose_timing=False):
        self.buf = buf
        self.agent_factory = agent_factory
        self.my_rank = my_rank
        self.num_workers = num_workers
        self.comm = comm
        self._verbose_timing = verbose_timing

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
        self.recv_bufs_gpu = {}  # GPU recv buffers for GPU-aware MPI path

        # Batched index arrays (concatenated across all peers)
        self._batched_send_indices_gpu = None  # CuPy int32, all send indices
        self._batched_recv_indices_gpu = None  # CuPy int32, all recv indices
        self._send_peer_order = []             # ordered list of dest ranks
        self._recv_peer_order = []             # ordered list of src ranks
        self._send_splits = []                 # cumulative split points for send
        self._recv_splits = []                 # cumulative split points for recv

        # Write-buffer ghost index (concatenated recv indices for write props)
        self._write_prop_mask = []  # indices into buf.sorted_write_indices that are visible

        self._gpu_aware_mpi = is_gpu_aware_mpi()
        self._is_initialized = False

    @property
    def is_initialized(self):
        return self._is_initialized

    def build_communication_maps(self):
        """Build pre-computed communication topology from CSR on GPU.

        Runs once after _build_gpu_buffers() or when topology changes.
        Scans CSR neighbor data to find cross-rank dependencies.

        Communication direction: when local agent A has neighbor B on a remote
        rank, A needs B's data.  We build a "need_from" map (which remote agent
        IDs this rank needs from each other rank), exchange requests via
        Alltoall + Isend/Irecv, then each rank builds send/recv index maps so
        that gpu_pack sends the *requested* agents and gpu_unpack writes them
        into the correct ghost slots.
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

        # Use a flag instead of early returns so all ranks reach the
        # collective Alltoall / Isend / Irecv calls below.
        has_cross_rank_work = True

        if self.total_stride == 0 or num_local == 0:
            has_cross_rank_work = False

        # ---- Identify which remote agents this rank needs ----
        # need_from_rank[src_rank] = np.array of unique remote agent IDs
        need_from_rank = {}

        if has_cross_rank_work:
            # 2. Download CSR for local agents from GPU
            cpu_offsets = buf.neighbor_offsets[:num_local + 1].get()
            cpu_offsets = np.asarray(cpu_offsets, dtype=np.int32)
            total_edges_local = int(cpu_offsets[num_local])

            if total_edges_local == 0:
                has_cross_rank_work = False

        if has_cross_rank_work:
            cpu_values_ids = buf.neighbor_values_ids[:total_edges_local].get()
            cpu_values_ids = np.asarray(cpu_values_ids, dtype=np.int32)

            # 3. Vectorized: expand (local_agent_idx, neighbor_id) pairs
            counts = np.diff(cpu_offsets)
            local_agent_indices = np.repeat(
                np.arange(num_local, dtype=np.int32), counts
            )

            # Filter invalid neighbor IDs (padding: -1)
            valid_mask = cpu_values_ids >= 0
            neighbor_ids = cpu_values_ids[valid_mask]

            if len(neighbor_ids) == 0:
                has_cross_rank_work = False

        if has_cross_rank_work:
            # Build dense rank lookup: rank_lookup[agent_id] = rank
            agent2rank = self.agent_factory._agent2rank
            if isinstance(agent2rank, np.ndarray):
                rank_lookup = agent2rank.astype(np.int32, copy=False)
                max_agent_id = len(rank_lookup) - 1
            else:
                max_agent_id = max(agent2rank.keys())
                rank_lookup = np.full(max_agent_id + 1, -1, dtype=np.int32)
                for aid, r in agent2rank.items():
                    rank_lookup[aid] = r

            # Vectorized rank lookup for all neighbors
            in_range = neighbor_ids <= max_agent_id
            neighbor_ranks = np.full(len(neighbor_ids), -1, dtype=np.int32)
            neighbor_ranks[in_range] = rank_lookup[neighbor_ids[in_range]]

            # Filter to cross-rank neighbors
            cross_mask = (neighbor_ranks != self.my_rank) & (neighbor_ranks >= 0)
            cross_neighbor_ids = neighbor_ids[cross_mask]
            cross_neighbor_ranks = neighbor_ranks[cross_mask]

            if len(cross_neighbor_ids) == 0:
                has_cross_rank_work = False

        if has_cross_rank_work:
            # Group by source rank: unique remote agent IDs needed from each
            for src_rank in np.unique(cross_neighbor_ranks):
                src_rank = int(src_rank)
                mask = cross_neighbor_ranks == src_rank
                unique_remote_ids = np.unique(cross_neighbor_ids[mask])
                need_from_rank[src_rank] = unique_remote_ids

        # ---- Phase 1: Exchange request counts (collective) ----
        request_counts = np.zeros(self.num_workers, dtype=np.int32)
        for src_rank, ids in need_from_rank.items():
            request_counts[src_rank] = len(ids)
        supply_counts = np.zeros(self.num_workers, dtype=np.int32)
        self.comm.Alltoall(request_counts, supply_counts)

        # ---- Phase 2: Exchange requested agent ID lists ----
        send_req_requests = []
        for src_rank, ids in need_from_rank.items():
            ids_np = ids.astype(np.int32)
            req = self.comm.Isend([ids_np, MPI.INT], dest=src_rank, tag=100)
            send_req_requests.append(req)

        # Receive: other ranks tell us which of our local agents they need
        requested_by_rank = {}
        recv_req_requests = []
        for dest_rank in range(self.num_workers):
            if dest_rank != self.my_rank and supply_counts[dest_rank] > 0:
                recv_buf = np.empty(int(supply_counts[dest_rank]), dtype=np.int32)
                requested_by_rank[dest_rank] = recv_buf
                req = self.comm.Irecv(
                    [recv_buf, MPI.INT], source=dest_rank, tag=100
                )
                recv_req_requests.append(req)

        MPI.Request.Waitall(send_req_requests + recv_req_requests)

        # ---- Phase 3: Build send / recv index maps ----
        self.send_indices_gpu = {}
        self.send_counts = {}
        self.recv_indices_gpu = {}
        self.recv_counts = {}

        # send_indices: map requested agent IDs to our local buffer indices
        for dest_rank, agent_ids_np in requested_by_rank.items():
            local_indices = np.array(
                [buf.agent_id_to_index[int(aid)] for aid in agent_ids_np],
                dtype=np.int32,
            )
            self.send_indices_gpu[dest_rank] = cp.array(
                local_indices, dtype=cp.int32
            )
            self.send_counts[dest_rank] = len(local_indices)

        # recv_indices: map the remote agents we requested to ghost slots
        for src_rank, remote_ids in need_from_rank.items():
            ghost_indices = np.array(
                [buf.agent_id_to_index[int(aid)] for aid in remote_ids],
                dtype=np.int32,
            )
            self.recv_indices_gpu[src_rank] = cp.array(
                ghost_indices, dtype=cp.int32
            )
            self.recv_counts[src_rank] = len(remote_ids)

        # 7. Pre-allocate send/recv CPU and GPU buffers
        self.send_bufs_cpu = {}
        self.send_bufs_gpu = {}
        self.recv_bufs_cpu = {}
        self.recv_bufs_gpu = {}

        for dest_rank, count in self.send_counts.items():
            size = count * self.total_stride
            self.send_bufs_gpu[dest_rank] = cp.empty(size, dtype=cp.float32)
            if not self._gpu_aware_mpi:
                self.send_bufs_cpu[dest_rank] = np.empty(size, dtype=np.float32)

        for src_rank, count in self.recv_counts.items():
            size = count * self.total_stride
            if self._gpu_aware_mpi:
                self.recv_bufs_gpu[src_rank] = cp.empty(size, dtype=cp.float32)
            else:
                self.recv_bufs_cpu[src_rank] = np.empty(size, dtype=np.float32)

        # ---- Build batched (concatenated) index arrays ----
        # Send: concatenate all per-peer send indices into one array
        self._send_peer_order = sorted(self.send_indices_gpu.keys())
        send_parts = [self.send_indices_gpu[r] for r in self._send_peer_order]
        if send_parts:
            self._batched_send_indices_gpu = cp.concatenate(send_parts)
            cumulative = 0
            self._send_splits = []
            for r in self._send_peer_order:
                cumulative += self.send_counts[r]
                self._send_splits.append(cumulative)
        else:
            self._batched_send_indices_gpu = cp.array([], dtype=cp.int32)
            self._send_splits = []

        # Recv: concatenate all per-peer recv indices into one array
        self._recv_peer_order = sorted(self.recv_indices_gpu.keys())
        recv_parts = [self.recv_indices_gpu[r] for r in self._recv_peer_order]
        if recv_parts:
            self._batched_recv_indices_gpu = cp.concatenate(recv_parts)
            cumulative = 0
            self._recv_splits = []
            for r in self._recv_peer_order:
                cumulative += self.recv_counts[r]
                self._recv_splits.append(cumulative)
        else:
            self._batched_recv_indices_gpu = cp.array([], dtype=cp.int32)
            self._recv_splits = []

        # Pre-compute which write-buffer indices are also visible (for ghost sync)
        visible_set = set(self.property_widths.keys())
        self._write_prop_mask = [
            i for i, prop_idx in enumerate(buf.sorted_write_indices)
            if prop_idx in visible_set
        ]

        # Pre-compute cumulative width offsets per property (for unpack slicing)
        self._prop_cum_widths = {}
        cum = 0
        for prop_idx in self.visible_prop_indices:
            self._prop_cum_widths[prop_idx] = cum
            cum += self.property_widths[prop_idx]

        self._is_initialized = True

    def _batched_gpu_pack(self):
        """Pack all peers' data with ONE gather per property, then split for MPI.

        For each property, does ONE GPU fancy-index gather using concatenated
        send indices, then splits the result per-peer and writes into the
        correct offset within each peer's send buffer.  This keeps the per-peer
        buffer layout (agent-grouped: [prop0(n), prop1(n), ...]) intact while
        reducing GPU gathers from V*P to V.
        """
        all_indices = self._batched_send_indices_gpu
        total_n = len(all_indices)
        if total_n == 0:
            return

        # Track per-peer write offset into send buffers
        peer_offsets = {r: 0 for r in self._send_peer_order}

        for prop_idx in self.visible_prop_indices:
            width = self.property_widths[prop_idx]
            # ONE gather for all peers
            gathered = self.buf.property_tensors[prop_idx][all_indices].ravel()
            # Split per-peer and write into per-peer send buffers
            agent_start = 0
            for i, dest_rank in enumerate(self._send_peer_order):
                n = self.send_counts[dest_rank]
                chunk = gathered[agent_start * width:(agent_start + n) * width]
                off = peer_offsets[dest_rank]
                self.send_bufs_gpu[dest_rank][off:off + n * width] = chunk
                peer_offsets[dest_rank] = off + n * width
                agent_start += n

        # CPU staging: one .get() per peer
        if not self._gpu_aware_mpi:
            for dest_rank in self._send_peer_order:
                self.send_bufs_cpu[dest_rank][:] = self.send_bufs_gpu[dest_rank].get()

    def mpi_exchange(self):
        """Non-blocking buffer-protocol MPI send/recv (no pickle).

        With GPU-aware MPI: passes CuPy device pointers directly to Isend/Irecv.
        Data moves GPU->NIC->GPU via GPU-Direct RDMA without CPU staging.
        Without GPU-aware MPI: falls back to CPU-staged numpy buffers.
        """
        if not self._verbose_timing:
            # Fast path: no timing overhead
            requests = []
            if self._gpu_aware_mpi:
                # GPU-aware path: MPI reads/writes GPU memory directly
                for dest_rank in self._send_peer_order:
                    sbuf = self.send_bufs_gpu[dest_rank]
                    requests.append(self.comm.Isend(
                        [sbuf, MPI.FLOAT], dest=dest_rank, tag=1))
                for src_rank in self._recv_peer_order:
                    rbuf = self.recv_bufs_gpu[src_rank]
                    requests.append(self.comm.Irecv(
                        [rbuf, MPI.FLOAT], source=src_rank, tag=1))
            else:
                # CPU staging path
                for dest_rank in self._send_peer_order:
                    sbuf = self.send_bufs_cpu[dest_rank]
                    requests.append(self.comm.Isend(
                        [sbuf, MPI.FLOAT], dest=dest_rank, tag=1))
                for src_rank in self._recv_peer_order:
                    rbuf = self.recv_bufs_cpu[src_rank]
                    requests.append(self.comm.Irecv(
                        [rbuf, MPI.FLOAT], source=src_rank, tag=1))
            MPI.Request.Waitall(requests)
        else:
            # Verbose path: detailed timing
            import time
            t_isend_start = time.perf_counter()
            send_requests = []
            if self._gpu_aware_mpi:
                for dest_rank in self._send_peer_order:
                    sbuf = self.send_bufs_gpu[dest_rank]
                    send_requests.append(self.comm.Isend(
                        [sbuf, MPI.FLOAT], dest=dest_rank, tag=1))
            else:
                for dest_rank in self._send_peer_order:
                    sbuf = self.send_bufs_cpu[dest_rank]
                    send_requests.append(self.comm.Isend(
                        [sbuf, MPI.FLOAT], dest=dest_rank, tag=1))
            t_isend_end = time.perf_counter()

            t_irecv_start = time.perf_counter()
            recv_requests = []
            if self._gpu_aware_mpi:
                for src_rank in self._recv_peer_order:
                    rbuf = self.recv_bufs_gpu[src_rank]
                    recv_requests.append(self.comm.Irecv(
                        [rbuf, MPI.FLOAT], source=src_rank, tag=1))
            else:
                for src_rank in self._recv_peer_order:
                    rbuf = self.recv_bufs_cpu[src_rank]
                    recv_requests.append(self.comm.Irecv(
                        [rbuf, MPI.FLOAT], source=src_rank, tag=1))
            t_irecv_end = time.perf_counter()

            t_wait_start = time.perf_counter()
            MPI.Request.Waitall(send_requests + recv_requests)
            t_wait_end = time.perf_counter()

            # Store detailed timing for retrieval by caller
            self._last_mpi_timing = {
                'mpi_isend_overhead': t_isend_end - t_isend_start,
                'mpi_irecv_overhead': t_irecv_end - t_irecv_start,
                'mpi_wait_time': t_wait_end - t_wait_start,
            }

    def _batched_gpu_unpack(self):
        """Unpack all peers' data with ONE scatter per property.

        Uploads received data to GPU (single transfer for CPU path), then for
        each property extracts the correct slice from each peer's section,
        concatenates on GPU, and does ONE scatter.  Per-peer buffers use
        agent-grouped layout: [prop0(n), prop1(n), ...], so we use pre-computed
        cumulative width offsets to find each property's data within each peer.
        Also fuses write-buffer ghost sync into the same loop.
        """
        all_ghost_indices = self._batched_recv_indices_gpu
        total_n = len(all_ghost_indices)
        if total_n == 0:
            return

        # Get per-peer recv data on GPU
        if self._gpu_aware_mpi:
            peer_recv_gpu = {r: self.recv_bufs_gpu[r] for r in self._recv_peer_order}
        else:
            # Single CPU→GPU upload for all peers
            cpu_parts = [self.recv_bufs_cpu[r] for r in self._recv_peer_order]
            combined = np.concatenate(cpu_parts) if len(cpu_parts) > 1 else cpu_parts[0]
            all_recv_gpu = cp.array(combined)
            # Split back into per-peer GPU views
            peer_recv_gpu = {}
            pos = 0
            for r in self._recv_peer_order:
                size = self.recv_counts[r] * self.total_stride
                peer_recv_gpu[r] = all_recv_gpu[pos:pos + size]
                pos += size

        # One scatter per property into property_tensors
        buf = self.buf
        for prop_idx in self.visible_prop_indices:
            width = self.property_widths[prop_idx]
            cum_w = self._prop_cum_widths[prop_idx]
            chunks = []
            for r in self._recv_peer_order:
                n = self.recv_counts[r]
                prop_start = n * cum_w
                chunks.append(peer_recv_gpu[r][prop_start:prop_start + n * width])
            prop_data = cp.concatenate(chunks) if len(chunks) > 1 else chunks[0]
            tensor = buf.property_tensors[prop_idx]
            if tensor.ndim > 1:
                prop_data = prop_data.reshape(total_n, width)
            tensor[all_ghost_indices] = prop_data  # ONE scatter

        # Fused write-buffer ghost sync: one scatter per writable visible prop
        for i in self._write_prop_mask:
            prop_idx = buf.sorted_write_indices[i]
            buf.write_buffers[i][all_ghost_indices] = \
                buf.property_tensors[prop_idx][all_ghost_indices]

    def exchange_ghost_data(self):
        """Full ghost exchange: batched GPU pack -> MPI -> batched GPU unpack."""
        if not self._is_initialized:
            return

        timing_data = {} if self._verbose_timing else None

        # Batched pack: one gather per property for all peers
        if self._verbose_timing:
            import time
            t_pack_start = time.perf_counter()
        self._batched_gpu_pack()
        if self._verbose_timing:
            timing_data['mpi_gpu_pack'] = time.perf_counter() - t_pack_start

        # GPU-aware path: ensure GPU packing is complete before MPI reads the buffers
        if self._gpu_aware_mpi and self._send_peer_order:
            if self._verbose_timing:
                t_sync_start = time.perf_counter()
            cp.cuda.get_current_stream().synchronize()
            if self._verbose_timing:
                timing_data['mpi_gpu_sync_pack'] = time.perf_counter() - t_sync_start

        # MPI exchange
        if self._send_peer_order or self._recv_peer_order:
            if self._verbose_timing:
                t_mpi_start = time.perf_counter()
            self.mpi_exchange()
            if self._verbose_timing:
                timing_data['mpi_exchange'] = time.perf_counter() - t_mpi_start
                # Get detailed breakdown from mpi_exchange
                if hasattr(self, '_last_mpi_timing'):
                    timing_data.update(self._last_mpi_timing)

        # Batched unpack + fused write-buffer ghost sync
        if self._verbose_timing:
            t_unpack_start = time.perf_counter()
        self._batched_gpu_unpack()
        if self._verbose_timing:
            timing_data['mpi_gpu_unpack'] = time.perf_counter() - t_unpack_start

            # Communication volume metrics
            timing_data['mpi_send_bytes'] = sum(
                self.send_counts[r] * self.total_stride * 4  # 4 bytes per float32
                for r in self._send_peer_order
            )
            timing_data['mpi_recv_bytes'] = sum(
                self.recv_counts[r] * self.total_stride * 4
                for r in self._recv_peer_order
            )
            timing_data['mpi_num_peers'] = len(self._send_peer_order)

            return timing_data
        return None
