"""
GPU-resident data structures for SAGESim Phase 1.

Provides persistent GPU buffer management to eliminate per-tick CPU rebuild cycles.
"""

import numpy as np
import cupy as cp


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
