"""
GPU-resident data structures for SAGESim Phase 1.

Provides persistent GPU buffer management to eliminate per-tick CPU rebuild cycles.
"""

import os
import time

from sagesim import internal_utils

import numpy as np
import cupy as cp
import cupyx
from mpi4py import MPI


def is_gpu_aware_mpi():
    """Detect if MPI library supports GPU-aware communication.

    Checks environment variables set by GPU-aware MPI implementations:
    - MPICH_GPU_SUPPORT_ENABLED=1 (Cray MPICH on Frontier / AMD MI250X)
    - OMPI_MCA_opal_cuda_support=true (Open MPI with CUDA/ROCm support)
    - SAGESIM_GPU_AWARE_MPI=1 (manual override for testing)
    """
    # Cray MPICH (Frontier / AMD MI250X): set by module or user
    if os.environ.get('MPICH_GPU_SUPPORT_ENABLED', '0') == '1':
        return True
    # Open MPI with CUDA/ROCm support
    if os.environ.get('OMPI_MCA_opal_cuda_support', '') == 'true':
        return True
    # Manual override for testing/debugging on systems without
    # native GPU-aware MPI — forces GPU-aware code path
    if os.environ.get('SAGESIM_GPU_AWARE_MPI', '0') == '1':
        return True
    return False


def plan_exchange_layout(fields, peer_order):
    """Two orders of one flat buffer that carries several fields to several peers.

    ``fields`` is ``[(width, {peer: rows}), ...]`` in field order and ``peer_order`` the
    peers in message order. Field-major puts a field's rows for every peer together,
    peers in message order: what one gather per field over the concatenated row
    indices produces. Peer-major puts a peer's fields together, fields in field order:
    what one contiguous message per peer needs. Returns
    ``(total, field_offsets, peer_offsets, peer_sizes, perm)`` with
    ``field_offsets[f] = (offset, rows)`` in the field-major buffer, ``peer_offsets[r]``
    and ``peer_sizes[r]`` in the peer-major one, and ``perm`` (int32) such that
    ``peer_major = field_major[perm]`` and ``field_major = peer_major[argsort(perm)]``.
    """
    field_offsets, peer_offsets, peer_sizes = [], {}, {}
    pos = 0
    for width, counts in fields:
        rows = sum(int(counts.get(r, 0)) for r in peer_order)
        field_offsets.append((pos, rows))
        pos += rows * width
    total = pos
    pos = 0
    for r in peer_order:
        peer_offsets[r] = pos
        peer_sizes[r] = sum(int(c.get(r, 0)) * w for w, c in fields)
        pos += peer_sizes[r]
    perm = np.empty(total, dtype=np.int32)
    cursor = [off for off, _ in field_offsets]      # next free position of each field (field-major)
    pos = 0                                         # next free position (peer-major)
    for r in peer_order:
        for f, (width, counts) in enumerate(fields):
            n = int(counts.get(r, 0)) * width
            perm[pos:pos + n] = np.arange(cursor[f], cursor[f] + n, dtype=np.int32)
            cursor[f] += n
            pos += n
    return total, field_offsets, peer_offsets, peer_sizes, perm


_segmented_add_kernel = None


def _segmented_add():
    """Kernel: arr[rows[u]] += contrib[offsets[u]:offsets[u+1]] summed in that order,
    starting from the row's current value -- the same float operations in the same
    order as adding each peer's chunk to the row in turn, so a reduce exchange gives
    bitwise the same sum however many peers deposit into one row. `arr` and `contrib`
    are (rows, stride) C-contiguous float32; launched with size = len(rows) * stride."""
    global _segmented_add_kernel
    if _segmented_add_kernel is None:
        _segmented_add_kernel = cp.ElementwiseKernel(
            'raw float32 contrib, raw int32 offsets, raw int32 rows, int32 stride',
            'raw float32 arr',
            '''
            const int u = i / stride, j = i - u * stride;
            const long long dst = (long long) rows[u] * stride + j;
            float s = arr[dst];
            for (int k = offsets[u]; k < offsets[u + 1]; ++k) {
                s += contrib[(long long) k * stride + j];
            }
            arr[dst] = s;
            ''',
            'sagesim_segmented_add')
    return _segmented_add_kernel


def _sorted_unique(a):
    """``np.unique(a)`` for a 1-D integer array, by a sort and a not-equal mask (unlike
    ``np.unique`` it would not collapse NaNs, so it is not for float ids).

    numpy 2.x's ``np.unique`` takes a hash path that is slower than sorting at every size
    and duplicate ratio measured (2.4x with heavy duplicates to 57x all distinct); on a
    38 M-entry neighbor list with 12.7 M distinct ids it took 9.5 s against 0.38 s.
    """
    s = np.sort(a)
    keep = np.empty(len(s), dtype=bool)
    keep[:1] = True
    np.not_equal(s[1:], s[:-1], out=keep[1:])
    return s[keep]


def _distinct_neighbor_ranks(neighbor_ids, agent2rank, local_ids=None, my_rank=-1):
    """The distinct ids in ``neighbor_ids`` and the owner rank of each (boundary-scoped).

    Returns ``(ids, ranks)``: the sorted distinct ids and an int32 rank per id, -1 for
    an id not in ``agent2rank``. Both callers need only these (which remote agents,
    owned by whom), not a rank per neighbor entry, so no per-entry map is built back
    (on Brunel's 38 M-entry neighbor list that was 0.9-1.2 s per call). When
    ``local_ids`` (this rank's own agent ids) is given, ids found there are assigned
    ``my_rank`` with a vectorised membership test, so the Python dict lookup runs only
    over ids owned elsewhere -- the partition boundary. Without ``local_ids`` every
    distinct id goes through the dict, which for a network whose agents are each
    other's neighbours (synapses of a soma) is the whole population.
    """
    if len(neighbor_ids) == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
    uniq = _sorted_unique(neighbor_ids)
    uniq_ranks = np.full(len(uniq), -1, dtype=np.int32)
    remote = np.arange(len(uniq))
    if local_ids is not None and len(local_ids) > 0:
        is_local = np.isin(uniq, np.asarray(local_ids, dtype=np.int64), assume_unique=True)
        uniq_ranks[is_local] = my_rank
        remote = np.flatnonzero(~is_local)
    if len(remote) > 0:
        uniq_ranks[remote] = np.fromiter(
            (agent2rank.get(int(a), -1) for a in uniq[remote]),
            dtype=np.int32, count=len(remote))
    return uniq, uniq_ranks


def discover_ghost_topology(all_neighbors, agent2rank, my_rank, num_workers=None,
                            local_ids=None):
    """Discover ghost agent IDs from local neighbor lists (CPU-only, no MPI).

    Vectorized scan of neighbor arrays to identify agents belonging to other
    ranks.  Returns the unique set of ghost IDs.  Actual data for these ghosts
    is filled later by CommunicationManager.exchange_ghost_data().

    :param all_neighbors: Ragged list of neighbor arrays (one per local agent)
    :param agent2rank: Dict mapping agent_id -> rank
    :param my_rank: This rank's ID
    :param num_workers: Number of ranks; with one rank there can be no ghosts and
        the scan is skipped entirely.
    :param local_ids: This rank's own agent ids (int64 array); lets the rank of
        locally owned neighbours be settled without a per-id dict lookup.
    :return: Sorted list of unique ghost agent IDs
    """
    if num_workers == 1 or not all_neighbors:
        return np.array([], dtype=np.int64)

    # Flatten ragged neighbor lists into one array
    if isinstance(all_neighbors[0], np.ndarray):
        flat = all_neighbors[0] if len(all_neighbors) == 1 else np.concatenate(all_neighbors)
    else:
        flat = np.array(
            [nid for sublist in all_neighbors for nid in sublist],
            dtype=np.float64,
        )

    if len(flat) == 0:
        return np.array([], dtype=np.int64)

    # Filter invalid entries (NaN padding, negative sentinels)
    if flat.dtype.kind == 'f':
        valid_mask = ~np.isnan(flat) & (flat >= 0)
    else:
        valid_mask = flat >= 0
    neighbor_ids = flat[valid_mask].astype(np.int64)

    if len(neighbor_ids) == 0:
        return np.array([], dtype=np.int64)

    # Owner rank per distinct neighbor (boundary-scoped: O(distinct remote ids)); the
    # ghosts are the distinct ids owned by another rank, already sorted.
    uniq, ranks = _distinct_neighbor_ranks(neighbor_ids, agent2rank, local_ids, my_rank)
    return uniq[(ranks != my_rank) & (ranks >= 0)]


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


class TableProperty:
    """Device form of an interned property: `table` (k, w) float32 rows and `codes`
    (capacity,) int32; row i is `table[codes[i]]`. Kernels receive the two arrays and
    the codegen rewrites their reads; host code goes through GPUBufferManager.row/rows."""

    def __init__(self, table, codes):
        self.table = table
        self.codes = codes

    @property
    def shape(self):
        return (int(self.codes.shape[0]), int(self.table.shape[1]))

    @property
    def ndim(self):
        return 2

    @property
    def dtype(self):
        return self.table.dtype

    @property
    def nbytes(self):
        return int(self.table.nbytes + self.codes.nbytes)

    def rows(self, sel):
        """Dense rows for an index/slice/array selection (a device gather)."""
        return cp.take(self.table, self.codes[sel], axis=0)


class GPUBufferManager:
    """Manages persistent GPU buffers for GPU-resident data.

    Allocated once, reused every tick. Eliminates per-tick cp.array() and .get() calls.
    Pre-allocates with slack to accommodate growth without reallocation.
    """

    def __init__(self, agent_slack_factor=1.5, csr_slack_factor=2.0, min_capacity=64):
        self.AGENT_SLACK_FACTOR = agent_slack_factor
        self.CSR_SLACK_FACTOR = csr_slack_factor
        self.MIN_CAPACITY = min_capacity
        self.is_initialized = False

        # Persistent GPU arrays
        self.property_tensors = []      # List of CuPy arrays, one per property
        self.property_alloc_stats = []  # per-property {prop_idx, shape, nbytes, seconds}
        self.write_buffers = []         # List of CuPy arrays for double-buffered properties
        self.neighbor_offsets = None    # CuPy int32 (CSR offsets)
        self.neighbor_values = None     # CuPy int32 (CSR values, local indices for kernel)
        self.neighbor_values_ids = None # CuPy int64 (CSR values, agent IDs for MPI)
        self.agent_ids_gpu = None       # CuPy array of all agent IDs (local + ghost)
        self.logical_ids_gpu = None    # CuPy array of logical IDs for stable RNG
        self.device_breed_local_write_bufs = []  # Write buffers for double-buffered BLAs
        self.device_globals = []  # list of CuPy arrays, one per registered global
        self.device_breed_locals = []   # list of CuPy arrays, one per breed-local array
        self.device_breed_local_idxs = []  # list of CuPy int32 index maps
        self.seed_gpu = None            # CuPy int32 scalar for framework-managed seed
        self.hash_map = None            # GPUHashMap instance
        self.barrier_counter = None     # CuPy uint32[1] for fused-kernel grid barrier
        # Event wake-lists (Model.configure_wake), keyed by priority value
        self.wake_stamps = {}           # priority -> CuPy int32[capacity]: last tick each row ran there (-1 never)
        self.wake_queues = {}           # priority -> CuPy int32 queue of buffer rows to step
        self.wake_counts = {}           # priority -> CuPy int32[1] queue length
        self.wake_emitters = {}         # priority -> CuPy int32 rows whose events wake someone
        self.wake_rev_offsets = {}      # priority -> CuPy int32 CSR offsets over wake_emitters
        self.wake_rev_values = {}       # priority -> CuPy int32 dependants (rows of woken breeds)

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
                                  convert_to_padded_func, interned=None, intern_func=None,
                                  empty_shapes=None):
        """Allocate property tensor GPU arrays with slack capacity.

        :param num_properties: Number of agent properties
        :param combined_lists: List of combined (local+ghost) property data
        :param agent_capacity: Pre-allocated capacity (>= num_total_agents)
        :param convert_to_padded_func: Function to convert ragged lists directly to padded GPU tensors
        :param interned: property indices stored as TableProperty (table + codes)
        :param intern_func: (prop_idx, column, capacity) -> (table_gpu, codes_gpu) for those
        :param empty_shapes: per-property trailing shapes (rows x 2 dims, 0 = absent)
            agreed across ranks in multi-rank runs, so a rank with no agents allocates
            the shapes the shared kernel expects
        """
        self.agent_capacity = agent_capacity
        self.property_tensors = []
        self.property_alloc_stats = []
        interned = set(interned or ())

        for i in range(num_properties):
            if i == 1:
                # Property 1 uses CSR, not a rectangular tensor
                self.property_tensors.append(None)
                continue
            _t0 = time.perf_counter()
            if i in interned:
                table, codes = intern_func(i, combined_lists[i], agent_capacity)
                padded = TableProperty(table, codes)
                path = "table_codes"
            else:
                if empty_shapes is not None and not combined_lists[i]:
                    padded = convert_to_padded_func(combined_lists[i], agent_capacity,
                                                    width=tuple(int(d) for d in empty_shapes[i]))
                else:
                    padded = convert_to_padded_func(combined_lists[i], agent_capacity)
                path = getattr(internal_utils, "LAST_CONVERSION_PATH", None)
            _elapsed = time.perf_counter() - _t0
            self.property_tensors.append(padded)
            # Per-property cost and footprint. The aggregate 'prop_tensors'
            # timer hides which column is expensive and how wide padding made it.
            self.property_alloc_stats.append({
                "prop_idx": i,
                "shape": tuple(int(d) for d in padded.shape),
                "nbytes": int(padded.nbytes),
                "seconds": _elapsed,
                "path": path,
            })

    # -- representation-agnostic host access to property rows --------------------

    def is_interned(self, prop_idx):
        return isinstance(self.property_tensors[prop_idx], TableProperty)

    def rows(self, prop_idx, sel):
        """Dense device rows of property `prop_idx` for an index/slice/array selection."""
        t = self.property_tensors[prop_idx]
        return t.rows(sel) if isinstance(t, TableProperty) else t[sel]

    def row_host(self, prop_idx, buf_idx):
        """One agent's row as a Python list (the read-back the getters return)."""
        t = self.property_tensors[prop_idx]
        if isinstance(t, TableProperty):
            return t.table[int(t.codes[buf_idx])].get().tolist()
        return t[buf_idx].get().tolist()

    def kernel_args_for(self, prop_idx):
        """The launch argument(s) a property contributes, in signature order."""
        t = self.property_tensors[prop_idx]
        return [t.table, t.codes] if isinstance(t, TableProperty) else [t]

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

        # Allocate the padded arrays on the device and copy the data in; padding
        # (zeros for offsets, -1 for values) is written by the device, so no padded
        # host copies of the ~edges-sized arrays are made.
        offsets_capacity = max(self.MIN_CAPACITY, int(num_total_agents * self.AGENT_SLACK_FACTOR)) + 1
        self.neighbor_offsets = cp.zeros(offsets_capacity, dtype=cp.int32)
        self.neighbor_offsets[:len(offsets_np)] = cp.asarray(np.asarray(offsets_np, dtype=np.int32))

        self.neighbor_values = cp.full(self.csr_values_capacity, -1, dtype=cp.int32)
        self.neighbor_values[:total_edges] = cp.asarray(np.asarray(values_np, dtype=np.int32))

        self.neighbor_values_ids = cp.full(self.csr_values_capacity, -1, dtype=cp.int64)
        self.neighbor_values_ids[:total_edges] = cp.asarray(np.asarray(values_ids_np, dtype=np.int64))

    def ensure_agent_capacity(self, needed):
        """Grow property tensors if needed > current capacity. Uses 2x doubling."""
        if needed <= self.agent_capacity:
            return

        new_capacity = max(needed, self.agent_capacity * 2)
        for i, tensor in enumerate(self.property_tensors):
            if tensor is None:
                continue
            if isinstance(tensor, TableProperty):
                new_codes = cp.zeros(new_capacity, dtype=cp.int32)
                new_codes[:self.agent_capacity] = tensor.codes
                self.property_tensors[i] = TableProperty(tensor.table, new_codes)
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
            new_vals_ids = cp.full(new_capacity, -1, dtype=cp.int64)
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

    def __init__(self, buf, agent_factory, my_rank, num_workers, comm, verbose_timing=False,
                 local_ids=None):
        self.buf = buf
        self.agent_factory = agent_factory
        self.my_rank = my_rank
        self.num_workers = num_workers
        self.comm = comm
        self._verbose_timing = verbose_timing
        self._local_ids = local_ids          # this rank's agent ids, for the boundary-scoped rank lookup

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

        # One buffer per direction for every peer (see _plan_buffers): a field-major
        # device copy that one gather per field fills, a peer-major copy that is the
        # concatenation of every peer's message, the permutation between them and,
        # without GPU-aware MPI, a pinned host copy of the peer-major one. The tick
        # then costs a fixed number of device operations however many peers there are.
        self._fwd = None      # {'send': plan, 'recv': plan} for visible properties + plain BLAs
        self._rev = None      # the same for reduce BLAs, roles swapped

        # Batched index arrays (concatenated across all peers)
        self._batched_send_indices_gpu = None  # CuPy int32, all send indices
        self._batched_recv_indices_gpu = None  # CuPy int32, all recv indices
        self._send_peer_order = []             # ordered list of dest ranks
        self._recv_peer_order = []             # ordered list of src ranks

        # Write-buffer ghost index (concatenated recv indices for write props)
        self._write_prop_mask = []  # indices into buf.sorted_write_indices that are visible

        # Breed-local array ghost exchange (combined into same MPI message)
        self._bla_configs = []       # per-BLA: {stride, send_counts, recv_counts, batched_send_rows, batched_recv_rows, send_splits, recv_splits} or None
        self._bla_initialized = False
        self._bla_extra_send_size = {}  # per-peer extra float32 elements for BLA data
        self._bla_extra_recv_size = {}

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

        # A neighbor-visible breed-local array is exchanged like a visible property, so
        # a model with only such an array (no visible property) still has ghost work.
        _has_visible_bla = any(info is not None
                               for info in getattr(buf, 'breed_local_ghost_info', []))
        if (self.total_stride == 0 and not _has_visible_bla) or num_local == 0:
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
            cpu_values_ids = np.asarray(cpu_values_ids, dtype=np.int64)

            # 3. Filter invalid neighbor IDs (padding: -1)
            valid_mask = cpu_values_ids >= 0
            neighbor_ids = cpu_values_ids[valid_mask]

            if len(neighbor_ids) == 0:
                has_cross_rank_work = False

        if has_cross_rank_work:
            # Owner rank per distinct neighbor (boundary-scoped: O(distinct neighbor ids)).
            uniq, uniq_ranks = _distinct_neighbor_ranks(
                neighbor_ids, self.agent_factory._agent2rank, self._local_ids, self.my_rank)

            # Keep the distinct ids owned by another rank (sorted, as uniq is)
            cross_mask = (uniq_ranks != self.my_rank) & (uniq_ranks >= 0)
            cross_neighbor_ids = uniq[cross_mask]
            cross_neighbor_ranks = uniq_ranks[cross_mask]

            if len(cross_neighbor_ids) == 0:
                has_cross_rank_work = False

        if has_cross_rank_work:
            # Group by source rank: the sorted distinct remote agent IDs needed from each
            for src_rank in np.unique(cross_neighbor_ranks):
                src_rank = int(src_rank)
                need_from_rank[src_rank] = cross_neighbor_ids[cross_neighbor_ranks == src_rank]

        # ---- Phase 1: Exchange request counts (collective) ----
        request_counts = np.zeros(self.num_workers, dtype=np.int32)
        for src_rank, ids in need_from_rank.items():
            request_counts[src_rank] = len(ids)
        supply_counts = np.zeros(self.num_workers, dtype=np.int32)
        self.comm.Alltoall(request_counts, supply_counts)

        # ---- Phase 2: Exchange requested agent ID lists ----
        send_req_requests = []
        for src_rank, ids in need_from_rank.items():
            ids_np = ids.astype(np.int64)
            req = self.comm.Isend([ids_np, MPI.INT64_T], dest=src_rank, tag=100)
            send_req_requests.append(req)

        # Receive: other ranks tell us which of our local agents they need
        requested_by_rank = {}
        recv_req_requests = []
        for dest_rank in range(self.num_workers):
            if dest_rank != self.my_rank and supply_counts[dest_rank] > 0:
                recv_buf = np.empty(int(supply_counts[dest_rank]), dtype=np.int64)
                requested_by_rank[dest_rank] = recv_buf
                req = self.comm.Irecv(
                    [recv_buf, MPI.INT64_T], source=dest_rank, tag=100
                )
                recv_req_requests.append(req)

        MPI.Request.Waitall(send_req_requests + recv_req_requests)

        # ---- Phase 2b: Exchange the BREED of each requested agent ----
        # register_remote_agents() carries only agent->rank; a remote agent's
        # breed lives with its owner. Breed-local-array ghost exchange is
        # per-breed, so the requester must know which of its ghost agents belong
        # to a given breed to size recv buffers and scatter values into the right
        # BLA rows. The owner already knows which of its (local) agents each rank
        # requested (requested_by_rank), so it reports their breeds here; the
        # requester records them into _agent2breed for its ghost slots. Reply
        # order matches the request order, so it aligns with need_from_rank.
        #
        # Ghost breeds are consumed ONLY by the neighbor-visible breed-local-array
        # sizing below, so skip this exchange entirely when the model has none.
        # `breed_local_ghost_info` is set in _build_gpu_buffers() before this call
        # and is the same value the BLA block reads later, so the gate matches
        # exactly when that block does work; it is symmetric across ranks (all
        # register the same BLAs), so all ranks skip or all participate.
        _bla_infos = getattr(buf, 'breed_local_ghost_info', [])
        if any(info is not None for info in _bla_infos):
            af = self.agent_factory
            breed_send_bufs = {}
            breed_send_requests = []
            for dest_rank, agent_ids_np in requested_by_rank.items():
                breeds_np = af.breeds_of(agent_ids_np)    # this rank's own agents
                breed_send_bufs[dest_rank] = breeds_np
                breed_send_requests.append(
                    self.comm.Isend([breeds_np, MPI.INT], dest=dest_rank, tag=101)
                )
            breed_recv_bufs = {}
            breed_recv_requests = []
            for src_rank, remote_ids in need_from_rank.items():
                rbuf = np.empty(len(remote_ids), dtype=np.int32)
                breed_recv_bufs[src_rank] = rbuf
                breed_recv_requests.append(
                    self.comm.Irecv([rbuf, MPI.INT], source=src_rank, tag=101)
                )
            MPI.Request.Waitall(breed_send_requests + breed_recv_requests)

            # Record ghost breeds locally (used by buf_breeds below for BLA sizing).
            for src_rank, remote_ids in need_from_rank.items():
                recv_breeds = breed_recv_bufs[src_rank]
                for aid, b in zip(remote_ids, recv_breeds):
                    af._agent2breed[int(aid)] = int(b)

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

        # ---- Build breed-local array ghost exchange maps ----
        bla_ghost_infos = getattr(buf, 'breed_local_ghost_info', [])
        self._bla_configs = []
        self._bla_extra_send_size = {}
        self._bla_extra_recv_size = {}

        # Built whenever this rank takes part in any exchange: it may hold no ghost
        # itself (has_cross_rank_work False) and still be asked for its agents by
        # another rank (send side of a plain array, receive side of a reduce array).
        if bla_ghost_infos and (self.send_indices_gpu or self.recv_indices_gpu):
            # Buffer_index -> breed_id (CPU), for the sent and received rows only: they are
            # the only rows read below, so breeds are looked up per exchanged agent rather
            # than per buffer row (12.7 M rows on Brunel, ~3 s). Sent rows are this rank's
            # agents (breed column), received rows are ghosts (breeds recorded above).
            send_rows = {r: idx.get().astype(np.int32) for r, idx in self.send_indices_gpu.items()}
            recv_rows = {r: idx.get().astype(np.int32) for r, idx in self.recv_indices_gpu.items()}
            exchanged = np.unique(np.concatenate(
                [np.zeros(0, dtype=np.int32), *send_rows.values(), *recv_rows.values()]))
            all_ids = np.asarray(buf.all_agent_ids_list)
            buf_breeds = np.full(buf.num_total_agents, -1, dtype=np.int32)
            buf_breeds[exchanged] = self.agent_factory.breeds_of(all_ids[exchanged])

            for bla_i, info in enumerate(bla_ghost_infos):
                if info is None:
                    self._bla_configs.append(None)
                    continue

                breed_ids = np.array(list(info.get('breed_ids') or [info['breed_id']]),
                                     dtype=np.int32)
                stride = 1
                for s in info['shape_per_agent']:
                    stride *= s

                bla_idx_map = buf.device_breed_local_idxs[bla_i].get()

                bla_send_counts = {}
                bla_recv_counts = {}
                bla_send_rows_per_peer = {}
                bla_recv_rows_per_peer = {}

                for dest_rank in self.send_indices_gpu:
                    cpu_indices = send_rows[dest_rank]
                    mask = np.isin(buf_breeds[cpu_indices], breed_ids)
                    filtered = cpu_indices[mask]
                    bla_rows = bla_idx_map[filtered]
                    bla_send_counts[dest_rank] = len(bla_rows)
                    bla_send_rows_per_peer[dest_rank] = bla_rows

                for src_rank in self.recv_indices_gpu:
                    cpu_indices = recv_rows[src_rank]
                    mask = np.isin(buf_breeds[cpu_indices], breed_ids)
                    filtered = cpu_indices[mask]
                    bla_rows = bla_idx_map[filtered]
                    bla_recv_counts[src_rank] = len(bla_rows)
                    bla_recv_rows_per_peer[src_rank] = bla_rows

                self._bla_configs.append({
                    'bla_i': bla_i,
                    'stride': stride,
                    'shape_per_agent': tuple(info['shape_per_agent']),
                    'reduce': bool(info.get('reduce')),
                    'send_counts': bla_send_counts,
                    'recv_counts': bla_recv_counts,
                    'bla_send_rows_per_peer': bla_send_rows_per_peer,
                    'bla_recv_rows_per_peer': bla_recv_rows_per_peer,
                })
                if info.get('reduce'):
                    # exchanged the other way round (ghost rows summed back into the
                    # owner's row, see _reverse_exchange): not part of the forward buffers
                    continue

                for r in self.send_counts:
                    self._bla_extra_send_size[r] = (
                        self._bla_extra_send_size.get(r, 0)
                        + bla_send_counts.get(r, 0) * stride)
                for r in self.recv_counts:
                    self._bla_extra_recv_size[r] = (
                        self._bla_extra_recv_size.get(r, 0)
                        + bla_recv_counts.get(r, 0) * stride)

        self._bla_initialized = any(c is not None for c in self._bla_configs)

        # ---- Build batched (concatenated) index arrays, peers in message order ----
        self._send_peer_order = sorted(self.send_indices_gpu.keys())
        send_parts = [self.send_indices_gpu[r] for r in self._send_peer_order]
        self._batched_send_indices_gpu = (cp.concatenate(send_parts) if send_parts
                                          else cp.array([], dtype=cp.int32))
        self._recv_peer_order = sorted(self.recv_indices_gpu.keys())
        recv_parts = [self.recv_indices_gpu[r] for r in self._recv_peer_order]
        self._batched_recv_indices_gpu = (cp.concatenate(recv_parts) if recv_parts
                                          else cp.array([], dtype=cp.int32))

        # Pre-compute which write-buffer indices are also visible (for ghost sync)
        visible_set = set(self.property_widths.keys())
        self._write_prop_mask = [
            i for i, prop_idx in enumerate(buf.sorted_write_indices)
            if prop_idx in visible_set
        ]

        # Batched BLA row arrays (one gather / scatter per BLA), peers in message order
        if self._bla_initialized:
            for bla_i, cfg in enumerate(self._bla_configs):
                if cfg is None:
                    continue
                cfg['batched_send_rows_per_peer'] = dict(cfg['bla_send_rows_per_peer'])
                cfg['batched_recv_rows_per_peer'] = dict(cfg['bla_recv_rows_per_peer'])
                send_parts = [cfg['bla_send_rows_per_peer'].get(r, np.array([], dtype=np.int32))
                              for r in self._send_peer_order]
                cfg['batched_send_rows'] = cp.array(
                    np.concatenate(send_parts) if send_parts else np.array([], dtype=np.int32),
                    dtype=cp.int32)
                recv_parts = [cfg['bla_recv_rows_per_peer'].get(r, np.array([], dtype=np.int32))
                              for r in self._recv_peer_order]
                cfg['batched_recv_rows'] = cp.array(
                    np.concatenate(recv_parts) if recv_parts else np.array([], dtype=np.int32),
                    dtype=cp.int32)
                del cfg['bla_send_rows_per_peer']
                del cfg['bla_recv_rows_per_peer']

        self._plan_forward_buffers()
        self._build_reverse_maps()
        self._is_initialized = True

    # ---- One buffer per direction, two orders of it ----
    #
    # Per tick the exchange used to do one gather per field and then, per peer, a
    # slice copy per field, a device-to-host copy, and on the way back per peer and per
    # field a slice, a concatenate and a scatter: O(peers x fields) small device
    # operations and 2 x peers device syncs, about 1 ms per peer on a laptop (notes in
    # SuperNeuroABM's EVENT_DRIVEN_NOTES section 16). Now the fields are gathered into
    # ONE field-major buffer (one `take` per field, into a view), permuted ONCE into a
    # peer-major buffer whose per-peer slices are the messages, copied to the host ONCE
    # (without GPU-aware MPI); the receive side mirrors it. The permutation is planned
    # at setup (plan_exchange_layout).

    def _plan_forward_buffers(self):
        """Forward direction: visible properties (every peer's rows), then plain
        breed-local arrays, in the order the per-peer messages used to carry them."""
        buf = self.buf
        fields, send_counts, recv_counts = [], [], []
        for prop_idx in self.visible_prop_indices:
            tensor = buf.property_tensors[prop_idx]
            if tensor is None:
                continue
            tail = tuple(tensor.shape[1:])
            fields.append(('prop', prop_idx, int(np.prod(tail)) if tail else 1, tail))
            send_counts.append(dict(self.send_counts))
            recv_counts.append(dict(self.recv_counts))
        for bla_i, cfg in enumerate(self._bla_configs):
            if cfg is None or cfg['reduce']:
                continue
            fields.append(('bla', bla_i, cfg['stride'], tuple(cfg['shape_per_agent'])))
            send_counts.append(dict(cfg['send_counts']))
            recv_counts.append(dict(cfg['recv_counts']))
        self._fwd = {'send': self._plan_buffers(fields, send_counts, self._send_peer_order),
                     'recv': self._plan_buffers(fields, recv_counts, self._recv_peer_order)}

    def _plan_buffers(self, fields, counts, peer_order):
        """Buffers and index plan for one direction. `fields[i]` is (kind, key, width,
        shape_tail) with per-peer row counts `counts[i]`; the plan carries the two device
        buffers, the permutation both ways, the per-peer message slices of the
        peer-major buffer and, without GPU-aware MPI, a pinned host copy."""
        total, field_off, peer_off, peer_size, perm = plan_exchange_layout(
            [(f[2], c) for f, c in zip(fields, counts)], peer_order)
        return {
            'total': total, 'peer_off': peer_off, 'peer_size': peer_size,
            'fields': [(kind, key, width, tail, off, rows)
                       for (kind, key, width, tail), (off, rows) in zip(fields, field_off)],
            'A': cp.empty(total, dtype=cp.float32),                       # field-major
            'B': cp.empty(total, dtype=cp.float32),                       # peer-major
            'perm': cp.asarray(perm),                                     # B = A[perm]
            'inv': cp.asarray(np.argsort(perm, kind='stable').astype(np.int32)),   # A = B[inv]
            'host': None if self._gpu_aware_mpi else cupyx.empty_pinned((total,), dtype=np.float32),
        }

    def _field_source(self, kind, key):
        buf = self.buf
        return buf.property_tensors[key] if kind == 'prop' else buf.device_breed_locals[key]

    @staticmethod
    def _gather_fields(plan, source_of, rows_of):
        """plan['A'][field segment] = source[rows] for every field: one `take` into a view."""
        A = plan['A']
        for kind, key, width, tail, off, rows in plan['fields']:
            if rows == 0:
                continue
            out = A[off:off + rows * width]
            if tail:
                out = out.reshape((rows,) + tail)
            cp.take(source_of(kind, key), rows_of(kind, key), axis=0, out=out)

    @staticmethod
    def _scatter_fields(plan, target_of, rows_of):
        """target[rows] = plan['A'][field segment] for every field: one scatter each."""
        A = plan['A']
        for kind, key, width, tail, off, rows in plan['fields']:
            if rows == 0:
                continue
            seg = A[off:off + rows * width]
            if tail:
                seg = seg.reshape((rows,) + tail)
            target_of(kind, key)[rows_of(kind, key)] = seg

    @staticmethod
    def _message(plan, r):
        """The slice of the peer-major buffer (device, or its host copy) that is peer r's message."""
        buf = plan['B'] if plan['host'] is None else plan['host']
        off = plan['peer_off'][r]
        return buf[off:off + plan['peer_size'][r]]

    # ---- Reduce-back-to-owner exchange for reduce breed-local arrays ----
    #
    # A reduce array's ghost rows receive deposits from local agents (atomic adds into
    # the write buffer, drained into the read buffer by the in-kernel write-back) but
    # nobody reads them here: their owner does. So, in the opposite direction of the
    # ordinary exchange, every rank sends the ghost rows it holds to the owner rank
    # (the forward "need_from" relation, reused), and the owner adds what arrives into
    # its own rows before its agents read them. The peer lists are the forward ones
    # with the roles swapped: I send to the ranks I receive ghosts from, I receive from
    # the ranks that hold ghosts of my agents.

    def _build_reverse_maps(self):
        self._rev_cfgs = [cfg for cfg in self._bla_configs if cfg is not None and cfg['reduce']]
        self._rev_send_peers = list(self._recv_peer_order)
        self._rev_recv_peers = list(self._send_peer_order)
        fields, send_counts, recv_counts = [], [], []
        for cfg in self._rev_cfgs:
            stride = cfg['stride']
            # ghost rows this rank holds, grouped by owner peer in _rev_send_peers order
            parts = [cfg['batched_recv_rows_per_peer'].get(r, np.array([], dtype=np.int32))
                     for r in self._rev_send_peers]
            cfg['rev_send_rows'] = cp.array(np.concatenate(parts) if parts else np.array([], dtype=np.int32),
                                            dtype=cp.int32)
            cfg['rev_send_counts'] = {r: int(cfg['recv_counts'].get(r, 0)) for r in self._rev_send_peers}
            # my own rows the peers deposit into, peer by peer in _rev_recv_peers order; a
            # row may be deposited into by several peers, so the add is a segmented sum
            # over its contributions in that order (bitwise what adding each peer's chunk
            # in turn gave): stable sort of the rows, one segment per distinct row
            rows_parts = [np.asarray(cfg['batched_send_rows_per_peer'].get(r, np.array([], dtype=np.int32)),
                                     dtype=np.int32) for r in self._rev_recv_peers]
            rows = np.concatenate(rows_parts) if rows_parts else np.array([], dtype=np.int32)
            cfg['rev_recv_counts'] = {r: int(cfg['send_counts'].get(r, 0)) for r in self._rev_recv_peers}
            order = np.argsort(rows, kind='stable')
            uniq, counts = np.unique(rows[order], return_counts=True)
            cfg['rev_add'] = {
                'order': cp.asarray(order.astype(np.int32)),
                'rows': cp.asarray(uniq.astype(np.int32)),
                'offsets': cp.asarray(np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)),
                'n_uniq': int(uniq.size),
                'contrib': cp.empty((int(rows.size), stride), dtype=cp.float32),
            }
            fields.append(('bla', cfg['bla_i'], stride, tuple(cfg['shape_per_agent'])))
            send_counts.append(cfg['rev_send_counts'])
            recv_counts.append(cfg['rev_recv_counts'])
            del cfg['batched_recv_rows_per_peer'], cfg['batched_send_rows_per_peer']
        self._rev = None
        if fields:
            self._rev = {'send': self._plan_buffers(fields, send_counts, self._rev_send_peers),
                         'recv': self._plan_buffers(fields, recv_counts, self._rev_recv_peers)}
        self._rev_initialized = bool(self._rev and (self._rev['send']['total'] or self._rev['recv']['total']))

    def _reverse_exchange(self):
        """Ghost rows of every reduce array -> their owner, added into the owner's row."""
        if not getattr(self, '_rev_initialized', False):
            return
        buf = self.buf
        send, recv = self._rev['send'], self._rev['recv']
        if send['total']:
            self._gather_fields(send, self._field_source,
                                lambda kind, key: self._bla_configs[key]['rev_send_rows'])
            cp.take(send['A'], send['perm'], out=send['B'])
            if send['host'] is not None:
                send['B'].get(out=send['host'])
            else:
                cp.cuda.get_current_stream().synchronize()
        # a pair exchanges a message only when it has rows for it, on both sides alike
        requests = []
        for r in self._rev_send_peers:
            if send['peer_size'][r]:
                requests.append(self.comm.Isend([self._message(send, r), MPI.FLOAT], dest=r, tag=2))
        for r in self._rev_recv_peers:
            if recv['peer_size'][r]:
                requests.append(self.comm.Irecv([self._message(recv, r), MPI.FLOAT], source=r, tag=2))
        if requests:
            MPI.Request.Waitall(requests)
        if recv['total']:
            if recv['host'] is not None:
                recv['B'].set(recv['host'])
            cp.take(recv['B'], recv['inv'], out=recv['A'])
            seg_add = _segmented_add()
            for (kind, key, width, tail, off, rows), cfg in zip(recv['fields'], self._rev_cfgs):
                if rows == 0:
                    continue
                add = cfg['rev_add']
                seg = recv['A'][off:off + rows * width].reshape(rows, width)
                cp.take(seg, add['order'], axis=0, out=add['contrib'])
                seg_add(add['contrib'], add['offsets'], add['rows'], np.int32(width),
                        buf.device_breed_locals[cfg['bla_i']], size=add['n_uniq'] * width)

    def _batched_gpu_pack(self):
        """Every peer's message in one go: one gather per field into the field-major
        buffer, one permutation into the peer-major buffer, one copy to the host
        (without GPU-aware MPI). Independent of the number of peers."""
        plan = self._fwd['send']
        if plan['total'] == 0:
            return
        self._gather_fields(
            plan, self._field_source,
            lambda kind, key: (self._batched_send_indices_gpu if kind == 'prop'
                               else self._bla_configs[key]['batched_send_rows']))
        cp.take(plan['A'], plan['perm'], out=plan['B'])
        if plan['host'] is not None:
            plan['B'].get(out=plan['host'])

    def mpi_exchange(self):
        """Non-blocking buffer-protocol MPI send/recv (no pickle), one message per peer:
        a slice of the peer-major buffer, on the device with GPU-aware MPI (GPU-Direct
        RDMA, no staging) and of its pinned host copy otherwise."""
        send, recv = self._fwd['send'], self._fwd['recv']
        if not self._verbose_timing:
            requests = [self.comm.Isend([self._message(send, r), MPI.FLOAT], dest=r, tag=1)
                        for r in self._send_peer_order]
            requests += [self.comm.Irecv([self._message(recv, r), MPI.FLOAT], source=r, tag=1)
                         for r in self._recv_peer_order]
            MPI.Request.Waitall(requests)
        else:
            import time
            t_isend_start = time.perf_counter()
            send_requests = [self.comm.Isend([self._message(send, r), MPI.FLOAT], dest=r, tag=1)
                             for r in self._send_peer_order]
            t_isend_end = time.perf_counter()
            t_irecv_start = time.perf_counter()
            recv_requests = [self.comm.Irecv([self._message(recv, r), MPI.FLOAT], source=r, tag=1)
                             for r in self._recv_peer_order]
            t_irecv_end = time.perf_counter()
            t_wait_start = time.perf_counter()
            MPI.Request.Waitall(send_requests + recv_requests)
            t_wait_end = time.perf_counter()
            self._last_mpi_timing = {
                'mpi_isend_overhead': t_isend_end - t_isend_start,
                'mpi_irecv_overhead': t_irecv_end - t_irecv_start,
                'mpi_wait_time': t_wait_end - t_wait_start,
            }

    def _batched_gpu_unpack(self):
        """The mirror of the pack: one copy to the device (without GPU-aware MPI), one
        permutation back to field-major, one scatter per field, plus the write-buffer
        ghost sync for visible properties a kernel writes."""
        plan = self._fwd['recv']
        if plan['total'] == 0:
            return
        buf = self.buf
        if plan['host'] is not None:
            plan['B'].set(plan['host'])
        cp.take(plan['B'], plan['inv'], out=plan['A'])
        self._scatter_fields(
            plan, self._field_source,
            lambda kind, key: (self._batched_recv_indices_gpu if kind == 'prop'
                               else self._bla_configs[key]['batched_recv_rows']))
        all_ghost_indices = self._batched_recv_indices_gpu
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

        # Reduce arrays go the other way: ghost rows' deposits back to their owner
        if self._verbose_timing:
            t_rev_start = time.perf_counter()
        self._reverse_exchange()
        if self._verbose_timing:
            timing_data['mpi_reverse'] = time.perf_counter() - t_rev_start

            # Communication volume metrics (property + BLA data, forward direction)
            timing_data['mpi_send_bytes'] = self._fwd['send']['total'] * 4
            timing_data['mpi_recv_bytes'] = self._fwd['recv']['total'] * 4
            timing_data['mpi_num_peers'] = len(self._send_peer_order)

            return timing_data
        return None
