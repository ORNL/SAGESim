"""
SuperNeuroABM basic Model class

"""

from typing import Dict, List, Callable, Set, Any, Union
import os
import re
import sys
from pathlib import Path
import importlib
import pickle
import math
import heapq
import warnings
import time

import ast
import inspect

import cupy as cp
import numpy as np
from mpi4py import MPI

from sagesim.agent import AgentFactory, Breed
from sagesim.space import Space
from sagesim.internal_utils import convert_to_equal_side_tensor, build_csr_from_ragged, build_csr_values_only, convert_to_padded_gpu_tensor
from sagesim.gpu_kernels import GPUBufferManager, GPUHashMap, CommunicationManager, is_gpu_aware_mpi, discover_ghost_topology


def convert_agent_ids_to_indices(data_tensor, agent_id_to_index_map):
    """
    Convert agent IDs in nested arrays to local indices using a hash map.

    :param data_tensor: Nested list structure containing agent IDs (can also contain sets)
    :param agent_id_to_index_map: Dictionary mapping agent_id -> local_index
    :return: Same structure with IDs replaced by local indices (-1 if not found)
    """
    # OPTIMIZATION: Build lookup arrays ONCE instead of for every agent!
    id_keys = np.array(list(agent_id_to_index_map.keys()), dtype=np.int32)
    id_values = np.array(list(agent_id_to_index_map.values()), dtype=np.int32)
    min_id = id_keys.min()
    max_id = id_keys.max()
    id_range = max_id - min_id + 1

    # Use dense array if not too sparse (< 3x overhead)
    use_dense = id_range < len(agent_id_to_index_map) * 3
    if use_dense:
        lookup_array = np.full(id_range, -1, dtype=np.int32)
        lookup_array[id_keys - min_id] = id_values
    else:
        # Sparse: use sorted arrays for binary search
        sort_idx = np.argsort(id_keys)
        sorted_keys = id_keys[sort_idx]
        sorted_values = id_values[sort_idx]

    result = []
    for agent_data in data_tensor:
        if isinstance(agent_data, np.ndarray):
            # FULLY VECTORIZED: Use pre-built lookup arrays (built once above)

            # Handle NaN values properly
            arr = agent_data.astype(np.float64, copy=False)
            valid_mask = ~np.isnan(arr)

            # Initialize output with -1 (invalid index)
            converted = np.full(arr.shape, -1, dtype=np.int32)

            if np.any(valid_mask):
                valid_ids = arr[valid_mask].astype(np.int32)

                # Use pre-built lookup arrays
                if use_dense:
                    # Dense lookup: O(1) array indexing
                    in_range = (valid_ids >= min_id) & (valid_ids <= max_id)
                    indices = np.full(len(valid_ids), -1, dtype=np.int32)
                    indices[in_range] = lookup_array[valid_ids[in_range] - min_id]
                else:
                    # Sparse IDs: use searchsorted (O(log n) per lookup)
                    positions = np.searchsorted(sorted_keys, valid_ids)
                    found = (positions < len(sorted_keys)) & (sorted_keys[positions] == valid_ids)
                    indices = np.where(found, sorted_values[positions], -1)

                converted[valid_mask] = indices

            result.append(converted.tolist())
        elif isinstance(agent_data, (list, tuple, set)):
            # Handle collections (list, tuple, set) with multiple connections
            converted_data = []
            for value in agent_data:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if not np.isnan(value):
                        # Convert ID to index, use -1 if not found
                        converted_data.append(agent_id_to_index_map.get(int(value), -1))
                    else:
                        converted_data.append(value)
                else:
                    converted_data.append(value)
            result.append(converted_data)
        else:
            # Single value
            if isinstance(agent_data, (int, float, np.integer, np.floating)):
                if not np.isnan(agent_data):
                    result.append(agent_id_to_index_map.get(int(agent_data), -1))
                else:
                    result.append(agent_data)
            else:
                result.append(agent_data)

    return result


def convert_agent_indices_to_ids(data_tensor, agent_index_to_id_list):
    """
    Convert local indices back to agent IDs in nested arrays.
    This is the reverse operation of convert_agent_ids_to_indices.

    :param data_tensor: Nested list/array structure containing local indices (can be 2D numpy array, list of arrays, or list of lists)
    :param agent_index_to_id_list: List where agent_index_to_id_list[index] = agent_id
    :return: Same structure with indices replaced by agent IDs (-1 remains -1)
    """
    # Convert to numpy array for vectorized operations (much faster)
    id_array = np.array(agent_index_to_id_list, dtype=np.int32)
    list_len = len(agent_index_to_id_list)

    # FAST PATH: If data_tensor is already a 2D numpy array, vectorize everything
    if isinstance(data_tensor, np.ndarray) and data_tensor.ndim == 2:
        # Create output array (start with copy of input to preserve NaN and special values)
        converted = data_tensor.copy()

        # Create mask for valid indices (not NaN, not -1, within bounds)
        # Use np.nan_to_num to avoid warning when comparing NaN
        with np.errstate(invalid='ignore'):
            is_valid_number = ~np.isnan(data_tensor)

        # Only process valid numbers
        if np.any(is_valid_number):
            valid_data = data_tensor[is_valid_number]
            # Convert to int safely (NaN already filtered out)
            valid_indices = valid_data.astype(np.int32)

            # Find which ones need conversion (not -1, within bounds)
            needs_conversion = (valid_indices >= 0) & (valid_indices < list_len)

            # Apply conversion using fancy indexing (very fast)
            if np.any(needs_conversion):
                indices_to_convert = valid_indices[needs_conversion]
                converted_values = id_array[indices_to_convert]

                # Put converted values back into output array
                # Create a temporary array for indexing
                temp = converted[is_valid_number]
                temp[needs_conversion] = converted_values
                converted[is_valid_number] = temp

        # OPTIMIZED: Return list of numpy arrays instead of list of lists
        # This avoids the expensive .tolist() conversion while remaining compatible
        # with list concatenation operations (e.g., local + received)
        result = [converted[i] for i in range(len(converted))]
        return result

    # SLOW PATH: For lists or mixed data structures, process row by row
    result = []
    for agent_data in data_tensor:
        if isinstance(agent_data, np.ndarray):
            # VECTORIZED path for numpy arrays - extremely fast
            arr = agent_data.astype(np.int32)

            # Create output array (start with original data)
            converted = np.where(
                np.isnan(agent_data),  # Keep NaN as NaN
                agent_data,
                np.where(
                    arr == -1,  # Keep -1 as -1
                    -1,
                    np.where(
                        (arr >= 0) & (arr < list_len),  # Valid indices
                        id_array[np.clip(arr, 0, list_len-1)],  # Convert to IDs
                        agent_data  # Out of bounds, keep original
                    )
                )
            )

            result.append(converted.tolist())
        elif isinstance(agent_data, (list, tuple, set)):
            # Handle collections (list, tuple, set) with multiple connections
            converted_data = []
            for value in agent_data:
                if isinstance(value, (int, float, np.integer, np.floating)):
                    if not np.isnan(value):
                        idx = int(value)
                        if idx == -1:
                            converted_data.append(-1)
                        elif 0 <= idx < list_len:
                            converted_data.append(agent_index_to_id_list[idx])
                        else:
                            converted_data.append(value)
                    else:
                        converted_data.append(value)
                else:
                    converted_data.append(value)
            result.append(converted_data)
        else:
            # Single value
            if isinstance(agent_data, (int, float, np.integer, np.floating)):
                if not np.isnan(agent_data):
                    idx = int(agent_data)
                    if idx == -1:
                        result.append(-1)
                    elif 0 <= idx < list_len:
                        result.append(agent_index_to_id_list[idx])
                    else:
                        result.append(agent_data)
                else:
                    result.append(agent_data)
            else:
                result.append(agent_data)

    return result


comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


def _build_param_to_property_index(param_names: list, num_properties: int) -> dict:
    """
    Build mapping from ORIGINAL step function parameter names to property indices.

    The last num_properties params map 1:1 to property indices.

    The parameter order in the user's step function is:
        tick, agent_index, globals, agent_ids,
        breeds, locations, prop2, prop3, ...
        ^-- property params (num_properties total) --^

    Returns dict: {param_name: property_index}
    """
    prop_params = param_names[-num_properties:]
    return {name: idx for idx, name in enumerate(prop_params)}


def _build_param_to_property_index_csr(param_names: list, num_properties: int) -> dict:
    """
    Build mapping from CSR-TRANSFORMED step function parameter names to property indices.

    After CSR transformation, property 1 (locations) is split into two parameters
    (neighbor_offsets, neighbor_values), so the function has num_properties + 1
    property-like parameters.

    The parameter order after CSR transformation:
        tick, agent_index, globals, agent_ids,
        breeds, neighbor_offsets, neighbor_values, prop2, prop3, ...
        ^-- property-like params (num_properties + 1 total) --^

    Returns dict: {param_name: property_index} where CSR params map to -1.
    """
    n_prop_params = num_properties + 1
    prop_params = param_names[-n_prop_params:]

    mapping = {}
    prop_idx = 0
    for i, name in enumerate(prop_params):
        if i == 1 or i == 2:
            mapping[name] = -1
        else:
            mapping[name] = prop_idx
            prop_idx += 1
            if prop_idx == 1:
                prop_idx = 2

    return mapping


def analyze_step_function_for_writes(step_func: Callable, num_properties: int,
                                      num_breed_local_params: int = 0) -> Set[int]:
    """Analyze step function to find which property indices need write buffers."""
    write_property_indices = set()

    source = inspect.getsource(step_func)
    tree = ast.parse(source)
    signature = inspect.signature(step_func)
    param_names = list(signature.parameters.keys())

    # Strip breed-local params from end before property mapping
    if num_breed_local_params > 0:
        param_names_for_prop = param_names[:-num_breed_local_params]
    else:
        param_names_for_prop = param_names

    # Build param name → property index mapping (for ORIGINAL user step function)
    param_to_prop = _build_param_to_property_index(param_names_for_prop, num_properties)

    # Get just the property param names for checking
    property_params = param_names[-num_properties:]

    def check_target_for_writes(target_node):
        if isinstance(target_node, ast.Name):
            # Direct assignment: param_name = value
            if target_node.id in param_to_prop:
                prop_idx = param_to_prop[target_node.id]
                write_property_indices.add(prop_idx)
        elif isinstance(target_node, ast.Subscript):
            # Subscript assignment: param_name[...] = value or nested subscripts
            check_target_for_writes(target_node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Name) and
                node.func.id == 'set_this_agent_data_from_tensor'):
                if len(node.args) >= 2:
                    tensor_arg = node.args[1]
                    if isinstance(tensor_arg, ast.Name):
                        tensor_name = tensor_arg.id
                        if tensor_name in param_to_prop:
                            write_property_indices.add(param_to_prop[tensor_name])
        elif isinstance(node, ast.Assign):
            # Check for all types of assignments to property parameters
            for target in node.targets:
                check_target_for_writes(target)
        elif isinstance(node, ast.AugAssign):
            # Check for augmented assignments (+=, -=, *=, etc.)
            check_target_for_writes(node.target)

    return write_property_indices


def analyze_step_function_for_bla_writes(step_func: Callable,
                                          bla_names: List[str]) -> Set[str]:
    """Analyze step function to find which breed-local array names are written to."""
    written_bla_names = set()
    source = inspect.getsource(step_func)
    tree = ast.parse(source)

    bla_name_set = set(bla_names)

    def check_target(target_node):
        if isinstance(target_node, ast.Name):
            if target_node.id in bla_name_set:
                written_bla_names.add(target_node.id)
        elif isinstance(target_node, ast.Subscript):
            check_target(target_node.value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                check_target(target)
        elif isinstance(node, ast.AugAssign):
            check_target(node.target)

    return written_bla_names


class Model:

    def __init__(
        self,
        space: Space,
        threads_per_block: int = 32,
        step_function_file_path: str = "step_func_code.py",
        verbose_timing: bool = False,
        agent_slack_factor: float = 1.5,
        csr_slack_factor: float = 2.0,
        min_capacity: int = 64,
    ) -> None:
        self._threads_per_block = threads_per_block
        self._step_function_file_path = os.path.abspath(step_function_file_path)
        self._verbose_timing = verbose_timing
        self._agent_slack_factor = agent_slack_factor
        self._csr_slack_factor = csr_slack_factor
        self._min_capacity = min_capacity
        self._agent_factory = AgentFactory(space, verbose_timing=verbose_timing)
        self._is_setup = False
        # Globals — each registered global is a separate named tensor
        self._global_tensors = []     # list of numpy arrays, registration order
        self._global_names = {}       # name → index in _global_tensors
        self._device_globals = []     # list of CuPy arrays (cached on GPU)
        self._globals_dirty = True
        self._seed = int(np.random.randint(0, 2**31))  # random by default
        self._logical_id_map = {}  # agent_id -> logical_id (for stable RNG)
        self.tick = 0
        self._write_property_indices = set()  # Cache for write property indices
        # Breed-local arrays — separate per-breed GPU tensors (not padded across breeds)
        self._breed_local_arrays = []  # list of dicts: {name, breed, shape_per_agent, neighbor_visible}
        # following may be set later in setup if distributed execution

    @property
    def verbose_timing(self) -> bool:
        """Enable timing verbose output for debugging performance."""
        return self._verbose_timing

    @verbose_timing.setter
    def verbose_timing(self, value: bool) -> None:
        self._verbose_timing = value
        self._agent_factory._verbose_timing = value

    def set_property_neighbor_visible(self, property_name: str, visible: bool) -> None:
        """Override neighbor_visible flag for a built-in property (e.g. 'breed').

        Must be called before setup().
        """
        if self._is_setup:
            raise RuntimeError("Cannot change neighbor_visible after setup()")
        af = self._agent_factory
        if property_name not in af._property_name_2_neighbor_visible:
            raise ValueError(f"Unknown property: {property_name}")
        af._property_name_2_neighbor_visible[property_name] = visible
        af._neighbor_visible_indices = []  # invalidate cache

    def register_breed(self, breed: Breed) -> None:
        if self._agent_factory.num_agents > 0:
            raise Exception(f"All breeds must be registered before agents are created!")
        self._agent_factory.register_breed(breed)

    def create_agent_of_breed(self, breed: Breed, add_to_space=True, rank: int = None, **kwargs) -> int:
        agent_id = self._agent_factory.create_agent(breed, rank=rank, **kwargs)
        if add_to_space:
            self.get_space().add_agent(agent_id)
        return agent_id

    def get_agent_property_value(self, id: int, property_name: str) -> Any:
        if self._is_setup and hasattr(self, '_gpu_buffers') and self._gpu_buffers.is_initialized:
            # Fast path: read single agent directly from GPU
            buf = self._gpu_buffers
            prop_idx = self._agent_factory._property_name_2_index[property_name]
            agent_rank = self._agent_factory._agent2rank[id]

            if agent_rank == MPI.COMM_WORLD.Get_rank():
                buf_idx = buf.agent_id_to_index[id]
                if prop_idx == 1:
                    # CSR/locations: read one agent's neighbor slice
                    start = int(buf.neighbor_offsets[buf_idx].get())
                    end = int(buf.neighbor_offsets[buf_idx + 1].get())
                    result = buf.neighbor_values_ids[start:end].get().tolist()
                else:
                    result = buf.property_tensors[prop_idx][buf_idx].get().tolist()
            else:
                result = None

            return MPI.COMM_WORLD.bcast(result, root=agent_rank)

        # Pre-setup path: use CPU-side data
        if self._is_setup:
            self._agent_factory._update_agent_property(
                self.__rank_local_agent_data_tensors, id, property_name
            )
        return self._agent_factory.get_agent_property_value(
            property_name=property_name, agent_id=id
        )

    def set_agent_property_value(self, id: int, property_name: str, value: Any) -> None:
        self._agent_factory.set_agent_property_value(
            property_name=property_name, agent_id=id, value=value
        )
        # GPU buffers are now stale — force rebuild on next tick
        if hasattr(self, '_gpu_buffers') and self._gpu_buffers.is_initialized:
            self._gpu_buffers.is_initialized = False
            self._cached_all_args = None

    def get_space(self) -> Space:
        return self._agent_factory._space

    def get_breed_data(self, breed_name, property_name, local=False):
        """Download property data for agents of a breed.

        :param breed_name: Name of the breed (e.g., "Tree", "Site")
        :param property_name: Property to download (e.g., "params", "states_db")
        :param local: If True, return local data only (no MPI gather).
                     If False (default), gather all data to rank 0.
        :return: If local=True: numpy array of local agents' data on ALL ranks
                 If local=False: numpy array of all agents' data on rank 0, None on other ranks
        """
        breed = self._agent_factory._breeds[breed_name]
        buf = self._gpu_buffers
        start, count = buf.breed_ranges.get(breed._breedidx, (0, 0))

        prop_idx = self._agent_factory._property_name_2_index[property_name]

        comm = MPI.COMM_WORLD

        # Return local data only if requested or if single-rank
        if local or comm.Get_size() == 1:
            if count > 0:
                return buf.property_tensors[prop_idx][start:start + count].get()
            return np.empty((0,), dtype=np.float32)

        # Multi-rank gather to rank 0
        return self._gatherv_breed_gpu(
            buf.property_tensors[prop_idx], start, count, comm)

    def get_breed_agent_ids(self, breed_name, local=False):
        """Download agent IDs for agents of a breed.

        :param breed_name: Name of the breed
        :param local: If True, return local data only (no MPI gather).
                     If False (default), gather all data to rank 0.
        :return: If local=True: numpy array of local agents' IDs on ALL ranks
                 If local=False: numpy array of all agents' IDs on rank 0, None on other ranks
        """
        breed = self._agent_factory._breeds[breed_name]
        buf = self._gpu_buffers
        start, count = buf.breed_ranges.get(breed._breedidx, (0, 0))

        comm = MPI.COMM_WORLD

        # Return local data only if requested or if single-rank
        if local or comm.Get_size() == 1:
            if count > 0:
                return buf.agent_ids_gpu[start:start + count].get()
            return np.empty((0,), dtype=np.float32)

        # Multi-rank gather to rank 0
        return self._gatherv_breed_gpu(
            buf.agent_ids_gpu, start, count, comm)

    def _gatherv_breed_gpu(self, gpu_tensor, start, count, comm):
        """Gatherv a slice of a GPU tensor to rank 0.

        Uses GPU-Direct MPI when available, CPU staging otherwise.
        Returns numpy array on rank 0, None on other ranks.
        """
        my_rank = comm.Get_rank()
        num_workers = comm.Get_size()

        # Width per agent (1D or 2D tensor)
        if gpu_tensor.ndim > 1:
            width = gpu_tensor.shape[1]
        else:
            width = 1
        local_elems = count * width

        # Gather counts from all ranks
        all_counts = np.empty(num_workers, dtype=np.int32)
        comm.Allgather(np.array([local_elems], dtype=np.int32), all_counts)
        total_elems = int(all_counts.sum())

        # Displacements for Gatherv
        displs = np.zeros(num_workers, dtype=np.int32)
        displs[1:] = np.cumsum(all_counts[:-1])

        # Get local slice (stay on GPU if possible)
        if count > 0:
            local_gpu = gpu_tensor[start:start + count]
            if local_gpu.ndim > 1:
                local_gpu = local_gpu.ravel()
        else:
            local_gpu = None

        gpu_aware = is_gpu_aware_mpi()

        if gpu_aware and local_gpu is not None:
            # GPU-Direct path: MPI reads GPU memory directly
            send_buf = [local_gpu, local_elems, MPI.FLOAT]
        else:
            # CPU staging: download local slice
            if local_gpu is not None:
                local_cpu = local_gpu.get()
            else:
                local_cpu = np.empty(0, dtype=np.float32)
            send_buf = [local_cpu, local_elems, MPI.FLOAT]

        if my_rank == 0:
            if gpu_aware:
                recv_gpu = cp.empty(total_elems, dtype=cp.float32)
                recv_buf = [recv_gpu, (all_counts, displs), MPI.FLOAT]
            else:
                recv_cpu = np.empty(total_elems, dtype=np.float32)
                recv_buf = [recv_cpu, (all_counts, displs), MPI.FLOAT]
        else:
            recv_buf = None

        comm.Gatherv(send_buf, recv_buf, root=0)

        if my_rank == 0:
            if gpu_aware:
                result = recv_gpu.get()
            else:
                result = recv_cpu
            if width > 1:
                result = result.reshape(-1, width)
            return result
        return None

    def get_agents_with(self, query: Callable) -> Set[List[Any]]:
        return self._agent_factory.get_agents_with(query=query)

    def register_global_property(self, property_name: str, value) -> int:
        """Register a global tensor. Value: scalar, list, or numpy array.
        Shape is preserved on GPU. Returns index (position in kernel params).
        """
        idx = len(self._global_tensors)
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        self._global_tensors.append(arr)
        self._global_names[property_name] = idx
        self._globals_dirty = True
        return idx

    def set_global_property_value(self, property_name: str, value) -> None:
        """Update a registered global in-place. Must match original shape."""
        idx = self._global_names[property_name]
        arr = np.atleast_1d(np.asarray(value, dtype=np.float64))
        self._global_tensors[idx][:] = arr
        self._globals_dirty = True

    def get_global_property_value(self, property_name: str):
        """Get the current value of a registered global."""
        idx = self._global_names[property_name]
        arr = self._global_tensors[idx]
        if arr.size == 1:
            return float(arr.flat[0])
        return arr.copy()

    @property
    def globals(self):
        """Dict view of registered globals: name → numpy array."""
        return {name: self._global_tensors[idx]
                for name, idx in self._global_names.items()}

    def register_breed_local_array(self, name: str, breed: Breed, shape_per_agent: tuple,
                                    neighbor_visible: bool = False) -> int:
        """Register a named per-breed GPU array accessible from step functions.

        Allocates one row per local agent of the given breed (+ ghost agents if
        neighbor_visible=True). The framework builds an index map so step functions
        can access the array via: array[idx_map[agent_index]].

        Double-buffering follows the same rules as properties: SAGESim detects
        writes and creates write buffers unless the array name is listed in
        a step function's no_double_buffer parameter.

        Args:
            name: Parameter name in step function signatures.
            breed: The breed this array belongs to. Determines row count.
            shape_per_agent: Column shape per agent (e.g. (50, 2) for 3D, (100,) for 2D).
            neighbor_visible: If True, ghost agents get rows and data is exchanged via MPI.
        """
        idx = len(self._breed_local_arrays)
        self._breed_local_arrays.append({
            'name': name,
            'breed': breed,
            'shape_per_agent': shape_per_agent,
            'neighbor_visible': neighbor_visible,
        })
        return idx

    def get_breed_local_array(self, name: str) -> np.ndarray:
        """Download a breed-local array from GPU to CPU."""
        idx = next(i for i, b in enumerate(self._breed_local_arrays) if b['name'] == name)
        return self._gpu_buffers.device_breed_locals[idx].get()

    def set_seed(self, seed: int) -> None:
        """Set the random seed for reproducibility. If not called, a random seed is used."""
        self._seed = int(seed)

    def set_agent_logical_id(self, agent_id: int, logical_id: int) -> None:
        """Set a stable logical ID used as the RNG key for this agent.

        By default, the RNG uses the global agent_id (creation-order) as
        the Philox counter key.  When logical IDs are set, the RNG uses
        them instead, making random sequences independent of agent
        creation order.  Call before setup().

        :param agent_id: Global agent ID returned by create_agent_of_breed
        :param logical_id: Stable identifier (e.g. site_slot, gap_index)
        """
        self._logical_id_map[int(agent_id)] = int(logical_id)

    def load_partition(self, partition_file: str, format: str = "auto") -> None:
        """Load network partition from file to optimize multi-worker performance.

        This method loads a pre-computed network partition that assigns agents to MPI ranks
        to minimize cross-worker communication. Must be called BEFORE creating any agents.

        For details on partition formats and usage, see AgentFactory.load_partition().

        :param partition_file: Path to partition file
        :param format: File format ('pickle', 'json', 'numpy', 'text', or 'auto')
        """
        self._agent_factory.load_partition(partition_file, format)

    def setup(self, use_gpu: bool = True, skip_priority_barriers=False) -> None:
        """
        Must be called before first simulate call.
        Initializes model and resets ticks. Readies step functions
        and for breeds.

        :param use_cuda: runs model in GPU mode.
        :param num_dask_worker: number of dask workers
        :param scheduler_fpath: specify if using external dask cluster. Else
            distributed.LocalCluster is set up.
        """
        import time
        t_setup_total_start = time.time()

        self._use_gpu = use_gpu
        self._skip_priority_barriers = skip_priority_barriers
        # Globals uploaded to GPU in _build_gpu_buffers()
        self.tick = 0

        ####
        # Create record of agent step functions by breed and priority
        self._breed_idx_2_step_func_by_priority: List[Dict[int, Callable]] = []
        self._priority_values: List[int] = []  # actual priority value per slot index
        heap_priority_breedidx_func = []
        for breed in self._agent_factory.breeds:
            for priority, func in breed.step_funcs.items():
                heap_priority_breedidx_func.append((priority, (breed._breedidx, func)))
        heapq.heapify(heap_priority_breedidx_func)
        last_priority = None
        while heap_priority_breedidx_func:
            priority, breed_idx_func = heapq.heappop(heap_priority_breedidx_func)
            if last_priority == priority:
                # same slot in self._breed_idx_2_step_func_by_priority
                self._breed_idx_2_step_func_by_priority[-1].update(
                    {breed_idx_func[0]: breed_idx_func[1]}
                )
            else:
                # new slot
                self._breed_idx_2_step_func_by_priority.append(
                    {breed_idx_func[0]: breed_idx_func[1]}
                )
                self._priority_values.append(priority)
                last_priority = priority


        # Collect all no_double_buffer property names from all breeds
        no_double_buffer_prop_names = set()
        for breed in self._agent_factory.breeds:
            no_double_buffer_prop_names.update(breed.no_double_buffer_props)

        # Convert property names to indices
        no_double_buffer_indices = set()
        for prop_name in no_double_buffer_prop_names:
            if prop_name in self._agent_factory._property_name_2_index:
                no_double_buffer_indices.add(
                    self._agent_factory._property_name_2_index[prop_name]
                )
            else:
                pass

        # Determine and cache write property indices once during setup
        self._write_property_indices = set()
        for breed_idx_2_step_func in self._breed_idx_2_step_func_by_priority:
            for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
                breed_step_func_impl, module_fpath = breed_step_func_info
                write_indices = analyze_step_function_for_writes(
                    breed_step_func_impl, self._agent_factory.num_properties,
                    num_breed_local_params=len(self._breed_local_arrays) * 2)
                # Note: num_breed_local_params=N*2 is for the ORIGINAL step func (before DB transform).
                # The write analysis runs on original source which has array+idx per BLA.
                self._write_property_indices.update(write_indices)

        # Exclude no_double_buffer properties from write buffer creation
        self._write_property_indices = self._write_property_indices - no_double_buffer_indices

        # Determine which breed-local arrays need write buffers
        bla_names = [bla['name'] for bla in self._breed_local_arrays]
        no_double_buffer_bla_names = no_double_buffer_prop_names & set(bla_names)
        self._write_bla_names = set()
        for breed_idx_2_step_func in self._breed_idx_2_step_func_by_priority:
            for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
                breed_step_func_impl, module_fpath = breed_step_func_info
                written = analyze_step_function_for_bla_writes(
                    breed_step_func_impl, bla_names)
                self._write_bla_names.update(written)
        self._write_bla_names = self._write_bla_names - no_double_buffer_bla_names


        # Sort write property indices for consistent ordering
        self._write_property_indices = sorted(self._write_property_indices)

        t_analysis_end = time.time()

        # Sort agents by breed for range-bounded kernel loops
        self._agent_factory.sort_by_breed()

        t_sort_end = time.time()

        # Generate agent data tensors early so we can inspect property shapes
        self.__rank_local_agent_data_tensors = (
            self._agent_factory._generate_agent_data_tensors()
        )

        t_tensors_end = time.time()

        # Compute property column counts for write-back code generation
        # 0 = scalar (1D array), >1 = number of columns (2D array)
        property_ndims = {}
        for prop_idx in self._write_property_indices:
            if prop_idx == 1:
                continue
            data = self.__rank_local_agent_data_tensors[prop_idx]
            if data and isinstance(data[0], (list, tuple, np.ndarray)):
                property_ndims[prop_idx] = len(data[0])
            else:
                property_ndims[prop_idx] = 0  # scalar

        t_codegen_start = time.time()
        if worker == 0:
            # Delete stale bytecode cache before writing new source — prevents
            # a previous process's .pyc from being loaded instead of the fresh .py
            _pyc = importlib.util.cache_from_source(self._step_function_file_path)
            if os.path.exists(_pyc):
                os.remove(_pyc)
            with open(self._step_function_file_path, "w", encoding="utf-8") as f:
                f.write(
                    generate_gpu_func(
                        len(self._global_tensors),
                        self._agent_factory.num_properties,
                        self._breed_idx_2_step_func_by_priority,
                        self._write_property_indices,
                        property_ndims,
                        extra_kernel_config=self._get_extra_kernel_config(),
                        skip_priority_barriers=self._skip_priority_barriers,
                        priority_values=self._priority_values,
                        global_scalar_flags=[t.size == 1 for t in self._global_tensors],
                        breed_local_names=[bla['name'] for bla in self._breed_local_arrays],
                        write_bla_names=self._write_bla_names,
                        write_bla_shapes={
                            bla['name']: bla['shape_per_agent']
                            for bla in self._breed_local_arrays
                            if bla['name'] in self._write_bla_names
                        } if self._write_bla_names else None,
                    )
                )
        comm.barrier()
        t_codegen_end = time.time()

        # Import and cache the step function once during setup
        # Suppress expected CuPy/Numba JIT compilation warnings
        t_jit_start = time.time()

        # Add user module directories to sys.path so generated code can import them
        added_paths = []
        for breed_step_funcs in self._breed_idx_2_step_func_by_priority:
            for breedidx, (func, module_fpath) in breed_step_funcs.items():
                module_dir = str(Path(module_fpath).parent)  # Already absolute from register_step_func
                if module_dir not in sys.path:
                    sys.path.insert(0, module_dir)
                    added_paths.append(module_dir)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", message=".*numba.*", category=Warning)
            abs_path = self._step_function_file_path  # Already absolute from __init__
            module_name = os.path.splitext(os.path.basename(abs_path))[0]
            # Reuse existing module if source unchanged — avoids CuPy JIT
            # recompilation which can produce non-deterministic CUDA binaries
            # (NVRTC compiles by function object identity, not source content).
            with open(abs_path, 'r') as f:
                source_hash = hash(f.read())
            existing = sys.modules.get(module_name)
            if existing is not None and getattr(existing, '_source_hash', None) == source_hash:
                step_func_module = existing
            else:
                sys.modules.pop(module_name, None)
                importlib.invalidate_caches()
                spec = importlib.util.spec_from_file_location(module_name, abs_path)
                step_func_module = importlib.util.module_from_spec(spec)
                step_func_module._source_hash = source_hash
                sys.modules[module_name] = step_func_module
                spec.loader.exec_module(step_func_module)
        self._step_func = step_func_module.stepfunc
        t_jit_end = time.time()
        ###

        # Print agent distribution summary
        if worker == 0:
            total_agents = self._agent_factory._num_agents
            agents_per_rank = {}
            a2r = self._agent_factory._agent2rank
            for agent_id in range(total_agents):
                r = int(a2r[agent_id]) if isinstance(a2r, np.ndarray) else a2r.get(agent_id, -1)
                agents_per_rank[r] = agents_per_rank.get(r, 0) + 1


        self._is_setup = True

        # Initialize GPU buffer manager (buffers allocated lazily on first tick)
        self._gpu_buffers = GPUBufferManager(
            self._agent_slack_factor, self._csr_slack_factor, self._min_capacity)
        self._gpu_aware_mpi = is_gpu_aware_mpi()

        t_setup_total_end = time.time()

    # ------------------------------------------------------------------
    # Overridable hooks for subclass-specific GPU kernel extensions
    # ------------------------------------------------------------------

    def _get_extra_kernel_config(self) -> dict:
        """Override to inject extra GPU kernel params and post-step code.
        Returns dict with optional keys:
          'extra_kernel_params': list[str]         — names for kernel signature
          'post_breed_step_code': list[tuple]      — [(code_lines, once_per_breed[, only_priority]), ...]
        """
        return {}

    def _build_bla_launch_args(self, buf):
        """Build kernel launch args for breed-local arrays.

        For each BLA, emits: array, [write_array + nrows if double-buffered], idx_map.
        """
        args = []
        for i, bla in enumerate(self._breed_local_arrays):
            args.append(buf.device_breed_locals[i])
            wb = buf.device_breed_local_write_bufs[i] if i < len(buf.device_breed_local_write_bufs) else None
            if wb is not None:
                args.append(wb)
                args.append(cp.float32(buf.device_breed_locals[i].shape[0]))
            args.append(buf.device_breed_local_idxs[i])
        return args

    def _prepare_kernel_extras(self, num_local_agents, sync_ticks) -> tuple:
        """Override to allocate/reset extra GPU buffers. Returns extra kernel args tuple."""
        return ()

    def _process_kernel_extras(self) -> None:
        """Override to download/process extra GPU data after kernel execution."""
        pass

    def _sync_gpu_to_agent_factory(self):
        """Download all GPU properties back to AgentFactory storage."""
        buf = self._gpu_buffers
        num_local = buf.num_local_agents
        idx_to_name = {v: k for k, v in self._agent_factory._property_name_2_index.items()}

        for prop_idx in range(self._agent_factory.num_properties):
            if prop_idx in (0, 1):  # breed (never changes), CSR/locations (skip)
                continue
            if buf.property_tensors[prop_idx] is None:
                continue
            prop_name = idx_to_name[prop_idx]
            gpu_data = buf.property_tensors[prop_idx][:num_local].get().tolist()
            self._agent_factory._property_name_2_agent_data_tensor[prop_name] = gpu_data

    def _regenerate_data_tensors(self):
        """Rebuild __rank_local_agent_data_tensors from AgentFactory."""
        self.__rank_local_agent_data_tensors = (
            self._agent_factory._generate_agent_data_tensors()
        )

    def reset(self) -> None:
        self.tick = 0
        self._globals_dirty = True  # re-upload on next _build_gpu_buffers

        # Sync GPU state back to AgentFactory before freeing
        if hasattr(self, '_gpu_buffers') and self._gpu_buffers.is_initialized:
            self._sync_gpu_to_agent_factory()

        self._regenerate_data_tensors()

        if hasattr(self, '_gpu_buffers'):
            self._gpu_buffers.free()
            self._gpu_buffers = GPUBufferManager(
                self._agent_slack_factor, self._csr_slack_factor, self._min_capacity)
        self._cached_all_args = None


    def simulate(
        self,
        ticks: int,
        sync_workers_every_n_ticks: int = 1,
    ) -> None:
        # Store ticks for verbose_timing to know when simulation ends
        self._last_simulate_ticks = ticks
        self._tick_timings = []

        if self._verbose_timing:
            import time
            t_barrier_start = time.perf_counter()
        comm.barrier()
        if self._verbose_timing and worker == 0:
            t_barrier_wait = time.perf_counter() - t_barrier_start
            print(f"[TIMING] MPI barrier wait at simulate start: {t_barrier_wait*1000:.2f} ms", flush=True)

        # Step function is cached during setup() - no need to reimport
        # Single worker: fuse all ticks in one kernel launch (no MPI sync needed)
        if num_workers == 1:
            self.worker_coroutine(ticks)
        else:
            # Multi-worker: need periodic synchronization
            original_sync_workers_every_n_ticks = sync_workers_every_n_ticks
            for time_chunk in range((ticks // original_sync_workers_every_n_ticks) + 1):
                if time_chunk == (ticks // original_sync_workers_every_n_ticks):
                    # Final chunk: handle remaining ticks
                    remaining_ticks = ticks - (
                        time_chunk * original_sync_workers_every_n_ticks
                    )
                    if remaining_ticks == 0:
                        break
                    sync_workers_every_n_ticks = remaining_ticks
                else:
                    # Regular chunk: use original batch size
                    sync_workers_every_n_ticks = original_sync_workers_every_n_ticks

                self.worker_coroutine(sync_workers_every_n_ticks)

    # ----------------------------------------------------------------
    # GPU-resident buffer helpers
    # ----------------------------------------------------------------

    @staticmethod
    def _create_zero_placeholder(sample):
        """Recursively create a zero-filled copy matching the structure of sample."""
        if isinstance(sample, np.ndarray):
            return np.zeros_like(sample)
        elif isinstance(sample, (list, tuple, set)):
            if len(sample) == 0:
                return []
            sample_list = list(sample) if isinstance(sample, set) else sample
            if isinstance(sample_list[0], (list, tuple, set, np.ndarray)):
                return [Model._create_zero_placeholder(elem) for elem in sample_list]
            else:
                return [0.0] * len(sample)
        else:
            return 0.0

    def _build_gpu_buffers(self, ghost_ids, num_local_agents, comm=None):
        """Build all persistent GPU buffers on first tick.

        Ghost agent slots are filled with placeholder zeros. Actual ghost
        values are populated by CommunicationManager.exchange_ghost_data()
        after build_communication_maps().
        """
        import time
        sub_timing = {}
        do_time = self._verbose_timing

        buf = self._gpu_buffers
        buf.num_local_agents = num_local_agents
        num_ghost = len(ghost_ids)


        # Compute per-priority breed ranges from sorted breed data
        if do_time:
            _t0 = time.time()
        breed_data = self.__rank_local_agent_data_tensors[0]  # property 0 = breed
        breed_ranges = {}  # breed_id -> (start, count)
        if num_local_agents > 0:
            breeds = np.array(breed_data, dtype=np.int32)
            unique, counts = np.unique(breeds, return_counts=True)
            start = 0
            for bid, cnt in zip(unique, counts):
                breed_ranges[int(bid)] = (start, int(cnt))
                start += int(cnt)

        buf.breed_ranges = breed_ranges

        buf.priority_ranges = {}
        for p_idx, breed_step_dict in enumerate(self._breed_idx_2_step_func_by_priority):
            min_start, max_end = num_local_agents, 0
            for bid in breed_step_dict.keys():
                if bid in breed_ranges:
                    s, c = breed_ranges[bid]
                    min_start = min(min_start, s)
                    max_end = max(max_end, s + c)
            if max_end > min_start:
                buf.priority_ranges[p_idx] = (min_start, max_end - min_start)
            else:
                buf.priority_ranges[p_idx] = (0, 0)

        if do_time:
            sub_timing['breed_ranges'] = time.time() - _t0
            _t0 = time.time()

        # 1. Build agent ID list and CPU hash map
        all_agent_ids_np = np.concatenate([self.__rank_local_agent_ids, ghost_ids])
        agent_id_to_index = {int(aid): idx for idx, aid in enumerate(all_agent_ids_np)}
        buf.all_agent_ids_list = all_agent_ids_np
        buf.agent_id_to_index = agent_id_to_index
        buf.num_total_agents = len(all_agent_ids_np)
        buf.prev_ghost_ids_set = set(ghost_ids)

        if do_time:
            sub_timing['id_cpu_dict'] = time.time() - _t0
            _t1 = time.time()

        # 2. Pre-allocate capacity with slack
        agent_capacity = max(buf.MIN_CAPACITY,
                             int(buf.num_total_agents * buf.AGENT_SLACK_FACTOR))

        # 3. Upload agent IDs to GPU with slack (pad on CPU, single memcpy)
        agent_ids_padded = np.full(agent_capacity, -1, dtype=np.float32)
        agent_ids_padded[:buf.num_total_agents] = all_agent_ids_np.astype(np.float32)
        buf.agent_ids_gpu = cp.array(agent_ids_padded)

        # 3b. Build logical_ids for stable RNG (defaults to agent_ids if not set)
        logical_ids_padded = agent_ids_padded.copy()
        if self._logical_id_map:
            for aid, lid in self._logical_id_map.items():
                idx = agent_id_to_index.get(int(aid))
                if idx is not None:
                    logical_ids_padded[idx] = float(lid)
        buf.logical_ids_gpu = cp.array(logical_ids_padded)

        if do_time:
            sub_timing['id_gpu_upload'] = time.time() - _t1
            _t1 = time.time()

        # 4. Upload global tensors and seed to GPU
        buf.device_globals = [cp.asarray(t) for t in self._global_tensors]
        buf.seed_gpu = cp.int32(self._seed)
        self._globals_dirty = False

        if do_time:
            sub_timing['id_global_data'] = time.time() - _t1
            _t1 = time.time()

        # 5. Build GPU hash map (reuse all_agent_ids_np instead of converting again)
        buffer_indices_np = np.arange(len(all_agent_ids_np), dtype=np.int32)
        hash_capacity = max(buf.MIN_CAPACITY, len(all_agent_ids_np) * 2)
        buf.hash_map = GPUHashMap(hash_capacity)
        buf.hash_map.build_from_arrays(all_agent_ids_np, buffer_indices_np)

        if do_time:
            sub_timing['id_gpu_hashmap'] = time.time() - _t1
            sub_timing['id_hashmap'] = time.time() - _t0
            _t0 = time.time()

        # 6. Combine local + ghost placeholder data and build GPU arrays
        combined_lists = []
        for i in range(self._agent_factory.num_properties):
            local_data = self.__rank_local_agent_data_tensors[i]

            if i == 1:
                # CSR: ghost agents get empty neighbor lists (never iterated by kernel)
                ghost_data = [[] for _ in range(num_ghost)]
            else:
                # Other properties: ghost agents get zero placeholders
                if local_data:
                    placeholder = self._create_zero_placeholder(local_data[0])
                else:
                    prop_names = list(self._agent_factory._property_name_2_defaults.keys())
                    placeholder = self._create_zero_placeholder(
                        self._agent_factory._property_name_2_defaults[prop_names[i]]
                    )
                ghost_data = [placeholder] * num_ghost

            combined = local_data + ghost_data

            if i == 1:
                # Build dual CSR: values with agent IDs (for MPI) and local indices (for kernel)
                offsets_np, values_ids_np = build_csr_from_ragged(combined)
                combined_indices = convert_agent_ids_to_indices(combined, agent_id_to_index)
                values_np = build_csr_values_only(combined_indices, offsets_np)
                buf.allocate_csr(offsets_np, values_np, values_ids_np, buf.num_total_agents)

            combined_lists.append(combined)

        if do_time:
            sub_timing['combined_data'] = time.time() - _t0
            _t0 = time.time()

        # 6b. Synchronize property widths across ranks (MPI_MAX) so all
        # ranks allocate tensors with identical shapes for ghost exchange.
        if comm is not None and comm.Get_size() > 1:
            num_props = self._agent_factory.num_properties
            local_widths = np.ones(num_props, dtype=np.int32)
            for i in range(num_props):
                if i == 1:
                    local_widths[i] = 0  # CSR, skip
                    continue
                data = combined_lists[i]
                if data and isinstance(data[0], (list, tuple, np.ndarray)):
                    local_widths[i] = max(
                        len(row) if isinstance(row, (list, tuple, np.ndarray)) else 1
                        for row in data
                    )
            global_widths = np.empty_like(local_widths)
            comm.Allreduce(local_widths, global_widths, op=MPI.MAX)
            # Pad rows to global max width where needed
            for i in range(num_props):
                if i == 1 or global_widths[i] <= 1:
                    continue
                gw = int(global_widths[i])
                for j in range(len(combined_lists[i])):
                    row = combined_lists[i][j]
                    if isinstance(row, (list, tuple)):
                        if len(row) < gw:
                            combined_lists[i][j] = list(row) + [0.0] * (gw - len(row))
                    elif isinstance(row, np.ndarray):
                        if len(row) < gw:
                            combined_lists[i][j] = list(row) + [0.0] * (gw - len(row))

        if do_time:
            sub_timing['mpi_sync'] = time.time() - _t0
            _t0 = time.time()

        # 7. Allocate property tensors on GPU with slack
        buf.allocate_property_tensors(
            self._agent_factory.num_properties,
            combined_lists,
            agent_capacity,
            convert_to_padded_gpu_tensor,
        )
        buf.agent_capacity = agent_capacity

        if do_time:
            sub_timing['prop_tensors'] = time.time() - _t0
            _t0 = time.time()

        # 8. Create write buffers
        sorted_write_indices = sorted(i for i in self._write_property_indices if i != 1)
        buf.allocate_write_buffers(sorted_write_indices)

        # 9. Allocate barrier counter for fused-kernel grid barrier
        # Must be int32 to match barrier arithmetic (int32 num_blocks_param, int literals)
        buf.barrier_counter = cp.zeros(1, dtype=cp.int32)

        if do_time:
            sub_timing['write_bufs'] = time.time() - _t0

        # 10. Allocate breed-local arrays and index maps (with ghost rows)
        buf.device_breed_locals = []
        buf.device_breed_local_idxs = []
        buf.breed_local_ghost_info = []  # per-BLA metadata for ghost exchange

        num_ghost = len(ghost_ids)
        agent2breed = self._agent_factory._agent2breed

        for bla in self._breed_local_arrays:
            breed_id = bla['breed']._breedidx
            start, count = breed_ranges.get(breed_id, (0, 0))

            ghost_rows = 0
            ghost_breed_mask = None
            if bla['neighbor_visible'] and num_ghost > 0:
                ghost_breed_ids = np.array(
                    [agent2breed.get(int(gid), -1) for gid in ghost_ids],
                    dtype=np.int32)
                ghost_breed_mask = (ghost_breed_ids == breed_id)
                ghost_rows = int(ghost_breed_mask.sum())

            total_rows = max(count + ghost_rows, 1)
            shape = (total_rows, *bla['shape_per_agent'])
            buf.device_breed_locals.append(cp.zeros(shape, dtype=cp.float32))

            # Vectorized index map: buffer_index → row in breed-local array
            idx_map = np.full(agent_capacity, -1, dtype=np.int32)
            if count > 0:
                idx_map[start:start + count] = np.arange(count, dtype=np.int32)

            if ghost_breed_mask is not None and ghost_rows > 0:
                ghost_offsets = np.where(ghost_breed_mask)[0]
                idx_map[num_local_agents + ghost_offsets] = np.arange(
                    count, count + ghost_rows, dtype=np.int32)

            buf.device_breed_local_idxs.append(cp.array(idx_map))

            # Store ghost metadata for CommunicationManager
            if bla['neighbor_visible']:
                buf.breed_local_ghost_info.append({
                    'breed_id': breed_id,
                    'local_rows': count,
                    'ghost_rows': ghost_rows,
                    'shape_per_agent': bla['shape_per_agent'],
                })
            else:
                buf.breed_local_ghost_info.append(None)

        # 11. Allocate write buffers for double-buffered breed-local arrays
        buf.device_breed_local_write_bufs = []
        for i, bla in enumerate(self._breed_local_arrays):
            if bla['name'] in self._write_bla_names:
                buf.device_breed_local_write_bufs.append(
                    buf.device_breed_locals[i].copy())
            else:
                buf.device_breed_local_write_bufs.append(None)

        buf.is_initialized = True

        return sub_timing if do_time else None

    def _download_local_data_to_cpu(self, num_local_agents):
        """Download only modified local agent data from GPU to CPU.

        Only downloads properties that the kernel could have written to.
        Unwritten properties retain their CPU-side values (unchanged by kernel).
        Property 1 (CSR/locations) is read-only — skip unless topology changed.
        """
        buf = self._gpu_buffers

        for prop_idx in buf.sorted_write_indices:
            self.__rank_local_agent_data_tensors[prop_idx] = \
                buf.property_tensors[prop_idx][:num_local_agents].get().tolist()

    def save(self, app: "Model", fpath: str) -> None:
        """
        Saves model. Must be overridden if additional data
        pertaining to application must be saved.

        :param fpath: file path to save pickle file at
        :param app_data: additional application data to be saved.
        """
        if "_agent_data_tensors" in app.__dict__:
            del app.__dict__["_agent_data_tensors"]
        with open(fpath, "wb") as fout:
            pickle.dump(app, fout)

    def load(self, fpath: str) -> "Model":
        """
        Loads model from pickle file.

        :param fpath: file path to pickle file.
        """
        with open(fpath, "rb") as fin:
            app = pickle.load(fin)
        return app

    # Define worker coroutine that executes cuda kernel
    # ------------------------------------------------------
    def worker_coroutine(
        self,
        sync_workers_every_n_ticks,
    ):
        """
        Coroutine that executes CUDA kernel with GPU-resident persistent buffers.

        On first call: builds all GPU buffers from scratch and stores them persistently.
        On subsequent calls: selectively downloads modified properties for MPI,
        exchanges data, selectively uploads ghost values, then runs the kernel.
        """
        import time
        t_start = time.time()

        # OPTIMIZATION: Cache agent IDs if topology hasn't changed (fused tick execution)
        # For single-worker simulations with fused ticks, agent dict remains unchanged
        current_dict_id = id(self._agent_factory._rank2agentid2agentidx[worker])
        if not hasattr(self, '_cached_rank_local_agent_ids'):
            # First call - build and cache
            self.__rank_local_agent_ids = np.array(
                list(self._agent_factory._rank2agentid2agentidx[worker].keys()),
                dtype=np.int64,
            )
            self._cached_rank_local_agent_ids = self.__rank_local_agent_ids
            self._cached_topology_dict_id = current_dict_id
        elif current_dict_id != self._cached_topology_dict_id:
            # Topology changed - rebuild cache
            self.__rank_local_agent_ids = np.array(
                list(self._agent_factory._rank2agentid2agentidx[worker].keys()),
                dtype=np.int64,
            )
            self._cached_rank_local_agent_ids = self.__rank_local_agent_ids
            self._cached_topology_dict_id = current_dict_id
        else:
            # Reuse cached array (common case for fused tick execution)
            self.__rank_local_agent_ids = self._cached_rank_local_agent_ids

        timing_data = {} if self._verbose_timing else None

        num_local_agents = len(self.__rank_local_agent_ids)
        threadsperblock = 128
        blockspergrid = int(math.ceil(num_local_agents / threadsperblock))

        buf = self._gpu_buffers

        # ============================================================
        # DATA PREPARATION: first-tick vs subsequent-tick paths
        # ============================================================
        t_data_prep_start = time.time()

        if not buf.is_initialized:
            # --- FIRST TICK: full build ---
            t_neighbor_start = time.time()
            rank_local_agents_neighbors = self.get_space()._neighbor_compute_func(
                self.__rank_local_agent_data_tensors[1]
            )
            t_neighbor_end = time.time()

            t_before_context = time.time()
            ghost_ids = discover_ghost_topology(
                rank_local_agents_neighbors,
                self._agent_factory._agent2rank,
                worker,
            )
            t_after_context = time.time()

            if self._verbose_timing:
                timing_data['neighbor'] = t_neighbor_end - t_neighbor_start
                timing_data['contextualize'] = t_after_context - t_before_context
                timing_data['num_neighbors'] = len(ghost_ids)

            t_build_start = time.time()
            gpu_sub_timing = self._build_gpu_buffers(ghost_ids, num_local_agents, comm)
            self._cached_all_args = None  # buffers rebuilt, invalidate cache
            t_build_end = time.time()

            if self._verbose_timing:
                timing_data['gpu_buffer_build'] = t_build_end - t_build_start
                if gpu_sub_timing:
                    timing_data['gpu_build_sub'] = gpu_sub_timing

            # Initialize CommunicationManager and fill ghost data from tick 1
            t_comm_init_start = time.time()
            if num_workers > 1:
                self._comm_manager = CommunicationManager(
                    buf, self._agent_factory, worker, num_workers, comm, verbose_timing=self._verbose_timing
                )
                self._comm_manager.build_communication_maps()
                mpi_timing = self._comm_manager.exchange_ghost_data()
                if self._verbose_timing and mpi_timing:
                    timing_data.update(mpi_timing)
            t_comm_init_end = time.time()

            if self._verbose_timing:
                timing_data['comm_init'] = t_comm_init_end - t_comm_init_start

        else:
            # --- SUBSEQUENT TICK: reuse existing GPU buffers ---
            t_before_context = time.time()

            if num_workers > 1 and hasattr(self, '_comm_manager') and self._comm_manager.is_initialized:
                mpi_timing = self._comm_manager.exchange_ghost_data()
                if self._verbose_timing and mpi_timing:
                    timing_data.update(mpi_timing)

            t_after_context = time.time()

            if self._verbose_timing:
                timing_data['contextualize'] = t_after_context - t_before_context

        t_data_prep_end = time.time()

        if self._verbose_timing:
            timing_data['data_prep'] = t_data_prep_end - t_data_prep_start

        # ============================================================
        # GPU KERNEL EXECUTION (fused: all ticks + priorities in one launch)
        # ============================================================

        # Build all_args from persistent GPU buffers (cached across ticks)
        if self._verbose_timing:
            t_kernel_args_start = time.time()
        if not hasattr(self, '_cached_all_args') or self._cached_all_args is None:
            all_args = []
            for i in range(self._agent_factory.num_properties):
                if i == 1:
                    all_args.append(buf.neighbor_offsets)
                    all_args.append(buf.neighbor_values)
                else:
                    all_args.append(buf.property_tensors[i])
            all_args = all_args + buf.write_buffers
            self._cached_all_args = all_args
        all_args = self._cached_all_args

        # Compute max co-resident blocks for grid barrier safety.
        # Grid barriers require ALL launched blocks to be co-resident on the GPU.
        # The hardware limit (MaxBlocksPerMultiprocessor) is the theoretical max,
        # but actual occupancy depends on per-kernel register and shared memory
        # pressure. Complex kernels (many step functions, large variable sets)
        # use more registers, reducing actual blocks per SM well below the
        # hardware limit. Launching more blocks than can be co-resident causes
        # deadlock: active blocks wait at barrier for non-resident blocks that
        # can't start until active blocks finish.
        # We use a conservative default (2 blocks/SM) that works reliably for
        # complex kernels. Can be tuned up for simpler kernels.

        # OPTIMIZATION: Cache GPU configuration (SM count detection)
        if not hasattr(self, '_cached_num_sms'):
            import os
            dev = cp.cuda.Device()
            attrs = dev.attributes
            num_sms = attrs['MultiProcessorCount']

            # AMD GPU detection: CuPy's HIP backend doesn't correctly report CU count
            # See CUPY_ISSUE.md for details
            if 'SAGESIM_NUM_SMS' in os.environ:
                num_sms = int(os.environ['SAGESIM_NUM_SMS'])
                if getattr(self, '_verbose', False):
                    print(f"[SAGESim] Using SAGESIM_NUM_SMS override: {num_sms} CUs")
            elif num_sms == 1:
                # Likely AMD GPU with incorrect CuPy detection
                props = cp.cuda.runtime.getDeviceProperties(dev.id)
                gpu_name = props['name'].decode('utf-8')
                if 'gfx90a' in gpu_name.lower() or 'mi250x' in gpu_name.lower() or 'mi210' in gpu_name.lower():
                    num_sms = 110
                    if getattr(self, '_verbose', False):
                        print(f"[SAGESim] Detected GPU: {gpu_name}")
                        print(f"[SAGESim] AMD MI250X/MI210 detected, using 110 CUs")

            self._cached_num_sms = num_sms

        num_sms = self._cached_num_sms
        # Conservative: 2 blocks/SM ensures co-residency even with high register pressure
        max_blocks_per_sm = getattr(self, '_max_blocks_per_sm', 2)
        max_grid_blocks = max_blocks_per_sm * num_sms
        effective_blocks = min(blockspergrid, max_grid_blocks)

        # Build range args for per-priority breed ranges
        range_args = []
        range_summary = []
        for p_idx in range(len(self._breed_idx_2_step_func_by_priority)):
            p_start, p_count = buf.priority_ranges.get(p_idx, (0, 0))
            range_args.append(cp.float32(p_start))
            range_args.append(cp.float32(p_count))
            range_summary.append(f"P{p_idx}:{p_count}")

        # Prepare subclass-specific extra kernel args (e.g. spike recording buffers)
        extra_kernel_args = self._prepare_kernel_extras(num_local_agents, sync_workers_every_n_ticks)

        if self._verbose_timing:
            timing_data['kernel_args_build'] = time.time() - t_kernel_args_start

        # Reset barrier counter and launch fused kernel
        t_kernel_launch_start = time.time()
        buf.barrier_counter[0] = 0
        # Re-upload globals if dirty (changed since last upload)
        if self._globals_dirty:
            buf.device_globals = [cp.asarray(t) for t in self._global_tensors]
            self._globals_dirty = False

        self._step_func[effective_blocks, threadsperblock](
            self.tick,
            buf.seed_gpu,
            *buf.device_globals,
            *all_args,
            sync_workers_every_n_ticks,
            cp.float32(num_local_agents),
            *range_args,
            buf.agent_ids_gpu,
            buf.logical_ids_gpu,
            buf.barrier_counter,
            cp.int32(effective_blocks),
            *extra_kernel_args,
            *self._build_bla_launch_args(buf),
        )
        if self._verbose_timing:
            t_kernel_launch_end = time.time()
            timing_data['kernel_launch_overhead'] = t_kernel_launch_end - t_kernel_launch_start

        # GPU synchronization
        t_gpu_sync_start = time.time()
        cp.cuda.get_current_stream().synchronize()
        t_gpu_sync_end = time.time()

        if self._verbose_timing:
            timing_data['gpu_sync'] = t_gpu_sync_end - t_gpu_sync_start
            timing_data['gpu_compute'] = (t_gpu_sync_end - t_kernel_launch_start) - timing_data['gpu_sync']

            # Grid barrier metrics
            num_barriers_hit = int(buf.barrier_counter[0].get())
            expected_barriers = len(self._breed_idx_2_step_func_by_priority) * sync_workers_every_n_ticks
            if not self._skip_priority_barriers:
                expected_barriers += sync_workers_every_n_ticks
            timing_data['grid_barriers_hit'] = num_barriers_hit
            timing_data['grid_barriers_expected'] = expected_barriers

        t_gpu_kernel_start = t_kernel_launch_start  # Keep for backward compat

        # Process subclass-specific extra data (e.g. download spike records)
        if self._verbose_timing:
            t_extras_start = time.time()
        self._process_kernel_extras()
        if self._verbose_timing:
            timing_data['process_extras'] = time.time() - t_extras_start

        # Final write-back: kernel does inter-tick write-backs on GPU,
        # but ensure property_tensors match write_buffers for post-kernel reads
        if self._verbose_timing:
            t_writeback_start = time.time()
        for i, prop_idx in enumerate(buf.sorted_write_indices):
            buf.property_tensors[prop_idx][:num_local_agents] = \
                buf.write_buffers[i][:num_local_agents]
        # Write-back for double-buffered breed-local arrays
        for i, wb in enumerate(buf.device_breed_local_write_bufs):
            if wb is not None:
                n = buf.device_breed_locals[i].shape[0]
                buf.device_breed_locals[i][:n] = wb[:n]
        if self._verbose_timing:
            timing_data['write_back'] = time.time() - t_writeback_start
        # No sync needed here: these are GPU→GPU copies. Next tick's
        # exchange_ghost_data() syncs before MPI reads the buffers.

        self.tick += sync_workers_every_n_ticks
        t_gpu_kernel_end = time.time()

        if self._verbose_timing:
            timing_data['gpu_kernel'] = t_gpu_kernel_end - t_gpu_kernel_start

        # ============================================================
        # POST-KERNEL: download for MPI next tick and user queries
        # ============================================================
        t_post_start = time.time()
        t_post_end = time.time()

        t_end = time.time()
        if self._verbose_timing:
            timing_data['post'] = t_post_end - t_post_start
            timing_data['total'] = t_end - t_start

            # Print per-rank timing (NO MPI aggregation to avoid deadlocks)
            # Build detailed first-tick breakdown if sub-timing is available
            prep_detail = ""
            if 'gpu_build_sub' in timing_data:
                sub = timing_data['gpu_build_sub']
                gpu_build_detail = (
                    f"gpu_build={timing_data.get('gpu_buffer_build', 0):.4f}s ("
                    f"breed_ranges={sub.get('breed_ranges', 0):.4f}s, "
                    f"id_hashmap={sub.get('id_hashmap', 0):.4f}s "
                    f"(cpu_dict={sub.get('id_cpu_dict', 0):.4f}s, "
                    f"gpu_upload={sub.get('id_gpu_upload', 0):.4f}s, "
                    f"global_data={sub.get('id_global_data', 0):.4f}s, "
                    f"gpu_hashmap={sub.get('id_gpu_hashmap', 0):.4f}s), "
                    f"combined_data={sub.get('combined_data', 0):.4f}s, "
                    f"mpi_sync={sub.get('mpi_sync', 0):.4f}s, "
                    f"prop_tensors={sub.get('prop_tensors', 0):.4f}s, "
                    f"write_bufs={sub.get('write_bufs', 0):.4f}s)"
                )
                prep_detail = (
                    f" [neighbor={timing_data.get('neighbor', 0):.4f}s, "
                    f"ghost_topo={timing_data.get('contextualize', 0):.4f}s, "
                    f"{gpu_build_detail}, "
                    f"comm_init={timing_data.get('comm_init', 0):.4f}s]"
                )

            if num_workers > 1:
                print(f"[Rank {worker}] Tick {self.tick}: "
                      f"total={timing_data['total']:.4f}s | "
                      f"prep={timing_data['data_prep']:.4f}s{prep_detail}, "
                      f"kern_args={timing_data.get('kernel_args_build', 0):.4f}s, "
                      f"gpu_compute={timing_data.get('gpu_compute', 0):.4f}s, "
                      f"gpu_sync={timing_data.get('gpu_sync', 0):.4f}s, "
                      f"extras={timing_data.get('process_extras', 0):.4f}s, "
                      f"writeback={timing_data.get('write_back', 0):.4f}s, "
                      f"mpi_pack={timing_data.get('mpi_gpu_pack', 0):.4f}s, "
                      f"mpi_wait={timing_data.get('mpi_wait_time', 0):.4f}s, "
                      f"mpi_unpack={timing_data.get('mpi_gpu_unpack', 0):.4f}s", flush=True)
            else:
                print(f"[Rank {worker}] Tick {self.tick}: "
                      f"total={timing_data['total']:.4f}s | "
                      f"prep={timing_data['data_prep']:.4f}s{prep_detail}, "
                      f"kern_args={timing_data.get('kernel_args_build', 0):.4f}s, "
                      f"gpu_compute={timing_data.get('gpu_compute', 0):.4f}s, "
                      f"gpu_sync={timing_data.get('gpu_sync', 0):.4f}s, "
                      f"extras={timing_data.get('process_extras', 0):.4f}s, "
                      f"writeback={timing_data.get('write_back', 0):.4f}s", flush=True)

            self._tick_timings.append(dict(timing_data))


class _CSRBodyTransformer(ast.NodeTransformer):
    """AST transformer that rewrites neighbor access patterns for CSR format.

    Handles these patterns in the step function body:
      - var = locations[agent_index]  → removed (var is tracked)
      - len(var)                      → neighbor_offsets[ai+1] - neighbor_offsets[ai]
      - var[i] != -1  (sentinel)      → removed from boolean conditions
      - var[i]                        → neighbor_values[neighbor_offsets[ai] + i]
      - locations[agent_index][i]     → neighbor_values[neighbor_offsets[ai] + i]
    """

    def __init__(self, locations_param, agent_index_param, neighbor_var):
        self.loc_param = locations_param
        self.agent_idx = agent_index_param
        self.nvar = neighbor_var  # May be None if no intermediate variable

    def _is_neighbor_var_subscript(self, node):
        """Check if node is neighbor_var[expr]."""
        return (self.nvar is not None and
                isinstance(node, ast.Subscript) and
                isinstance(node.value, ast.Name) and
                node.value.id == self.nvar)

    def _is_locations_agent_subscript(self, node):
        """Check if node is locations[agent_index]."""
        return (isinstance(node, ast.Subscript) and
                isinstance(node.value, ast.Name) and
                node.value.id == self.loc_param and
                isinstance(node.slice, ast.Name) and
                node.slice.id == self.agent_idx)

    def _make_num_neighbors(self):
        """AST for: neighbor_offsets[agent_index + 1] - neighbor_offsets[agent_index]"""
        return ast.BinOp(
            left=ast.Subscript(
                value=ast.Name(id='neighbor_offsets', ctx=ast.Load()),
                slice=ast.BinOp(
                    left=ast.Name(id=self.agent_idx, ctx=ast.Load()),
                    op=ast.Add(),
                    right=ast.Constant(value=1)
                ),
                ctx=ast.Load()
            ),
            op=ast.Sub(),
            right=ast.Subscript(
                value=ast.Name(id='neighbor_offsets', ctx=ast.Load()),
                slice=ast.Name(id=self.agent_idx, ctx=ast.Load()),
                ctx=ast.Load()
            )
        )

    def _make_csr_access(self, index_expr):
        """AST for: neighbor_values[neighbor_offsets[agent_index] + expr]"""
        return ast.Subscript(
            value=ast.Name(id='neighbor_values', ctx=ast.Load()),
            slice=ast.BinOp(
                left=ast.Subscript(
                    value=ast.Name(id='neighbor_offsets', ctx=ast.Load()),
                    slice=ast.Name(id=self.agent_idx, ctx=ast.Load()),
                    ctx=ast.Load()
                ),
                op=ast.Add(),
                right=index_expr
            ),
            ctx=ast.Load()
        )

    def _is_sentinel_check(self, node):
        """Check if node is: var[expr] != -1 or var[expr] == -1."""
        if not isinstance(node, ast.Compare):
            return False
        if len(node.ops) != 1 or len(node.comparators) != 1:
            return False
        if not isinstance(node.ops[0], (ast.NotEq, ast.Eq)):
            return False
        if not self._is_neighbor_var_subscript(node.left):
            return False
        comp = node.comparators[0]
        if isinstance(comp, ast.UnaryOp) and isinstance(comp.op, ast.USub):
            if isinstance(comp.operand, ast.Constant) and comp.operand.value == 1:
                return True
        if isinstance(comp, ast.Constant) and comp.value == -1:
            return True
        return False

    def visit_Assign(self, node):
        """Remove: var = locations[agent_index]"""
        if (self.nvar is not None and
            len(node.targets) == 1 and
            isinstance(node.targets[0], ast.Name) and
            node.targets[0].id == self.nvar and
            self._is_locations_agent_subscript(node.value)):
            return None  # Remove the assignment
        return self.generic_visit(node)

    def visit_BoolOp(self, node):
        """Remove sentinel checks from And conditions, then visit remaining children."""
        if isinstance(node.op, ast.And):
            new_values = []
            for val in node.values:
                if self._is_sentinel_check(val):
                    continue  # Remove sentinel check
                new_val = self.visit(val)
                if new_val is not None:
                    new_values.append(new_val)
            if len(new_values) == 0:
                return ast.Constant(value=True)
            elif len(new_values) == 1:
                return new_values[0]
            node.values = new_values
            return node
        return self.generic_visit(node)

    def visit_Call(self, node):
        """Replace: len(neighbor_var) → CSR num_neighbors.
        Also replace bare 'locations' in function call args with CSR arrays."""
        self.generic_visit(node)
        if (isinstance(node.func, ast.Name) and
            node.func.id == 'len' and
            len(node.args) == 1):
            arg = node.args[0]
            if isinstance(arg, ast.Name) and self.nvar and arg.id == self.nvar:
                return self._make_num_neighbors()
            if self._is_locations_agent_subscript(arg):
                return self._make_num_neighbors()

        # Replace bare 'locations' forwarded to sub-function calls
        # e.g., other_func(..., locations, ...) → other_func(..., neighbor_offsets, neighbor_values, ...)
        new_args = []
        changed = False
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id == self.loc_param:
                new_args.append(ast.Name(id='neighbor_offsets', ctx=ast.Load()))
                new_args.append(ast.Name(id='neighbor_values', ctx=ast.Load()))
                changed = True
            else:
                new_args.append(arg)
        if changed:
            node.args = new_args

        return node

    def visit_Subscript(self, node):
        """Replace: neighbor_var[expr] or locations[agent_index][expr] → CSR access."""
        self.generic_visit(node)
        if self._is_neighbor_var_subscript(node):
            return self._make_csr_access(node.slice)
        if (isinstance(node.value, ast.Subscript) and
            self._is_locations_agent_subscript(node.value)):
            return self._make_csr_access(node.slice)
        return node


def _find_forwarded_location_funcs(step_func, num_properties):
    """Find device functions called from step_func that receive the locations parameter.

    When a step function forwards 'locations' to a helper function (e.g., a dispatcher
    pattern), the helper also needs CSR transformation. This function identifies such
    helpers by scanning the step function's AST for calls that pass the locations
    parameter as a bare argument.

    Returns list of (func_name, func_object) pairs.
    """
    source = inspect.getsource(step_func)
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break

    if func_def is None:
        return []

    param_names = [arg.arg for arg in func_def.args.args]
    if len(param_names) < 6:
        return []

    locations_param = param_names[5]  # Property 1

    # Find all calls that pass locations as a bare argument
    called_funcs = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            for arg in node.args:
                if isinstance(arg, ast.Name) and arg.id == locations_param:
                    if isinstance(node.func, ast.Name):
                        called_funcs.add(node.func.id)
                    break

    # Resolve function names to actual function objects via the step function's module
    module = inspect.getmodule(step_func)
    result = []
    for func_name in called_funcs:
        func_obj = getattr(module, func_name, None)
        if func_obj is not None and callable(func_obj):
            result.append((func_name, func_obj))

    return result


def _auto_transform_csr(source: str, num_properties: int, n_globals: int = 1) -> str:
    """
    Auto-transform a user's step function to use CSR format for property 1 (locations).

    The user writes their step function with a single 'locations' parameter.
    This function automatically:
      1. Replaces the locations parameter with neighbor_offsets, neighbor_values
      2. Transforms body access patterns (loops, indexing, sentinel checks)

    This keeps the user-facing API unchanged while using efficient CSR internally.
    """
    tree = ast.parse(source)

    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break

    if func_def is None:
        return source

    param_names = [arg.arg for arg in func_def.args.args]

    # Standard params: tick, agent_index, [n_globals global params], agent_ids
    # Property params start after standard params
    # Property 0 (breeds) is first, property 1 (locations) is second
    n_standard = 2 + n_globals + 1  # tick, agent_index, g0..gN, agent_ids
    loc_idx = n_standard + 1  # property 0 (breeds) + 1

    if len(param_names) <= loc_idx:
        return source

    agent_index_param = param_names[1]
    locations_param = param_names[loc_idx]  # Property 1
    func_def.args.args[loc_idx] = ast.arg(arg='neighbor_offsets')
    func_def.args.args.insert(loc_idx + 1, ast.arg(arg='neighbor_values'))

    # Step 2: Find `var = locations[agent_index]` assignment to track the local variable
    neighbor_var = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Subscript):
                val = node.value
                if (isinstance(val.value, ast.Name) and val.value.id == locations_param and
                    isinstance(val.slice, ast.Name) and val.slice.id == agent_index_param):
                    neighbor_var = target.id
                    break

    # Step 3: Transform the body
    transformer = _CSRBodyTransformer(locations_param, agent_index_param, neighbor_var)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree)


def _inject_seed(source: str) -> str:
    """Inject _seed param and prepend _seed to rand_* calls in a function source.

    Also replaces `agent_index` with `agent_ids[agent_index]` in rand_* calls
    so the PRNG keys on the global agent ID (rank-agnostic determinism).
    """
    tree = ast.parse(source)
    func_def = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_def = node
            break
    if func_def is None:
        return source
    # Insert _seed param at position 2 (after tick, agent_index) — skip if already present
    existing_params = {arg.arg for arg in func_def.args.args}
    if '_seed' not in existing_params:
        func_def.args.args.insert(2, ast.arg(arg='_seed'))
    # Insert logical_ids param right after agent_ids — skip if already present
    if 'logical_ids' not in existing_params:
        # Find agent_ids position and insert logical_ids after it
        agent_ids_pos = None
        for i, arg in enumerate(func_def.args.args):
            if arg.arg == 'agent_ids':
                agent_ids_pos = i
                break
        if agent_ids_pos is not None:
            func_def.args.args.insert(agent_ids_pos + 1, ast.arg(arg='logical_ids'))
    # Prepend _seed and replace agent_index with logical_ids[agent_index] in rand_* calls
    # logical_ids defaults to agent_ids if no logical IDs are set by the user,
    # but allows stable RNG keys independent of agent creation order.
    _rand_funcs = {'rand_uniform_philox', 'rand_uniform_xorshift', 'rand_normal', 'rand_normal_bounded'}
    # Build AST node for logical_ids[agent_index]
    def _make_agent_id_lookup():
        return ast.Subscript(
            value=ast.Name(id='logical_ids', ctx=ast.Load()),
            slice=ast.Name(id='agent_index', ctx=ast.Load()),
            ctx=ast.Load(),
        )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _rand_funcs:
                node.args.insert(0, ast.Name(id='_seed', ctx=ast.Load()))
                # After seed insertion, args are: [_seed, tick, agent_index, salt, ...]
                # Replace arg at index 2 (agent_index) with agent_ids[agent_index]
                if (len(node.args) > 2
                        and isinstance(node.args[2], ast.Name)
                        and node.args[2].id == 'agent_index'):
                    node.args[2] = _make_agent_id_lookup()
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _gen_barrier_code(indent):
    """Generate software grid barrier code lines for the fused kernel.

    Uses atomic counter pattern: each block signals completion, then spins
    until all blocks have arrived. Two threadfences bracket the atomic
    operations to ensure global memory visibility.
    """
    return [
        f"{indent}jit.syncthreads()",
        f"{indent}if jit.threadIdx.x == 0:",
        f"{indent}\tjit.threadfence()",
        f"{indent}\tjit.atomic_add(barrier_counter, 0, 1)",
        f"{indent}\t_barrier_target = (barrier_id + 1) * num_blocks_param",
        f"{indent}\twhile jit.atomic_add(barrier_counter, 0, 0) < _barrier_target:",
        f"{indent}\t\tpass",
        f"{indent}\tjit.threadfence()",
        f"{indent}jit.syncthreads()",
        f"{indent}barrier_id = barrier_id + 1",
    ]


def _gen_bla_kernel_params(breed_local_names, write_bla_names):
    """Generate kernel parameter list for breed-local arrays.

    For each BLA: array, [write_array + nrows if double-buffered], idx_map.
    The nrows param is used for in-kernel write-back (not passed to step funcs).
    """
    params = []
    write_bla = write_bla_names or set()
    for n in (breed_local_names or []):
        params.append(f"{n},")
        if n in write_bla:
            params.append(f"write_{n},")
            params.append(f"{n}_nrows,")
        params.append(f"{n}_idx,")
    return params


def _gen_bla_step_func_args(breed_local_names, write_bla_names):
    """Generate step function call args for breed-local arrays.

    For each BLA: array, [write_array if double-buffered], idx_map.
    Matches the order in _gen_bla_kernel_params.
    """
    args = []
    write_bla = write_bla_names or set()
    for n in (breed_local_names or []):
        args.append(n)
        if n in write_bla:
            args.append(f"write_{n}")
        args.append(f"{n}_idx")
    return args


def generate_gpu_func(
    n_globals: int,
    n_properties: int,
    breed_idx_2_step_func_by_priority: List[List[Union[int, Callable]]],
    write_property_indices: Set[int],
    property_ndims: dict = None,
    extra_kernel_config: dict = None,
    skip_priority_barriers=False,
    priority_values: list = None,
    global_scalar_flags: list = None,
    breed_local_names: list = None,
    write_bla_names: set = None,
    write_bla_shapes: dict = None,
) -> str:
    """
    Generate GPU function string with double buffering support for race condition prevention.
    
    This function now includes double buffering to prevent race conditions from shared mutable 
    agent data tensors across agents in the same rank. It:
    1. Uses pre-analyzed write property indices to determine which properties need write buffers
    2. Creates write buffer parameters for properties that have assignments  
    3. Generates modified step functions that write to separate buffers
    4. Extracts and includes all necessary imports from original step function files

    cupy jit.rawkernel does not like us passing *args into
    them. This is because the Python function
    will be compiled by cupy.jit and the parameter arguments
    type and count must be set at jit compilation time.
    However, SAGESim users will have varying numbers of
    properties in their step functions, which means
    our cuda kernel's parameter count would also be variable.
    Normally, we'd just define the stepfunc with *args, but
    due to the above constraints we have to infer the number of
    arguments from the user defined breed step functions,
    rewrite the overall stepfunc as a string and then pass it
    into cupy.jit to be compiled.

    This function returns a str representation of stepfunc cupy jit.rawkernel:

        step_funcs_code = generate_gpu_func(
                    len(agent_data_tensors),
                    breed_idx_2_step_func_by_priority,
                )
    This function can then be written to a file and imported using
    spec_from_file_location. For example, if you write the code to a file
    called step_func_code.py, you can import it as below:

        import importlib.util
        spec = importlib.util.spec_from_file_location("step_func_code", "/abs/path/step_func_code.py")
        step_func_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(step_func_module)
        stepfunc = step_func_module.stepfunc
    Then you can run the stepfunc as a jit.rawkernel as below:

        stepfunc[blockspergrid, threadsperblock](
                device_global_data_vector,
                *agent_data_tensors,  # Now includes write buffers
                current_tick,
                sync_workers_every_n_ticks,
            )
        )

    :param n_properties: int total number of agent properties
    :param breed_idx_2_step_func_by_priority: List of List. Each inner List
        first element is the breedidx and second element is a tuple of the user defined
        step function, and the file where it is defined.
        The major list elements are ordered in decreasing order of execution
        priority
    :param write_property_indices: Set of property indices that require write buffers
        (pre-analyzed from all step functions)
    :return: str representation of stepfunc cuda kernal with double buffering
        that can be written to file or imported directly.

    """
    
    def generate_modified_step_func_code(step_func: Callable, write_indices: Set[int], num_properties: int,
                                         write_bla_names_set: set = None) -> str:
        """Generate modified step function code with CSR transformation and write buffer parameters.

        Phase 1: Auto-transform locations parameter to CSR (neighbor_offsets, neighbor_values)
        Phase 2: Add double buffering for writable properties
        """
        source = inspect.getsource(step_func)

        # Phase 1: CSR auto-transformation
        # Replaces locations param with neighbor_offsets, neighbor_values
        # and transforms body access patterns (loops, indexing, sentinel checks)
        source = _auto_transform_csr(source, num_properties, n_globals)

        # Phase 1b: Add _seed param and inject into rand_* calls
        source = _inject_seed(source)

        # Phase 2: Double buffering
        # Parse the CSR-transformed source
        tree = ast.parse(source)
        func_def = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_def = node
                break

        param_names = [arg.arg for arg in func_def.args.args]

        # Build CSR-aware mapping (function now has num_properties + 1 property-like params)
        # Breed-local array params (array + idx_map each) are appended after properties,
        # so we must strip them before the property-index mapping.
        n_bla = len(breed_local_names or []) * 2  # 2 params per BLA: array + idx_map
        if n_bla > 0:
            # Strip breed-local params from end so _build_param_to_property_index_csr
            # sees property params at the tail as expected
            param_names_for_prop = param_names[:-n_bla]
        else:
            param_names_for_prop = param_names
        param_to_prop = _build_param_to_property_index_csr(param_names_for_prop, num_properties)
        n_prop_params = num_properties + 1
        property_params = param_names_for_prop[-n_prop_params:]

        # Create mapping from property parameter names to write parameter names
        param_to_write_param = {}
        for param_name in property_params:
            prop_idx = param_to_prop.get(param_name, -1)
            if prop_idx >= 0 and prop_idx in write_indices:
                param_to_write_param[param_name] = f"write_{param_name}"

        # Also add BLA write names (breed-local arrays with double buffering)
        bla_write_params = {}
        if write_bla_names_set:
            for param_name in param_names:
                if param_name in write_bla_names_set:
                    bla_write_params[param_name] = f"write_{param_name}"
            param_to_write_param.update(bla_write_params)

        if not param_to_write_param:
            return source  # No double buffering needed

        # Add write parameters to function signature (AST-based)
        # Property write params: append at end (existing behavior, kernel passes them after properties)
        # BLA write params: insert after read name (kernel passes them interleaved)
        for param_name, write_param_name in param_to_write_param.items():
            if param_name in bla_write_params:
                # BLA: insert write_name right after the read name
                for i, arg in enumerate(func_def.args.args):
                    if arg.arg == param_name:
                        func_def.args.args.insert(i + 1, ast.arg(arg=write_param_name))
                        break
            else:
                # Property: append at end
                func_def.args.args.append(ast.arg(arg=write_param_name))

        # Replace parameter names with write parameter names in WRITE contexts only
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                # Replace in set_this_agent_data_from_tensor calls
                if (isinstance(node.func, ast.Name) and
                    node.func.id == 'set_this_agent_data_from_tensor' and
                    len(node.args) >= 2):
                    tensor_arg = node.args[1]
                    if isinstance(tensor_arg, ast.Name) and tensor_arg.id in param_to_write_param:
                        node.args[1] = ast.Name(id=param_to_write_param[tensor_arg.id], ctx=ast.Load())

            elif isinstance(node, ast.Assign):
                # Replace in all types of assignments to param_name
                def replace_param_in_target(target_node):
                    if isinstance(target_node, ast.Name):
                        if target_node.id in param_to_write_param:
                            target_node.id = param_to_write_param[target_node.id]
                    elif isinstance(target_node, ast.Subscript):
                        replace_param_in_target(target_node.value)

                for target in node.targets:
                    replace_param_in_target(target)

            elif isinstance(node, ast.AugAssign):
                # Convert augmented assignments to regular assignments for double buffering
                # e.g., write_param[i] += 1 becomes write_param[i] = param[i] + 1
                def convert_aug_assign_target(target_node):
                    if isinstance(target_node, ast.Name):
                        if target_node.id in param_to_write_param:
                            original_target = ast.Name(id=target_node.id, ctx=ast.Load())
                            write_target = ast.Name(id=param_to_write_param[target_node.id], ctx=ast.Store())
                            new_value = ast.BinOp(left=original_target, op=node.op, right=node.value)
                            return ast.Assign(targets=[write_target], value=new_value)
                    elif isinstance(target_node, ast.Subscript):
                        base = target_node
                        while isinstance(base, ast.Subscript):
                            base = base.value
                        if isinstance(base, ast.Name) and base.id in param_to_write_param:
                            import copy
                            read_target = copy.deepcopy(target_node)
                            read_target.ctx = ast.Load()
                            write_target = copy.deepcopy(target_node)
                            write_target.ctx = ast.Store()
                            current = write_target
                            while isinstance(current, ast.Subscript):
                                if isinstance(current.value, ast.Name):
                                    current.value.id = param_to_write_param[current.value.id]
                                    break
                                current = current.value
                            new_value = ast.BinOp(left=read_target, op=node.op, right=node.value)
                            return ast.Assign(targets=[write_target], value=new_value)
                    return None

                new_assign = convert_aug_assign_target(node.target)
                if new_assign:
                    node.__class__ = ast.Assign
                    node.targets = new_assign.targets
                    node.value = new_assign.value
                    if hasattr(node, 'op'):
                        delattr(node, 'op')
                    if hasattr(node, 'target'):
                        delattr(node, 'target')

        return ast.unparse(tree)

    
    # Generate global arguments (one per registered global tensor)
    global_args = [f"g{i}" for i in range(n_globals)]

    # For step function calls: auto-extract scalars with [0]
    step_func_global_args = []
    for i in range(n_globals):
        if global_scalar_flags and i < len(global_scalar_flags) and global_scalar_flags[i]:
            step_func_global_args.append(f"g{i}[0]")
        else:
            step_func_global_args.append(f"g{i}")

    # Generate read arguments (original properties)
    # Property 1 (locations) is replaced by two CSR arrays: neighbor_offsets, neighbor_values
    read_args = []
    for i in range(n_properties):
        if i == 1:
            read_args.append("neighbor_offsets")
            read_args.append("neighbor_values")
        else:
            read_args.append(f"a{i}")

    # Generate write arguments for properties that need write buffers
    # Property 1 (CSR) is never written in the kernel
    write_args = [f"write_a{i}" for i in sorted(write_property_indices) if i != 1]

    # Combine all arguments
    args = read_args + write_args
    
    # Generate modified step functions
    step_sources = [
        "from sagesim.jit_extensions import install_jit_extensions",
        "install_jit_extensions()",
    ]
    imported_modules = set()

    modified_step_functions = []
    transformed_helpers = set()  # Track already-transformed helper functions
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            step_func_name = getattr(breed_step_func_impl, "__name__", repr(callable))
            modified_step_func_name = f"{step_func_name}_double_buffer"

            # Generate modified step function
            modified_step_func_code = generate_modified_step_func_code(
                breed_step_func_impl, write_property_indices, n_properties,
                write_bla_names_set=write_bla_names)
            modified_step_func_code = modified_step_func_code.replace(
                f"def {step_func_name}(",
                f"def {modified_step_func_name}("
            )
            modified_step_functions.append(modified_step_func_code)

            # Transform helper functions that receive forwarded locations parameter
            # (e.g., dispatcher pattern where step func calls other device functions)
            forwarded_funcs = _find_forwarded_location_funcs(breed_step_func_impl, n_properties)
            for helper_name, helper_obj in forwarded_funcs:
                if helper_name not in transformed_helpers:
                    helper_source = inspect.getsource(helper_obj)
                    transformed_helper = _auto_transform_csr(helper_source, n_properties, n_globals)
                    transformed_helper = _inject_seed(transformed_helper)
                    modified_step_functions.append(transformed_helper)
                    transformed_helpers.add(helper_name)

            module_fpath = Path(module_fpath).absolute()
            module_name = module_fpath.stem
            if module_fpath not in imported_modules:
                step_sources.append(f"from {module_name} import *")
                imported_modules.add(module_fpath)

    step_sources = "\n".join(step_sources)
    all_modified_step_functions = "\n\n".join(modified_step_functions)
    joined_args = ",".join(args)

    if property_ndims is None:
        property_ndims = {}

    # ================================================================
    # Generate fused kernel body: persistent threads + grid barriers
    # All priorities and ticks execute in a single kernel launch.
    # ================================================================
    tick_body = []
    _extra_seen_breeds = set()  # For once_per_breed dedup of extra post-step code

    for priority_idx, breed_idx_2_step_func in enumerate(breed_idx_2_step_func_by_priority):
        # Range-bounded persistent thread loop for this priority
        p = priority_idx
        tick_body.append(f"\t\tagent_index = thread_id")
        tick_body.append(f"\t\twhile agent_index < priority_{p}_count:")
        tick_body.append(f"\t\t\t_real_idx = int(agent_index) + int(priority_{p}_start)")
        tick_body.append(f"\t\t\tbreed_id = a0[_real_idx]")

        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            step_func_name = getattr(breed_step_func_impl, "__name__", repr(callable))
            modified_step_func_name = f"{step_func_name}_double_buffer"

            tick_body.append(f"\t\t\tif breed_id == {breedidx}:")
            tick_body.append(f"\t\t\t\t{modified_step_func_name}(")
            tick_body.append("\t\t\t\t\tthread_local_tick,")
            tick_body.append("\t\t\t\t\t_real_idx,")
            tick_body.append("\t\t\t\t\t_seed,")
            if global_args:
                tick_body.append(f"\t\t\t\t\t{','.join(step_func_global_args)},")
            tick_body.append("\t\t\t\t\tagent_ids,")
            tick_body.append("\t\t\t\t\tlogical_ids,")
            # Property args (read + write buffers)
            tick_body.append(f"\t\t\t\t\t{','.join(args)},")
            # Breed-local arrays and their index maps (with write buffers for double-buffered BLAs)
            if breed_local_names:
                bla_args = _gen_bla_step_func_args(breed_local_names, write_bla_names)
                tick_body.append(f"\t\t\t\t\t{','.join(bla_args)},")
            tick_body.append("\t\t\t\t)")

            # Inject extra post-step code from subclass config
            if extra_kernel_config:
                for entry in extra_kernel_config.get('post_breed_step_code', []):
                    code_lines, once_per_breed = entry[0], entry[1]
                    only_priority = entry[2] if len(entry) > 2 else None
                    if only_priority is not None and priority_idx != only_priority:
                        continue
                    if once_per_breed and breedidx in _extra_seen_breeds:
                        continue
                    for line in code_lines:
                        tick_body.append(f"\t\t\t\t{line}")
                    if once_per_breed:
                        _extra_seen_breeds.add(breedidx)

        tick_body.append("\t\t\tagent_index = agent_index + total_threads")

        # Grid barrier after this priority
        is_last_priority = (priority_idx == len(breed_idx_2_step_func_by_priority) - 1)
        should_skip = False
        if skip_priority_barriers is True and not is_last_priority:
            should_skip = True
        elif isinstance(skip_priority_barriers, set) and not is_last_priority:
            if priority_values and priority_values[priority_idx] in skip_priority_barriers:
                should_skip = True

        if not should_skip:
            tick_body.append("")
            tick_body += _gen_barrier_code("\t\t")

    # Write-back: copy write buffers to read buffers (end of tick)
    writeback_props = sorted(i for i in write_property_indices if i != 1)
    writeback_blas = [n for n in (breed_local_names or [])
                      if write_bla_names and n in write_bla_names
                      and write_bla_shapes and n in write_bla_shapes]

    if writeback_props or writeback_blas:
        # Property write-back
        if writeback_props:
            tick_body.append("")
            tick_body.append("\t\tagent_index = thread_id")
            tick_body.append("\t\twhile agent_index < num_rank_local_agents:")

            for prop_idx in writeback_props:
                ncols = property_ndims.get(prop_idx, 0)
                if ncols > 1:
                    tick_body.append(f"\t\t\tfor _wb_j in range({ncols}):")
                    tick_body.append(f"\t\t\t\ta{prop_idx}[agent_index][_wb_j] = write_a{prop_idx}[agent_index][_wb_j]")
                else:
                    tick_body.append(f"\t\t\ta{prop_idx}[agent_index] = write_a{prop_idx}[agent_index]")

            tick_body.append("\t\t\tagent_index = agent_index + total_threads")

        # Breed-local array write-back
        for bla_name in writeback_blas:
            shape = write_bla_shapes[bla_name]
            tick_body.append("")
            tick_body.append(f"\t\t_bla_row = thread_id")
            tick_body.append(f"\t\twhile _bla_row < {bla_name}_nrows:")
            if len(shape) == 1:
                tick_body.append(f"\t\t\tfor _bla_j in range({shape[0]}):")
                tick_body.append(f"\t\t\t\t{bla_name}[_bla_row][_bla_j] = write_{bla_name}[_bla_row][_bla_j]")
            elif len(shape) == 2:
                tick_body.append(f"\t\t\tfor _bla_j in range({shape[0]}):")
                tick_body.append(f"\t\t\t\tfor _bla_k in range({shape[1]}):")
                tick_body.append(f"\t\t\t\t\t{bla_name}[_bla_row][_bla_j][_bla_k] = write_{bla_name}[_bla_row][_bla_j][_bla_k]")
            tick_body.append(f"\t\t\t_bla_row = _bla_row + total_threads")

        # Single barrier after all write-backs
        tick_body.append("")
        tick_body += _gen_barrier_code("\t\t")

    joined_tick_body = "\n".join(tick_body)

    # Build range parameter names for kernel signature
    num_priorities = len(breed_idx_2_step_func_by_priority)
    range_params = []
    for p in range(num_priorities):
        range_params.extend([f"priority_{p}_start", f"priority_{p}_count"])
    joined_range_params = ",".join(range_params)

    joined_global_args = ",".join(global_args) + "," if global_args else ""

    func = [
        "# Auto-generated fused GPU kernel with grid barriers",
        "# All priorities and ticks in a single kernel launch",
        "",
        step_sources,
        "",
        "# Modified step functions with double buffering",
        all_modified_step_functions,
        "",
        "@jit.rawkernel(device='cuda')",
        "def stepfunc(",
        "global_tick,",
        "_seed,",
        joined_global_args,
        joined_args + ",",
        "sync_workers_every_n_ticks,",
        "num_rank_local_agents,",
        joined_range_params + "," if joined_range_params else "",
        "agent_ids,",
        "logical_ids,",
        "barrier_counter,",
        "num_blocks_param,",
        *[f"{p}," for p in (extra_kernel_config or {}).get('extra_kernel_params', [])],
        *_gen_bla_kernel_params(breed_local_names, write_bla_names),
        "):",
        "\tthread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x",
        "\ttotal_threads = jit.gridDim.x * jit.blockDim.x",
        "\tbarrier_id = 0",
        "",
        "\tfor tick in range(sync_workers_every_n_ticks):",
        "\t\tthread_local_tick = int(global_tick) + tick",
        "",
        joined_tick_body,
    ]

    func = "\n".join(func)
    return func
