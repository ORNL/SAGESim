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
from sagesim.internal_utils import convert_to_equal_side_tensor, build_csr_from_ragged
from sagesim.gpu_kernels import GPUBufferManager, GPUHashMap, CommunicationManager, is_gpu_aware_mpi


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


def analyze_step_function_for_writes(step_func: Callable, num_properties: int) -> Set[int]:
    """Analyze step function to find which property indices need write buffers."""
    write_property_indices = set()

    source = inspect.getsource(step_func)
    tree = ast.parse(source)
    signature = inspect.signature(step_func)
    param_names = list(signature.parameters.keys())

    # Build param name → property index mapping (for ORIGINAL user step function)
    param_to_prop = _build_param_to_property_index(param_names, num_properties)

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


class Model:

    def __init__(
        self,
        space: Space,
        threads_per_block: int = 32,
        step_function_file_path: str = "step_func_code.py",
        verbose_timing: bool = False,
        verbose_mpi_transfer: bool = False,
    ) -> None:
        self._threads_per_block = threads_per_block
        self._step_function_file_path = os.path.abspath(step_function_file_path)
        self._verbose_timing = verbose_timing
        self._verbose_mpi_transfer = verbose_mpi_transfer
        self._agent_factory = AgentFactory(space, verbose_timing=verbose_timing, verbose_mpi_transfer=verbose_mpi_transfer)
        self._is_setup = False
        self.globals = {}
        self.tick = 0
        self._write_property_indices = set()  # Cache for write property indices
        # following may be set later in setup if distributed execution

    @property
    def verbose_timing(self) -> bool:
        """Enable timing verbose output for debugging performance."""
        return self._verbose_timing

    @verbose_timing.setter
    def verbose_timing(self, value: bool) -> None:
        self._verbose_timing = value
        self._agent_factory._verbose_timing = value

    @property
    def verbose_mpi_transfer(self) -> bool:
        """Enable MPI transfer verbose output to track bytes sent/received between workers."""
        return self._verbose_mpi_transfer

    @verbose_mpi_transfer.setter
    def verbose_mpi_transfer(self, value: bool) -> None:
        self._verbose_mpi_transfer = value
        self._agent_factory._verbose_mpi_transfer = value

    def register_breed(self, breed: Breed) -> None:
        if self._agent_factory.num_agents > 0:
            raise Exception(f"All breeds must be registered before agents are created!")
        self._agent_factory.register_breed(breed)

    def create_agent_of_breed(self, breed: Breed, add_to_space=True, **kwargs) -> int:
        agent_id = self._agent_factory.create_agent(breed, **kwargs)
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

    def get_space(self) -> Space:
        return self._agent_factory._space

    def get_breed_data(self, breed_name, property_name):
        """Download property data for all agents of a breed, gathered across MPI ranks.

        Uses bulk GPU download (no per-agent Python calls). For multi-worker,
        each rank downloads its local agents, then MPI Allgather combines them.

        :param breed_name: Name of the breed (e.g., "Tree", "Site")
        :param property_name: Property to download (e.g., "params", "states_db")
        :return: numpy array, shape (total_count, property_width)
        """
        breed = self._agent_factory._breeds[breed_name]
        buf = self._gpu_buffers
        start, count = buf.breed_ranges.get(breed._breedidx, (0, 0))

        prop_idx = self._agent_factory._property_name_2_index[property_name]
        if count > 0:
            local_data = buf.property_tensors[prop_idx][start:start + count].get()
        else:
            local_data = np.empty((0,), dtype=np.float32)

        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            return local_data
        all_data = comm.allgather(local_data)
        return np.concatenate(all_data, axis=0)

    def get_breed_agent_ids(self, breed_name):
        """Download agent IDs for all agents of a breed, gathered across MPI ranks.

        :param breed_name: Name of the breed
        :return: numpy array of agent IDs (float32)
        """
        breed = self._agent_factory._breeds[breed_name]
        buf = self._gpu_buffers
        start, count = buf.breed_ranges.get(breed._breedidx, (0, 0))

        if count > 0:
            local_ids = buf.agent_ids_gpu[start:start + count].get()
        else:
            local_ids = np.empty((0,), dtype=np.float32)

        comm = MPI.COMM_WORLD
        if comm.Get_size() == 1:
            return local_ids
        all_ids = comm.allgather(local_ids)
        return np.concatenate(all_ids)

    def get_agents_with(self, query: Callable) -> Set[List[Any]]:
        return self._agent_factory.get_agents_with(query=query)

    def register_global_property(
        self, property_name: str, value: Union[float, int]
    ) -> None:
        self.globals[property_name] = value

    def set_global_property_value(
        self, property_name: str, value: Union[float, int]
    ) -> None:
        self.globals[property_name] = value

    def get_global_property_value(self, property_name: str) -> Union[float, int]:
        return self.globals[property_name]

    def register_reduce_function(self, reduce_func: Callable) -> None:
        self._reduce_func = reduce_func

    def load_partition(self, partition_file: str, format: str = "auto") -> None:
        """Load network partition from file to optimize multi-worker performance.

        This method loads a pre-computed network partition that assigns agents to MPI ranks
        to minimize cross-worker communication. Must be called BEFORE creating any agents.

        For details on partition formats and usage, see AgentFactory.load_partition().

        :param partition_file: Path to partition file
        :param format: File format ('pickle', 'json', 'numpy', 'text', or 'auto')
        """
        self._agent_factory.load_partition(partition_file, format)

    def setup(self, use_gpu: bool = True) -> None:
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
        # Generate global data tensor
        self._global_data_vector = list(self.globals.values())
        self.tick = 0

        ####
        # Create record of agent step functions by breed and priority
        self._breed_idx_2_step_func_by_priority: List[Dict[int, Callable]] = []
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
                write_indices = analyze_step_function_for_writes(breed_step_func_impl, self._agent_factory.num_properties)
                self._write_property_indices.update(write_indices)

        # Exclude no_double_buffer properties from write buffer creation
        self._write_property_indices = self._write_property_indices - no_double_buffer_indices


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
            with open(self._step_function_file_path, "w", encoding="utf-8") as f:
                f.write(
                    generate_gpu_func(
                        self._agent_factory.num_properties,
                        self._breed_idx_2_step_func_by_priority,
                        self._write_property_indices,
                        property_ndims,
                        extra_kernel_config=self._get_extra_kernel_config(),
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
            importlib.invalidate_caches()
            abs_path = self._step_function_file_path  # Already absolute from __init__
            module_name = os.path.splitext(os.path.basename(abs_path))[0]
            sys.modules.pop(module_name, None)  # Evict stale module for repeated setup() calls (#47)
            spec = importlib.util.spec_from_file_location(module_name, abs_path)
            step_func_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = step_func_module
            spec.loader.exec_module(step_func_module)
        self._step_func = step_func_module.stepfunc
        t_jit_end = time.time()
        ###

        # Print agent distribution summary
        if worker == 0:
            total_agents = self._agent_factory._num_agents
            agents_per_rank = {}
            for agent_id in range(total_agents):
                rank = self._agent_factory._agent2rank.get(agent_id, -1)
                agents_per_rank[rank] = agents_per_rank.get(rank, 0) + 1


        self._is_setup = True

        # Initialize GPU buffer manager (buffers allocated lazily on first tick)
        self._gpu_buffers = GPUBufferManager()
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
        self._global_data_vector = list(self.globals.values())

        # Sync GPU state back to AgentFactory before freeing
        if hasattr(self, '_gpu_buffers') and self._gpu_buffers.is_initialized:
            self._sync_gpu_to_agent_factory()

        self._regenerate_data_tensors()

        if hasattr(self, '_gpu_buffers'):
            self._gpu_buffers.free()
            self._gpu_buffers = GPUBufferManager()


    def simulate(
        self,
        ticks: int,
        sync_workers_every_n_ticks: int = 1,
    ) -> None:
        comm.barrier()

        # Step function is cached during setup() - no need to reimport
        # Optimization: Single worker doesn't need synchronization overhead
        # Just run all ticks in one batch
        if num_workers == 1:
            # Single worker optimization: no MPI sync overhead
            self.worker_coroutine(ticks)
        else:
            # Multi-worker: need periodic synchronization
            # Repeatedly execute worker coroutine until simulation
            # has run for the right amount of ticks
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

    def _build_gpu_buffers(self, received_neighbor_ids, received_neighbor_adts, num_local_agents):
        """Build all persistent GPU buffers on first tick.

        Performs the same work as the original per-tick data prep, but stores
        results in self._gpu_buffers for reuse on subsequent ticks.
        """
        buf = self._gpu_buffers
        buf.num_local_agents = num_local_agents
        num_ghost = len(received_neighbor_ids)

        # Compute per-priority breed ranges from sorted breed data
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

        # 1. Build agent ID list and CPU hash map
        all_agent_ids_list = self.__rank_local_agent_ids + received_neighbor_ids
        agent_id_to_index = {int(agent_id): idx for idx, agent_id in enumerate(all_agent_ids_list)}
        buf.all_agent_ids_list = all_agent_ids_list
        buf.agent_id_to_index = agent_id_to_index
        buf.num_total_agents = len(all_agent_ids_list)
        buf.prev_ghost_ids_set = set(received_neighbor_ids)

        # 2. Pre-allocate capacity with slack
        agent_capacity = max(GPUBufferManager.MIN_CAPACITY,
                             int(buf.num_total_agents * GPUBufferManager.AGENT_SLACK_FACTOR))

        # 3. Upload agent IDs to GPU with slack
        agent_ids_padded = np.full(agent_capacity, -1, dtype=np.float32)
        agent_ids_padded[:buf.num_total_agents] = np.array(all_agent_ids_list, dtype=np.float32)
        buf.agent_ids_gpu = cp.array(agent_ids_padded)

        # 4. Upload global data vector
        buf.global_data_vector = cp.array(self._global_data_vector)

        # 5. Build GPU hash map
        agent_ids_np = np.array(all_agent_ids_list, dtype=np.int64)
        buffer_indices_np = np.arange(len(all_agent_ids_list), dtype=np.int32)
        hash_capacity = max(GPUBufferManager.MIN_CAPACITY, len(all_agent_ids_list) * 2)
        buf.hash_map = GPUHashMap(hash_capacity)
        buf.hash_map.build_from_arrays(agent_ids_np, buffer_indices_np)

        # 6. Combine local + neighbor data and build GPU arrays
        combined_lists = []
        for i in range(self._agent_factory.num_properties):
            received_data = received_neighbor_adts[i]

            # Handle non-neighbor-visible properties
            if received_data and received_data[0] is None:
                local_data = self.__rank_local_agent_data_tensors[i]
                placeholder = self._create_zero_placeholder(local_data[0]) if local_data else 0.0
                received_data = [placeholder for _ in range(len(received_data))]

            combined = self.__rank_local_agent_data_tensors[i] + received_data

            if i == 1:
                # Build dual CSR: values with agent IDs (for MPI) and local indices (for kernel)
                combined_for_ids = list(combined)  # preserve original agent IDs
                combined_indices = convert_agent_ids_to_indices(combined, agent_id_to_index)
                offsets_np, values_np = build_csr_from_ragged(combined_indices)
                _, values_ids_np = build_csr_from_ragged(combined_for_ids)
                buf.allocate_csr(offsets_np, values_np, values_ids_np, buf.num_total_agents)

            combined_lists.append(combined)

        # 7. Allocate property tensors on GPU with slack
        buf.allocate_property_tensors(
            self._agent_factory.num_properties,
            combined_lists,
            agent_capacity,
            convert_to_equal_side_tensor,
        )
        buf.agent_capacity = agent_capacity

        # 8. Create write buffers
        sorted_write_indices = sorted(i for i in self._write_property_indices if i != 1)
        buf.allocate_write_buffers(sorted_write_indices)

        # 9. Allocate barrier counter for fused-kernel grid barrier
        # Must be int32 to match barrier arithmetic (int32 num_blocks_param, int literals)
        buf.barrier_counter = cp.zeros(1, dtype=cp.int32)

        buf.is_initialized = True

    def _upload_ghost_values(self, received_neighbor_ids, received_neighbor_adts, num_local_agents):
        """Fast path: ghost set unchanged, only update property values in existing GPU slots."""
        buf = self._gpu_buffers
        num_ghosts = len(received_neighbor_ids)
        if num_ghosts == 0:
            return

        ghost_start = num_local_agents

        for prop_idx in range(self._agent_factory.num_properties):
            if prop_idx == 1:
                # CSR (locations) — unchanged when ghost set is stable
                continue

            received_data = received_neighbor_adts[prop_idx]
            if received_data and received_data[0] is None:
                # Non-visible property — GPU already has placeholders
                continue

            tensor = buf.property_tensors[prop_idx]
            if tensor is None:
                continue

            # Build ghost values in buffer order and upload as a batch
            # received_neighbor_ids may be in different order than all_agent_ids_list
            if tensor.ndim == 1:
                ghost_cpu = np.zeros(num_ghosts, dtype=np.float32)
                for ghost_idx, ghost_id in enumerate(received_neighbor_ids):
                    buffer_idx = buf.agent_id_to_index.get(int(ghost_id))
                    if buffer_idx is not None:
                        local_ghost_offset = buffer_idx - ghost_start
                        ghost_cpu[local_ghost_offset] = float(received_data[ghost_idx])
                buf.property_tensors[prop_idx][ghost_start:ghost_start + num_ghosts] = cp.array(ghost_cpu)
            else:
                # For multi-dim properties: download current ghost region, update, re-upload
                ghost_region = tensor[ghost_start:ghost_start + num_ghosts].get()
                for ghost_idx, ghost_id in enumerate(received_neighbor_ids):
                    buffer_idx = buf.agent_id_to_index.get(int(ghost_id))
                    if buffer_idx is not None:
                        local_ghost_offset = buffer_idx - ghost_start
                        val = received_data[ghost_idx]
                        if isinstance(val, (list, tuple)):
                            arr = np.array(val, dtype=np.float32)
                            ghost_region[local_ghost_offset, :len(arr)] = arr
                        elif isinstance(val, np.ndarray):
                            ghost_region[local_ghost_offset, :len(val)] = val.astype(np.float32)
                        else:
                            ghost_region[local_ghost_offset, 0] = float(val)
                buf.property_tensors[prop_idx][ghost_start:ghost_start + num_ghosts] = cp.array(ghost_region)

            # Also update write buffers for ghost region if this property has one
            if prop_idx in buf.sorted_write_indices:
                wb_idx = buf.sorted_write_indices.index(prop_idx)
                buf.write_buffers[wb_idx][ghost_start:ghost_start + num_ghosts] = \
                    buf.property_tensors[prop_idx][ghost_start:ghost_start + num_ghosts]

    def _rebuild_gpu_buffers(self, received_neighbor_ids, received_neighbor_adts, num_local_agents):
        """Rebuild path: ghost set or topology changed. Rebuild CSR, hash map, ghost region.

        Local agent data in property_tensors[0:num_local] stays on GPU untouched.
        """
        buf = self._gpu_buffers
        num_ghost = len(received_neighbor_ids)
        buf.num_local_agents = num_local_agents

        # 1. Rebuild agent ID list and CPU hash map
        all_agent_ids_list = self.__rank_local_agent_ids + received_neighbor_ids
        agent_id_to_index = {int(agent_id): idx for idx, agent_id in enumerate(all_agent_ids_list)}
        buf.all_agent_ids_list = all_agent_ids_list
        buf.agent_id_to_index = agent_id_to_index
        buf.num_total_agents = len(all_agent_ids_list)
        buf.prev_ghost_ids_set = set(received_neighbor_ids)

        # 2. Ensure capacity
        buf.ensure_agent_capacity(buf.num_total_agents)

        # 3. Rebuild GPU hash map
        agent_ids_np = np.array(all_agent_ids_list, dtype=np.int64)
        buffer_indices_np = np.arange(len(all_agent_ids_list), dtype=np.int32)
        hash_capacity = max(GPUBufferManager.MIN_CAPACITY, len(all_agent_ids_list) * 2)
        if buf.hash_map is None or hash_capacity > buf.hash_map.capacity:
            if buf.hash_map is not None:
                buf.hash_map.free()
            buf.hash_map = GPUHashMap(hash_capacity)
        buf.hash_map.build_from_arrays(agent_ids_np, buffer_indices_np)

        # 4. Update agent IDs on GPU
        agent_ids_padded = np.full(buf.agent_capacity, -1, dtype=np.float32)
        agent_ids_padded[:buf.num_total_agents] = np.array(all_agent_ids_list, dtype=np.float32)
        buf.agent_ids_gpu = cp.array(agent_ids_padded)

        # 5. Rebuild property data for ghost region and CSR
        for i in range(self._agent_factory.num_properties):
            received_data = received_neighbor_adts[i]

            if received_data and received_data[0] is None:
                local_data = self.__rank_local_agent_data_tensors[i]
                placeholder = self._create_zero_placeholder(local_data[0]) if local_data else 0.0
                received_data = [placeholder for _ in range(len(received_data))]

            combined = self.__rank_local_agent_data_tensors[i] + received_data

            if i == 1:
                # Rebuild dual CSR
                combined_for_ids = list(combined)
                combined_indices = convert_agent_ids_to_indices(combined, agent_id_to_index)
                offsets_np, values_np = build_csr_from_ragged(combined_indices)
                _, values_ids_np = build_csr_from_ragged(combined_for_ids)
                total_edges = len(values_np)
                buf.ensure_csr_capacity(total_edges)
                # Write into existing GPU arrays
                buf.neighbor_offsets[:len(offsets_np)] = cp.array(offsets_np)
                buf.neighbor_values[:total_edges] = cp.array(values_np)
                buf.neighbor_values_ids[:total_edges] = cp.array(values_ids_np)
            else:
                # Rebuild combined tensor and update ghost region on GPU
                # Local region [0:num_local] already correct on GPU from kernel writes
                tensor = convert_to_equal_side_tensor(combined)
                if tensor.ndim == 1:
                    buf.property_tensors[i][:len(tensor)] = tensor
                else:
                    buf.property_tensors[i][:tensor.shape[0]] = tensor

        # 6. Rebuild write buffers
        buf.write_buffers = []
        for prop_idx in buf.sorted_write_indices:
            buf.write_buffers.append(buf.property_tensors[prop_idx].copy())

        # 7. Rebuild communication maps to match new topology
        if hasattr(self, '_comm_manager') and num_workers > 1:
            self._comm_manager.build_communication_maps()

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

        self.__rank_local_agent_ids = list(
            self._agent_factory._rank2agentid2agentidx[worker].keys()
        )

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
            (
                self.__rank_local_agent_ids,
                self.__rank_local_agent_data_tensors,
                received_neighbor_ids,
                received_neighbor_adts,
            ) = self._agent_factory.contextualize_agent_data_tensors(
                self.__rank_local_agent_data_tensors,
                self.__rank_local_agent_ids,
                rank_local_agents_neighbors,
            )
            t_after_context = time.time()

            if self._verbose_timing:
                timing_data['neighbor'] = t_neighbor_end - t_neighbor_start
                timing_data['contextualize'] = t_after_context - t_before_context
                timing_data['num_neighbors'] = len(received_neighbor_ids)

            # Update num_local_agents after contextualize (may reorder)
            num_local_agents = len(self.__rank_local_agent_ids)
            blockspergrid = int(math.ceil(num_local_agents / threadsperblock))

            t_build_start = time.time()
            self._build_gpu_buffers(received_neighbor_ids, received_neighbor_adts, num_local_agents)
            t_build_end = time.time()


            # Initialize CommunicationManager for GPU-direct MPI on subsequent ticks
            if num_workers > 1:
                self._comm_manager = CommunicationManager(
                    buf, self._agent_factory, worker, num_workers, comm
                )
                self._comm_manager.build_communication_maps()

        else:
            # --- SUBSEQUENT TICK: reuse existing GPU buffers ---
            t_before_context = time.time()

            if num_workers > 1 and hasattr(self, '_comm_manager') and self._comm_manager.is_initialized:
                self._comm_manager.exchange_ghost_data()

            t_after_context = time.time()

            if self._verbose_timing:
                timing_data['contextualize'] = t_after_context - t_before_context

            # Update global_data_vector on GPU
            # GPU-aware path: already on GPU from last tick's in-place Allreduce
            if not self._gpu_aware_mpi or num_workers <= 1:
                buf.global_data_vector = cp.array(self._global_data_vector)

        t_data_prep_end = time.time()

        if self._verbose_timing:
            timing_data['data_prep'] = t_data_prep_end - t_data_prep_start

        # ============================================================
        # GPU KERNEL EXECUTION (fused: all ticks + priorities in one launch)
        # ============================================================

        # Build all_args from persistent GPU buffers
        all_args = []
        for i in range(self._agent_factory.num_properties):
            if i == 1:
                all_args.append(buf.neighbor_offsets)
                all_args.append(buf.neighbor_values)
            else:
                all_args.append(buf.property_tensors[i])
        all_args = all_args + buf.write_buffers

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
        dev = cp.cuda.Device()
        attrs = dev.attributes
        num_sms = attrs['MultiProcessorCount']
        # Conservative: 2 blocks/SM ensures co-residency even with high register pressure
        max_blocks_per_sm = getattr(self, '_max_blocks_per_sm', 2)
        max_grid_blocks = max_blocks_per_sm * num_sms
        effective_blocks = min(blockspergrid, max_grid_blocks)

        t_gpu_kernel_start = time.time()

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

        # Reset barrier counter and launch fused kernel
        buf.barrier_counter[0] = 0
        self._step_func[effective_blocks, threadsperblock](
            self.tick,
            buf.global_data_vector,
            *all_args,
            sync_workers_every_n_ticks,
            cp.float32(num_local_agents),
            *range_args,
            buf.agent_ids_gpu,
            buf.barrier_counter,
            cp.int32(effective_blocks),
            *extra_kernel_args,
        )
        cp.cuda.get_current_stream().synchronize()

        # Process subclass-specific extra data (e.g. download spike records)
        self._process_kernel_extras()

        # Final write-back: kernel does inter-tick write-backs on GPU,
        # but ensure property_tensors match write_buffers for post-kernel reads
        for i, prop_idx in enumerate(buf.sorted_write_indices):
            buf.property_tensors[prop_idx][:num_local_agents] = \
                buf.write_buffers[i][:num_local_agents]
        cp.cuda.get_current_stream().synchronize()

        self.tick += sync_workers_every_n_ticks
        t_gpu_kernel_end = time.time()

        if self._verbose_timing:
            timing_data['gpu_kernel'] = t_gpu_kernel_end - t_gpu_kernel_start

        # ============================================================
        # POST-KERNEL: download for MPI next tick and user queries
        # ============================================================
        t_post_start = time.time()
        t_post_end = time.time()

        # Ensure all workers have finished before syncing neighbor data
        t_before_barrier = time.time()
        if num_workers > 1:
            comm.barrier()
        t_after_barrier = time.time()

        # Global data vector: allreduce across ranks
        t_before_allreduce = time.time()
        if num_workers > 1 and self._gpu_aware_mpi:
            # GPU-aware path: in-place Allreduce directly on GPU buffer (no CPU staging)
            gdv = buf.global_data_vector
            mpi_type = MPI.DOUBLE if gdv.dtype == cp.float64 else MPI.FLOAT
            comm.Allreduce(MPI.IN_PLACE, [gdv, mpi_type], op=MPI.MAX)
            # Store CPU copy for fallback queries and single-worker path
            self._global_data_vector = gdv.get().tolist()
        else:
            # CPU staging path: download, pickle-based allreduce, store for next tick upload
            global_cpu = buf.global_data_vector.tolist()
            self._global_data_vector = comm.allreduce(
                global_cpu, op=reduce_global_data_vector
            )
        t_after_allreduce = time.time()

        t_end = time.time()
        if self._verbose_timing:
            timing_data['post'] = t_post_end - t_post_start
            timing_data['sync'] = (t_after_barrier - t_before_barrier) + (t_after_allreduce - t_before_allreduce)
            timing_data['total'] = t_end - t_start

            print(f"[Rank {worker}] Tick {self.tick}: "
                  f"total={timing_data['total']:.3f}s | "
                  f"prep={timing_data['data_prep']:.3f}s, "
                  f"gpu={timing_data['gpu_kernel']:.3f}s, "
                  f"post={timing_data['post']:.3f}s, "
                  f"sync={timing_data['sync']:.3f}s", flush=True)



def reduce_global_data_vector(A, B):
    values = np.stack([A, B], axis=1)
    return np.max(values, axis=1)


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


def _auto_transform_csr(source: str, num_properties: int) -> str:
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

    # Standard params: tick, agent_index, globals, agent_ids (indices 0-3)
    # Property params start at index 4
    # Property 1 (locations) is at index 5 (4 standard + property 0)
    if len(param_names) < 6:
        return source

    agent_index_param = param_names[1]
    locations_param = param_names[4 + 1]  # Property 1

    # Step 1: Replace locations parameter with neighbor_offsets, neighbor_values
    loc_idx = 5
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


def generate_gpu_func(
    n_properties: int,
    breed_idx_2_step_func_by_priority: List[List[Union[int, Callable]]],
    write_property_indices: Set[int],
    property_ndims: dict = None,
    extra_kernel_config: dict = None,
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
    
    def generate_modified_step_func_code(step_func: Callable, write_indices: Set[int], num_properties: int) -> str:
        """Generate modified step function code with CSR transformation and write buffer parameters.

        Phase 1: Auto-transform locations parameter to CSR (neighbor_offsets, neighbor_values)
        Phase 2: Add double buffering for writable properties
        """
        source = inspect.getsource(step_func)

        # Phase 1: CSR auto-transformation
        # Replaces locations param with neighbor_offsets, neighbor_values
        # and transforms body access patterns (loops, indexing, sentinel checks)
        source = _auto_transform_csr(source, num_properties)

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
        param_to_prop = _build_param_to_property_index_csr(param_names, num_properties)
        n_prop_params = num_properties + 1
        property_params = param_names[-n_prop_params:]

        # Create mapping from property parameter names to write parameter names
        param_to_write_param = {}
        for param_name in property_params:
            prop_idx = param_to_prop.get(param_name, -1)
            if prop_idx >= 0 and prop_idx in write_indices:
                param_to_write_param[param_name] = f"write_{param_name}"

        if not param_to_write_param:
            return source  # No double buffering needed

        # Add write parameters to function signature (AST-based)
        for write_param_name in param_to_write_param.values():
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
            modified_step_func_code = generate_modified_step_func_code(breed_step_func_impl, write_property_indices, n_properties)
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
                    transformed_helper = _auto_transform_csr(helper_source, n_properties)
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
            tick_body.append("\t\t\t\t\tdevice_global_data_vector,")
            tick_body.append("\t\t\t\t\tagent_ids,")
            tick_body.append(f"\t\t\t\t\t{','.join(args)},")
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
        tick_body.append("")
        tick_body += _gen_barrier_code("\t\t")

    # Write-back: copy write buffers to read buffers (end of tick)
    writeback_props = sorted(i for i in write_property_indices if i != 1)
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

        # Final barrier (ensures write-back visible before next tick)
        tick_body.append("")
        tick_body += _gen_barrier_code("\t\t")

    joined_tick_body = "\n".join(tick_body)

    # Build range parameter names for kernel signature
    num_priorities = len(breed_idx_2_step_func_by_priority)
    range_params = []
    for p in range(num_priorities):
        range_params.extend([f"priority_{p}_start", f"priority_{p}_count"])
    joined_range_params = ",".join(range_params)

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
        "device_global_data_vector,",
        joined_args + ",",
        "sync_workers_every_n_ticks,",
        "num_rank_local_agents,",
        joined_range_params + "," if joined_range_params else "",
        "agent_ids,",
        "barrier_counter,",
        "num_blocks_param,",
        *[f"{p}," for p in (extra_kernel_config or {}).get('extra_kernel_params', [])],
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
