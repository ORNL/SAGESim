"""
SuperNeuroABM basic Model class

"""

from typing import Dict, List, Callable, Set, Any, Union
import os
import re
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
from sagesim.internal_utils import convert_to_equal_side_tensor


def convert_agent_ids_to_indices(data_tensor, agent_id_to_index_map):
    """
    Convert agent IDs in nested arrays to local indices using a hash map.

    :param data_tensor: Nested list structure containing agent IDs (can also contain sets)
    :param agent_id_to_index_map: Dictionary mapping agent_id -> local_index
    :return: Same structure with IDs replaced by local indices (-1 if not found)
    """
    result = []
    for agent_data in data_tensor:
        if isinstance(agent_data, (list, tuple, set)):
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

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


def analyze_step_function_for_writes(step_func: Callable, num_properties: int) -> Set[int]:
    """Analyze step function to find which property indices need write buffers."""
    write_property_indices = set()

    source = inspect.getsource(step_func)
    tree = ast.parse(source)
    signature = inspect.signature(step_func)
    param_names = list(signature.parameters.keys())
    
    # In SAGESim, the actual agent properties are the last N parameters
    # where N = num_properties
    property_params = param_names[-num_properties:]  # Take last N parameters as properties
    
    def check_target_for_writes(target_node):
        if isinstance(target_node, ast.Name):
            # Direct assignment: param_name = value
            if target_node.id in property_params:
                property_index = property_params.index(target_node.id)
                write_property_indices.add(property_index)
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
                        if tensor_name in property_params:
                            property_index = property_params.index(tensor_name)
                            write_property_indices.add(property_index)
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
    ) -> None:
        self._threads_per_block = threads_per_block
        self._step_function_file_path = step_function_file_path
        self._agent_factory = AgentFactory(space)
        self._is_setup = False
        self.globals = {}
        self.tick = 0
        self._write_property_indices = set()  # Cache for write property indices
        # following may be set later in setup if distributed execution


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

    def get_space(self) -> Space:
        return self._agent_factory._space

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


        # Determine and cache write property indices once during setup
        self._write_property_indices = set()
        for breed_idx_2_step_func in self._breed_idx_2_step_func_by_priority:
            for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
                breed_step_func_impl, module_fpath = breed_step_func_info
                write_indices = analyze_step_function_for_writes(breed_step_func_impl, self._agent_factory.num_properties)
                self._write_property_indices.update(write_indices)

        # Sort write property indices for consistent ordering
        self._write_property_indices = sorted(self._write_property_indices)

        if worker == 0:
            with open(self._step_function_file_path, "w") as f:
                f.write(
                    generate_gpu_func(
                        self._agent_factory.num_properties,
                        self._breed_idx_2_step_func_by_priority,
                        self._write_property_indices,
                    )
                )
        comm.barrier()

        # Import and cache the step function once during setup
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            step_func_module = importlib.import_module(
                os.path.splitext(self._step_function_file_path)[0]
            )
        self._step_func = step_func_module.stepfunc
        ###

        # Generate agent data tensors
        self.__rank_local_agent_data_tensors = (
            self._agent_factory._generate_agent_data_tensors()
        )
        self._is_setup = True

    def reset(self) -> None:
        # Generate global data tensor
        self._global_data_vector = list(self.globals.values())
        self.tick = 0
        
        # Generate agent data tensors
        self.__rank_local_agent_data_tensors = (
            self._agent_factory._generate_agent_data_tensors()
        )


    def simulate(
        self,
        ticks: int,
        sync_workers_every_n_ticks: int = 1,
    ) -> None:
        simulate_start = time.time()
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

        print(f"[TIMING] Total simulate time: {time.time() - simulate_start:.4f}s")

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
        Corountine that exec's cuda kernel. This coroutine should
        eventually be distributed among dask workers with agent
        data partitioning and data reduction.

        :param device_global_data_vector: cuda device array containing
            SAGESim global properties
        :param agent_data_tensors: listof property data tensors defined by user.
            Contains all agent info. Each inner list represents a particular property and may
            itself be a multidimensional list. This is also where the
            cuda kernels will make modifications as agent properties
            are updated.
        :param current_tick: Current simulation tick
        :param sync_workers_every_n_ticks: number of ticks to forward
            the simulation by
        :param agent_ids: agents to process by this cudakernel call
        """

        self.__rank_local_agent_ids = list(
            self._agent_factory._rank2agentid2agentidx[worker].keys()
        )
        threadsperblock = 32
        blockspergrid = int(
            math.ceil(len(self.__rank_local_agent_ids) / threadsperblock)
        )

        rank_local_agents_neighbors = self.get_space()._neighbor_compute_func(
            self.__rank_local_agent_data_tensors[1]
        )

        context_start = time.time()
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
        print(f"[TIMING] Contextualize agent data: {time.time() - context_start:.4f}s")

        tensor_convert_start = time.time()
        rank_local_agent_and_neighbor_adts = [
            convert_to_equal_side_tensor(
                self.__rank_local_agent_data_tensors[i] + received_neighbor_adts[i]
            )
            for i in range(self._agent_factory.num_properties)
        ]
        print(f"[TIMING] Convert to equal side tensor: {time.time() - tensor_convert_start:.4f}s")

        self._global_data_vector = cp.array(self._global_data_vector)

        # Create agent ID to local index mapping for fast lookups
        all_agent_ids_list = self.__rank_local_agent_ids + received_neighbor_ids
        agent_id_to_index = {int(agent_id): idx for idx, agent_id in enumerate(all_agent_ids_list)}

        rank_local_agent_and_non_local_neighbor_ids = cp.array(all_agent_ids_list)

        # Convert agent IDs to local indices in locations array (property index 1)
        # This eliminates the need for linear search in GPU kernels
        id_convert_start = time.time()

        # Property index 1 is 'locations' (neighbors/connections) - this is standard in SAGESim
        # Keep original locations unchanged, create converted copy for kernel
        locations_for_kernel = None
        if len(rank_local_agent_and_neighbor_adts) > 1:
            original_locations = rank_local_agent_and_neighbor_adts[1]
            # Convert on CPU (fast with hash map), then transfer to GPU
            locations_cpu = original_locations if not isinstance(original_locations, cp.ndarray) else original_locations.get()

            locations_as_indices = convert_agent_ids_to_indices(
                locations_cpu, agent_id_to_index
            )
            # Create new array for kernel - don't modify original!
            locations_for_kernel = cp.array(locations_as_indices)

        id_convert_time = time.time() - id_convert_start
        if id_convert_time > 0.1:
            print(f"[TIMING] ID to index conversion: {id_convert_time:.4f}s")

        # Create write buffers for properties that need them
        write_buffers = []
        for prop_idx in self._write_property_indices:
            # Create a copy of the tensor for writing
            write_buffer = cp.array(rank_local_agent_and_neighbor_adts[prop_idx])
            write_buffers.append(write_buffer)

        # Prepare all arguments: read tensors + write tensors
        # Use converted locations for kernel, but keep original in rank_local_agent_and_neighbor_adts
        all_args = []
        for i, tensor in enumerate(rank_local_agent_and_neighbor_adts):
            if i == 1 and locations_for_kernel is not None:
                all_args.append(locations_for_kernel)  # Use converted indices for kernel
            else:
                all_args.append(tensor)  # Use original
        all_args = all_args + write_buffers

        # CROSS-BREED SYNCHRONIZATION: Process breeds sequentially within each tick
        for tick_offset in range(sync_workers_every_n_ticks):
            current_tick = self.tick + tick_offset

            # Execute each breed priority group separately with synchronization
            for priority_idx, priority_group in enumerate(self._breed_idx_2_step_func_by_priority):
                # Execute step functions for this breed priority group only
                self._step_func[blockspergrid, threadsperblock](
                    current_tick,
                    self._global_data_vector,
                    *all_args,
                    1,  # Process exactly 1 tick
                    cp.float32(len(self.__rank_local_agent_ids)),
                    rank_local_agent_and_non_local_neighbor_ids,
                    priority_idx,  # Which priority group to execute
                )

                # Synchronization barrier: Wait for all blocks to complete this breed phase
                cp.cuda.Stream.null.synchronize()
            ####===============================================
            # Copy write buffers back to read buffers after ALL priority groups complete
            # This ensures all priorities execute with the same read buffer (state at start of tick)
            for i, prop_idx in enumerate(self._write_property_indices):
                rank_local_agent_and_neighbor_adts[prop_idx][:len(self.__rank_local_agent_ids)] = write_buffers[i][:len(self.__rank_local_agent_ids)]
            ####===============================================

            # Final synchronization before next tick
            cp.cuda.Stream.null.synchronize()

        # Update global tick counter after all threads have completed
        self.tick += sync_workers_every_n_ticks
        cp.get_default_memory_pool().free_all_blocks()

        # Convert GPU tensors back to CPU lists for next iteration
        gpu_to_cpu_start = time.time()
        num_agents = len(self.__rank_local_agent_ids)
        self.__rank_local_agent_data_tensors = [
            rank_local_agent_and_neighbor_adts[i][:num_agents].tolist()
            for i in range(self._agent_factory.num_properties)
        ]
        print(f"[TIMING] GPU to CPU conversion: {time.time() - gpu_to_cpu_start:.4f}s")

        """worker_agent_and_neighbor_data_tensors = (
            self._agent_factory.reduce_agent_data_tensors(
                worker_agent_and_neighbor_data_tensors,
                agent_and_neighbor_ids_in_subcontext,
                self._reduce_func,
            )
        )
        """
        self._global_data_vector = comm.allreduce(
            self._global_data_vector.tolist(), op=reduce_global_data_vector
        )


def reduce_global_data_vector(A, B):
    values = np.stack([A, B], axis=1)
    return np.max(values, axis=1)


def generate_gpu_func(
    n_properties: int,
    breed_idx_2_step_func_by_priority: List[List[Union[int, Callable]]],
    write_property_indices: Set[int],
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
    This function can then be directly loaded using importlib or written to a
    file and imported. For example, if you write the code to a file
    called step_func_code.py, you can import it as below:

        import importlib
        step_func_module = importlib.import_module("step_func_code")
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
        """Generate modified step function code with write buffer parameters."""
        source = inspect.getsource(step_func)
        signature = inspect.signature(step_func)
        param_names = list(signature.parameters.keys())
        
        # Use the same approach as analyze_step_function_for_writes
        property_params = param_names[-num_properties:]  # Take last N parameters as properties
        
        # Create mapping from property parameter names to write parameter names
        param_to_write_param = {}
        for i, param_name in enumerate(property_params):
            if i in write_indices:
                param_to_write_param[param_name] = f"write_{param_name}"
        
        if not param_to_write_param:
            return source  # No modifications needed
        
        # Step 1: Add write parameters to function signature
        # Find last parameter and add write parameters
        last_param = property_params[-1]
        write_params_list = list(param_to_write_param.values())

        # Replace the closing ): with write parameters + ):
        if f'{last_param},\n):' in source:
            # Multi-line with trailing comma
            write_params_str = '    ' + ',\n    '.join(write_params_list) + ',\n'
            modified_source = source.replace(f'{last_param},\n):', f'{last_param},\n{write_params_str}):')
        elif f'{last_param}\n):' in source:
            # Multi-line without trailing comma
            write_params_str = ',\n    ' + ',\n    '.join(write_params_list) + ',\n'
            modified_source = source.replace(f'{last_param}\n):', f'{last_param}{write_params_str}):')
        else:
            # Fallback: try to find the closing ): and insert before it
            write_params_str = ',\n    ' + ',\n    '.join(write_params_list)
            modified_source = source.replace('\n):', f'{write_params_str},\n):')


        # Step 2: Replace parameter names with write parameter names in WRITE contexts only
        tree = ast.parse(modified_source)
        
        # Walk through AST and replace write operations
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
                        # Direct assignment: param_name = value
                        if target_node.id in param_to_write_param:
                            target_node.id = param_to_write_param[target_node.id]
                    elif isinstance(target_node, ast.Subscript):
                        # Subscript assignment: param_name[...] = value or nested subscripts
                        replace_param_in_target(target_node.value)

                for target in node.targets:
                    replace_param_in_target(target)

            elif isinstance(node, ast.AugAssign):
                # Convert augmented assignments to regular assignments for double buffering
                # e.g., write_param[i] += 1 becomes write_param[i] = param[i] + 1
                def convert_aug_assign_target(target_node):
                    if isinstance(target_node, ast.Name):
                        if target_node.id in param_to_write_param:
                            # Create new assignment: write_param = param + value
                            original_target = ast.Name(id=target_node.id, ctx=ast.Load())
                            write_target = ast.Name(id=param_to_write_param[target_node.id], ctx=ast.Store())

                            new_value = ast.BinOp(
                                left=original_target,
                                op=node.op,
                                right=node.value
                            )

                            new_assign = ast.Assign(targets=[write_target], value=new_value)
                            return new_assign
                    elif isinstance(target_node, ast.Subscript):
                        # Check if the base is a parameter that needs write buffering
                        base = target_node
                        while isinstance(base, ast.Subscript):
                            base = base.value
                        if isinstance(base, ast.Name) and base.id in param_to_write_param:
                            # Create new assignment: write_param[...] = param[...] + value
                            # Clone the entire target structure for read (original param)
                            import copy
                            read_target = copy.deepcopy(target_node)
                            read_target.ctx = ast.Load()

                            # Clone the entire target structure for write (write param)
                            write_target = copy.deepcopy(target_node)
                            write_target.ctx = ast.Store()
                            # Find and replace the base name with write version
                            current = write_target
                            while isinstance(current, ast.Subscript):
                                if isinstance(current.value, ast.Name):
                                    current.value.id = param_to_write_param[current.value.id]
                                    break
                                current = current.value

                            new_value = ast.BinOp(
                                left=read_target,
                                op=node.op,
                                right=node.value
                            )

                            new_assign = ast.Assign(targets=[write_target], value=new_value)
                            return new_assign
                    return None

                new_assign = convert_aug_assign_target(node.target)
                if new_assign:
                    # Replace the AugAssign node with the new Assign node
                    # We need to track this for later replacement
                    node.__class__ = ast.Assign
                    node.targets = new_assign.targets
                    node.value = new_assign.value
                    # Remove the op and target attributes
                    if hasattr(node, 'op'):
                        delattr(node, 'op')
                    if hasattr(node, 'target'):
                        delattr(node, 'target')

        modified_source = ast.unparse(tree)
        
        return modified_source

    
    # Generate read arguments (original properties)
    read_args = [f"a{i}" for i in range(n_properties)]
    
    # Generate write arguments for properties that need write buffers
    write_args = [f"write_a{i}" for i in sorted(write_property_indices)]
    
    # Combine all arguments
    args = read_args + write_args
    
    # Generate modified step functions and simulation loop
    sim_loop = []
    step_sources = ["import os", "import sys"]
    imported_modules = set()

    modified_step_functions = []
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

            module_fpath = Path(module_fpath).absolute()
            module_name = module_fpath.stem
            if module_fpath not in imported_modules:
                step_sources += [
                    f"module_path = os.path.abspath('{module_fpath.parent}')",
                    "if module_path not in sys.path:",
                    "\tsys.path.append(module_path)",
                    f"from {module_name} import *",
                ]
                imported_modules.add(module_fpath)
            
            # Generate step function call
            sim_loop += [
                f"if breed_id == {breedidx}:",
                f"\t{modified_step_func_name}(",
                "\t\tthread_local_tick,",
                "\t\tagent_index,",
                "\t\tdevice_global_data_vector,",
                "\t\tagent_ids,",
                f"\t\t{','.join(args)},",
                "\t)",
            ]

    step_sources = "\n".join(step_sources)

    # Add modified step functions
    all_modified_step_functions = "\n\n".join(modified_step_functions)
    
    # Preprocess parts that would break in f-strings
    joined_sim_loop = "\n\t\t\t".join(sim_loop)
    joined_args = ",".join(args)

    # Generate breed-specific execution logic with proper indentation
    breed_execution_code = []
    for priority_idx, breed_idx_2_step_func in enumerate(breed_idx_2_step_func_by_priority):
        breed_execution_code.append(f"\t\t\tif current_priority_index == {priority_idx}:")
        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            step_func_name = getattr(breed_step_func_impl, "__name__", repr(callable))
            modified_step_func_name = f"{step_func_name}_double_buffer"

            breed_execution_code.append(f"\t\t\t\tif breed_id == {breedidx}:")
            breed_execution_code.append(f"\t\t\t\t\t{modified_step_func_name}(")
            breed_execution_code.append("\t\t\t\t\t\tthread_local_tick,")
            breed_execution_code.append("\t\t\t\t\t\tagent_index,")
            breed_execution_code.append("\t\t\t\t\t\tdevice_global_data_vector,")
            breed_execution_code.append("\t\t\t\t\t\tagent_ids,")
            breed_execution_code.append(f"\t\t\t\t\t\t{','.join(args)},")
            breed_execution_code.append("\t\t\t\t\t)")

    joined_breed_execution = "\n".join(breed_execution_code)

    func = [
        "# Auto-generated GPU kernel with cross-breed synchronization",
        "# Contains all necessary imports and modified step functions",
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
        "agent_ids,",
        "current_priority_index,",
        "):",
        "\tthread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x",
        "\tagent_index = thread_id",
        "\tif agent_index < num_rank_local_agents:",
        "\t\tbreed_id = a0[agent_index]",
        "\t\tfor tick in range(sync_workers_every_n_ticks):",
        "\t\t\tthread_local_tick = int(global_tick) + tick",
        "",
        joined_breed_execution,
    ]

    func = "\n".join(func)
    return func
