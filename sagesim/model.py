"""
SuperNeuroABM basic Model class

"""

from typing import Dict, List, Callable, Set, Any, Union
import os
from pathlib import Path
import importlib
import pickle
import math
import heapq
import warnings

import ast
import inspect

import cupy as cp
import numpy as np
from mpi4py import MPI

from sagesim.agent import AgentFactory, Breed
from sagesim.space import Space
from sagesim.internal_utils import convert_to_equal_side_tensor

comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


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

    def _analyze_step_function_for_writes(self, step_func: Callable) -> Set[int]:
        """Analyze step function to find which property indices need write buffers."""
        write_property_indices = set()

        source = inspect.getsource(step_func)
        tree = ast.parse(source)
        signature = inspect.signature(step_func)
        param_names = list(signature.parameters.keys())
        
        # In SAGESim, the actual agent properties are the last N parameters
        # where N = self._agent_factory.num_properties
        # The property parameters start after standard parameters
        num_properties = self._agent_factory.num_properties
        property_params = param_names[-num_properties:]  # Take last N parameters as properties
        
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
                # Check for direct tensor assignments like state_tensor[agent_index] = value
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name):
                            tensor_name = target.value.id
                            if tensor_name in property_params:
                                property_index = property_params.index(tensor_name)
                                write_property_indices.add(property_index)

        return write_property_indices

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

        # Generate global data tensor
        self._global_data_vector = list(self.globals.values())
        
        # Determine and cache write property indices once during setup
        self._write_property_indices = set()
        for breed_idx_2_step_func in self._breed_idx_2_step_func_by_priority:
            for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
                breed_step_func_impl, module_fpath = breed_step_func_info
                write_indices = self._analyze_step_function_for_writes(breed_step_func_impl)
                self._write_property_indices.update(write_indices)
        
        if worker == 0:
            with open(self._step_function_file_path, "w") as f:
                f.write(
                    generate_gpu_func(
                        self._agent_factory.num_properties,
                        self._breed_idx_2_step_func_by_priority,
                    )
                )
        comm.barrier()
        # Generate agent data tensors
        self.__rank_local_agent_data_tensors = (
            self._agent_factory._generate_agent_data_tensors()
        )
        self._is_setup = True

    def simulate(
        self,
        ticks: int,
        sync_workers_every_n_ticks: int = 1,
    ) -> None:
        comm.barrier()
        # Import the package using module package
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            step_func_module = importlib.import_module(
                os.path.splitext(self._step_function_file_path)[0]
            )

        # Access the step function using the module
        self._step_func = step_func_module.stepfunc

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
        rank_local_agent_and_neighbor_adts = [
            convert_to_equal_side_tensor(
                self.__rank_local_agent_data_tensors[i] + received_neighbor_adts[i]
            )
            for i in range(self._agent_factory.num_properties)
        ]
        self._global_data_vector = cp.array(self._global_data_vector)
        rank_local_agent_and_non_local_neighbor_ids = cp.array(
            self.__rank_local_agent_ids + received_neighbor_ids
        )
        
        # Create write buffers for properties that need them
        write_buffers = []
        for prop_idx in sorted(self._write_property_indices):
            # Create a copy of the tensor for writing
            write_buffer = cp.array(rank_local_agent_and_neighbor_adts[prop_idx])
            write_buffers.append(write_buffer)
        
        # Prepare all arguments: read tensors + write tensors
        all_args = list(rank_local_agent_and_neighbor_adts) + write_buffers
        
        self._step_func[blockspergrid, threadsperblock](
            self.tick,
            self._global_data_vector,
            *all_args,
            sync_workers_every_n_ticks,
            cp.float32(len(self.__rank_local_agent_ids)),
            rank_local_agent_and_non_local_neighbor_ids,
        )
        
        # Copy write buffers back to read buffers BEFORE extracting data
        for i, prop_idx in enumerate(sorted(self._write_property_indices)):
            rank_local_agent_and_neighbor_adts[prop_idx] = write_buffers[i]
        
        # Update global tick counter after all threads have completed
        self.tick += sync_workers_every_n_ticks
        cp.get_default_memory_pool().free_all_blocks()
        num_agents = len(self.__rank_local_agent_ids)
        self.__rank_local_agent_data_tensors = [
            rank_local_agent_and_neighbor_adts[i][:num_agents].tolist()
            for i in range(self._agent_factory.num_properties)
        ]
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
) -> str:
    """
    Generate GPU function string with double buffering support for race condition prevention.
    
    This function now includes double buffering to prevent race conditions from shared mutable 
    agent data tensors across agents in the same rank. It:
    1. Analyzes step functions to identify assignment operations to tensor properties
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
    :return: str representation of stepfunc cuda kernal with double buffering
        that can be written to file or imported directly.

    """

    def extract_imports_from_file(file_path: str) -> List[str]:
        """Extract import statements from a Python file."""
        imports = []
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.asname:
                            imports.append(f"import {alias.name} as {alias.asname}")
                        else:
                            imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    names = []
                    for alias in node.names:
                        if alias.asname:
                            names.append(f"{alias.name} as {alias.asname}")
                        else:
                            names.append(alias.name)
                    
                    if len(names) == 1:
                        imports.append(f"from {module} import {names[0]}")
                    else:
                        imports.append(f"from {module} import {', '.join(names)}")
        except Exception:
            # If we can't parse imports, add common ones
            imports = [
                "import cupy as cp",
                "import random", 
                "from sagesim.utils import get_this_agent_data_from_tensor, set_this_agent_data_from_tensor, get_neighbor_data_from_tensor"
            ]
        return imports
    
    def analyze_step_function_for_writes(step_func: Callable) -> Set[int]:
        """Analyze step function to find which property indices need write buffers."""
        write_property_indices = set()

        source = inspect.getsource(step_func)
        tree = ast.parse(source)
        signature = inspect.signature(step_func)
        param_names = list(signature.parameters.keys())
        standard_params = {'tick', 'agent_index', 'globals', 'agent_ids', 'breeds', 'locations'}
        property_params = [p for p in param_names if p not in standard_params]
        
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
                # Check for direct tensor assignments like state_tensor[agent_index] = value
                for target in node.targets:
                    if isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name):
                            tensor_name = target.value.id
                            if tensor_name in property_params:
                                property_index = property_params.index(tensor_name)
                                write_property_indices.add(property_index)

        return write_property_indices
    
    def generate_modified_step_func_code(step_func: Callable, write_indices: Set[int]) -> str:
        """Generate modified step function code with write buffer parameters."""
        try:
            source = inspect.getsource(step_func)
            lines = source.split('\n')
            signature = inspect.signature(step_func)
            param_names = list(signature.parameters.keys())
            standard_params = {'tick', 'agent_index', 'globals', 'agent_ids', 'breeds', 'locations'}
            property_params = [p for p in param_names if p not in standard_params]
            
            # Create mapping from property parameter names to write parameter names
            param_to_write_param = {}
            for i, param_name in enumerate(property_params):
                if i in write_indices:
                    param_to_write_param[param_name] = f"write_{param_name}"
            
            modified_lines = []
            in_function_def = False
            signature_complete = False
            in_set_call = False
            set_call_lines = []
            
            for line in lines:
                # Handle function signature
                if 'def ' in line and step_func.__name__ in line:
                    in_function_def = True
                    modified_lines.append(line)
                    continue
                    
                if in_function_def and not signature_complete:
                    if line.strip().endswith('):'):
                        if modified_lines and modified_lines[-1].strip().endswith(','):
                            # Previous line has comma, add write parameters
                            if param_to_write_param:
                                indent = len(line) - len(line.lstrip())
                                for param_name, write_param_name in param_to_write_param.items():
                                    modified_lines.append(" " * indent + write_param_name + ",")
                                modified_lines.append(line)
                            else:
                                modified_lines.append(line)
                        else:
                            # Need to add comma and write parameters
                            if param_to_write_param:
                                indent = len(line) - len(line.lstrip())
                                base_line = line.rstrip()[:-2] + ","
                                modified_lines.append(base_line)
                                for param_name, write_param_name in param_to_write_param.items():
                                    modified_lines.append(" " * indent + write_param_name + ",")
                                modified_lines.append(" " * indent + "):")
                            else:
                                modified_lines.append(line)
                        signature_complete = True
                        continue
                    else:
                        modified_lines.append(line)
                        continue
                
                # Handle multi-line set_this_agent_data_from_tensor calls
                if 'set_this_agent_data_from_tensor(' in line:
                    in_set_call = True
                    set_call_lines = [line]
                    continue
                elif in_set_call:
                    set_call_lines.append(line)
                    if ')' in line:
                        in_set_call = False
                        full_call = '\n'.join(set_call_lines)
                        
                        # Transform the call
                        for param_name, write_param_name in param_to_write_param.items():
                            if f', {param_name},' in full_call:
                                full_call = full_call.replace(f', {param_name},', f', {write_param_name},')
                            elif f'({param_name},' in full_call:
                                full_call = full_call.replace(f'({param_name},', f'({write_param_name},')
                            elif f', {param_name} ' in full_call:
                                full_call = full_call.replace(f', {param_name} ', f', {write_param_name} ')
                        
                        modified_lines.extend(full_call.split('\n'))
                        set_call_lines = []
                    continue
                
                # Handle direct tensor assignments like state_tensor[agent_index] = value
                line_modified = line
                for param_name, write_param_name in param_to_write_param.items():
                    if f'{param_name}[' in line:
                        line_modified = line_modified.replace(f'{param_name}[', f'{write_param_name}[')
                
                modified_lines.append(line_modified)
            
            return '\n'.join(modified_lines)
        except Exception:
            return inspect.getsource(step_func)
    
    # Analyze which properties need write buffers across all step functions
    all_write_property_indices = set()
    all_imports = set() 
    processed_files = set()
    
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            
            # Analyze this step function for write operations
            write_indices = analyze_step_function_for_writes(breed_step_func_impl)
            all_write_property_indices.update(write_indices)
            
            # Extract imports from the original file
            if str(module_fpath) not in processed_files:
                file_imports = extract_imports_from_file(str(module_fpath))
                all_imports.update(file_imports)
                processed_files.add(str(module_fpath))
    
    # Generate read arguments (original properties)
    read_args = [f"a{i}" for i in range(n_properties)]
    
    # Generate write arguments for properties that need write buffers
    write_args = [f"write_a{i}" for i in sorted(all_write_property_indices)]
    
    # Combine all arguments
    all_args = read_args + write_args
    
    # Generate modified step functions and simulation loop
    sim_loop = []
    modified_step_functions = []
    
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            step_func_name = getattr(breed_step_func_impl, "__name__", repr(callable))
            modified_step_func_name = f"{step_func_name}_double_buffer"
            
            # Generate modified step function
            write_indices = analyze_step_function_for_writes(breed_step_func_impl)
            modified_step_func_code = generate_modified_step_func_code(breed_step_func_impl, write_indices)
            modified_step_func_code = modified_step_func_code.replace(
                f"def {step_func_name}(",
                f"def {modified_step_func_name}("
            )
            modified_step_functions.append(modified_step_func_code)
            
            # Generate step function call
            sim_loop += [
                f"if breed_id == {breedidx}:",
                f"\t{modified_step_func_name}(",
                "\t\tthread_local_tick,",
                "\t\tagent_index,",
                "\t\tdevice_global_data_vector,",
                "\t\tagent_ids,",
                f"\t\t{','.join(all_args)},",
                "\t)",
            ]
    
    # Create clean import section
    import_lines = []
    basic_imports = []
    from_imports = []
    
    seen = set()
    for imp in sorted(all_imports):
        if imp not in seen:
            seen.add(imp)
            if imp.startswith('import '):
                basic_imports.append(imp)
            elif imp.startswith('from '):
                from_imports.append(imp)
    
    import_lines = basic_imports + from_imports
    step_sources = "\n".join(import_lines)
    
    # Add modified step functions
    all_modified_step_functions = "\n\n".join(modified_step_functions)
    
    # Preprocess parts that would break in f-strings
    joined_sim_loop = "\n\t\t\t".join(sim_loop)
    joined_args = ",".join(all_args)

    func = [
        "# Auto-generated GPU kernel with double buffering",
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
        "):",
        "\tthread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x",
        "\tagent_index = thread_id",
        "\tif agent_index < num_rank_local_agents:",
        "\t\tbreed_id = a0[agent_index]",
        "\t\tfor tick in range(sync_workers_every_n_ticks):",
        f"\n\t\t\tthread_local_tick = int(global_tick) + tick",
        f"\n\t\t\t{joined_sim_loop}",
    ]

    func = "\n".join(func)
    return func
