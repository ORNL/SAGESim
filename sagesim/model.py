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
        self._step_func[blockspergrid, threadsperblock](
            self.tick,
            self._global_data_vector,
            *rank_local_agent_and_neighbor_adts,
            sync_workers_every_n_ticks,
            cp.float32(len(self.__rank_local_agent_ids)),
            rank_local_agent_and_non_local_neighbor_ids,
        )
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
                *agent_data_tensors,
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
    :return: str representation of stepfunc cuda kernal
        that can be written to file or imported directly.

    """
    args = [f"a{i}" for i in range(n_properties)]
    sim_loop = []
    step_sources = ["import os", "import sys"]
    imported_modules = set()
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func_info in breed_idx_2_step_func.items():
            breed_step_func_impl, module_fpath = breed_step_func_info
            step_func_name = getattr(breed_step_func_impl, "__name__", repr(callable))
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
            sim_loop += [
                f"if breed_id == {breedidx}:",
                f"\t{step_func_name}(",
                "\t\tthread_local_tick,",
                "\t\tagent_index,",
                "\t\tdevice_global_data_vector,",
                "\t\tagent_ids,",
                f"\t\t{','.join(args)},",
                "\t)",
            ]
    step_sources = "\n".join(step_sources)

    # Preprocess parts that would break in f-strings
    joined_sim_loop = "\n\t\t\t".join(sim_loop)
    joined_args = ",".join(args)

    func = [
        "from cupyx import jit",
        step_sources,
        "\n\n@jit.rawkernel(device='cuda')",
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
