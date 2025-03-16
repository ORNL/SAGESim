"""
SuperNeuroABM basic Model class

"""

from typing import Dict, List, Callable, Set, Any, Union
import inspect
import importlib.machinery
import importlib
import os
import pickle
import math
import heapq
import warnings

import cupy as cp
import numpy as np
from mpi4py import MPI

from sagesim.agent import (
    AgentFactory,
    Breed,
    decontextualize_agent_data_tensors,
    contextualize_agent_data_tensors,
)
from sagesim.space import Space, NetworkSpace

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
        self._globals = {"tick": 0}
        # following may be set later in setup if distributed execution

    def register_breed(self, breed: Breed) -> None:
        if self._agent_factory.num_agents > 0:
            raise Exception(f"All breeds must be registered before agents are created!")
        self._agent_factory.register_breed(breed)

    def create_agent_of_breed(self, breed: Breed, **kwargs) -> int:
        agent_id = self._agent_factory.create_agent(breed, **kwargs)
        return agent_id

    def get_agent_property_value(self, id: int, property_name: str) -> Any:
        current_tick = self._global_data_vector[0]
        self._agent_factory._update_agent_property(
            self._agent_data_tensors, id, property_name, current_tick
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
        self._globals[property_name] = value

    def get_global_property_value(self, property_name: str) -> Union[float, int]:
        return self._globals[property_name]

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
        self._global_data_vector = list(self._globals.values())
        with open(self._step_function_file_path, "w") as f:
            f.write(
                generate_gpu_func(
                    self._agent_factory.num_properties,
                    self._breed_idx_2_step_func_by_priority,
                )
            )

    def simulate(
        self,
        ticks: int,
        sync_workers_every_n_ticks: int = 1,
    ) -> None:

        # TODO Remove the following commeneted code once Summit-tested
        # Generate agent data tensors
        self._agent_data_tensors = self._agent_factory._generate_agent_data_tensors()

        if worker == 0:
            # Chunk agent ids
            all_agent_ids = [int(i) for i in range(len(self._agent_data_tensors[0]))]
            chunk_size = len(all_agent_ids) // num_workers
            agent_ids_chunks = [
                all_agent_ids[i * chunk_size : (i + 1) * chunk_size]
                for i in range(num_workers - 1)
            ] + [all_agent_ids[(num_workers - 1) * chunk_size :]]
        else:
            agent_ids_chunks = None

        # Repeatedly execute worker coroutine untill simulation
        # has run for the right amount of ticks
        for time_chunk in range((ticks // sync_workers_every_n_ticks) + 1):
            if time_chunk == (ticks // sync_workers_every_n_ticks):
                num_time_chunks = ticks // sync_workers_every_n_ticks
                sync_workers_every_n_ticks = (
                    ticks - sync_workers_every_n_ticks * num_time_chunks
                )
                if sync_workers_every_n_ticks == 0:
                    break
            elif ticks % sync_workers_every_n_ticks:
                sync_workers_every_n_ticks = ticks % sync_workers_every_n_ticks

            worker_agent_ids_chunk = comm.scatter(agent_ids_chunks, root=0)
            (
                worker_global_data_vector,
                worker_agent_data_tensors,
                worker_agent_ids_result,
            ) = worker_coroutine(
                worker_agent_ids_chunk,
                self._global_data_vector,
                self._agent_data_tensors,
                self.get_space()._neighbor_compute_func,
                sync_workers_every_n_ticks,
                self._step_function_file_path,
            )

            self._global_data_vector = comm.allreduce(
                worker_global_data_vector, op=reduce_global_data_vector
            )

            self._agent_data_tensors = comm.allreduce(
                worker_agent_data_tensors, op=reduce_agent_data_tensors
            )

            # TODO convert global data vector to list of data tensors too
            # Time must always be sync'd

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


def reduce_global_data_vector(A, B):
    values = np.stack([A, B], axis=1)
    return np.max(values, axis=1)


def reduce_agent_data_tensors_(A, B):
    result = []
    # breed would be same as first
    result.append(A[0])
    # network would be same as first
    result.append(A[1])
    # state would be max value. Infected superceeds susceptible.
    states = np.stack([A[2], B[2]], axis=1)
    new_state = np.max(states, axis=1)
    result.append(new_state)
    return result


def reduce_agent_data_tensors(A, B):
    return A


def smap(func_args):
    return func_args[0](*func_args[1])


def generate_gpu_func(
    n_properties: int,
    breed_idx_2_step_func_by_priority: List[List[Union[int, Callable]]],
) -> str:
    """
    Numba cuda jit does not like us passing *args into
    cuda kernels. This is because the Python function
    will be compiled by cuda.jit and the parameter arguments
    type and count must be set at jit compilation time.
    However, SAGESim users will have varying numbers of
    properties in their step functions, which means
    our cuda kernel's parameter count would also be variable.
    Normally, we'd just define the stepfunc with *args, but
    due to the above constraints we have to infer the number of
    arguments from the user defined breed step functions,
    rewrite the overall stepfunc as a string and then pass it
    into cuda.jit to be compiled.

    This function returns a str representation of stepfunc cuda kernal
        that can be compiled using exec as below:

        step_funcs_code = generate_gpu_func(
                    len(agent_data_tensors),
                    breed_idx_2_step_func_by_priority,
                )
                exec(step_funcs_code)
    And run using exec as below:

        exec(
            cuda.jit(stepfunc)[blockspergrid, threadsperblock](
                device_global_data_vector,
                *agent_data_tensors,
                current_tick,
                sync_workers_every_n_ticks,
            )
        )

    Note: exec will pick up your current context so when you call
    exec to run the cuda kernel device_global_data_vector, agent_data_tensors,
    current_tick, sync_workers_every_n_ticks, must have been defined
    within the current scope.

    :param n_properties: int total number of agent properties
    :param breed_idx_2_step_func_by_priority: List of List. Each inner List
        first element is the breedidx and second element is the user defined
        step function. The major list elements are ordered in decreasing
        order of priority
    :param agent_ids: list of agents to execute this stepfunc kernel on
    :return: str representation of stepfunc cuda kernal
        that can be compiled using exec

    """
    args = [f"a{i}" for i in range(n_properties)]
    sim_loop = ""
    step_sources = ""
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func in breed_idx_2_step_func.items():
            step_source = inspect.getsource(breed_step_func)
            step_sources += (
                "\n\nimport cupy as cp\nimport math\n\n@jit.rawkernel(device='cuda')\n"
                + step_source
            )
            step_func_name = getattr(breed_step_func, "__name__", repr(callable))
            # step_source = step_source.splitlines(True)
            sim_loop += f"""
            \n\t\t\t\tif breed_id == {breedidx}:
            \t\t{step_func_name}(
            \t\t\tagent_id,
            \t\t\tagents_index_in_subcontext,
            \t\t\tdevice_global_data_vector,
            \t\t\t{','.join(args)},
            \t\t)
            #cuda.syncthreads()

            """
    func = f"""
    \nfrom random import random
    \nfrom cupyx import jit
    \n\n{step_sources}
    \n\n@jit.rawkernel()
def stepfunc(
    device_global_data_vector,
    {','.join(args)},
    sync_workers_every_n_ticks,
    agent_ids,
    agents_index_in_subcontext,
    ):
        thread_id = jit.blockIdx.x * jit.blockDim.x + jit.threadIdx.x
        #g = cuda.cg.this_grid()
        agent_id = thread_id        
        if agent_id < agent_ids.shape[0]:
            breed_id = a0[agent_id]                
            for tick in range(sync_workers_every_n_ticks):
                {sim_loop}                
                if thread_id == 0:
                    device_global_data_vector[0] += 1
    """
    func = func.replace("\t", "    ")
    return func  # compile(func, "<string>", "exec")


# Define worker coroutine that executes cuda kernel
# ------------------------------------------------------
def worker_coroutine(
    agent_ids,
    global_data_vector,
    agent_data_tensors,
    neighbor_compute_func,
    sync_workers_every_n_ticks,
    step_function_file_path,
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

    # Contextualize
    # Import the package using module package
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        step_func_module = importlib.import_module(
            os.path.splitext(step_function_file_path)[0]
        )

    # Access the step function using the module
    step_func = step_func_module.stepfunc

    threadsperblock = 32
    blockspergrid = int(math.ceil(len(agent_ids) / threadsperblock))

    # Get all neighbors of agents in agent_ids
    all_neighbors = np.unique(neighbor_compute_func(agent_data_tensors[1], agent_ids))
    all_neighbors = all_neighbors[~np.isnan(all_neighbors)].astype(int)

    # Contextualize agent data tensors
    (
        agent_ids_in_subcontext,
        agent_index_in_subcontextualized_adts,
        contextualized_agent_data_tensors,
    ) = contextualize_agent_data_tensors(agent_data_tensors, all_neighbors, agent_ids)

    # Pickling won't work on ctypes so have to let the
    # dask worker send the ADTs and GDT to GPU device
    device_global_data_vector = cp.asarray(global_data_vector)
    device_agent_data_tensors_subcontext = [
        cp.asarray(adt) for adt in contextualized_agent_data_tensors
    ]
    device_agent_ids = cp.asarray(agent_ids)

    # TODO document what this does
    device_agents_index_in_subcontext = cp.asarray(
        agent_index_in_subcontextualized_adts
    )
    # Execute cuda kernel. Unfortunately this seems to have to also
    # be performed in a string
    step_func[blockspergrid, threadsperblock](
        device_global_data_vector,
        *device_agent_data_tensors_subcontext,
        sync_workers_every_n_ticks,
        device_agent_ids,
        device_agents_index_in_subcontext,
    )

    # cuda.synchronize()
    # TODO consider using update_agents_properties as it's
    # best case time complexity is lower than copy_to_host()
    # agent_subcontext_indices = agent_index_in_subcontextualized_adts[agent_ids].astype(int)
    for i in range(len(agent_data_tensors)):
        agent_data_tensors[i][agent_ids_in_subcontext] = (
            device_agent_data_tensors_subcontext[i].get()
        )

    return (
        device_global_data_vector,
        agent_data_tensors,
        agent_ids,
    )
