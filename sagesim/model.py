"""
SuperNeuroABM basic Model class

"""

from typing import Dict, List, Callable, Set, Any, Union, Optional
import math
import heapq
from multiprocessing import Pool, Manager
import inspect
import time
import numpy as np
import importlib
import os

from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states
from tqdm import tqdm
from distributed import get_worker, get_client


from sagesim.agent import (
    AgentFactory,
    Breed,
    decontextualize_agent_data_tensors,
    contextualize_agent_data_tensors,
)
from sagesim.util import (
    init_dask_cluster,
    compress_tensor,
    convert_to_equal_side_tensor,
)
from sagesim.space import Space, NetworkSpace


class Model:
    THREADSPERBLOCK = 32
    STEPFUNCPATH = "step_func_code.py"

    def __init__(self, space: Space) -> None:
        self._agent_factory = AgentFactory(space)
        self._globals = {"tick": 0}
        # following may be set later in setup if distributed execution
        self._distributed = False
        self._num_workers = 0
        self._client = None

    def register_breed(self, breed: Breed) -> None:
        if self._agent_factory.num_agents > 0:
            raise Exception(f"Breeds must be registered before agents are created!")
        self._agent_factory.register_breed(breed)

    def create_agent_of_breed(self, breed: Breed, **kwargs) -> int:
        agent_id = self._agent_factory.create_agent(breed, **kwargs)
        return agent_id

    def get_agent_property_value(self, id: int, property_name: str) -> Any:
        return self._agent_factory.get_agent_property_value(
            property_name=property_name, agent_id=id
        )

    def set_agent_property_value(
        self,
        id: int,
        property_name: str,
        value: Any,
        dims: List[int] = None,
    ) -> None:
        self._agent_factory.set_agent_property_value(
            property_name=property_name, agent_id=id, value=value, dims=dims
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

    def setup(
        self,
        use_cuda: bool = True,
        num_dask_worker: int = None,
        scheduler_fpath: Optional[str] = None,
    ) -> None:
        """
        Must be called before first simulate call.
        Initializes model and resets ticks. Readies step functions
        and for breeds.

        :param use_cuda: runs model in GPU mode.
        :param num_dask_worker: number of dask workers
        :param scheduler_fpath: specify if using external dask cluster. Else
            distributed.LocalCluster is set up.
        """
        self._use_cuda = use_cuda
        if self._use_cuda:
            if not cuda.is_available():
                raise EnvironmentError(
                    "CUDA requested but no cuda installation detected."
                )
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

        # If distributed init dask cluster and get client
        if num_dask_worker:
            self._distributed = True
            self._num_workers = num_dask_worker
            self._client = init_dask_cluster(
                num_workers=num_dask_worker, scheduler_fpath=scheduler_fpath
            )
        else:
            self._num_workers = 1
        # Generate global data tensor
        self._global_data_vector = list(self._globals.values())
        with open(Model.STEPFUNCPATH, "w") as f:
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
        self._agent_data_tensors = self._agent_factory.generate_agent_data_tensors()

        # Chunk agent ids
        all_agent_ids = [int(i) for i in range(len(self._agent_data_tensors[0]))]
        chunk_size = len(all_agent_ids) // self._num_workers
        agent_ids_chunks = [
            all_agent_ids[i * chunk_size : (i + 1) * chunk_size]
            for i in range(self._num_workers - 1)
        ] + [all_agent_ids[(self._num_workers - 1) * chunk_size :]]
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
            agent_chunk_futures = []
            for chunk_index, agent_ids_chunk in enumerate(agent_ids_chunks):
                agent_futures_chunk = self._client.submit(
                    worker_coroutine,
                    self._global_data_vector,
                    self._agent_data_tensors,
                    agent_ids_chunk,
                    self.get_space()._neighbor_compute_func,
                    sync_workers_every_n_ticks,
                )
                agent_chunk_futures.append(agent_futures_chunk)

            subsolutions = self._client.gather(agent_chunk_futures)
            (
                global_data_vector_versions,
                agent_data_tensors_versions,
                agent_ids_versions,
            ) = zip(*subsolutions)
            self._global_data_vector = global_data_vector_versions[0]
            for property_id in range(self._agent_factory.num_properties):
                for agent_data_tensors_version, agent_ids_version in zip(
                    agent_data_tensors_versions, agent_ids_versions
                ):

                    self._agent_data_tensors[property_id][agent_ids_version] = (
                        agent_data_tensors_version[property_id]
                    )

        self._agent_factory.update_agents_properties(self._agent_data_tensors)

        return


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
    for breed_idx_2_step_func in breed_idx_2_step_func_by_priority:
        for breedidx, breed_step_func in breed_idx_2_step_func.items():
            step_source = inspect.getsource(breed_step_func)
            step_func_name = getattr(breed_step_func, "__name__", repr(callable))
            step_source = "\t\t\t\t".join(step_source.splitlines(True))
            sim_loop += f"""
            \n\t\t\tif breed_id == {breedidx}:
            \n\t\t\t\t{step_source}
            \t{step_func_name}(
            \t\tagent_id,
            \t\tagents_index_in_subcontext,
            \t\trng_states,
            \t\tdevice_global_data_vector,
            \t\t{','.join(args)},
            \t)
            g.sync()
            #cuda.syncthreads()

            """
    func = f"""
    \nfrom numba import cuda
    \nfrom sagesim.step_func_utils import *
    \n@cuda.jit\ndef stepfunc(
    device_global_data_vector,
    {','.join(args)},
    sync_workers_every_n_ticks,
    agent_ids,
    agents_index_in_subcontext,
    rng_states,
    ):
        thread_id = int(cuda.grid(1))
        g = cuda.cg.this_grid()
        agent_id = thread_id
        if agent_id >= len(a0):
            return
        valid = False
        for i in range(len(agent_ids)):
            if agent_id == agent_ids[i]:
                valid = True
                break
        if not valid:
            return
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
    global_data_vector,
    agent_data_tensors,
    agent_ids,
    neighbor_compute_func,
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
    # Contextualize
    # Import the package using module package
    step_func_module = importlib.import_module(os.path.splitext(Model.STEPFUNCPATH)[0])

    # Access the step function using the module
    step_func = step_func_module.stepfunc

    threadsperblock = 32
    blockspergrid = int(math.ceil(len(agent_ids) / threadsperblock))
    rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=1)
    start = time.time()
    all_neighbors = np.unique(neighbor_compute_func(agent_data_tensors[1], agent_ids))
    # ...

    all_neighbors = np.unique(neighbor_compute_func(agent_data_tensors[1], agent_ids))
    all_neighbors = all_neighbors[~np.isnan(all_neighbors)].astype(int)
    (
        agents_index_in_subcontext,
        agent_data_tensors,
    ) = contextualize_agent_data_tensors(agent_data_tensors, all_neighbors, agent_ids)
    # sync_workers_every_n_ticks block of time on a worker

    # Pickling won't work on ctypes so have to let the
    # dask worker send the ADTs and GDT to GPU device
    device_global_data_vector = cuda.to_device(global_data_vector)
    device_agent_data_tensors = [cuda.to_device(adt) for adt in agent_data_tensors]
    device_agent_ids = cuda.to_device(agent_ids)
    # TODO document what this does
    device_agents_index_in_subcontext = cuda.to_device(agents_index_in_subcontext)
    # Execute cuda kernel. Unfortunately this seems to have to also
    # be performed in a string
    step_func[blockspergrid, threadsperblock](
        device_global_data_vector,
        *device_agent_data_tensors,
        sync_workers_every_n_ticks,
        device_agent_ids,
        device_agents_index_in_subcontext,
        rng_states,
    )
    cuda.synchronize()
    # TODO consider using update_agents_properties as it's
    # best case time complexity is lower than copy_to_host()
    agent_subcontext_indices = agents_index_in_subcontext[agent_ids].astype(int)
    return (
        device_global_data_vector.copy_to_host(),
        [
            adt.copy_to_host()[agent_subcontext_indices]
            for adt in device_agent_data_tensors
        ],
        agent_ids,
    )
