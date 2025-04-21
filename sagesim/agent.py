from __future__ import annotations
import math
from typing import Any, Callable, Iterable, List, Dict, Union, Tuple, Set
from collections import OrderedDict
from copy import copy
import time

import numpy as np
import cupy as cp
import pickle
from mpi4py import MPI

from sagesim.breed import Breed
from sagesim.util import (
    compress_tensor,
    convert_to_equal_side_tensor,
)
from sagesim.space import Space, NetworkSpace
from pathlib import Path


comm = MPI.COMM_WORLD
num_workers = comm.Get_size()
worker = comm.Get_rank()


class AgentFactory:
    def __init__(self, space: Space) -> None:
        self._breeds: Dict[str, Breed] = OrderedDict()
        self._space: Space = space
        self._space._agent_factory = self
        self._num_breeds = 0
        self._num_agents = 0
        self._property_name_2_agent_data_tensor = OrderedDict(
            {"breed": [], "locations": []}
        )
        self._property_name_2_defaults = OrderedDict(
            {
                "breed": 0,
                "locations": self._space._locations_defaults,
            }
        )
        self._property_name_2_index = {
            "breed": 0,
            "locations": 1,
        }
        self._agent2rank = {}  # global
        self._rank2agentid2agentidx = {}  # global

        self._current_rank = 0

    @property
    def breeds(self) -> List[Breed]:
        """
        Returns the breeds registered in the model

        :return: A list of currently registered breeds.

        """
        return self._breeds.values()

    @property
    def num_agents(self) -> int:
        """
        Returns number of agents. Agents are not removed if they are killed at the
            moment.

        """
        return self._num_agents

    @property
    def num_properties(self) -> int:
        """
        Returns number of properties, equivalent to the number
        of agent data tensors.

        """
        return len(self._property_name_2_agent_data_tensor)

    def register_breed(self, breed: Breed) -> None:
        """
        Registered agent breed in the model so that agents can be created under
            this definition.

        :param breed: Breed definition of agent

        """
        breed._breedidx = self._num_breeds
        self._num_breeds += 1
        self._breeds[breed.name] = breed
        for property_name, default in breed.properties.items():
            self._property_name_2_index[property_name] = len(
                self._property_name_2_agent_data_tensor
            )
            self._property_name_2_agent_data_tensor[property_name] = []
            self._property_name_2_defaults[property_name] = default

    def create_agent(self, breed: Breed, **kwargs) -> int:
        """
        Creates and agent of the given breed initialized with the properties given in
            **kwargs.

        :param breed: Breed definition of agent
        :param **kwargs: named arguments of agent properties. Names much match properties
            already registered in breed.
        :return: Agent ID

        """

        agent_id = self._num_agents
        # Assign agents to rank in round robin fashion across available workers.
        self._agent2rank[agent_id] = self._current_rank
        agentid2agentidx_of_current_rank = self._rank2agentid2agentidx.get(
            self._current_rank, OrderedDict()
        )
        agentid2agentidx_of_current_rank[agent_id] = len(
            self._property_name_2_agent_data_tensor["locations"]
        )
        self._rank2agentid2agentidx[self._current_rank] = (
            agentid2agentidx_of_current_rank
        )

        self._current_rank += 1
        if self._current_rank >= num_workers:
            self._current_rank = 0

        if worker == self._agent2rank[agent_id]:
            # Only the worker that owns this agent will create and store the agent data.
            # This is to avoid unnecessary data duplication across workers.
            if breed.name not in self._breeds:
                raise ValueError(f"Fatal: unregistered breed {breed.name}")
            property_names = self._property_name_2_agent_data_tensor.keys()
            for property_name in property_names:
                if property_name == "breed":
                    breed = self._breeds[breed.name]
                    self._property_name_2_agent_data_tensor[property_name].append(
                        breed._breedidx
                    )
                else:
                    default_value = copy(self._property_name_2_defaults[property_name])
                    self._property_name_2_agent_data_tensor[property_name].append(
                        kwargs.get(property_name, default_value)
                    )

        self._num_agents += 1

        return agent_id

    def get_agent_property_value(self, property_name: str, agent_id: int) -> Any:
        """
        Returns the value of the specified property_name of the agent with
            agent_id

        :param property_name: str name of property as registered in the breed.
        :param agent_id: Agent's id as returned by create_agent
        :return: value of property_name property for agent of agent_id
        """
        agent_rank = self._agent2rank[agent_id]
        if agent_rank == worker:
            subcontextidx = self._rank2agentid2agentidx.get(worker).get(agent_id)
            result = self._property_name_2_agent_data_tensor[property_name][
                subcontextidx
            ]
        else:
            result = None
        result = comm.bcast(result, root=agent_rank)

        return result

    def set_agent_property_value(
        self,
        property_name: str,
        agent_id: int,
        value: Any,
    ) -> None:
        """
        Sets the property of property_name for the agent with agent_id with
            value.
        :param property_name: str name of property as registered in the breed.
        :param agent_id: Agent's id as returned by create_agent
        :param value: New value for property
        """
        if worker == self._agent2rank[agent_id]:
            if property_name not in self._property_name_2_agent_data_tensor:
                raise ValueError(f"{property_name} not a property of any breed")
            subcontextidx = self._rank2agentid2agentidx.get(worker).get(agent_id)
            self._property_name_2_agent_data_tensor[property_name][
                subcontextidx
            ] = value

    def get_agents_with(self, query: Callable) -> Dict[int, List[Any]]:
        """
        Returns an Dict, key: agent_id value: List of properties, of the agents that satisfy
            the query. Query must be a callable that returns a boolean and accepts **kwargs
            where arguments may with breed property names may be accepted and used to form
            query logic.

        :param query: Callable that takes agent data as dict and returns List of agent data
        :return: Dict of agent_id: List of properties

        """
        raise NotImplementedError(
            "get_agents_with not implemented in base AgentFactory class. "
            "This should be implemented in subclasses."
        )
        matching_agents = {}
        property_names = self._property_name_2_agent_data_tensor.keys()
        for agent_id in range(self._num_agents):
            agent_properties = {
                property_name: self._property_name_2_agent_data_tensor[property_name][
                    agent_id
                ]
                for property_name in property_names
            }
            if query(**agent_properties):
                matching_agents[agent_id] = agent_properties
        return matching_agents

    def _generate_agent_data_tensors(
        self,
    ) -> Union[List[cp.ndarray],]:
        converted_agent_data_tensors = []
        for property_name in self._property_name_2_agent_data_tensor.keys():
            converted_agent_data_tensors.append(
                convert_to_equal_side_tensor(
                    self._property_name_2_agent_data_tensor[property_name]
                )
            )

        return converted_agent_data_tensors

    def _update_agent_property(
        self,
        regularized_agent_data_tensors: List[cp.ndarray],
        agent_id: int,
        property_name: str,
    ) -> None:
        if worker == self._agent2rank[agent_id]:
            subcontextidx = self._rank2agentid2agentidx.get(worker).get(agent_id)
            property_idx = self._property_name_2_index[property_name]
            adt = regularized_agent_data_tensors[property_idx]
            value = (
                compress_tensor(adt[subcontextidx], min_axis=0)
                if type(adt[subcontextidx]) == Iterable
                else adt[subcontextidx]
            )

            self._property_name_2_agent_data_tensor[property_name][
                subcontextidx
            ] = value

    def contextualize_agent_data_tensors(
        self,
        agent_data_tensors,
        agent_ids_chunk,
        all_neighbors,
        first_tick: bool = False,
    ) -> Tuple[Set[int], List[cp.array]]:
        """
        Chunks agent data tensors so that each distributed worker does not
        get more data than the agents that worker processes actually need.

        :return: 2-tuple.
            1. agent_ids_chunks: List of Lists of agent_ids to be processed
                by each worker.
            3. agent_data_tensors_subcontexts: subcontext of agent_data_tensors
                required by agents of agent_ids_chunks to be processed by a worker
        """
        # Write my agent info to file
        if worker == 0:
            start_time = time.time()
            print(agent_data_tensors[0].shape, len(agent_ids_chunk), flush=True)

        # Write agent_data_tensors to a file
        # Create a temporary directory if it does not exist
        temp_dir = Path(f"./tmp/")
        if first_tick:
            # Only write if first tick, after that the files are updated upon reduce anyway.
            for agent_idx, agent_id in enumerate(agent_ids_chunk):
                agent_info = [
                    agent_data_tensors[prop_idx][agent_idx].tolist()
                    for prop_idx in range(self.num_properties)
                ]
                with open(temp_dir / f"{int(agent_id)}.pkl", "wb") as f:
                    pickle.dump(agent_info, f)
        comm.barrier()

        if worker == 0:
            print(
                f"Time to write agent data: {time.time() - start_time:.6f} seconds",
                flush=True,
            )

        if worker == 0:
            start_time = time.time()

        neighbor_ids = []
        found_neighbors = set()
        neighbor_adts = [[] for _ in range(self.num_properties)]
        # Read neighbor data from files
        for agent_neighborids in all_neighbors:
            for neighbor_id in agent_neighborids.tolist():
                if cp.isnan(neighbor_id):
                    # Last neighbor reached
                    break
                if neighbor_id in found_neighbors:
                    # Skip if already processed this neighbor
                    continue
                neighbor_rank = self._agent2rank[int(neighbor_id)]
                if neighbor_rank == worker:
                    # Already in agent_data_tensors, no need to read again
                    continue
                found_neighbors.add(neighbor_id)
                neighbor_ids.append(neighbor_id)

        if worker == 0:
            print(len(neighbor_ids), len(found_neighbors), flush=True)
        for neighbor_id in neighbor_ids:
            # Read neighbor data from file
            with open(temp_dir / f"{int(neighbor_id)}.pkl", "rb") as f:
                try:
                    neighbor_data = pickle.load(f)
                except EOFError:
                    print(neighbor_id, "EOFError", flush=True)
                    exit()
                for prop_idx in range(self.num_properties):
                    neighbor_adts[prop_idx].append(neighbor_data[prop_idx])

        if worker == 0:
            print(
                f"Time to read agent data: {time.time() - start_time:.6f} seconds",
                flush=True,
            )

        if worker == 0:
            start_time = time.time()

        agent_and_neighbor_adts = [
            convert_to_equal_side_tensor(
                agent_data_tensors[i].tolist() + neighbor_adts[i]
            )
            for i in range(self.num_properties)
        ]
        agent_and_neighbor_ids = agent_ids_chunk + neighbor_ids
        comm.barrier()
        if worker == 0:
            print(
                f"Time to postprocess recv: {time.time() - start_time:.6f} seconds",
                flush=True,
            )

        return (
            agent_and_neighbor_ids,
            agent_and_neighbor_adts,
        )

    def reduce_agent_data_tensors(
        self,
        agent_and_neighbor_data_tensors,
        agent_and_neighbor_ids_in_subcontext,
        reduce_func: Callable = None,
    ) -> List[cp.ndarray]:

        num_agents_this_rank = len(self._rank2agentid2agentidx.get(worker).keys())
        agent_ids = agent_and_neighbor_ids_in_subcontext[:num_agents_this_rank]
        neighbor_ids = agent_and_neighbor_ids_in_subcontext[num_agents_this_rank:]

        agent_data_tensors = [
            agent_and_neighbor_data_tensors[prop_idx][:num_agents_this_rank].tolist()
            for prop_idx in range(self.num_properties)
        ]
        neighbor_adts = [
            agent_and_neighbor_data_tensors[i][num_agents_this_rank:].tolist()
            for i in range(self.num_properties)
        ]

        temp_dir = Path(f"./tmp/")
        # First write this rank agents data to files
        if worker == 0:
            start_time = time.time()
        for agent_idx, agent_id in enumerate(agent_ids):
            agent_info = [
                agent_data_tensors[prop_idx][agent_idx]
                for prop_idx in range(self.num_properties)
            ]
            # write agent data reduce and write back
            with open(temp_dir / f"{int(agent_id)}.pkl", "wb") as f:
                pickle.dump(agent_info, f)
        comm.barrier()
        if worker == 0:
            print(
                f"Time to write agent data to prep for reduce: {time.time() - start_time:.6f} seconds",
                flush=True,
            )

        # Next, read, reduce, and write neighbor data to files
        if worker == 0:
            start_time = time.time()
        for neighbor_idx, neighbor_id in enumerate(neighbor_ids):
            neighbor_info_new = [
                neighbor_adts[prop_idx][neighbor_idx]
                for prop_idx in range(self.num_properties)
            ]
            # read agent data reduce and write back
            with open(temp_dir / f"{int(neighbor_id)}.pkl", "rb+") as f:
                neighbor_info_original = pickle.load(f)
                # reduce
                neighbor_info_reduced = reduce_func(
                    neighbor_info_original, neighbor_info_new
                )
                pickle.dump(neighbor_info_reduced, f)
        comm.barrier()
        if worker == 0:
            print(
                f"Time to reduce and write neighbors: {time.time() - start_time:.6f} seconds",
                flush=True,
            )

        if worker == 0:
            start_time = time.time()

        # Read version of your agents that other ranks have written
        for agent_id, agent_idx in self._rank2agentid2agentidx.get(worker).items():
            with open(temp_dir / f"{int(agent_id)}.pkl", "rb") as f:
                agent_data_reduced = pickle.load(f)
                for prop_idx in range(self.num_properties):
                    agent_data_tensors[prop_idx][agent_idx] = agent_data_reduced[
                        prop_idx
                    ]
        comm.barrier()
        if worker == 0:
            print(
                f"Time to read reduced adt version: {time.time() - start_time:.6f} seconds",
                flush=True,
            )

        return agent_data_tensors
