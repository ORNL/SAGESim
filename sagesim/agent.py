from __future__ import annotations
import math
from typing import Any, Callable, Iterable, List, Dict, Union, Tuple
from collections import OrderedDict
from copy import copy
from time import time

import numpy as np
import cupy as cp

from sagesim.breed import Breed
from sagesim.util import (
    compress_tensor,
    convert_to_equal_side_tensor,
)
from sagesim.space import Space, NetworkSpace


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
        self._agent_connectivity: Dict[int, Dict[int, float]] = {}

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
        agent_id = self._num_agents
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
        return self._property_name_2_agent_data_tensor[property_name][agent_id]

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

        if property_name not in self._property_name_2_agent_data_tensor:
            raise ValueError(f"{property_name} not a property of any breed")
        self._property_name_2_agent_data_tensor[property_name][agent_id] = value

    def get_agents_with(self, query: Callable) -> Dict[int, List[Any]]:
        """
        Returns an Dict, key: agent_id value: List of properties, of the agents that satisfy
            the query. Query must be a callable that returns a boolean and accepts **kwargs
            where arguments may with breed property names may be accepted and used to form
            query logic.

        :param query: Callable that takes agent data as dict and returns List of agent data
        :return: Dict of agent_id: List of properties

        """
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
        tick: int,
    ) -> None:
        property_idx = self._property_name_2_index[property_name]
        adt = regularized_agent_data_tensors[property_idx]
        value = (
            compress_tensor(adt[agent_id], min_axis=0)
            if type(adt[agent_id]) == Iterable
            else adt[agent_id]
        )

        self._property_name_2_agent_data_tensor[property_name][agent_id] = value


def contextualize_agent_data_tensors(
    agent_data_tensors: List[List[Any]],
    all_neighbors,
    agent_ids_chunk: List[int],
) -> Tuple[List[List[int]], List[List[int]], List[List[List[Any]]]]:
    """
    Chunks agent data tensors so that each distributed worker does not
    get more data than the agents that worker processes actually need.

    :return: 3-tuple.
        1. agent_ids_chunks: List of Lists of agent_ids to be processed
            by each worker.
        2. agents_index_in_subcontext: A single List, len equal to number of agents
            and entries specifying which index in agent data tensor
            chunk is occupied by agent_id corresponding to index of this
            list.
        3. agent_data_tensors_subcontexts: subcontext of agent_data_tensors
            required by agents of agent_ids_chunks to be processed by a worker
    """
    agent_ids_in_subcontext = set.union(
        set(agent_ids_chunk), set(all_neighbors.tolist())
    )
    """for agent_id in agent_ids_chunk:
        agent_ids_in_subcontext.add(agent_id)
        neighbors = all_neighbors.get(agent_id, set())
        agent_ids_in_subcontext.update(neighbors)"""
    agent_ids_in_subcontext = sorted(agent_ids_in_subcontext)
    agent_data_tensors_subcontext = [
        adt[agent_ids_in_subcontext] for adt in agent_data_tensors
    ]
    agents_index_in_subcontext = []
    index_in_subcontext = 0
    for agent_id in range(len(agent_data_tensors[0])):
        if agent_id not in agent_ids_in_subcontext:
            agents_index_in_subcontext.append(math.nan)
        else:
            agents_index_in_subcontext.append(int(index_in_subcontext))
            index_in_subcontext += 1
    return (
        np.array(agent_ids_in_subcontext).astype(int),
        np.array(agents_index_in_subcontext),
        agent_data_tensors_subcontext,
    )


def decontextualize_agent_data_tensors(
    agents_to_be_processed: List[int],
    agent_data_tensors_versions: List[List[List[Any]]],
    agents_index_in_new_versions,
    previous_agent_data_tensors_full: List[List[Any]],
    reduction_function: Callable = None,
) -> Dict[int, List[Any]]:
    """Does the opposite of contextualize_agent_data_tensors
    and recombines partitions using the given reduce
    function"""

    def simple_reduce_func(previous_agent, agent_versions):
        """using this for now TODO move to API for user
        defined reduce functions"""
        if len(agent_versions) > 0:
            reduced_agent = agent_versions[0]
        else:
            reduced_agent = previous_agent
        return reduced_agent

    start = time()
    print("starting reduce partition previous adts")
    total_num_agents = len(previous_agent_data_tensors_full[0])
    processed_set = set(agents_to_be_processed)
    counter = 0
    agents_index_in_partition_of_prev = np.full(
        (total_num_agents,), math.nan, dtype=int
    )
    for agent_id in range(total_num_agents):
        if agent_id in processed_set:
            agents_index_in_partition_of_prev[agent_id] = counter
            counter += 1
    previous_agent_data_tensors_partition = [
        [padt_fullcontext[aid] for aid in agents_to_be_processed]
        for padt_fullcontext in previous_agent_data_tensors_full
    ]
    print(f"Reduce partition previous adts  took {time() - start} seconds")

    start = time()
    print("starting reduce compress partition previous adts")
    previous_agent_data_tensors_partition = [
        compress_tensor(adt) for adt in previous_agent_data_tensors_partition
    ]
    print(f"finished compress partition previous adts {time() - start} seconds")
    n_properties = len(previous_agent_data_tensors_partition)
    n_versions = len(agent_data_tensors_versions)

    """new_agent_data_tensors_partition_versions = []
    for version_id, nadts_version in enumerate(agent_data_tensors_versions):
        nadts_version_partition = []
        for property_id in range(n_properties):
            nadt_version_partition = []
            for agent_id in agents_to_be_processed:
                agent_index = agents_index_in_new_versions[version_id][agent_id]
                if math.isnan(agent_index):
                    continue
                agent_index = int(agent_index)
                nadt_version_partition.append(nadts_version[property_id][agent_index])
            nadt_version_partition = compress_tensor(nadt_version_partition)
            nadts_version_partition.append(nadt_version_partition)
        new_agent_data_tensors_partition_versions.append(nadts_version_partition)"""

    agent_data_tensors_versions = [
        [compress_tensor(adt) for adt in adts_version]
        for adts_version in agent_data_tensors_versions
    ]

    agent2data = {}
    for agent_id in agents_to_be_processed:
        agent_versions = []
        for version_id in range(n_versions):
            agent_version = []
            for property_id in range(n_properties):
                agent_index = agents_index_in_new_versions[version_id][agent_id]
                if math.isnan(agent_index):
                    continue
                agent_index = int(agent_index)
                agent_version.append(
                    agent_data_tensors_versions[version_id][property_id][agent_index]
                )
            if len(agent_version) > 0:
                agent_versions.append(agent_version)
        previous_agent = [
            previous_agent_data_tensors_partition[property_id][
                int(agents_index_in_partition_of_prev[agent_id])
            ]
            for property_id in range(n_properties)
        ]
        new_agent = simple_reduce_func(previous_agent, agent_versions)
        agent2data[agent_id] = new_agent
    return agent2data
