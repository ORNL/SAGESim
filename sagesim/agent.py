from __future__ import annotations
import math
from typing import Any, Callable, Iterable, List, Dict, Union, Tuple, Set
from collections import OrderedDict
from copy import copy
from time import time

import numpy as np
import cupy as cp
from mpi4py import MPI

from sagesim.breed import Breed
from sagesim.util import (
    compress_tensor,
    convert_to_equal_side_tensor,
)
from sagesim.space import Space, NetworkSpace


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
        self._agent_connectivity: Dict[int, Dict[int, float]] = {}
        self._agent2rank = {}  # global
        self._rank2agentids = {}  # global
        self._this_rank_agent2subcontextidx = OrderedDict()  # local

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
        agents_in_current_rank = self._rank2agentids.get(self._current_rank, [])
        agents_in_current_rank.append(agent_id)
        self._rank2agentids[self._current_rank] = agents_in_current_rank

        self._current_rank += 1
        if self._current_rank >= num_workers:
            self._current_rank = 0

        if worker == self._agent2rank[agent_id]:
            # Only the worker that owns this agent will create and store the agent data.
            # This is to avoid unnecessary data duplication across workers.
            if breed.name not in self._breeds:
                raise ValueError(f"Fatal: unregistered breed {breed.name}")
            self._this_rank_agent2subcontextidx[agent_id] = len(
                self._property_name_2_agent_data_tensor["locations"]
            )
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
            subcontextidx = self._this_rank_agent2subcontextidx.get(agent_id)
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
            subcontextidx = self._this_rank_agent2subcontextidx.get(agent_id)
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
            subcontextidx = self._this_rank_agent2subcontextidx.get(agent_id)
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
        self, agent_data_tensors, agentandneighbors
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
        neighborrankandagentadts = [
            (
                self._agent2rank[neighbor_id],
                [
                    adt[self._this_rank_agent2subcontextidx[agent_id]]
                    for adt in agent_data_tensors
                ],
            )
            for agent_id, neighbor_id in agentandneighbors
        ]
        neighborrank2agentids = OrderedDict()
        send_info = OrderedDict()
        for i, (rank, adts) in enumerate(neighborrankandagentadts):
            agent_id = agentandneighbors[i][0]
            if agent_id not in neighborrank2agentids.get(rank, []):
                neighborrank2agentids.setdefault(rank, []).append(agent_id)
                send_info.setdefault(rank, []).append([adt.tolist() for adt in adts])

        for rank, adts in send_info.items():
            # send_adts = [list(v) for v in list(zip(*adts))]
            send_info[rank] = (adts, neighborrank2agentids[rank])

        received_neighbor_adts = []
        received_neighbor_ids = []
        for from_rank in range(num_workers):
            if from_rank == worker:
                # Send the data to the other workers in chunks
                for to_rank in range(num_workers):
                    if to_rank == worker:
                        # Don't send to self
                        continue

                    if to_rank in send_info:
                        # Send the data for this rank
                        agent_data_tensor_to_send = send_info[to_rank]
                    else:
                        # Send None for ranks without agent data
                        agent_data_tensor_to_send = None

                    if agent_data_tensor_to_send is not None:
                        # Break the data into chunks
                        chunk_size = 1024  # Define a fixed chunk size
                        num_chunks = len(agent_data_tensor_to_send) // chunk_size + (
                            1 if len(agent_data_tensor_to_send) % chunk_size > 0 else 0
                        )
                        comm.send(
                            num_chunks,
                            dest=to_rank,
                            tag=(worker * num_workers) + to_rank,
                        )
                        for i in range(num_chunks):
                            chunk = agent_data_tensor_to_send[
                                i * chunk_size : (i + 1) * chunk_size
                            ]
                            comm.send(
                                chunk,
                                dest=to_rank,
                                tag=(worker * num_workers) + to_rank + i,
                            )
                    else:
                        # Send zero chunks to indicate no data
                        comm.send(0, dest=to_rank, tag=(worker * num_workers) + to_rank)

            else:
                # Receive the data from the other workers in chunks
                num_chunks = comm.recv(
                    source=from_rank, tag=(from_rank * num_workers) + worker
                )
                if num_chunks > 0:
                    received_data_chunks = []
                    for i in range(num_chunks):
                        chunk = comm.recv(
                            source=from_rank, tag=(from_rank * num_workers) + worker + i
                        )
                        received_data_chunks.extend(chunk)
                    received_neighbor_adts.extend(received_data_chunks[0])
                    received_neighbor_ids.extend(received_data_chunks[1])

            comm.barrier()

        received_neighbor_adts = [list(v) for v in list(zip(*received_neighbor_adts))]

        agent_and_neighbor_adts = [
            convert_to_equal_side_tensor(
                agent_data_tensors[i].tolist() + received_neighbor_adts[i]
            )
            for i in range(self.num_properties)
        ]
        agent_and_neighbor_ids = (
            self._rank2agentids.get(worker, []) + received_neighbor_ids
        )

        return (
            agent_and_neighbor_ids,
            agent_and_neighbor_adts,
        )

    def reduce_agent_data_tensors(
        self, agent_data_tensors, agent_ids_in_subcontext, reduce_func: Callable = None
    ) -> List[cp.ndarray]:
        # Worker knows which neighbors it has data for
        # Send neighbors data along with neighbor ids to responsible rank
        # Recv your agents data based on neighboring ranks
        # Reduce and integrate versions of your agents if there are duplicates received

        num_agents_this_rank = len(self._this_rank_agent2subcontextidx.keys())
        neighbor_ids = agent_ids_in_subcontext[num_agents_this_rank:]

        neighbor_adts = [
            agent_data_tensors[i][num_agents_this_rank:].tolist()
            for i in range(self.num_properties)
        ]

        to_ranks = [self._agent2rank[nid] for nid in neighbor_ids]

        neighborrankandidandadts = list(
            zip(
                to_ranks,
                neighbor_ids,
                list(
                    zip(
                        *neighbor_adts,
                    )
                ),
            ),
        )

        # Find rank of neighbors

        neighborrank2neighborids = OrderedDict()
        send_info = OrderedDict()
        for rank, neighbor_id, adts in neighborrankandidandadts:
            if neighbor_id not in neighborrank2neighborids.get(rank, []):
                neighborrank2neighborids.setdefault(rank, []).append(neighbor_id)
                send_info.setdefault(rank, []).append([adt for adt in adts])

        for rank, adts in send_info.items():
            # send_adts = [list(v) for v in list(zip(*adts))]
            send_info[rank] = dict(zip(neighborrank2neighborids[rank], adts))

        receivedagentcopies2adts = {}
        for from_rank in range(num_workers):
            if from_rank == worker:
                for to_rank in range(num_workers):
                    if to_rank != worker:
                        if to_rank in send_info:
                            data_to_send = send_info[to_rank]
                            chunk_size = 1024  # Define a fixed chunk size
                            num_chunks = len(data_to_send) // chunk_size + (
                                1 if len(data_to_send) % chunk_size > 0 else 0
                            )
                            comm.send(
                                num_chunks,
                                dest=to_rank,
                                tag=(from_rank * num_workers) + to_rank,
                            )
                            for i in range(num_chunks):
                                chunk = {
                                    k: v
                                    for k, v in list(data_to_send.items())[
                                        i * chunk_size : (i + 1) * chunk_size
                                    ]
                                }
                                comm.send(
                                    chunk,
                                    dest=to_rank,
                                    tag=(from_rank * num_workers) + to_rank + i,
                                )
                        else:
                            comm.send(
                                0, dest=to_rank, tag=(from_rank * num_workers) + to_rank
                            )
            else:
                num_chunks = comm.recv(
                    source=from_rank, tag=(from_rank * num_workers) + worker
                )
                if num_chunks > 0:
                    received_data = {}
                    for i in range(num_chunks):
                        chunk = comm.recv(
                            source=from_rank,
                            tag=(from_rank * num_workers) + worker + i,
                        )
                        received_data.update(chunk)
                    for neighbor_id, adts in received_data.items():
                        receivedagentcopies2adts.setdefault(neighbor_id, []).append(
                            adts
                        )

            comm.barrier()

        for agent_id, adts_versions in receivedagentcopies2adts.items():
            if agent_id not in self._this_rank_agent2subcontextidx:
                raise ValueError(
                    f"Fatal: agent_id {agent_id} not found in rank {worker}'s subcontext"
                )
            subcontextidx = self._this_rank_agent2subcontextidx[agent_id]
            for adts_version in adts_versions:
                original_adts = [adt[subcontextidx] for adt in agent_data_tensors]
                reduce_result = reduce_func(original_adts, adts_version)
                for prop_idx, adt in enumerate(agent_data_tensors):
                    adt[subcontextidx] = reduce_result[prop_idx]

        return agent_data_tensors
