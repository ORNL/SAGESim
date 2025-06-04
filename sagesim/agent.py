from __future__ import annotations
from typing import Any, Callable, Iterable, List, Dict
from collections import OrderedDict
from copy import copy
import sys

import numpy as np
from mpi4py import MPI

from sagesim.breed import Breed
from sagesim.internal_utils import (
    compress_tensor,
)
from sagesim.space import Space


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

        self._prev_agent_data = {}

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
            if property_name in self._property_name_2_agent_data_tensor:
                # If the property is already registered, just update the default value
                self._property_name_2_defaults[property_name] = default
            else:
                # Register the new property
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
    ) -> List[List[Any]]:
        """converted_agent_data_tensors = []
        for property_name in self._property_name_2_agent_data_tensor.keys():
            converted_agent_data_tensors.append(
                convert_to_equal_side_tensor(
                    self._property_name_2_agent_data_tensor[property_name]
                )
            )

        return converted_agent_data_tensors"""
        return list(self._property_name_2_agent_data_tensor.values())

    def _update_agent_property(
        self,
        regularized_agent_data_tensors,
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
        self, agent_data_tensors, agent_ids_chunk, all_neighbors
    ):
        """
        Chunks agent data tensors so that each distributed worker does not
        get more data than the agents that worker processes actually need.

        :return: 2-tuple.
            1. agent_ids_chunks: List of Lists of agent_ids to be processed
                by each worker.
            3. agent_data_tensors_subcontexts: subcontext of agent_data_tensors
                required by agents of agent_ids_chunks to be processed by a worker
        """

        neighborrank2agentidandadt = {}
        neighborrankandagentidsvisited = set()
        num_agents_this_rank = len(agent_ids_chunk)
        for agent_idx in range(num_agents_this_rank):
            agent_id = agent_ids_chunk[agent_idx]
            agent_adts = [adt[agent_idx] for adt in agent_data_tensors]
            if agent_id not in self._prev_agent_data:
                self._prev_agent_data[agent_id] = agent_adts
            else:
                # If the agent data has not changed, skip sending it
                agent_changed = False
                for prop_idx in range(self.num_properties):
                    if not np.array_equal(
                        agent_adts[prop_idx],
                        self._prev_agent_data[agent_id][prop_idx],
                        equal_nan=True,
                    ):
                        agent_changed = True
                        break
                if agent_changed:
                    # Update the previous agent data
                    self._prev_agent_data[agent_id] = agent_adts
                else:
                    # Skip sending this agent if its data has not changed
                    continue

            for neighbor_id in all_neighbors[agent_idx]:
                if np.isnan(neighbor_id):
                    break
                neighbor_rank = self._agent2rank[int(neighbor_id)]
                if neighbor_rank == worker:
                    # Don't send to self
                    continue
                if (neighbor_rank, agent_id) not in neighborrankandagentidsvisited:
                    # Don't send the same agent to the same rank multiple times
                    neighborrankandagentidsvisited.add((neighbor_rank, agent_id))
                    if neighbor_rank not in neighborrank2agentidandadt.keys():
                        neighborrank2agentidandadt[neighbor_rank] = []
                    neighborrank2agentidandadt[neighbor_rank].append(
                        (agent_id, agent_adts)
                    )

        received_neighbor_adts = []
        received_neighbor_ids = []
        # Send chunk nums
        sends_num_chunks = []
        torank2numchunks = {}
        total_num_chunks = 0
        other_ranks_to = [(worker + i) % num_workers for i in range(1, num_workers)]
        other_ranks_from = [(worker + i) % num_workers for i in range(1, num_workers)]
        # Calculate chunk_size to ensure each chunk is <= 128 bytes
        # Estimate the size of a single value in neighborrank2agentidandadt
        if neighborrank2agentidandadt:
            sample_value = next(iter(neighborrank2agentidandadt.values()))[0]
            estimated_value_size = sys.getsizeof(
                sample_value
            )  # Approximate size in bytes
            chunk_size = max(
                1, 128 // estimated_value_size
            )  # Ensure at least one value per chunk
        else:
            chunk_size = 0  # Default to 1 if no data is present
        for to_rank in other_ranks_to:
            if to_rank in neighborrank2agentidandadt:
                # Send the data for this rank
                data_to_send_to_rank = neighborrank2agentidandadt[to_rank]
                # Break the data into chunks

                num_chunks = len(data_to_send_to_rank) // chunk_size + (
                    1 if len(data_to_send_to_rank) % chunk_size > 0 else 0
                )
                total_num_chunks += num_chunks
                torank2numchunks[to_rank] = num_chunks
                sends_num_chunks.append(
                    comm.isend(
                        num_chunks,
                        dest=to_rank,
                        tag=0,
                    )
                )
            else:
                # No data to send to this rank
                torank2numchunks[to_rank] = 0
                sends_num_chunks.append(
                    comm.isend(
                        0,
                        dest=to_rank,
                        tag=0,
                    )
                )
        # Receive num_chunks from all ranks
        recvs_num_chunks_requests = []
        for from_rank in other_ranks_from:
            recvs_num_chunks_requests.append(comm.irecv(source=from_rank, tag=0))

        MPI.Request.waitall(sends_num_chunks)
        recvs_num_chunks = MPI.Request.waitall(recvs_num_chunks_requests)

        # Send the chunks
        send_chunk_requests = []
        for to_rank in other_ranks_to:
            if to_rank in neighborrank2agentidandadt:
                # Send the data for this rank
                data_to_send_to_rank = neighborrank2agentidandadt[to_rank]
                num_chunks = torank2numchunks[to_rank]
                for i in range(num_chunks):
                    chunk = data_to_send_to_rank[i * chunk_size : (i + 1) * chunk_size]
                    send_chunk_request = comm.isend(
                        chunk,
                        dest=to_rank,
                        tag=i + 1,
                    )
                    if i >= len(send_chunk_requests):
                        send_chunk_requests.append([])
                    send_chunk_requests[i].append(send_chunk_request)
        # Receive the chunks
        recv_chunk_requests = []
        for i, from_rank in enumerate(other_ranks_from):
            num_chunks = recvs_num_chunks[i]
            for j in range(num_chunks):
                received_chunk_request = comm.irecv(source=from_rank, tag=j + 1)
                if j >= len(recv_chunk_requests):
                    recv_chunk_requests.append([])
                recv_chunk_requests[j].append(received_chunk_request)

        received_data = []
        num_send_chunk_requests = len(send_chunk_requests)
        num_recv_chunk_requests = len(recv_chunk_requests)
        for i in range(max(num_send_chunk_requests, num_recv_chunk_requests)):
            if i < num_send_chunk_requests:
                MPI.Request.waitall(send_chunk_requests[i])
            if i < num_recv_chunk_requests:
                received_data_ranks_chunk = MPI.Request.waitall(recv_chunk_requests[i])
                for received_data_rank_chunk in received_data_ranks_chunk:
                    received_data.extend(received_data_rank_chunk)

        # Process received chunks
        received_neighbor_adts = [[] for _ in range(self.num_properties)]
        received_neighbor_ids = []
        for neighbor_idx, (neighbor_id, adts) in enumerate(received_data):
            received_neighbor_ids.append(neighbor_id)
            for prop_idx in range(self.num_properties):
                received_neighbor_adts[prop_idx].append(adts[prop_idx])

        return (
            agent_ids_chunk,
            agent_data_tensors,
            received_neighbor_ids,
            received_neighbor_adts,
        )

    def reduce_agent_data_tensors(
        self,
        agent_and_neighbor_data_tensors,
        agent_and_neighbor_ids_in_subcontext,
        reduce_func: Callable = None,
    ):

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

        # Find rank of neighbors
        neighbors_visited = set()
        neighborrank2neighboridandadt = OrderedDict()
        for neighbor_idx, neighbor_id in enumerate(neighbor_ids):
            if neighbor_id not in neighbors_visited:
                if np.isnan(neighbor_id):
                    continue
                neighbors_visited.add(neighbor_id)
                neighbor_rank = self._agent2rank[neighbor_id]
                neighbor_adt = [
                    neighbor_adt[neighbor_idx] for neighbor_adt in neighbor_adts
                ]
                if neighbor_rank not in neighborrank2neighboridandadt:
                    neighborrank2neighboridandadt[neighbor_rank] = []
                neighborrank2neighboridandadt[neighbor_rank].append(
                    (neighbor_id, neighbor_adt)
                )

        # Estimate the size of a single value in neighborrank2neighboridandadt
        if neighborrank2neighboridandadt:
            sample_value = next(iter(neighborrank2neighboridandadt.values()))[0]
            estimated_value_size = sys.getsizeof(
                sample_value
            )  # Approximate size in bytes
            chunk_size = max(
                1, 1024 // estimated_value_size
            )  # Ensure at least one value per chunk
        else:
            chunk_size = 1
        # Send chunk nums
        sends_num_chunks_requests = []
        torank2numchunks = {}
        other_ranks = [(worker + i) % num_workers for i in range(1, num_workers)]
        for to_rank in other_ranks:
            if to_rank in neighborrank2neighboridandadt:
                # Send the data for this rank
                data_to_send_to_rank = neighborrank2neighboridandadt[to_rank]
                # Break the data into chunks
                num_chunks = len(data_to_send_to_rank) // chunk_size + (
                    1 if len(data_to_send_to_rank) % chunk_size > 0 else 0
                )
                torank2numchunks[to_rank] = num_chunks
                sends_num_chunks_requests.append(
                    comm.isend(
                        num_chunks,
                        dest=to_rank,
                        tag=0,
                    )
                )
        # Receive num_chunks from all ranks
        recvs_num_chunks_requests = []
        for from_rank in other_ranks:
            recvs_num_chunks_requests.append(comm.irecv(source=from_rank, tag=0))
        MPI.Request.waitall(sends_num_chunks_requests)
        recv_chunk_nums = MPI.Request.waitall(recvs_num_chunks_requests)

        # Send the chunks
        send_chunk_requests = []
        for to_rank in other_ranks:
            if to_rank in neighborrank2neighboridandadt:
                # Send the data for this rank
                data_to_send_to_rank = neighborrank2neighboridandadt[to_rank]
                num_chunks = torank2numchunks[to_rank]
                for i in range(num_chunks):
                    chunk = data_to_send_to_rank[i * chunk_size : (i + 1) * chunk_size]
                    send_chunk_request = comm.isend(
                        chunk,
                        dest=to_rank,
                        tag=i + 1,
                    )
                    if i >= len(send_chunk_requests):
                        send_chunk_requests.append([])
                    send_chunk_requests[i].append(send_chunk_request)
        # Receive the chunks
        recv_chunk_requests = []
        for i, from_rank in enumerate(other_ranks):
            num_chunks = recv_chunk_nums[i]
            for j in range(num_chunks):
                received_chunk_request = comm.irecv(source=from_rank, tag=j + 1)
                if j >= len(recv_chunk_requests):
                    recv_chunk_requests.append([])
                recv_chunk_requests[j].append(received_chunk_request)

        received_data = []
        num_send_chunk_requests = len(send_chunk_requests)
        num_recv_chunk_requests = len(recv_chunk_requests)
        for i in range(max(num_send_chunk_requests, num_recv_chunk_requests)):
            if i < num_send_chunk_requests:
                MPI.Request.waitall(send_chunk_requests[i])
            if i < num_recv_chunk_requests:
                received_data_ranks_chunk = MPI.Request.waitall(recv_chunk_requests[i])
                for received_data_rank_chunk in received_data_ranks_chunk:
                    received_data.extend(received_data_rank_chunk)

        for agent_id, modified_adts in received_data:
            agent_idx = self._rank2agentid2agentidx[worker][agent_id]
            original_adts = [adt[agent_idx] for adt in agent_data_tensors]
            reduce_result = reduce_func(original_adts, modified_adts)
            for prop_idx in range(self.num_properties):
                agent_data_tensors[prop_idx][agent_idx] = reduce_result[prop_idx]
