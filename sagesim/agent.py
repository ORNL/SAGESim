from __future__ import annotations
from typing import Any, Callable, Iterable, List, Dict, Optional
from collections import OrderedDict
from copy import copy
import sys
import pickle
import json
from pathlib import Path

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
    def __init__(self, space: Space, verbose_timing: bool = False, verbose_mpi_transfer: bool = False) -> None:
        self._breeds: Dict[str, Breed] = OrderedDict()
        self._space: Space = space
        self._verbose_timing = verbose_timing
        self._verbose_mpi_transfer = verbose_mpi_transfer
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
        # Track which properties need to be sent to neighbors during MPI sync
        # Default: breed and locations are NOT neighbor-visible (never read by neighbors)
        self._property_name_2_neighbor_visible = OrderedDict(
            {"breed": False, "locations": False}
        )
        self._neighbor_visible_indices: List[int] = []  # Cached list of neighbor-visible property indices
        self._agent2rank = {}  # global
        self._rank2agentid2agentidx = {}  # global

        self._current_rank = 0
        self._partition_loaded = False  # Flag to track if partition is loaded
        self._partition_mapping: Optional[Dict[int, int]] = None  # agent_id -> rank mapping

        self._prev_agent_data = {}

        # MPI stats tracking for benchmarking
        self._mpi_stats = {
            'total_bytes_sent': 0,
            'total_bytes_recv': 0,
            'contextualize_total_time': 0.0,
        }

    def clear_agent_data_cache(self):
        """Clear the previous agent data cache to force resending all agent data.

        This is necessary to avoid stale data issues in MPI synchronization where
        the cache check happens before GPU kernel execution in the same tick.
        """
        self._prev_agent_data = {}

    def load_partition_from_dict(self, partition_dict: Dict[int, int]) -> None:
        """Load partition directly from a dictionary (no file I/O).

        This is useful for dynamic networks where partitions are regenerated frequently.

        :param partition_dict: Dictionary mapping agent_id -> rank
        :raises ValueError: If partition is loaded after agents are created
        """
        if self._num_agents > 0:
            raise ValueError(
                "Partition must be loaded BEFORE creating any agents. "
                f"Currently {self._num_agents} agents already exist."
            )

        # Validate partition
        if not partition_dict:
            raise ValueError("Partition dictionary is empty")

        # Check that ranks are valid (0 to num_workers-1)
        invalid_ranks = {r for r in partition_dict.values() if r < 0 or r >= num_workers}
        if invalid_ranks:
            raise ValueError(
                f"Partition contains invalid ranks: {sorted(invalid_ranks)}. "
                f"Valid ranks are 0 to {num_workers-1} (num_workers={num_workers})"
            )

        self._partition_mapping = partition_dict
        self._partition_loaded = True

        if worker == 0:
            num_agents_in_partition = len(partition_dict)
            agents_per_rank = {}
            for agent_id, rank in partition_dict.items():
                agents_per_rank[rank] = agents_per_rank.get(rank, 0) + 1

            if self._verbose_timing:
                print(f"[SAGESim] Loaded partition from dictionary")
                print(f"[SAGESim] Number of agents in partition: {num_agents_in_partition}")
                print(f"[SAGESim] Agents per rank: {dict(sorted(agents_per_rank.items()))}")

    def load_partition(self, partition_file: str, format: str = "auto") -> None:
        """Load network partition from file.

        This method loads a pre-computed network partition that maps agent IDs to MPI ranks.
        When a partition is loaded, agents will be assigned to ranks according to the partition
        instead of using round-robin assignment. This can significantly reduce cross-worker
        communication overhead (see SAGESIM_OVERHEAD_ANALYSIS.md).

        Must be called BEFORE creating any agents.

        Supported formats:
        - 'pickle': Python pickle format - Dict[int, int] mapping agent_id -> rank
        - 'json': JSON format - {"agent_id": rank, ...}
        - 'numpy': NumPy format (.npy) - 1D array where index=agent_id, value=rank
        - 'text': Plain text format - one line per agent: "agent_id rank"
        - 'auto': Automatically detect format from file extension

        Example pickle format:
            {0: 0, 1: 0, 2: 1, 3: 1, ...}  # Agent 0,1 on rank 0; Agent 2,3 on rank 1

        Example text format:
            0 0
            1 0
            2 1
            3 1

        :param partition_file: Path to partition file
        :param format: File format ('pickle', 'json', 'numpy', 'text', or 'auto')
        :raises ValueError: If partition file is loaded after agents are created
        :raises FileNotFoundError: If partition file does not exist
        :raises ValueError: If partition format is invalid
        """
        if self._num_agents > 0:
            raise ValueError(
                "Partition must be loaded BEFORE creating any agents. "
                f"Currently {self._num_agents} agents already exist."
            )

        partition_path = Path(partition_file)
        if not partition_path.exists():
            raise FileNotFoundError(f"Partition file not found: {partition_file}")

        # Auto-detect format from extension
        if format == "auto":
            suffix = partition_path.suffix.lower()
            if suffix == ".pkl" or suffix == ".pickle":
                format = "pickle"
            elif suffix == ".json":
                format = "json"
            elif suffix == ".npy":
                format = "numpy"
            elif suffix == ".txt" or suffix == ".dat":
                format = "text"
            else:
                raise ValueError(
                    f"Cannot auto-detect format for extension '{suffix}'. "
                    "Please specify format explicitly."
                )

        # Load partition based on format
        if format == "pickle":
            with open(partition_path, "rb") as f:
                partition_mapping = pickle.load(f)
            if not isinstance(partition_mapping, dict):
                raise ValueError(
                    f"Pickle partition must be Dict[int, int], got {type(partition_mapping)}"
                )
        elif format == "json":
            with open(partition_path, "r") as f:
                raw_mapping = json.load(f)
            # Convert string keys to int if needed
            partition_mapping = {int(k): int(v) for k, v in raw_mapping.items()}
        elif format == "numpy":
            partition_array = np.load(partition_path)
            if partition_array.ndim != 1:
                raise ValueError(
                    f"NumPy partition must be 1D array, got shape {partition_array.shape}"
                )
            # Convert array to dict: index -> value
            partition_mapping = {i: int(rank) for i, rank in enumerate(partition_array)}
        elif format == "text":
            partition_mapping = {}
            with open(partition_path, "r") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue  # Skip empty lines and comments
                    parts = line.split()
                    if len(parts) != 2:
                        raise ValueError(
                            f"Invalid text partition format at line {line_num}: '{line}'. "
                            "Expected 'agent_id rank'"
                        )
                    try:
                        agent_id = int(parts[0])
                        rank = int(parts[1])
                        partition_mapping[agent_id] = rank
                    except ValueError as e:
                        raise ValueError(
                            f"Invalid integer at line {line_num}: '{line}'. Error: {e}"
                        )
        else:
            raise ValueError(
                f"Unknown partition format: '{format}'. "
                "Supported formats: 'pickle', 'json', 'numpy', 'text', 'auto'"
            )

        # Validate partition
        if not partition_mapping:
            raise ValueError("Partition file is empty or contains no valid mappings")

        # Check that ranks are valid (0 to num_workers-1)
        invalid_ranks = {r for r in partition_mapping.values() if r < 0 or r >= num_workers}
        if invalid_ranks:
            raise ValueError(
                f"Partition contains invalid ranks: {sorted(invalid_ranks)}. "
                f"Valid ranks are 0 to {num_workers-1} (num_workers={num_workers})"
            )

        self._partition_mapping = partition_mapping
        self._partition_loaded = True

        if worker == 0:
            num_agents_in_partition = len(partition_mapping)
            agents_per_rank = {}
            for agent_id, rank in partition_mapping.items():
                agents_per_rank[rank] = agents_per_rank.get(rank, 0) + 1

            if self._verbose_timing:
                print(f"[SAGESim] Loaded network partition from: {partition_file}")
                print(f"[SAGESim] Partition format: {format}")
                print(f"[SAGESim] Number of agents in partition: {num_agents_in_partition}")
                print(f"[SAGESim] Agents per rank: {dict(sorted(agents_per_rank.items()))}")

                # Calculate partition quality metrics
                max_agents = max(agents_per_rank.values()) if agents_per_rank else 0
                min_agents = min(agents_per_rank.values()) if agents_per_rank else 0
                avg_agents = num_agents_in_partition / num_workers if num_workers > 0 else 0
                imbalance = (max_agents - min_agents) / avg_agents if avg_agents > 0 else 0

                print(f"[SAGESim] Load balance - Max: {max_agents}, Min: {min_agents}, "
                      f"Avg: {avg_agents:.1f}, Imbalance: {imbalance:.2%}")

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

    def _build_neighbor_visible_indices(self) -> None:
        """Build cached list of property indices that are neighbor-visible.

        Called lazily on first contextualization or can be called explicitly after setup.
        """
        self._neighbor_visible_indices = [
            idx for name, idx in self._property_name_2_index.items()
            if self._property_name_2_neighbor_visible.get(name, True)
        ]

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
            # Get neighbor_visible flag from breed (default True for backward compatibility)
            neighbor_visible = breed.prop2neighbor_visible.get(property_name, True)

            if property_name in self._property_name_2_agent_data_tensor:
                # If the property is already registered, just update the default value
                self._property_name_2_defaults[property_name] = default
                # Update neighbor_visible (use OR to be conservative - if any breed marks it visible, it's visible)
                self._property_name_2_neighbor_visible[property_name] = (
                    self._property_name_2_neighbor_visible.get(property_name, False) or neighbor_visible
                )
            else:
                # Register the new property
                self._property_name_2_index[property_name] = len(
                    self._property_name_2_agent_data_tensor
                )
                self._property_name_2_agent_data_tensor[property_name] = []
                self._property_name_2_defaults[property_name] = default
                self._property_name_2_neighbor_visible[property_name] = neighbor_visible

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

        # Assign agent to rank: use partition if loaded, otherwise round-robin
        if self._partition_loaded and agent_id in self._partition_mapping:
            # Use pre-loaded partition
            assigned_rank = self._partition_mapping[agent_id]
        else:
            # Fall back to round-robin assignment
            assigned_rank = self._current_rank
            self._current_rank += 1
            if self._current_rank >= num_workers:
                self._current_rank = 0

        self._agent2rank[agent_id] = assigned_rank
        agentid2agentidx_of_current_rank = self._rank2agentid2agentidx.get(
            assigned_rank, OrderedDict()
        )
        agentid2agentidx_of_current_rank[agent_id] = len(
            self._property_name_2_agent_data_tensor["locations"]
        )
        self._rank2agentid2agentidx[assigned_rank] = (
            agentid2agentidx_of_current_rank
        )

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

        Uses selective property synchronization to only send properties marked
        as neighbor_visible=True, reducing MPI communication overhead.

        :return: 2-tuple.
            1. agent_ids_chunks: List of Lists of agent_ids to be processed
                by each worker.
            3. agent_data_tensors_subcontexts: subcontext of agent_data_tensors
                required by agents of agent_ids_chunks to be processed by a worker
        """
        import math
        import time
        t_ctx_start = time.time()

        # Build neighbor-visible indices cache if not already built
        if not self._neighbor_visible_indices and self.num_properties > 0:
            self._build_neighbor_visible_indices()

        if self._verbose_mpi_transfer:
            total_bytes_sent = 0
            total_bytes_recv = 0

        neighborrank2agentidandadt = {}
        neighborrankandagentidsvisited = set()
        num_agents_this_rank = len(agent_ids_chunk)

        # OPTIMIZATION: Convert all_neighbors to lists ONCE to avoid slow numpy scalar iteration
        if isinstance(all_neighbors, list) and len(all_neighbors) > 0 and isinstance(all_neighbors[0], np.ndarray):
            # Convert numpy arrays to lists (much faster iteration)
            all_neighbors_list = [arr.tolist() for arr in all_neighbors]
        else:
            all_neighbors_list = all_neighbors

        for agent_idx in range(num_agents_this_rank):
            agent_id = agent_ids_chunk[agent_idx]
            # OPTIMIZATION: Only collect neighbor-visible properties for sending
            # This reduces MPI message size significantly
            agent_adts_visible = [
                agent_data_tensors[idx][agent_idx]
                for idx in self._neighbor_visible_indices
            ] if self._neighbor_visible_indices else []
            # Keep full adts for local cache comparison
            agent_adts = [adt[agent_idx] for adt in agent_data_tensors]
            agent_neighbors = all_neighbors_list[agent_idx]

            # Check if this agent has neighbors on other workers
            has_cross_worker_neighbors = False
            for neighbor_id in agent_neighbors:
                # Use math.isnan for Python floats (faster than np.isnan)
                if math.isnan(neighbor_id):
                    break
                if int(neighbor_id) < 0:  # Skip invalid/external agent IDs
                    continue
                neighbor_rank = self._agent2rank[int(neighbor_id)]
                if neighbor_rank != worker:
                    has_cross_worker_neighbors = True
                    break

            # Optimization: Skip sending if data hasn't changed AND agent has no cross-worker neighbors
            # Agents with cross-worker neighbors must always be sent to maintain agent_id_to_index mapping
            if not has_cross_worker_neighbors:
                if agent_id not in self._prev_agent_data:
                    self._prev_agent_data[agent_id] = agent_adts
                else:
                    # If the agent data has not changed, skip sending it
                    agent_changed = False
                    for prop_idx in range(self.num_properties):
                        current_property_adt = agent_adts[prop_idx]
                        previous_property_adt = self._prev_agent_data[agent_id][prop_idx]

                        # OPTIMIZATION: Skip expensive comparison for large arrays (property 1 = locations)
                        # Property 1 contains neighbor connections which can be very large (1000+ elements)
                        # np.array_equal on these is too slow (27M element comparisons!)
                        # Instead, always mark as changed for property 1 if it's a numpy array
                        if prop_idx == 1 and isinstance(current_property_adt, np.ndarray):
                            # Always send property 1 if it's a large numpy array
                            agent_changed = True
                            break

                        # Compare based on type to handle ordered (list) vs unordered (set) neighbors
                        properties_equal = False
                        if isinstance(current_property_adt, set):
                            # For sets (unordered neighbors), order doesn't matter
                            properties_equal = current_property_adt == previous_property_adt
                        else:
                            # For lists, tuples, arrays (ordered neighbors), order matters
                            properties_equal = np.array_equal(
                                current_property_adt,
                                previous_property_adt,
                                equal_nan=True,
                            )

                        if not properties_equal:
                            agent_changed = True
                            break
                    if agent_changed:
                        # Update the previous agent data
                        self._prev_agent_data[agent_id] = agent_adts
                    else:
                        # Skip sending this agent if its data has not changed
                        # and it has no neighbors on other workers
                        continue

            # Build list of ranks this agent should be sent to
            for neighbor_id in agent_neighbors:
                if math.isnan(neighbor_id):
                    break
                if int(neighbor_id) < 0:  # Skip invalid/external agent IDs
                    continue
                neighbor_rank = self._agent2rank[int(neighbor_id)]
                if neighbor_rank == worker:
                    # Don't send to self
                    continue

                if (neighbor_rank, agent_id) not in neighborrankandagentidsvisited:
                    # Don't send the same agent to the same rank multiple times
                    neighborrankandagentidsvisited.add((neighbor_rank, agent_id))
                    if neighbor_rank not in neighborrank2agentidandadt.keys():
                        neighborrank2agentidandadt[neighbor_rank] = []
                    # OPTIMIZATION: Only send neighbor-visible properties
                    neighborrank2agentidandadt[neighbor_rank].append(
                        (agent_id, agent_adts_visible)
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

        t_before_wait1 = time.time()
        MPI.Request.waitall(sends_num_chunks)
        recvs_num_chunks = MPI.Request.waitall(recvs_num_chunks_requests)
        t_after_wait1 = time.time()

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
                    if self._verbose_mpi_transfer:
                        total_bytes_sent += len(pickle.dumps(chunk))
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

        t_before_wait2 = time.time()
        for i in range(max(num_send_chunk_requests, num_recv_chunk_requests)):
            if i < num_send_chunk_requests:
                MPI.Request.waitall(send_chunk_requests[i])
            if i < num_recv_chunk_requests:
                received_data_ranks_chunk = MPI.Request.waitall(recv_chunk_requests[i])
                for received_data_rank_chunk in received_data_ranks_chunk:
                    received_data.extend(received_data_rank_chunk)
        t_after_wait2 = time.time()

        # Process received chunks
        # OPTIMIZATION: Reconstruct full property list from neighbor-visible subset
        received_neighbor_adts = [[] for _ in range(self.num_properties)]
        received_neighbor_ids = []

        # Build mapping: prop_idx -> visible_idx (position in adts_visible)
        prop_idx_to_visible_idx = {
            prop_idx: visible_idx
            for visible_idx, prop_idx in enumerate(self._neighbor_visible_indices)
        }

        for neighbor_idx, (neighbor_id, adts_visible) in enumerate(received_data):
            received_neighbor_ids.append(neighbor_id)
            if self._verbose_mpi_transfer:
                total_bytes_recv += len(pickle.dumps((neighbor_id, adts_visible)))

            # Reconstruct full property list from neighbor-visible subset
            for prop_idx in range(self.num_properties):
                if prop_idx in prop_idx_to_visible_idx:
                    # This property was sent - get its value from adts_visible
                    visible_idx = prop_idx_to_visible_idx[prop_idx]
                    received_neighbor_adts[prop_idx].append(adts_visible[visible_idx])
                else:
                    # This property was not sent - use None as placeholder
                    # (it won't be read by neighbors anyway)
                    received_neighbor_adts[prop_idx].append(None)

        t_ctx_end = time.time()

        # Timing verbose output
        if self._verbose_timing:
            print(f"  [Rank {worker}] [contextualize] Total={t_ctx_end - t_ctx_start:.3f}s (wait_counts={t_after_wait1 - t_before_wait1:.3f}s, wait_chunks={t_after_wait2 - t_before_wait2:.3f}s)", flush=True)
            self._mpi_stats['contextualize_total_time'] += (t_ctx_end - t_ctx_start)

        # MPI transfer tracking (accumulate stats silently, summary printed at end)
        if self._verbose_mpi_transfer:
            self._mpi_stats['total_bytes_sent'] += total_bytes_sent
            self._mpi_stats['total_bytes_recv'] += total_bytes_recv
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
