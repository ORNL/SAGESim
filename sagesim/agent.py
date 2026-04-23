from __future__ import annotations
from typing import Any, Callable, Iterable, List, Dict, Optional
from collections import OrderedDict
from copy import copy
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
    def __init__(self, space: Space, verbose_timing: bool = False) -> None:
        self._breeds: Dict[str, Breed] = OrderedDict()
        self._space: Space = space
        self._verbose_timing = verbose_timing
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
        # Default: breed is neighbor-visible (safe default); locations uses CSR (not exchanged)
        self._property_name_2_neighbor_visible = OrderedDict(
            {"breed": True, "locations": False}
        )
        self._neighbor_visible_indices: List[int] = []  # Cached list of neighbor-visible property indices
        self._agent2rank = {}  # global
        self._agent2breed = {}  # global: agent_id -> breed_id (populated by all ranks)
        self._rank2agentid2agentidx = {}  # global

        self._current_rank = 0
        self._partition_loaded = False  # Flag to track if partition is loaded
        self._partition_mapping: Optional[Dict[int, int]] = None  # agent_id -> rank mapping
        self._partition_preallocated = False  # True after _init_partition pre-allocates tensors

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

    def create_agent(self, breed: Breed, rank: int = None, agent_id: int = None, **kwargs) -> int:
        """
        Creates and agent of the given breed initialized with the properties given in
            **kwargs.

        :param breed: Breed definition of agent
        :param rank: Target rank for this agent. If provided, overrides partition and
            round-robin. If None, uses partition mapping or round-robin fallback.
        :param agent_id: Explicit global agent ID. If provided, uses this ID instead of
            auto-incrementing. Useful for partition-based loading where IDs are pre-assigned.
            If None (default), auto-assigns the next sequential ID.
        :param **kwargs: named arguments of agent properties. Names much match properties
            already registered in breed.
        :return: Agent ID

        """

        # Fast path: partition mode with pre-allocated tensors
        if agent_id is not None and self._partition_preallocated:
            local_idx = self._rank2agentid2agentidx.get(worker, {}).get(agent_id)
            if local_idx is not None:
                self._agent2breed[agent_id] = self._breeds[breed.name]._breedidx
                for prop_name in self._property_name_2_agent_data_tensor:
                    if prop_name == "breed":
                        self._property_name_2_agent_data_tensor[prop_name][local_idx] = (
                            self._breeds[breed.name]._breedidx
                        )
                    else:
                        default_value = copy(self._property_name_2_defaults[prop_name])
                        self._property_name_2_agent_data_tensor[prop_name][local_idx] = (
                            kwargs.get(prop_name, default_value)
                        )
            return agent_id

        # Standard path
        if agent_id is None:
            agent_id = self._num_agents
            self._num_agents += 1
        else:
            self._num_agents = max(self._num_agents, agent_id + 1)

        # Assign agent to rank: explicit > partition > round-robin
        if rank is not None:
            assigned_rank = rank
        elif self._partition_loaded and agent_id in self._partition_mapping:
            assigned_rank = self._partition_mapping[agent_id]
        else:
            assigned_rank = self._current_rank
            self._current_rank += 1
            if self._current_rank >= num_workers:
                self._current_rank = 0

        self._agent2rank[agent_id] = assigned_rank
        self._agent2breed[agent_id] = self._breeds[breed.name]._breedidx
        agentid2agentidx_of_current_rank = self._rank2agentid2agentidx.get(
            assigned_rank, OrderedDict()
        )
        agentid2agentidx_of_current_rank[agent_id] = len(
            self._property_name_2_agent_data_tensor["locations"]
        )
        self._rank2agentid2agentidx[assigned_rank] = (
            agentid2agentidx_of_current_rank
        )

        if worker == assigned_rank:
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

    def sort_by_breed(self):
        """Sort local agents by breed_id for GPU kernel range optimization.

        Reorders _property_name_2_agent_data_tensor and updates
        _rank2agentid2agentidx to reflect sorted positions.
        """
        old_mapping = self._rank2agentid2agentidx.get(worker)
        if not old_mapping:
            return

        agent_ids = list(old_mapping.keys())
        n = len(agent_ids)
        if n == 0:
            return

        # Get breed data (property index 0)
        breed_data = self._property_name_2_agent_data_tensor["breed"]
        old_indices = [old_mapping[aid] for aid in agent_ids]
        breeds = [breed_data[idx] for idx in old_indices]

        # Stable sort by breed
        sorted_order = sorted(range(n), key=lambda i: breeds[i])
        sort_perm = [old_indices[sorted_order[i]] for i in range(n)]

        # Reorder all property data lists
        for prop_name in self._property_name_2_agent_data_tensor:
            old_list = self._property_name_2_agent_data_tensor[prop_name]
            self._property_name_2_agent_data_tensor[prop_name] = [
                old_list[sort_perm[i]] for i in range(n)
            ]

        # Rebuild _rank2agentid2agentidx with sorted positions
        new_mapping = OrderedDict()
        for new_idx, old_pos in enumerate(sorted_order):
            new_mapping[agent_ids[old_pos]] = new_idx
        self._rank2agentid2agentidx[worker] = new_mapping

    def register_remote_agents(self, remote_agent_ranks: dict) -> None:
        """Register rank info for remote agents referenced by local agents.

        Only records agent_id → rank mapping. No property data is allocated.
        Used for partition-based loading where each rank only creates local
        agents but needs to know the rank of remote neighbors for MPI exchange.

        :param remote_agent_ranks: Dict mapping remote agent_id → rank
        """
        for agent_id, rank in remote_agent_ranks.items():
            self._agent2rank[agent_id] = rank

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

