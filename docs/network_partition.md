# Network Partitioning in SAGESim

## Overview

SAGESim supports loading pre-computed network partitions to optimize multi-worker performance. By assigning agents that communicate frequently to the same worker, you can significantly reduce MPI communication overhead.

**Key Benefits:**
- Reduces cross-worker edges (ghost agent communication)
- Decreases MPI overhead by minimizing neighbor data exchange
- Enables efficient scaling with multiple workers

## Quick Start

### 1. Generate a Partition

```python
from sagesim.partition_utils import generate_metis_partition, save_partition
import networkx as nx

# Create or load your network graph
G = nx.karate_club_graph()  # Example graph

# Generate partition for 4 workers
partition = generate_metis_partition(G, num_workers=4)

# Save partition
save_partition(partition, "partition.pkl")
```

### 2. Load Partition in Your Model

```python
from sagesim.model import Model
from sagesim.space import NetworkSpace

# Create model
space = NetworkSpace()
model = Model(space)

# Load partition BEFORE creating agents
model.load_partition("partition.pkl")

# Now create agents as usual
# Agents will be assigned to ranks according to the partition
for i in range(num_agents):
    model.create_agent_of_breed(my_breed, ...)

# Setup and run
model.setup(use_gpu=True)
model.simulate(num_ticks)
```

## Partition Utilities

SAGESim provides utility functions in `sagesim/partition_utils.py`:

### `generate_metis_partition(graph, num_workers)`

Generate a network partition using the METIS algorithm.

```python
from sagesim.partition_utils import generate_metis_partition

partition = generate_metis_partition(
    graph,              # NetworkX graph
    num_workers=4,      # Number of MPI workers
    node_id_attr=None,  # Node attribute for agent IDs (optional)
    recursive=True,     # Use recursive bisection (recommended)
)
# Returns: {agent_id: rank, ...}
```

**Requirements:**
- `pip install metis networkx`
- On HPC systems, may need: `module load metis`

### `save_partition(partition, filepath, format)`

Save partition to file.

```python
from sagesim.partition_utils import save_partition

# Auto-detect format from extension
save_partition(partition, "partition.pkl")   # Pickle format
save_partition(partition, "partition.json")  # JSON format
save_partition(partition, "partition.npy")   # NumPy format
save_partition(partition, "partition.txt")   # Text format
```

### `analyze_partition(partition, graph)`

Print partition quality metrics.

```python
from sagesim.partition_utils import analyze_partition

analyze_partition(partition, G)
# Output:
# Partition Analysis:
#   Total agents: 34
#   Number of workers: 4
#   Agents per worker:
#     Rank 0:      9 agents (26.5%)
#     Rank 1:      8 agents (23.5%)
#     Rank 2:      9 agents (26.5%)
#     Rank 3:      8 agents (23.5%)
#   Balance:
#     Min: 8, Max: 9, Avg: 8.5
#     Imbalance: 11.76%
#   Edge cuts: 12/78 (15.4%)
```

## Supported File Formats

### Pickle Format (`.pkl`, `.pickle`)
```python
partition = {
    0: 0,  # Agent 0 -> Rank 0
    1: 0,  # Agent 1 -> Rank 0
    2: 1,  # Agent 2 -> Rank 1
    3: 1,  # Agent 3 -> Rank 1
}
```

### JSON Format (`.json`)
```json
{
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 1
}
```

### NumPy Format (`.npy`)
```python
# Array where index = agent_id, value = rank
partition = np.array([0, 0, 1, 1, 2, 2, ...])
```

### Text Format (`.txt`)
```
# agent_id rank
0 0
1 0
2 1
3 1
```

## Alternative: NetworkX Community Detection

If METIS is not available, you can use NetworkX community detection:

```python
import networkx as nx
from networkx.algorithms import community

G = nx.Graph()
G.add_edges_from(edges)

# Detect communities
communities = community.greedy_modularity_communities(G)

# Assign communities to workers
num_workers = 4
partition = {}
for agent_id in G.nodes():
    for comm_idx, comm in enumerate(communities):
        if agent_id in comm:
            partition[agent_id] = comm_idx % num_workers
            break

# Save and use as shown above
```

## Important Notes

1. **Load Before Creating Agents**: The partition must be loaded BEFORE any agents are created:
   ```python
   # CORRECT
   model.load_partition("partition.pkl")
   model.create_agent_of_breed(...)

   # WRONG - partition won't be applied
   model.create_agent_of_breed(...)
   model.load_partition("partition.pkl")
   ```

2. **Agent ID Mapping**: Agent IDs in the partition file must match the order agents are created (0, 1, 2, ...).

3. **Missing Agents**: If an agent_id is not in the partition, it falls back to round-robin assignment.

4. **Rank Validation**: All ranks in the partition must be valid (0 to num_workers-1).

## Partition Quality Guidelines

**Good partition characteristics:**
- **Load balance**: Each worker has roughly equal agents (imbalance < 10%)
- **Low edge cut ratio**: Most edges are within same worker (< 15% cross-worker)

**Measuring edge cut ratio:**
```python
def measure_edge_cut(graph, partition):
    """Measure partition quality."""
    total_edges = 0
    cross_worker_edges = 0

    for u, v in graph.edges():
        total_edges += 1
        if partition.get(u) != partition.get(v):
            cross_worker_edges += 1

    ratio = cross_worker_edges / total_edges if total_edges > 0 else 0
    print(f"Edge cut ratio: {ratio:.1%}")
    return ratio
```

## Command-Line Tool

SAGESim includes a CLI tool for generating partitions from network files:

```bash
python -m sagesim.generate_partition \
    --input network.graphml \
    --output partition.pkl \
    --method metis \
    --workers 4
```

**Options:**
- `--method`: `metis` (recommended), `community`, `random`, `roundrobin`
- `--format`: `pickle`, `json`, `numpy`, `text`, or `auto` (detect from extension)
- `--workers`: Number of MPI workers

## See Also

- `sagesim/partition_utils.py` - Partition utility functions (for programmatic use)
- `sagesim/generate_partition.py` - CLI tool for generating partitions
- `gpu_cpu_data_flow.md` - How data flows in distributed simulation
- `selective_property_synchronization.md` - Reducing MPI message size
