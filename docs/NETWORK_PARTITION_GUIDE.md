# Network Partition Loading in SAGESim

## Overview

SAGESim now supports loading pre-computed network partitions to optimize multi-worker performance. This feature addresses the MPI communication overhead identified in the [SAGESIM_OVERHEAD_ANALYSIS.md](SAGESIM_OVERHEAD_ANALYSIS.md).

**Key Benefits:**
- Reduces ghost agent communication by 10-20× (from P_cross ≈ 0.75 to P_cross ≈ 0.05)
- Decreases MPI overhead from ~465ms to ~30ms per contextualization (with good partitioning)
- Enables efficient weak scaling with 90%+ efficiency up to 64+ workers
- Improves strong scaling from <30% to 75-95% efficiency

## Quick Start

### 1. Load a Partition File

```python
from sagesim import Model, Space

# Create model
space = Space()
model = Model(space)

# Load partition BEFORE creating agents
model.load_partition("network_partition.pkl")

# Now create agents as usual
# Agents will be assigned to ranks according to the partition
for i in range(num_agents):
    model.create_agent_of_breed(my_breed, ...)
```

### 2. Supported File Formats

SAGESim supports multiple partition file formats:

#### Pickle Format (`.pkl`, `.pickle`)
```python
import pickle

partition = {
    0: 0,  # Agent 0 -> Rank 0
    1: 0,  # Agent 1 -> Rank 0
    2: 1,  # Agent 2 -> Rank 1
    3: 1,  # Agent 3 -> Rank 1
    # ...
}

with open("partition.pkl", "wb") as f:
    pickle.dump(partition, f)
```

#### JSON Format (`.json`)
```json
{
    "0": 0,
    "1": 0,
    "2": 1,
    "3": 1
}
```

#### NumPy Format (`.npy`)
```python
import numpy as np

# Array where index = agent_id, value = rank
partition = np.array([0, 0, 1, 1, 2, 2, ...])
np.save("partition.npy", partition)
```

#### Text Format (`.txt`, `.dat`)
```
# agent_id rank
0 0
1 0
2 1
3 1
```

## Generating Partitions

### Using METIS (Recommended)

For graph-based partitioning with minimal edge cuts:

```python
import networkx as nx
import metis

# Create your network graph
G = nx.DiGraph()
G.add_edges_from([(0, 1), (1, 2), ...])

# Convert to METIS format (undirected adjacency list)
adj_list = [list(G.neighbors(n)) for n in G.nodes()]

# Partition with METIS
num_workers = 4
_, partition = metis.part_graph(adj_list, nparts=num_workers, recursive=True)

# Save partition
import pickle
partition_dict = {agent_id: rank for agent_id, rank in enumerate(partition)}
with open("metis_partition.pkl", "wb") as f:
    pickle.dump(partition_dict, f)
```

### Using NetworkX Community Detection

```python
import networkx as nx
from networkx.algorithms import community

# Create graph
G = nx.Graph()  # Use undirected for community detection
G.add_edges_from(edges)

# Detect communities
communities = community.greedy_modularity_communities(G)

# Assign communities to workers
num_workers = 4
partition = {}
for agent_id in G.nodes():
    # Find which community this agent belongs to
    for comm_idx, comm in enumerate(communities):
        if agent_id in comm:
            partition[agent_id] = comm_idx % num_workers
            break

# Save partition
import pickle
with open("community_partition.pkl", "wb") as f:
    pickle.dump(partition, f)
```

### Manual/Custom Partitioning

```python
def create_partition_by_properties(nodes_data, num_workers):
    """Custom partition based on node properties."""
    partition = {}

    # Example: Partition based on node type
    for agent_id, data in nodes_data.items():
        if data['type'] == 'input':
            rank = 0  # All input nodes on rank 0
        elif data['type'] == 'hidden':
            rank = (agent_id % (num_workers - 1)) + 1  # Distribute hidden
        else:
            rank = num_workers - 1  # Output nodes on last rank

        partition[agent_id] = rank

    return partition
```

## Complete Example with SuperNeuroABM

```python
import networkx as nx
from superneuroabm.io.nx import model_from_nx_graph
import pickle

# 1. Load your network
graph = nx.read_graphml("network.graphml")

# 2. Generate partition using METIS
import metis
adj_list = [list(graph.neighbors(n)) for n in graph.nodes()]
num_workers = 4
_, metis_partition = metis.part_graph(adj_list, nparts=num_workers)

# Convert to dict with actual node IDs
node_ids = list(graph.nodes())
partition_dict = {node_ids[i]: rank for i, rank in enumerate(metis_partition)}

# 3. Save partition
with open("network_partition.pkl", "wb") as f:
    pickle.dump(partition_dict, f)

# 4. Load partition in SAGESim
model = model_from_nx_graph(graph)

# IMPORTANT: Load partition BEFORE model.setup()
model.load_partition("network_partition.pkl")

# 5. Setup and run
model.setup(use_gpu=True)
model.simulate(ticks=100)
```

## Partition Quality Metrics

When loading a partition, SAGESim prints quality metrics:

```
[SAGESim] Loaded network partition from: partition.pkl
[SAGESim] Partition format: pickle
[SAGESim] Number of agents in partition: 10000
[SAGESim] Agents per rank: {0: 2500, 1: 2500, 2: 2500, 3: 2500}
[SAGESim] Load balance - Max: 2500, Min: 2500, Avg: 2500.0, Imbalance: 0.00%
```

**Good partition characteristics:**
- **Imbalance < 10%**: Balanced load across workers
- **Edge cut ratio < 0.10**: Most neighbors on same rank (measure P_cross)

To measure edge cut ratio:
```python
def measure_edge_cut(graph, partition):
    """Measure partition quality."""
    total_edges = 0
    cross_worker_edges = 0

    for u, v in graph.edges():
        total_edges += 1
        if partition[u] != partition[v]:
            cross_worker_edges += 1

    edge_cut_ratio = cross_worker_edges / total_edges if total_edges > 0 else 0
    print(f"Edge cut ratio (P_cross): {edge_cut_ratio:.3f}")
    return edge_cut_ratio
```

## Important Notes

1. **Load Before Creating Agents**: The partition must be loaded BEFORE any agents are created:
   ```python
   # CORRECT
   model.load_partition("partition.pkl")
   model.create_agent_of_breed(...)

   # WRONG - Will raise ValueError
   model.create_agent_of_breed(...)
   model.load_partition("partition.pkl")  # Error!
   ```

2. **Agent ID Mapping**: Agent IDs in the partition file must match the order agents are created:
   - Agent created 1st → agent_id = 0
   - Agent created 2nd → agent_id = 1
   - etc.

3. **Missing Agents**: If an agent_id is not in the partition, it falls back to round-robin assignment.

4. **Rank Validation**: All ranks in the partition must be valid (0 to num_workers-1).

## Expected Performance Improvements

Based on the overhead analysis (see `SAGESIM_OVERHEAD_ANALYSIS.md`):

### MPI Overhead Reduction
- **Bad partition** (P_cross = 0.75): ~465 ms/contextualization
- **Good partition** (P_cross = 0.05): ~30 ms/contextualization
- **Speedup**: **15× faster** MPI communication

### Weak Scaling
- Without partition: <50% efficiency (unusable)
- With partition: **90-95% efficiency** up to 64+ workers

### Strong Scaling
- Without partition: <30% efficiency at W=4
- With partition: **75-95% efficiency** up to W=16-64

## Troubleshooting

### ValueError: Partition must be loaded BEFORE creating any agents
**Solution**: Call `model.load_partition()` immediately after creating the model, before any agents.

### ValueError: Partition contains invalid ranks
**Solution**: Ensure all ranks in partition file are between 0 and num_workers-1.

### FileNotFoundError: Partition file not found
**Solution**: Check the file path. Use absolute paths if needed.

### Agent count mismatch
**Solution**: Ensure the partition file contains entries for all agents you plan to create, in order.

## References

- [SAGESIM_OVERHEAD_ANALYSIS.md](SAGESIM_OVERHEAD_ANALYSIS.md) - Detailed overhead analysis
- [METIS Documentation](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
- NetworkX Community Detection: https://networkx.org/documentation/stable/reference/algorithms/community.html

## Future Enhancements

The following features are planned for future releases:

1. **Native METIS Integration**: Built-in partition generation directly in SAGESim
2. **Dynamic Repartitioning**: Adapt partitions during simulation
3. **Partition Caching**: Automatic partition file generation and caching
4. **GPU-Aware MPI**: Direct GPU-to-GPU transfers to further reduce overhead
