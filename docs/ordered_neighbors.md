# Ordered Neighbors in NetworkSpace

## Overview

By default, `NetworkSpace` stores neighbors as **unordered sets**, which means neighbor order is arbitrary and may vary between runs. For most agent-based models, this is fine since agents typically interact with all their neighbors equally.

However, some applications require **deterministic neighbor ordering**, where the position of a neighbor in the list has semantic meaning.

## When to Use Ordered Neighbors

Use `ordered=True` when:

✅ **Neighbor position matters**
- Reading from first neighbor only
- Directional relationships (e.g., parent-child, predecessor-successor)
- Priority-based interactions (process neighbors in specific order)

✅ **Reproducibility is critical**
- Need identical results across different runs
- Debugging neighbor-dependent behavior

❌ **Don't use ordered neighbors when:**
- All neighbors are equivalent
- Only need to check "is neighbor infected?" (position doesn't matter)
- Performance is critical (sets are slightly faster for membership testing)

## Usage

### Basic Setup

```python
from sagesim.space import NetworkSpace
from sagesim.model import Model

# Ordered neighbors (insertion order preserved)
space_ordered = NetworkSpace(ordered=True)

# Unordered neighbors (default, arbitrary order)
space_unordered = NetworkSpace(ordered=False)  # or just NetworkSpace()

class MyModel(Model):
    def __init__(self):
        space = NetworkSpace(ordered=True)  # Enable ordered neighbors
        super().__init__(space)
```

### Connecting Agents

```python
# Connection order matters with ordered=True
space.connect_agents(agent_1, agent_0)  # Agent 0 becomes first neighbor
space.connect_agents(agent_1, agent_2)  # Agent 2 becomes second neighbor
# agent_1's neighbors: [0, 2] (in that order)

# Duplicates are automatically prevented
space.connect_agents(agent_1, agent_0)  # Ignored - already connected
# agent_1's neighbors: still [0, 2] (no duplicates)
```

### Accessing Neighbors in Step Functions

```python
from cupyx import jit

@jit.rawkernel(device="cuda")
def step_func(tick, agent_index, globals, agent_ids, breeds, locations, ...):
    neighbor_indices = locations[agent_index]

    # With ordered=True, you can rely on position:
    if len(neighbor_indices) > 0 and neighbor_indices[0] != -1:
        first_neighbor = neighbor_indices[0]  # Always the first connected neighbor
        # Do something with first neighbor specifically

    # Access second neighbor
    if len(neighbor_indices) > 1 and neighbor_indices[1] != -1:
        second_neighbor = neighbor_indices[1]
        # Do something with second neighbor
```

## Implementation Details

### Data Structures

| Mode | Storage | Duplicates | Order | Performance |
|------|---------|------------|-------|-------------|
| `ordered=False` | `set` | Prevented | Arbitrary | Faster membership tests |
| `ordered=True` | `list` | Prevented | Insertion order | Sequential access |

### Behavior Differences

```python
# Unordered mode (default)
space = NetworkSpace(ordered=False)
space.connect_agents(0, 1)
space.connect_agents(0, 2)
neighbors = space.get_neighbors(0)
# Result: {1, 2} - order may vary (could be {2, 1})

# Ordered mode
space = NetworkSpace(ordered=True)
space.connect_agents(0, 1)
space.connect_agents(0, 2)
neighbors = space.get_neighbors(0)
# Result: [1, 2] - always in insertion order
```

## Example Use Cases

### 1. Directional Flow (e.g., Information Cascade)

```python
# Agent reads value from upstream neighbor (first in list)
@jit.rawkernel(device="cuda")
def read_from_upstream(tick, agent_index, globals, agent_ids, breeds, locations, values):
    neighbors = locations[agent_index]
    if len(neighbors) > 0 and neighbors[0] != -1:
        upstream_neighbor = neighbors[0]  # First neighbor is upstream source
        values[agent_index] = values[upstream_neighbor]
```

### 2. Priority-Based Interactions

```python
# Agent preferentially interacts with first neighbor, falls back to second
@jit.rawkernel(device="cuda")
def priority_interaction(tick, agent_index, globals, agent_ids, breeds, locations, resources):
    neighbors = locations[agent_index]

    # Try first neighbor (highest priority)
    if len(neighbors) > 0 and neighbors[0] != -1:
        if resources[neighbors[0]] > 0:
            # Interact with first neighbor
            return

    # Fall back to second neighbor
    if len(neighbors) > 1 and neighbors[1] != -1:
        if resources[neighbors[1]] > 0:
            # Interact with second neighbor
            return
```

### 3. Parent-Child Relationships

```python
# First neighbor is parent, rest are children
model = MyModel()
parent = model.create_agent()
child_1 = model.create_agent()
child_2 = model.create_agent()

# Connect parent first, then children
model.connect_agents(agent=child_1, neighbor=parent)  # Parent first
model.connect_agents(agent=child_1, neighbor=child_2) # Sibling second
# child_1's neighbors: [parent, child_2]
```

## Migration Guide

### Updating Existing Code

If your code **doesn't care about neighbor order**, no changes needed:
```python
# This works the same way with both ordered modes
space = NetworkSpace()  # Default: ordered=False
```

If your code **relies on specific neighbor positions**, update to:
```python
# Explicitly request ordered neighbors
space = NetworkSpace(ordered=True)
```

### Performance Considerations

- **Ordered mode**: Slightly slower for large neighbor lists (O(n) duplicate checking)
- **Unordered mode**: Faster membership tests (O(1) average)
- In practice: **Negligible difference** for typical ABM neighbor counts (<100 neighbors/agent)

## See Also

- Example: `examples/ordered_neighbors_example.py`
- Source: `sagesim/space.py` - `NetworkSpace` class
