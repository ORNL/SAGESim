# Selective Property Synchronization

This document explains SAGESim's optimization for reducing MPI communication overhead by only synchronizing properties that neighbors actually need to read.

---

## Overview

In distributed agent-based simulations, workers must exchange agent data via MPI so that each worker has access to neighbor agent properties. However, not all properties are read by neighbors - some are purely internal state.

**Selective Property Synchronization** allows users to mark properties as `neighbor_visible=False`, excluding them from MPI transfers and reducing communication bandwidth.

---

## The Problem: Unnecessary Data Transfer

### Default Behavior (Without Optimization)

Every tick, workers exchange ALL properties for neighbor agents:

```
Worker 0 sends to Worker 1:
  Agent 5: [breed, locations, health, internal_counter, debug_flag, ...]
  Agent 8: [breed, locations, health, internal_counter, debug_flag, ...]
  ...
```

But if Worker 1's agents only READ the `health` property from neighbors, sending `internal_counter` and `debug_flag` wastes bandwidth.

### Impact on Large Simulations

For a simulation with:
- 1 million agents
- 10 properties per agent
- 50% of agents have cross-worker neighbors
- 4 bytes per property value

**Without optimization:**
```
Data per tick = 500,000 agents × 10 properties × 4 bytes = 20 MB per worker pair
```

**With optimization (only 3 neighbor-visible properties):**
```
Data per tick = 500,000 agents × 3 properties × 4 bytes = 6 MB per worker pair
```

**Reduction: 70% less MPI traffic**

---

## The Solution: neighbor_visible Flag

### API: Registering Properties

**Location:** `sagesim/breed.py:37-58`

```python
def register_property(
    self,
    name: str,
    default: Union[int, float, List] = nan,
    max_dims: Optional[List[int]] = None,
    neighbor_visible: bool = True,  # NEW PARAMETER
) -> None:
    """
    Register a property for this breed.

    :param name: Property name
    :param default: Default value for the property
    :param max_dims: Optional maximum dimensions for the property
    :param neighbor_visible: If True (default), this property will be sent to
        neighboring workers during MPI synchronization. Set to False for properties
        that are never read by neighbors to reduce communication overhead.
    """
```

### Usage Example

```python
class Neuron(Breed):
    def __init__(self):
        super().__init__("Neuron")

        # Properties that neighbors READ - must be visible
        self.register_property("voltage", default=0.0, neighbor_visible=True)
        self.register_property("fired", default=False, neighbor_visible=True)

        # Properties that are INTERNAL ONLY - neighbors never read these
        self.register_property("refractory_counter", default=0, neighbor_visible=False)
        self.register_property("learning_rate", default=0.01, neighbor_visible=False)
        self.register_property("debug_trace", default=[], neighbor_visible=False)
```

### Built-in Properties

By default, `breed` and `locations` are marked as NOT neighbor-visible:

**Location:** `sagesim/agent.py:48-51`

```python
# Track which properties need to be sent to neighbors during MPI sync
# Default: breed and locations are NOT neighbor-visible (never read by neighbors)
self._property_name_2_neighbor_visible = OrderedDict(
    {"breed": False, "locations": False}
)
```

**Why?**
- `breed`: Neighbors typically don't need to know an agent's breed type (they interact based on properties, not breed identity)
- `locations`: Each agent already knows its own neighbors; it doesn't need to receive the neighbor's neighbor list

---

## Implementation Details

### Step 1: Build Neighbor-Visible Index Cache

**Location:** `sagesim/agent.py:288-302`

At simulation start, the system builds a cached list of which property indices are neighbor-visible:

```python
def _build_neighbor_visible_indices(self) -> None:
    """Build cached list of property indices that are neighbor-visible."""
    self._neighbor_visible_indices = [
        idx for name, idx in self._property_name_2_index.items()
        if self._property_name_2_neighbor_visible.get(name, True)
    ]
```

**Example:**
```
Properties:        [breed, locations, health, counter, fired]
Indices:           [0,     1,         2,      3,       4    ]
neighbor_visible:  [False, False,     True,   False,   True ]

_neighbor_visible_indices = [2, 4]  # Only health and fired
```

### Step 2: Filter Properties During MPI Send

**Location:** `sagesim/agent.py:534-539`

When preparing data to send to other workers, only neighbor-visible properties are included:

```python
# OPTIMIZATION: Only collect neighbor-visible properties for sending
# This reduces MPI message size significantly
agent_adts_visible = [
    agent_data_tensors[idx][agent_idx]
    for idx in self._neighbor_visible_indices
] if self._neighbor_visible_indices else []
```

**Before (all properties):**
```python
# Sending: [breed_val, locations_val, health_val, counter_val, fired_val]
# Size: 5 properties
```

**After (only neighbor-visible):**
```python
# Sending: [health_val, fired_val]
# Size: 2 properties (60% reduction)
```

### Step 3: Send Filtered Data via MPI

**Location:** `sagesim/agent.py:618-621`

Only the visible properties are added to the MPI send queue:

```python
# OPTIMIZATION: Only send neighbor-visible properties
neighborrank2agentidandadt[neighbor_rank].append(
    (agent_id, agent_adts_visible)  # Filtered list, not full list
)
```

### Step 4: Reconstruct Full Property List on Receive

**Location:** `sagesim/agent.py:725-749`

The receiving worker must reconstruct the full property list, inserting `None` placeholders for non-visible properties:

```python
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
```

**Example reconstruction:**
```
Received: (agent_id=5, adts_visible=[0.8, True])  # health, fired

Reconstructed full list:
  prop 0 (breed):     None  # Not sent
  prop 1 (locations): None  # Not sent
  prop 2 (health):    0.8   # From adts_visible[0]
  prop 3 (counter):   None  # Not sent
  prop 4 (fired):     True  # From adts_visible[1]
```

### Step 5: Handle None Placeholders in Data Preparation

**Location:** `sagesim/model.py:627-640`

During GPU data preparation, `None` placeholders are replaced with zero-filled arrays matching the expected structure:

```python
# Handle non-neighbor-visible properties: replace None with placeholder values
# For properties not sent during MPI sync, received_neighbor_adts contains None
if received_data and received_data[0] is None:
    # Use placeholder matching the structure of local data
    local_data = self.__rank_local_agent_data_tensors[i]
    if local_data:
        placeholder = create_zero_placeholder(local_data[0])
    else:
        placeholder = 0.0
    received_data = [placeholder for _ in range(len(received_data))]
```

**Why zero placeholders?**
- GPU arrays must have valid numeric values (not Python `None`)
- The step function won't read these values anyway (property is not neighbor-visible)
- Zero is a safe default that won't cause GPU errors

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SELECTIVE PROPERTY SYNCHRONIZATION                        │
└─────────────────────────────────────────────────────────────────────────────┘

WORKER 0 (Sending)                           WORKER 1 (Receiving)
─────────────────────                        ─────────────────────

Agent 5 properties:
┌─────────────────────────────┐
│ breed:     0                │ ─┐
│ locations: [10, 15]         │  │ NOT sent
│ health:    0.8              │ ─┼─► Filter ─────┐
│ counter:   42               │  │               │
│ fired:     True             │ ─┘               │
└─────────────────────────────┘                  │
                                                 ▼
                              ┌─────────────────────────────────┐
                              │ MPI Send:                       │
                              │   (agent_id=5, [0.8, True])     │
                              │                                 │
                              │ Only 2 values instead of 5!     │
                              └─────────────────────────────────┘
                                                 │
                                                 │ Network
                                                 ▼
                              ┌─────────────────────────────────┐
                              │ MPI Receive:                    │
                              │   (agent_id=5, [0.8, True])     │
                              └─────────────────────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────┐
                              │ Reconstruct full property list: │
                              │   breed:     None → 0 (placeholder)
                              │   locations: None → [] (placeholder)
                              │   health:    0.8  (from received)
                              │   counter:   None → 0 (placeholder)
                              │   fired:     True (from received)
                              └─────────────────────────────────┘
                                                 │
                                                 ▼
                              ┌─────────────────────────────────┐
                              │ GPU Kernel:                     │
                              │   - Reads health[neighbor] ✓    │
                              │   - Reads fired[neighbor] ✓     │
                              │   - Never reads counter/breed   │
                              └─────────────────────────────────┘
```

---

## Multiple Breeds: Visibility Merging

When multiple breeds share a property name, the `neighbor_visible` flags are merged using OR logic:

**Location:** `sagesim/agent.py:325-328`

```python
# Update neighbor_visible (use OR to be conservative - if any breed marks it visible, it's visible)
self._property_name_2_neighbor_visible[property_name] = (
    self._property_name_2_neighbor_visible.get(property_name, False) or neighbor_visible
)
```

**Example:**
```python
# Breed A marks 'energy' as NOT visible
breed_a.register_property("energy", default=100, neighbor_visible=False)

# Breed B marks 'energy' as visible (neighbors need to read it)
breed_b.register_property("energy", default=50, neighbor_visible=True)

# Result: 'energy' IS neighbor-visible (True OR False = True)
# Conservative approach: if ANY breed needs it visible, it's visible for all
```

**Why OR logic?**
- Safety: Better to send unnecessary data than to miss required data
- Correctness: If one breed's step function reads a neighbor's property, that property must be sent regardless of the neighbor's breed

---

## Best Practices

### 1. Analyze Your Step Functions

Before marking properties as `neighbor_visible=False`, verify they are never read from neighbors:

```python
def step(model, agent, health, energy, internal_state, neighbors):
    # This reads 'health' from neighbors - MUST be visible
    for n in neighbors:
        if health[n] < 50:
            ...

    # This only reads own 'internal_state' - can be NOT visible
    if internal_state[agent] > 100:
        ...
```

### 2. Mark Internal Counters as Not Visible

```python
# Good candidates for neighbor_visible=False:
breed.register_property("tick_counter", default=0, neighbor_visible=False)
breed.register_property("random_seed", default=0, neighbor_visible=False)
breed.register_property("debug_info", default=[], neighbor_visible=False)
breed.register_property("accumulated_reward", default=0.0, neighbor_visible=False)
```

### 3. Keep Interaction Properties Visible

```python
# Must remain neighbor_visible=True:
breed.register_property("health", default=100, neighbor_visible=True)
breed.register_property("infected", default=False, neighbor_visible=True)
breed.register_property("signal_strength", default=0.0, neighbor_visible=True)
```

### 4. Use Verbose Mode to Verify

Enable MPI transfer logging to see the bandwidth reduction:

```python
model = Model()
model.set_verbose_mpi_transfer(True)
model.simulate(100)

# Output shows bytes sent/received - compare with/without optimization
```

---

## Performance Impact

### Bandwidth Reduction

| Scenario | Properties | Visible | Reduction |
|----------|------------|---------|-----------|
| SIR Model | 5 | 2 | 60% |
| Neural Network | 10 | 3 | 70% |
| Complex ABM | 20 | 5 | 75% |

### When to Use

| Situation | Recommendation |
|-----------|----------------|
| Few properties (< 5) | Minor benefit, optional |
| Many properties (> 10) | Significant benefit, recommended |
| Large neighbor lists | High benefit (multiplied by neighbor count) |
| High-latency network | Critical for performance |

---

## Limitations

1. **Cannot change at runtime**: `neighbor_visible` is set at property registration and cannot be changed during simulation.

2. **Conservative merging**: If any breed needs a property visible, it's visible for all breeds sharing that property name.

3. **No automatic detection**: The system cannot automatically determine which properties are read by neighbors; users must specify manually.

4. **Placeholder overhead**: Non-visible properties still occupy space in GPU arrays (filled with zeros), so memory usage is not reduced - only network bandwidth.

---

## Code References

| Component | File | Lines |
|-----------|------|-------|
| Property registration API | `breed.py` | 37-58 |
| Default visibility (breed, locations) | `agent.py` | 48-51 |
| Build visible indices cache | `agent.py` | 288-302 |
| Filter during MPI send | `agent.py` | 534-539 |
| Reconstruct on receive | `agent.py` | 725-749 |
| Handle None placeholders | `model.py` | 627-640 |
| Visibility merging (OR logic) | `agent.py` | 325-328 |
