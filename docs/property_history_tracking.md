# Property History Tracking

This guide explains how to track and save the history of agent properties during simulation.

## Overview

SAGESim uses a **circular buffer** mechanism to track property history. When you register a buffer property, it stores the last N states, where N is the buffer size you specify.

## Setting Up History Tracking

### Step 1: Register a Buffer Property

In your Breed class, register a list property to serve as the history buffer:

```python
class MyBreed(Breed):
    def __init__(self) -> None:
        super().__init__("MyBreed")

        # Register the property you want to track
        self.register_property("state", 1)  # Current state

        # Register a buffer to store history (size = number of ticks to remember)
        self.register_property("state_history_buffer", [0.0] * 10)  # Stores last 10 states
```

### Step 2: Write to Buffer in Step Function

In your step function, use modulo indexing to write to the circular buffer:

```python
@jit.rawkernel(device="cuda")
def step_func(
    tick,
    agent_index,
    globals,
    agent_ids,
    breeds,
    locations,
    state_tensor,
    state_history_buffer,  # Buffer tensor passed automatically
):
    # Your simulation logic here...

    # Save current state to history buffer using circular indexing
    buffer_idx = tick % len(state_history_buffer[agent_index])
    state_history_buffer[agent_index][buffer_idx] = state_tensor[agent_index]
```

## How Circular Buffer Works

The buffer uses modulo arithmetic (`tick % buffer_size`) to determine where to write:

| Buffer Size | Ticks Stored | Behavior |
|-------------|--------------|----------|
| 1 | Only final state | Overwrites every tick |
| 5 | Last 5 ticks | Ticks 0-4, then overwrites starting at index 0 |
| 10 | Last 10 ticks | Full history if simulation runs 10 ticks |
| N | Last N ticks | Circular overwrite when tick >= N |

### Example: Buffer Size 5 with 10 Ticks

```
Tick 0: buffer[0] = state  → [S, _, _, _, _]
Tick 1: buffer[1] = state  → [S, S, _, _, _]
Tick 2: buffer[2] = state  → [S, S, S, _, _]
Tick 3: buffer[3] = state  → [S, S, S, S, _]
Tick 4: buffer[4] = state  → [S, S, S, S, S]
Tick 5: buffer[0] = state  → [S*, S, S, S, S]  ← Overwrites tick 0
Tick 6: buffer[1] = state  → [S*, S*, S, S, S] ← Overwrites tick 1
...
```

After 10 ticks, the buffer contains states from ticks 5-9 (the last 5 ticks).

## Retrieving History After Simulation

### Single Worker

```python
# Run simulation
model.simulate(num_ticks)

# Get history for a specific agent
history = model.get_agent_property_value(agent_id, "state_history_buffer")
```

### Multiple Workers (MPI)

When running with MPI, each worker only has access to agents it owns. Use `comm.gather()` to collect all histories:

```python
from mpi4py import MPI

comm = MPI.COMM_WORLD
worker = comm.Get_rank()

# Each worker collects its own agents' histories
local_histories = {}
for agent_id in range(num_agents):
    history = model.get_agent_property_value(agent_id, "state_history_buffer")
    if history is not None:
        local_histories[agent_id] = history

# Gather all histories to worker 0
all_histories_list = comm.gather(local_histories, root=0)

# Merge on worker 0
if worker == 0:
    all_histories = {}
    for hist in all_histories_list:
        all_histories.update(hist)

    # Now all_histories contains data from all agents
    # Save to CSV, analyze, etc.
```

## Choosing Buffer Size

| Use Case | Recommended Buffer Size |
|----------|------------------------|
| Only need final state | 1 |
| Memory constrained | Match analysis window size |
| Full history needed | >= num_ticks |
| Debugging/visualization | >= num_ticks |

## Complete Example

See `tests/test_worker_sync.py` for a complete working example that:
- Tracks state history for an SIR model
- Uses a 10-tick buffer
- Saves results to CSV with MPI support
