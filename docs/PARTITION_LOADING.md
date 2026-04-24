# SAGESim: Distributed Model Creation

## Overview

SAGESim is a distributed agent-based modeling framework. It manages agents on GPUs across MPI ranks, handles neighbor data exchange, and executes step functions. SAGESim knows nothing about what agents represent — that is the application's job.

## What SAGESim Expects

Each agent has:
- **ID** — a globally unique integer
- **Breed** — defines which properties the agent has, their defaults, and the step function
- **Property values** — per-agent values (each agent of the same breed can have different values)
- **Connections** — directed neighbor relationships (agent A reads agent B's `neighbor_visible` properties)

## Two Paths for Creating Agents

### Path 1: One at a Time (`create_agent_of_breed`)

For small-scale or interactive use. Application creates agents in a loop:

```python
model = MyModel()
# Register breeds first, then create agents one by one
agent_id = model.create_agent_of_breed(
    breed=my_breed,
    agent_id=42,          # optional explicit ID (auto-increment if omitted)
    rank=0,               # optional explicit rank (round-robin if omitted)
    hyperparameters=[0.02, 5.0],
    internal_state=[0.0, -65.0],
)
```

Each call appends to internal data structures. Works but slow for 10K+ agents.

### Path 2: Bulk (`build_from_local_data`)

For distributed/partition-based loading. Application prepares all agents, connections, and remote rank info, then hands everything to SAGESim in one call:

```python
agents = [
    {'id': 0, 'breed': breed_a, 'properties': {'hp': [1.0, 2.0], 'state': [0.0]}},
    {'id': 1, 'breed': breed_a, 'properties': {'hp': [1.5, 2.5], 'state': [0.0]}},
    {'id': 100, 'breed': breed_b, 'properties': {'hp': [3.0], 'state': [0.0, 0.0]}},
]

connections = [
    (100, 0),   # agent 100 reads agent 0's neighbor_visible properties
    (100, 1),   # agent 100 reads agent 1's neighbor_visible properties
    (0, 100),   # agent 0 reads agent 100's neighbor_visible properties
]

remote_agent_ranks = {
    500: 3,     # agent 500 lives on rank 3 (for MPI ghost exchange)
}

model.build_from_local_data(agents, connections, remote_agent_ranks)
```

SAGESim pre-allocates all tensors and fills them in bulk. Fast for any scale.

### What SAGESim Does Internally

1. **Sparse space** — creates neighbor list containers for local agents only (dict-based, not a global array)
2. **Agent factory** — builds index mapping (agent_id -> local tensor index), allocates property tensors
3. **Property fill** — writes each agent's property values into the pre-allocated tensors
4. **Connections** — populates neighbor lists (CSR format on GPU)
5. **Remote ranks** — registers which rank owns each remote agent for MPI ghost exchange

### What SAGESim Does NOT Do

- Interpret what agents mean (soma, synapse, tree, gap, site — that's the application)
- Load or parse partition files (that's the application)
- Decide how to partition the network (that's external tools like METIS)
- Compute property values from domain data (that's the application)

## Runtime (After Agent Creation)

1. `model.setup(use_gpu=True)` — builds GPU buffers, discovers ghost topology, initializes MPI communication
2. `model.simulate(ticks=N)` or `model.step()` — executes GPU kernels, exchanges ghost data via MPI each tick
3. `neighbor_visible` properties are automatically synced across ranks — the application's step functions read neighbor data transparently
