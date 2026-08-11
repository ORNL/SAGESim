# SAGESim: Distributed Model Creation

## Overview

SAGESim is a distributed agent-based modeling framework. It manages agents on GPUs across MPI ranks, handles neighbor data exchange, and executes step functions. SAGESim knows nothing about what agents represent — that is the application's job.

## What SAGESim Expects

Each agent has:
- **ID** — a globally unique integer
- **Breed** — defines which properties the agent has, their defaults, and the step function
- **Property values** — per-agent values (each agent of the same breed can have different values). Property names and defaults are defined by the breed; the application sets per-agent values.
- **Connections** — neighbor relationships. Can be directed (one-way) or undirected (bidirectional).

## Two Paths for Creating Agents

### Path 1: One at a Time (`create_agent_of_breed`)

For small-scale or interactive use. Application creates agents in a loop:

```python
model = MyModel()
# Register breeds first, then create agents one by one.
# Keyword arguments are the property names defined by the breed.
agent_id = model.create_agent_of_breed(
    breed=my_breed,
    agent_id=42,          # optional explicit ID (auto-increment if omitted)
    rank=0,               # optional explicit rank (round-robin if omitted)
    my_property=[0.02, 5.0],       # application-defined property
    another_property=[0.0, -65.0], # application-defined property
)
```

Each call appends to internal data structures. Works but slow for 10K+ agents.

### Path 2: Bulk (`build_from_local_data`)

For distributed/partition-based loading. Each MPI rank creates only its own local agents. The application reads a graph (from GraphML, pickle, CSV, etc.), partitions it (e.g., using METIS), and then each rank prepares its local agents, connections, and remote rank info, then hands everything to SAGESim in one call.

#### How it works across MPI ranks

Each rank calls `build_from_local_data()` independently with only its own local data:

```
Rank 0: agents [0, 1, 2],   connections among them + edges to remote agents
Rank 1: agents [3, 4, 5],   connections among them + edges to remote agents
Rank 2: agents [6, 7, 8],   connections among them + edges to remote agents
```

When a local agent connects to an agent on another rank, that remote agent's properties are needed during simulation. The `remote_agent_ranks` dict tells SAGESim which rank owns each remote neighbor so it can set up MPI ghost exchange.

#### Example: 2-rank partition

Suppose the application partitions a 6-node graph into 2 ranks, with an edge between agent 2 (rank 0) and agent 3 (rank 1):

```python
# --- Rank 0 ---
agents = [
    {'id': 0, 'breed': my_breed, 'properties': {'state': [1.0]}},
    {'id': 1, 'breed': my_breed, 'properties': {'state': [2.0]}},
    {'id': 2, 'breed': my_breed, 'properties': {'state': [3.0]}},
]

connections = [
    (0, 1),   # local edge
    (2, 3),   # cross-rank edge: agent 2 (local) connects to agent 3 (on rank 1)
]

# Agent 3 is not local — tell SAGESim which rank owns it
remote_agent_ranks = {3: 1}

model.build_from_local_data(agents, connections, remote_agent_ranks)
```

```python
# --- Rank 1 ---
agents = [
    {'id': 3, 'breed': my_breed, 'properties': {'state': [4.0]}},
    {'id': 4, 'breed': my_breed, 'properties': {'state': [5.0]}},
    {'id': 5, 'breed': my_breed, 'properties': {'state': [6.0]}},
]

connections = [
    (3, 2),   # cross-rank edge: agent 3 (local) connects to agent 2 (on rank 0)
    (4, 5),   # local edge
]

remote_agent_ranks = {2: 0}

model.build_from_local_data(agents, connections, remote_agent_ranks)
```

After `model.setup()`, SAGESim discovers the ghost topology from `remote_agent_ranks`: rank 0 needs agent 3's data from rank 1, and rank 1 needs agent 2's data from rank 0. Every simulation tick, SAGESim exchanges ghost agent data via MPI automatically.

#### The `directed` parameter

The `directed` parameter controls how connections are interpreted:
- **`directed=False`** (default) — each `(a, b)` creates a bidirectional connection. Both agents can see each other. Use for undirected graphs where each edge appears once in the edge list.
- **`directed=True`** — each `(a, b)` creates a one-way connection. Only agent `a` can see agent `b`. Use when the application has already resolved edge directions (e.g., adding explicit reverse edges where needed).

SAGESim pre-allocates all tensors and fills them in bulk. Fast for any scale.

### What `build_from_local_data()` does

When a rank calls `build_from_local_data(agents, connections, remote_agent_ranks)`, SAGESim performs these steps:

1. **Create sparse space** — calls `space.add_local_agents(local_ids)` to create neighbor list containers for local agents only. Uses a dict `{agent_id: set()}` instead of a global array, so agent IDs don't need to be contiguous or start at 0.

2. **Build agent factory mappings** — for each local agent, records:
   - `agent_id → local tensor index` (the position in the property arrays)
   - `agent_id → rank` (this rank owns it)
   - `agent_id → breed index` (which breed this agent belongs to)

3. **Fill property tensors in bulk** — for each property registered by the breed(s), builds a list of values across all local agents. If an agent's `properties` dict doesn't include a property, the breed's default value is used.

4. **Register remote agent ranks** — records `agent_id → rank` for each remote agent in `remote_agent_ranks`. This tells `setup()` where to find these agents for MPI ghost exchange.

5. **Create connections** — for each `(a, b)` in the connections list, calls `space.connect_agents(a, b, directed=directed)`. This populates the neighbor lists that step functions use to access neighbor data.

After `build_from_local_data()`, the application calls `model.setup()` which transfers everything to GPU buffers and sets up MPI communication channels based on the remote agent rank info.

### What SAGESim Does NOT Do

- Interpret what agents mean (soma, synapse, tree, gap, site — that's the application)
- Load or parse graph files (that's the application)
- Decide how to partition the network (that's external tools like METIS)
- Compute property values from domain data (that's the application)

## Runtime (After Agent Creation)

1. `model.setup()` — builds GPU buffers, discovers ghost topology from `remote_agent_ranks`, initializes MPI communication channels
2. `model.simulate(ticks=N)` or `model.step()` — each tick: executes GPU kernels, then exchanges ghost agent data across ranks via MPI
3. Step functions read neighbor data transparently — whether the neighbor is local or on another rank, the access pattern is the same
