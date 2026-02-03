# Getting Started with SAGESim

**SAGESim** (Scalable Agent-Based GPU-Enabled Simulator) is a scalable, pure-Python, general-purpose agent-based modeling framework that supports both distributed computing and GPU acceleration.

This tutorial walks through how to build and run agent-based simulations using SAGESim. The core idea centers on subclassing the `Model` class to define your custom model.

> For architecture details, see [Architecture Overview](architecture_overview.md). For synchronization details, see [Synchronization and Double Buffering](synchronization_and_double_buffering.md).

---

## Defining a Custom Model Class

Building a custom model class that subclasses the base `Model` class is the core part of using SAGESim. This enables access to the built-in `simulate()` method to execute your simulations.

The model class is responsible for:
- **Registering Breeds**: Register breeds in the model's `__init__()` method using `register_breed()`.
- **Registering Global Properties**: Register shared properties using `register_global_property()`.
- **Creating and Connecting Agents**: Use `create_agent_of_breed()` and `connect_agents()`.

### Example Model

```python
from sagesim.model import Model
from sagesim.space import NetworkSpace

class MyModel(Model):

    def __init__(self, p_infection=0.2) -> None:
        space = NetworkSpace()
        super().__init__(space)

        # Register breeds
        self._my_breed = MyBreed()
        self.register_breed(breed=self._my_breed)

        # Register global properties
        self.register_global_property("p_infection", p_infection)

    def create_agent(self, state):
        agent_id = self.create_agent_of_breed(self._my_breed, state=state)
        return agent_id

    def connect_agents(self, agent_0, agent_1):
        self.get_space().connect_agents(agent_0, agent_1)
```

---

## Defining a Breed Class

Every agent in SAGESim belongs to a specific *breed*. To define a breed, subclass the `Breed` class:

- **Register properties** using `self.register_property(name, default_value)`.
- **Register step functions** using `self.register_step_func(func, file_path, priority)`.

### Example Breed

```python
from sagesim.breed import Breed

class MyBreed(Breed):

    def __init__(self) -> None:
        super().__init__("MyBreed")
        self.register_property("state", 1)  # Default value = 1
        self.register_step_func(my_step_func, __file__, priority=0)
```

---

## Writing Step Functions

A step function defines how an agent behaves during each simulation tick. It must be decorated with `@jit.rawkernel(device="cuda")`.

### Step Function Signature

```python
from cupyx import jit

@jit.rawkernel(device="cuda")
def my_step_func(
    tick,           # Current simulation tick
    agent_index,    # Index of this agent in the arrays
    globals,        # Global properties array
    agent_ids,      # Agent ID array
    breeds,         # Breed ID array
    locations,      # Neighbor indices array
    state,          # User-defined property arrays...
):
    """Agent behavior logic goes here."""
    # Read current state
    current_state = state[agent_index]

    # Access neighbors
    neighbor_indices = locations[agent_index]

    # Update state
    state[agent_index] = new_value
```

### Important Rules

1. **Required parameters**: `tick`, `agent_index`, `globals`, `agent_ids`, `breeds`, and `locations` must always be included in this exact order.

2. **All properties included**: All registered properties from all breeds must be in the signature, even if not used.

3. **Property order**: Properties appear in breed registration order, then property registration order within each breed.

---

## CuPy Kernel Limitations

SAGESim uses CuPy's `jit.rawkernel` for GPU execution. When writing step functions, be aware of these constraints:

| Limitation | Workaround |
|------------|------------|
| NaN checks don't work normally | Use `x != x` to check for NaN |
| No dicts or custom objects | Use arrays and primitives only |
| No `*args` or `**kwargs` | Use fixed argument lists |
| No nested functions | Define helpers at module level |
| No `for-each` loops | Use `for i in range(n)` |
| No `return` statements | Write results to arrays |
| No `break` or `continue` | Use boolean flags |
| No variable reassignment in scopes | Declare variables at top level |
| No `-1` indexing | Use `len(array) - 1` |

See [CuPy documentation](https://docs.cupy.dev/en/stable/reference/routines.html) for supported operations.

---

## Running a Simulation

### Single Worker, Single GPU (Recommended for Small Simulations)

If your simulation fits in one GPU's memory, use a single worker for best performance:

```python
# Run with: python my_simulation.py

# Create model and agents
model = MyModel(p_infection=0.2)
for i in range(1000):
    model.create_agent(state=1)

# Connect agents
for i in range(999):
    model.connect_agents(i, i + 1)

# Setup and run
model.setup(use_gpu=True)
model.simulate(ticks=100, sync_workers_every_n_ticks=1)

# Get results
for agent_id in range(10):
    state = model.get_agent_property_value(agent_id, "state")
    print(f"Agent {agent_id}: state={state}")
```

### Multiple Workers, Multiple GPUs (For Large Simulations)

For simulations that exceed single GPU memory, distribute across multiple GPUs with one worker per GPU:

```bash
# 4 workers on 4 GPUs (one worker per GPU)
mpirun -n 4 python my_simulation.py
```

> **Recommendation: One Worker = One GPU**
>
> While MPI can run multiple workers on a single GPU, this is **not recommended** due to:
> - MPI communication overhead between workers
> - GPU memory contention
> - No performance benefit over single-worker execution
>
> For best performance, use one MPI worker per physical GPU. If your simulation fits in one GPU, use a single worker (`python my_simulation.py`). Only use multiple workers when distributing across multiple physical GPUs.

---

## HPC Deployment

SAGESim is designed for HPC clusters where each compute node has multiple GPUs. The key principle is **one MPI rank per GPU**.

### Sample SLURM Script (Frontier)

```bash
#!/bin/bash
#SBATCH -A your_account
#SBATCH -J sagesim_run
#SBATCH -o logs/sagesim_%j.out
#SBATCH -e logs/sagesim_%j.err
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -N 10

# Load modules
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/5.7.1
module load craype-accel-amd-gfx90a

# Activate environment
source activate your_env_name

# Run simulation (8 GPUs per node)
num_nodes=10
num_mpi_ranks=$((8 * num_nodes))

srun -N${num_nodes} -n${num_mpi_ranks} -c7 \
     --ntasks-per-gpu=1 --gpu-bind=closest \
     python3 -u ./run.py
```

### Best Practices

- **Match MPI ranks to GPUs**: Set `num_ranks = gpus_per_node * num_nodes`
- **Use GPU binding**: `--gpu-bind=closest` reduces memory latency
- **Isolate runs**: Use job-specific output directories
- **Log management**: Include `%j` in log filenames for job ID

---

## Next Steps

- [Architecture Overview](architecture_overview.md) - System design and data flow
- [Synchronization and Double Buffering](synchronization_and_double_buffering.md) - Race condition prevention
- [Network Partitioning](network_partition.md) - Load balancing for distributed execution
- [Runtime Optimizations](runtime_optimizations.md) - Performance tuning
