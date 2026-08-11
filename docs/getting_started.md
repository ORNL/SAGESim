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
import cupy as cp
from cupyx import jit

from sagesim.math_utils import rand_uniform_philox

@jit.rawkernel(device="cuda")
def my_step_func(
    tick,           # Current simulation tick
    agent_index,    # Index of this agent in the arrays
    p_infection,    # One parameter per registered global, in registration order
    agent_ids,      # Agent ID array
    breeds,         # Breed ID array
    locations,      # Neighbor indices array
    state,          # User-defined property arrays...
):
    """Susceptible agents catch the state from any infected neighbor."""
    # Neighbor indices for this agent (SAGESim pre-converts agent IDs to indices)
    neighbor_indices = locations[agent_index]

    # Read this agent's own state
    agent_state = int(state[agent_index])

    if agent_state == 1:  # susceptible
        i = 0
        infected = False
        while i < len(neighbor_indices) and not cp.isnan(neighbor_indices[i]) and not infected:
            neighbor_index = int(neighbor_indices[i])
            if int(state[neighbor_index]) == 2:  # infected neighbor
                if rand_uniform_philox(tick, agent_index, 1) < p_infection:
                    state[agent_index] = 2
                    infected = True
            i += 1
```

### Important Rules

1. **Parameter order**: SAGESim passes arguments in exactly this order, and the signature must
   match it exactly:

   ```
   tick, agent_index, <one per registered global>, agent_ids, breeds, locations, <one per registered property>
   ```

   `tick`, `agent_index`, `agent_ids`, `breeds` and `locations` are always present. **Globals are
   not** — there is one parameter per global registered with `register_global_property()`, in
   registration order, and none at all if the model registers none. A model with no globals takes
   `(tick, agent_index, agent_ids, breeds, locations, ...)`.

2. **All properties included**: All registered properties from all breeds must be in the signature, even if not used.

3. **Property order**: Properties appear in breed registration order, then property registration order within each breed.

4. **A mismatch fails late**: an extra or missing parameter is not caught at `setup()`. It raises
   at the first `simulate()` call as
   `TypeError: <name>_double_buffer_N() takes X positional arguments but Y were given`.

5. **Use SAGESim's RNG, not `random.random()`**: draw random numbers with
   `rand_uniform_philox(tick, agent_index, salt)` from `sagesim.math_utils` (or
   `rand_uniform_xorshift`, `rand_normal`, `rand_normal_bounded`). SAGESim rewrites these calls to
   inject the run seed and key them on the agent's stable logical ID, so draws vary per tick and
   per agent and are reproducible across runs and rank counts. `salt` is any small integer that
   distinguishes one call site from another.

   Python's `random.random()` is **not** rewritten. Inside a kernel it does not advance with the
   tick, so an agent draws the same value every step — a probabilistic model built on it silently
   stalls rather than erroring. Measured on a 60-agent chain with `p_infection=0.2` over 50 ticks:
   `rand_uniform_philox` infected 13 agents, `random.random()` infected 0 beyond the seed agent.

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

If your simulation fits in one GPU's memory, use a single worker for best performance.

The model, breed and step function above, plus the driver code below, all go in **one file** —
`my_simulation.py`. Two requirements are easy to miss:

- **Driver code must sit under `if __name__ == "__main__":`.** `setup()` generates a kernel source
  file that imports your module to recover the step function, so anything at module level runs a
  second time during that import.
- **Agents must exist before `setup()`.** It inspects registered agent data to determine each
  property's shape.

```python
# Run with: python my_simulation.py

if __name__ == "__main__":
    # Create model and agents
    model = MyModel(p_infection=0.2)
    for i in range(1000):
        model.create_agent(state=1)

    # Connect agents
    for i in range(999):
        model.connect_agents(i, i + 1)

    # Infect agent 0 so there is something to spread
    model.set_agent_property_value(0, "state", 2)

    # Setup and run
    model.setup()
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

# Generic MPI (OpenMPI / MPICH)
mpirun -n 4 python my_simulation.py

# SLURM sites such as ORNL Frontier, where mpirun is not available
srun -n 4 --ntasks-per-gpu=1 --gpu-bind=closest python my_simulation.py
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
- [Partition Loading](partition_loading.md) - Building a model from per-rank partitions
- [Runtime Optimizations](runtime_optimizations.md) - Performance tuning
