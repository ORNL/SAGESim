![SAGESim](SAGESim-inline-tag-color.png)

# SAGESim - Scalable Agent-based GPU-Enabled Simulator

**SAGESim** is the first scalable, pure-Python, general-purpose agent-based modeling framework that supports both distributed computing and GPU acceleration. Designed for high-performance computing (HPC) environments, SAGESim enables simulations with millions of agents by combining MPI-level parallelism across multiple GPUs with GPU-level parallelism using thousands of threads per device.

## Key Features

- **Dual-Level Parallelism**: MPI distribution across multiple GPUs + GPU thread parallelism for individual agents
- **Pure Python**: Write agent behaviors in Python using CuPy's JIT-compiled GPU kernels
- **Scalable**: From laptop GPUs to HPC clusters with thousands of GPUs
- **Network-Based Models**: Built-in support for agent networks with automatic neighbor data synchronization
- **Distributed Construction**: Each rank builds only its own partition — the full graph is never materialized on any one rank
- **GPU-Resident State**: Persistent GPU buffers and CSR neighbor storage, so agent data stays on the device across ticks
- **GPU-Aware MPI**: Direct GPU-to-GPU transfers where the MPI implementation supports it, with automatic detection
- **Fused Ticks**: All ticks and priorities in a single kernel launch, synchronized by in-kernel grid barriers
- **Double Buffering**: Race condition prevention for concurrent agent interactions
- **Flexible Properties**: Support for scalar and nested list properties with automatic padding

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA drivers **or** AMD GPU with ROCm
- MPI implementation (OpenMPI, MPICH, etc.)

Tested on ORNL Frontier with ROCm 7.2.0 and CuPy 14.0.1 — see
[Frontier setup](docs/frontier_setup_rocm720_cupy1401.md) for a known-good environment.

## Installation

Your system might require specific steps to install `mpi4py` and/or `cupy` depending on your hardware. In that case, use your system's recommended instructions to install these dependencies first.

```bash
# Install SAGESim
pip install sagesim

# Or install from source
git clone https://github.com/ORNL/sagesim.git
cd sagesim
pip install .
```

### Dependencies

Resolved automatically by `pip install sagesim`:

- `networkx` - Graph/network handling
- `numpy` - CPU array operations
- `awkward` - Ragged array support

**Not** installed automatically, because the correct build depends on your GPU and MPI stack —
install these yourself first:

- `cupy` - GPU array computing (choose the CUDA or ROCm build matching your hardware)
- `mpi4py` - MPI bindings for Python (must be built against your system MPI)

## Quick Start

Steps 1-4 below build up a **single file** — call it `my_simulation.py`. The step function has to
live in an importable module, and the driver code has to sit behind a `__main__` guard (step 3
explains why), so splitting the pieces across files or a REPL will not work.

### 1. Define a Breed (Agent Type)

```python
from cupyx import jit
from sagesim.breed import Breed

@jit.rawkernel(device="cuda")
def my_step_func(tick, agent_index, agent_ids, breeds, locations, health):
    """Agent behavior: heal by 1 each tick"""
    health[agent_index] = health[agent_index] + 1

class MyBreed(Breed):
    def __init__(self):
        super().__init__("MyBreed")
        self.register_property("health", 100)  # Initial value
        self.register_step_func(my_step_func, __file__, priority=0)
```

SAGESim calls your step function with a fixed parameter order, and the argument list must match
it exactly:

```
tick, agent_index, <one per registered global>, agent_ids, breeds, locations, <one per registered property>
```

The breed above registers no globals and one property, hence
`(tick, agent_index, agent_ids, breeds, locations, health)`. Declaring a parameter that does not
correspond to a registered global or property fails at the first `simulate()` call with
`TypeError: ... takes N positional arguments but M were given`.

### 2. Define a Model

```python
from sagesim.model import Model
from sagesim.space import NetworkSpace

class MyModel(Model):
    def __init__(self):
        super().__init__(NetworkSpace())
        self._breed = MyBreed()
        self.register_breed(self._breed)

    def create_agent(self, health):
        return self.create_agent_of_breed(self._breed, health=health)

    def connect_agents(self, agent_a, agent_b):
        self.get_space().connect_agents(agent_a, agent_b)
```

### 3. Run the Simulation

Driver code must sit under an `if __name__ == "__main__":` guard. `setup()` generates a kernel
source file that imports your module to pick up the step function, so anything at module level
runs a second time during that import.

```python
if __name__ == "__main__":
    # Create model and agents
    model = MyModel()
    for i in range(1000):
        model.create_agent(health=100)

    # Connect agents in a network
    for i in range(999):
        model.connect_agents(i, i + 1)

    # Setup and run
    model.setup()
    model.simulate(ticks=100, sync_workers_every_n_ticks=1)
```

Agents must exist before `setup()` — it inspects the registered agent data to determine each
property's shape.

### 4. Read the Results

Continuing inside the same `__main__` block:

```python
    # One agent
    health = model.get_agent_property_value(0, property_name="health")

    # Or every agent of a breed at once
    all_health = model.get_breed_data("MyBreed", "health")
```

Each agent healed 1 per tick for 100 ticks, so `health` is `200.0`.

### Running on multiple GPUs

The same script runs unchanged under MPI, one rank per GPU:

```bash
# Generic MPI (OpenMPI / MPICH)
mpirun -n 4 python my_simulation.py

# SLURM sites such as ORNL Frontier, where mpirun is not available
srun -n 4 --ntasks-per-gpu=1 --gpu-bind=closest python my_simulation.py
```

## Run Example: SIR Epidemic Model

Runs on a single GPU, no MPI launcher needed:

```bash
git clone https://github.com/ORNL/sagesim.git
cd sagesim/examples/sir
python run.py
```

The parameters (20 agents, 2 initial connections, 10 ticks) are set at the top of the
`if __name__ == "__main__":` block in `run.py` — there are no command-line flags; edit the file
to change them. To spread the same run across 4 GPUs:

```bash
# Generic MPI (OpenMPI / MPICH)
mpirun -n 4 python run.py

# SLURM sites such as ORNL Frontier, where mpirun is not available
srun -n 4 --ntasks-per-gpu=1 --gpu-bind=closest python run.py
```

## Testing

```bash
pip install ".[test]"

pytest                 # full suite
pytest -m benchmark    # timing benchmarks, deselected by default
```

Every test executes GPU kernels, so the suite skips itself entirely when no device is visible
rather than failing.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture_overview.md) | System design, MPI distribution, GPU threading |
| [Getting Started](docs/getting_started.md) | Step-by-step guide to building models |
| [Double Buffering](docs/synchronization_and_double_buffering.md) | Race condition prevention mechanisms |
| [Partition Loading](docs/partition_loading.md) | Building a model from per-rank partitions |
| [Runtime Optimizations](docs/runtime_optimizations.md) | Performance tuning techniques |
| [Overhead Analysis](docs/overhead_analysis.md) | Where per-tick time actually goes |
| [Selective Sync](docs/selective_property_synchronization.md) | Reducing MPI overhead |
| [Property History](docs/property_history_tracking.md) | Tracking property changes over time |
| [Ordered Neighbors](docs/ordered_neighbors.md) | Ordered neighbor storage for agent networks |
| [GPU-CPU Data Flow](docs/gpu_cpu_data_flow.md) | Data flow between CPU and GPU |
| [GPU Communication Redesign](docs/gpu_communication_redesign.md) | GPU-resident buffers and CSR-based ghost exchange |
| [Frontier Setup](docs/frontier_setup_rocm720_cupy1401.md) | Known-good ROCm 7.2.0 / CuPy 14.0.1 environment |

## HPC Deployment

SAGESim is designed for HPC clusters. Example SLURM script for ORNL Frontier:

```bash
#!/bin/bash
#SBATCH -A <your_project>
#SBATCH -N 10
#SBATCH -t 00:30:00

num_nodes=10
num_mpi_ranks=$((8 * num_nodes))  # 8 GPUs per node

srun -N${num_nodes} -n${num_mpi_ranks} -c7 \
     --ntasks-per-gpu=1 --gpu-bind=closest \
     python3 -u ./run.py
```

## CuPy JIT Kernel Limitations

When writing step functions, be aware of these `cupyx.jit.rawkernel` constraints:

- **NaN checks**: Use `x != x` (inequality to self)
- **No dicts/objects**: Only primitive types and arrays
- **No `*args`/`**kwargs`**: Fixed argument lists only
- **No nested functions**: Define helpers at module level
- **Use CuPy, not NumPy**: Use `cupy` data types and routines in kernels
- **`for` loops**: Must use `range()` iterator only
- **No `return`**: Side effects via array writes only
- **No `break`/`continue`**: Use boolean flags instead
- **No variable reassignment in scopes**: Declare at top level
- **No `-1` indexing**: Use `len(array) - 1` instead
- **Random numbers**: use `rand_uniform_philox(tick, agent_index, salt)` from `sagesim.math_utils`, not `random.random()` — see below

See [CuPy documentation](https://docs.cupy.dev/en/stable/reference/routines.html) for supported operations.

### Random numbers in step functions

Draw with `rand_uniform_philox(tick, agent_index, salt)` (or `rand_uniform_xorshift`,
`rand_normal`, `rand_normal_bounded`) from `sagesim.math_utils`. SAGESim rewrites these calls to
inject the run seed and key them on the agent's stable logical ID, so draws vary per tick and per
agent and stay reproducible across runs and rank counts. `salt` is any small integer
distinguishing one call site from another.

Python's `random.random()` is not rewritten and does not advance with the tick, so an agent draws
the same value on every step. A probabilistic model built on it stalls silently instead of
raising.

## Project Structure

```
sagesim/
├── sagesim/               # Core library
│   ├── model.py           # Model class, simulation loop, GPU kernel generation
│   ├── gpu_kernels.py     # GPU buffer manager, GPU hash map, MPI communication manager
│   ├── agent.py           # Agent factory, rank assignment, agent data tensors
│   ├── breed.py           # Breed definition, property registration
│   ├── space.py           # NetworkSpace for agent topology
│   ├── math_utils.py      # Math helpers callable from step kernels
│   ├── utils.py           # Agent/neighbor data accessors for step kernels
│   ├── partition_utils.py # Helpers for per-rank partition loading
│   ├── internal_utils.py  # Array conversion and CSR construction
│   └── jit_extensions.py  # CuPy JIT builtins (e.g. threadfence)
├── examples/              # Example models (SIR epidemic model)
├── scaling_tests/         # Weak scaling harness and SLURM launcher
├── docs/                  # Comprehensive documentation
└── tests/                 # Test suite
```

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/ORNL/sagesim) for issues and pull requests.

## License

MIT License - Oak Ridge National Laboratory
