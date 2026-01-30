![SAGESim](SAGESim-inline-tag-color.png)

# SAGESim - Scalable Agent-based GPU-Enabled Simulator

**SAGESim** is the first scalable, pure-Python, general-purpose agent-based modeling framework that supports both distributed computing and GPU acceleration. Designed for high-performance computing (HPC) environments, SAGESim enables simulations with millions of agents by combining MPI-level parallelism across multiple GPUs with GPU-level parallelism using thousands of threads per device.

## Key Features

- **Dual-Level Parallelism**: MPI distribution across multiple GPUs + GPU thread parallelism for individual agents
- **Pure Python**: Write agent behaviors in Python using CuPy's JIT-compiled GPU kernels
- **Scalable**: From laptop GPUs to HPC clusters with thousands of GPUs
- **Network-Based Models**: Built-in support for agent networks with automatic neighbor data synchronization
- **Double Buffering**: Race condition prevention for concurrent agent interactions
- **Graph Partitioning**: Load pre-computed partitions to minimize cross-worker communication
- **Flexible Properties**: Support for scalar and nested list properties with automatic padding

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA drivers **or** AMD GPU with ROCm 5.7.1+
- MPI implementation (OpenMPI, MPICH, etc.)

## Installation

Your system might require specific steps to install `mpi4py` and/or `cupy` depending on your hardware. In that case, use your system's recommended instructions to install these dependencies first.

```bash
# Install SAGESim
pip install sagesim

# Or install from source
git clone https://github.com/ORNL/sagesim.git
cd sagesim
pip install -e .
```

### Dependencies

- `cupy` - GPU array computing
- `mpi4py` - MPI bindings for Python
- `networkx` - Graph/network handling
- `numpy` - CPU array operations
- `awkward` - Ragged array support

## Quick Start

### 1. Define a Breed (Agent Type)

```python
from cupyx import jit
from sagesim.breed import Breed

@jit.rawkernel(device="cuda")
def my_step_func(tick, agent_index, globals, agent_ids, breeds, locations, health):
    """Agent behavior: heal by 1 each tick"""
    health[agent_index] = health[agent_index] + 1

class MyBreed(Breed):
    def __init__(self):
        super().__init__("MyBreed")
        self.register_property("health", 100)  # Initial value
        self.register_step_func(my_step_func, __file__, priority=0)
```

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

```python
# Create model and agents
model = MyModel()
for i in range(1000):
    model.create_agent(health=100)

# Connect agents in a network
for i in range(999):
    model.connect_agents(i, i + 1)

# Setup and run
model.setup(use_gpu=True)
model.simulate(ticks=100, sync_workers_every_n_ticks=1)
```

### 4. Run with MPI (Multiple GPUs)

```bash
mpirun -n 4 python my_simulation.py
```

## Run Example: SIR Epidemic Model

```bash
git clone https://github.com/ORNL/sagesim.git
cd sagesim/examples/sir
mpirun -n 4 python run.py --num_agents 10000 --percent_init_connections 0.1 --num_nodes 1
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/architecture_overview.md) | System design, MPI distribution, GPU threading |
| [Getting Started](docs/getting_started.md) | Step-by-step guide to building models |
| [Double Buffering](docs/synchronization_and_double_buffering.md) | Race condition prevention mechanisms |
| [Network Partitioning](docs/network_partition.md) | Loading pre-computed partitions for load balancing |
| [Runtime Optimizations](docs/runtime_optimizations.md) | Performance tuning techniques |
| [Selective Sync](docs/selective_property_synchronization.md) | Reducing MPI overhead |
| [Property History](docs/property_history_tracking.md) | Tracking property changes over time |

## HPC Deployment

SAGESim is designed for HPC clusters. Example SLURM script for ORNL Frontier:

```bash
#!/bin/bash
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

See [CuPy documentation](https://docs.cupy.dev/en/stable/reference/routines.html) for supported operations.

## Project Structure

```
sagesim/
├── sagesim/           # Core library
│   ├── model.py       # Model class, simulation loop, GPU kernel generation
│   ├── agent.py       # Agent factory, MPI data synchronization
│   ├── breed.py       # Breed definition, property registration
│   ├── space.py       # NetworkSpace for agent topology
│   └── internal_utils.py  # Array conversion utilities
├── examples/          # Example models (SIR epidemic model)
├── docs/              # Comprehensive documentation
└── tests/             # Test suite
```

## Contributing

Contributions are welcome! Please see the [GitHub repository](https://github.com/ORNL/sagesim) for issues and pull requests.

## License

MIT License - Oak Ridge National Laboratory
