# Changelog

All notable changes to SAGESim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.1] - 2026-08-25

Documentation-only release. No library code changed between 0.7.0 and 0.7.1, so upgrading requires
no changes to model code.

### Fixed
- **Step function signature** in `README.md` and `docs/getting_started.md` — SAGESim passes one
  parameter per registered global, in registration order, and none at all when the model registers
  no globals. The documented signature previously showed a single `globals` array, which raises
  `TypeError: <name>_double_buffer_N() takes X positional arguments but Y were given` at the first
  `simulate()` call. The real order is
  `tick, agent_index, <one per registered global>, agent_ids, breeds, locations, <one per property>`
- **`__main__` guard**: documented that driver code must sit under `if __name__ == "__main__":`,
  because `setup()` generates a kernel source file that imports the caller's module to recover the
  step function, so module-level code runs a second time during that import
- **Agent creation ordering**: documented that agents must exist before `setup()`, which infers
  each property's shape from the registered agent data
- **SIR example invocation**: `examples/sir/run.py` takes no command-line flags — parameters are set
  in its `if __name__ == "__main__":` block. The previously documented
  `--num_agents 10000 --percent_init_connections 0.1 --num_nodes 1` flags were silently ignored
- **SLURM launch**: added `srun -n 4 --ntasks-per-gpu=1 --gpu-bind=closest` alongside `mpirun`, for
  sites such as ORNL Frontier where `mpirun` is unavailable
- **Random numbers in step functions**: documented that `random.random()` is not rewritten by the
  AST pass and does not advance with the tick, so every agent draws the same value on every step and
  probabilistic models stall silently instead of raising. Use `rand_uniform_philox`,
  `rand_uniform_xorshift`, `rand_normal` or `rand_normal_bounded` from `sagesim.math_utils`
- Corrected the stale step-function parameter order in the `_build_param_to_property_index()` and
  `_build_param_to_property_index_csr()` docstrings in `sagesim/model.py`

## [0.7.0] - 2026-08-11

> **Upgrading from 0.6.x:** `model.setup(use_gpu=True)` → `model.setup()`.

### Added
- **Distributed model construction**: each rank builds only its own partition — no global
  graph is ever materialized. `Model.build_from_local_data()`, `Model.build_from_local_columns()`,
  `NetworkSpace.set_prebuilt_csr()`, `NetworkSpace.add_local_agents()`, `NetworkSpace.bulk_connect()`,
  `AgentFactory.register_remote_agents()`
- **Local-only accessors**: `get_local_agent_property_value()`, `set_local_agent_property_value()`,
  and a `local=` flag on `get_breed_data()` / `get_breed_agent_ids()` to read this rank's agents
  without a collective
- **`set_agent_logical_id()`**: stable per-agent identifier for reproducible RNG independent of
  partitioning
- **Caller-supplied agent IDs**: `agent_id=` on `create_agent()` / `create_agent_of_breed()`
- **Breed info exchange** when a neighbor-visible breed-local array is present
- **First-tick topology cache** for models whose network does not change
- **Test suite**: pytest suite under `tests/`, with GPU-absent auto-skip and a `benchmark`
  marker deselected by default (`pytest -m benchmark` to run timing tests)
- **Docs**: `docs/frontier_setup_rocm720_cupy1401.md` (ROCm 7.2.0 / CuPy 14.0.1 setup),
  `docs/partition_loading.md`

### Changed
- Agent IDs are `int64` throughout, lifting the previous magnitude limit
- `_agent2rank` on the GPU now holds only local and ghost entries instead of all ranks
- `GPUBufferManager` accepts slack and minimum-capacity tuning parameters
- `convert_agent_ids_to_indices()` gained `return_arrays=`
- Single-worker tick fusing is bypassed when `verbose_timing=True`, so per-tick timing rows are
  emitted instead of one row covering the whole run

### Removed
- **`Model.setup(use_gpu=...)`** — execution is always on GPU; there is no CPU backend
- **`sagesim/generate_partition.py`** in full (`partition_with_metis`, `partition_with_communities`,
  `partition_random`, `partition_round_robin`, `save_partition`, `analyze_partition`) and
  `Model.load_partition()` / `Model.load_partition_from_dict()`. Partitioning now happens outside
  SAGESim; see `docs/partition_loading.md`
- `AgentFactory.bulk_add_agents()`, `AgentFactory.bulk_register_agents()`,
  `AgentFactory.create_agent_at_index()`

### Fixed
- Registering an agent ID that has already exited is now rejected instead of silently accepted

## [0.6.0] - 2026-03-27

### Added
- **CSR format for neighbor lists**: Compact Sparse Row representation for efficient neighbor traversal on GPU
- **GPU-resident buffers**: Persistent GPU buffer management (`GPUBufferManager`, `GPUHashMap`) eliminating per-tick CPU-GPU transfers
- **GPU-aware MPI**: Direct GPU-to-GPU communication via buffer-protocol MPI with automatic detection of GPU-aware MPI environments
- **Single kernel launch**: Replaced Python tick+priority loop with a single kernel launch for reduced overhead
- **`breed_local_arrays`**: New API for breeding agents with array-valued properties
- **Globals as tensors**: Redesigned globals to accept tensors; redesigned random seed handling so users don't manage it manually
- **`math_utils.py`**: High-level math utility functions for use in step kernels
- **`get_breed_data()` API**: Bulk download property values for all agents of a breed
- **`get_agent_property_value`**: Now reads directly from GPU buffers
- **Overridable hook methods**: 3 new hook methods for customizing simulation lifecycle
- **`post_breed_step_code()`**: Inject code only after a specific priority
- **Agents sorted by breed**: Only agents of required breeds run on each step function
- **Scaling tests**: Weak scaling test scripts for HPC benchmarking

### Changed
- Threads per block increased from 32 to 128 for better GPU occupancy
- Immediate GPU allocation strategy replacing deferred allocation
- GPU buffer build refactored to remove for-loops (vectorized)
- Pack/send/receive communication refactored for efficiency
- Removed first-tick contextualization overhead
- `CuPy` experimental feature warnings suppressed

### Fixed
- Kernel hang issues during GPU execution
- GPU hang during time tracking
- CSR format correctness fix
- Import resolution issues
- Allow user to skip barrier between priorities

## [0.5.0] - 2025-02-03

### Added
- **Single-worker optimization**: Optimized performance for single-worker (non-MPI) execution
- **Optional double buffering**: Added `no_double_buffer` option for scenarios where race conditions are not a concern (#43)
- **Selective property synchronization**: Reduce MPI overhead by only synchronizing properties that have changed (#41)
- **Network partition support**: Load pre-computed METIS partitions for better load balancing (#39)
- **Ordered neighbors**: Support for maintaining neighbor order in location data
- **GPU-aware MPI**: Direct GPU-to-GPU communication on HPC environments
- **Verbose timing options**: Separate MPI transfer and computation timing output

### Changed
- Optimized CPU-GPU data transfer with vectorized contextualization (#37)
- Optimized `convert_id2index` and `index2id` for reduced runtime
- Write buffers now copied back to read buffers once per tick after all priority groups complete
- Replaced linear search with hash-map for locations lookup in step functions

### Fixed
- Bug fix in `create_zero_placeholder()` for locations data with set type
- Worker sync issue: ghost agents now always sent to corresponding workers regardless of data changes
- Locations with `-1` neighbor now skip but continue looping; NaN (padded) values break the loop

### Documentation
- Added comprehensive documentation for all major features
- New guides: selective property synchronization, ordered neighbors, network partitioning

## [0.4.0.dev1] - 2024

### Added
- Runtime optimization features
- Property history tracking
- Enhanced double buffering documentation

## [0.3.0] - 2024

### Added
- SIR epidemic model example with Jupyter tutorial
- HPC deployment support (SLURM scripts for ORNL Frontier)
- Improved agent network synchronization

## [0.2.0] - 2024

### Added
- Initial public release
- Core simulation framework with MPI + GPU support
- NetworkSpace for agent topology
- Breed and property registration system
- Double buffering for race condition prevention
