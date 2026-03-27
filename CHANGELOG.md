# Changelog

All notable changes to SAGESim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
