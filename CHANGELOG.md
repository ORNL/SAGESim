# Changelog

All notable changes to SAGESim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
