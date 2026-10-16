# SAGESim — working notes for Claude Code

SAGESim is a general GPU agent-based-modelling framework (CuPy kernels generated from
per-breed step functions, one persistent fused kernel per launch with software grid
barriers, MPI ghost exchange between ranks). SuperNeuroABM (`../superneuroabm`) is its
main client and keeps the running performance log in `docs/EVENT_DRIVEN_NOTES.md`
there; `docs/gpu_cpu_data_flow.md` here describes the data flow, the ghost exchange in
both directions and the single-buffer layout.

## Rules of the work (the owner's)

- Every change must be general-purpose ABM: justify it as helping any ABM, not only
  spiking networks. Discuss feature or semantic changes with the owner first.
- Measure before optimizing; prove equivalence by bitwise diff or measurement.
- Never `git stash`; use `git checkout <sha> -- <path>` for a before/after state.
- Multi-rank tests run on one GPU with `mpirun --oversubscribe -n N`, N in 1 2 3 4;
  a rank-0-only readback deadlocks because `get_agent_property_value` is collective.

## Environment and gate

```sh
PY=/home/xxz/miniforge3/envs/sna-dev/bin/python     # laptop env; both repos editable
$PY -m pytest -q tests/                              # seconds
PYTHON=$PY bash tests/run_mpi_tests.sh               # ranks 1 2 3 4
```

`SAGESIM_DIRECT_LAUNCH=0` disables the direct kernel launch (sagesim/launch.py);
`SAGESIM_GPU_AWARE_MPI=1` forces the device-buffer MPI path (needs a CUDA/ROCm-aware
MPI; set automatically from `MPICH_GPU_SUPPORT_ENABLED=1` / Open MPI's cuda flag).
Generated `step_func_code_*.py` files land in the working directory (gitignored).
