#!/bin/bash
#SBATCH -A LRN088
#SBATCH -J sagesim_weak_scaling
#SBATCH -o /lustre/orion/proj-shared/lrn088/objective3/xxz/SAGESim/scaling_tests/output/weak_scaling_%j.out
#SBATCH -t 02:00:00
#SBATCH -q debug
#SBATCH -N 1

# SAGESim Weak Scaling Test
# Double loop: outer loop over nodes (N), inner loop over GPUs per node (n)
#
# Usage examples:
#   Test 1-10 nodes with 8 GPUs each:  Set MAX_NODES=10, GPUS_PER_NODE="8"
#   Test 1 node with 1,2,4,8 GPUs:     Set MAX_NODES=1, GPUS_PER_NODE="1 2 4 8"
#   Test 1-5 nodes with 8 GPUs each:   Set MAX_NODES=5, GPUS_PER_NODE="8"

unset SLURM_EXPORT_ENV

echo "=========================================="
echo "SAGESim Weak Scaling Test"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated Nodes: $SLURM_JOB_NUM_NODES"
echo ""

# Load modules
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/6.4.1
module load craype-accel-amd-gfx90a

# Activate environment
source activate /lustre/orion/proj-shared/lrn088/objective3/envs/superneuroabm_cupy13.6_env_xxz

# Use local sagesim
export PYTHONPATH=/lustre/orion/proj-shared/lrn088/objective3/xxz/SAGESim:$PYTHONPATH

cd /lustre/orion/proj-shared/lrn088/objective3/xxz/SAGESim/scaling_tests

# Create output directory
mkdir -p output

# ==================== CONFIGURATION ====================
# Test parameters (constant per rank for weak scaling)
# Sized for quick testing and clear GPU-aware MPI comparison
AGENTS_PER_RANK=50000
INTRA_DEGREE=10
CROSS_EDGES=2000
NUM_NEIGHBORS=1
TICKS=10
SYNC_TICKS=1
SEED=42

# Node range: test from 1 to MAX_NODES
MAX_NODES=$SLURM_JOB_NUM_NODES

# GPUs per node to test (space-separated list)
# Examples:
#   GPUS_PER_NODE="8"           -> Use all 8 GPUs per node
#   GPUS_PER_NODE="1 2 4 8"     -> Test with 1, 2, 4, and 8 GPUs per node
GPUS_PER_NODE="1 2 3 4 5 6 7 8"

# MPI mode (set to 0 for GPU-CPU-GPU, 1 for GPU-aware MPI)
export MPICH_GPU_SUPPORT_ENABLED=0
# =======================================================

echo "Configuration:"
echo "  Agents per rank: $AGENTS_PER_RANK (constant)"
echo "  Intra-cluster degree: $INTRA_DEGREE"
echo "  Cross-cluster edges: $CROSS_EDGES"
echo "  Neighbor clusters: $NUM_NEIGHBORS"
echo "  Simulation ticks: $TICKS"
echo "  Sync interval: $SYNC_TICKS (every tick)"
echo "  GPU-aware MPI: $([ $MPICH_GPU_SUPPORT_ENABLED -eq 1 ] && echo 'ENABLED' || echo 'DISABLED')"
echo ""
echo "Test Range:"
echo "  Nodes: 1 to $MAX_NODES"
echo "  GPUs per node: $GPUS_PER_NODE"
echo ""

# Double loop: outer loop over nodes (N), inner loop over GPUs per node (n)
for NNODES in $(seq 1 $MAX_NODES); do
    for NGPUS_PER_NODE in $GPUS_PER_NODE; do
        # Calculate total GPUs and agents
        TOTAL_GPUS=$((NNODES * NGPUS_PER_NODE))
        TOTAL_AGENTS=$((TOTAL_GPUS * AGENTS_PER_RANK))

        echo ""
        echo "=========================================="
        echo "Test: ${NNODES} node(s) × ${NGPUS_PER_NODE} GPU(s)/node = ${TOTAL_GPUS} GPUs"
        echo "  Total agents: ${TOTAL_AGENTS}"
        echo "=========================================="

        # Run the test
        srun -N${NNODES} -n${TOTAL_GPUS} -c7 --ntasks-per-gpu=1 --gpu-bind=closest \
            python weak_scaling_sir.py \
            --agents-per-rank $AGENTS_PER_RANK \
            --intra-cluster-degree $INTRA_DEGREE \
            --cross-cluster-edges $CROSS_EDGES \
            --num-neighbor-clusters $NUM_NEIGHBORS \
            --ticks $TICKS \
            --sync-ticks $SYNC_TICKS \
            --seed $SEED \
            $([ $MPICH_GPU_SUPPORT_ENABLED -eq 0 ] && echo '--no-gpu-aware-mpi' || echo '')

        echo ""
    done
done

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo ""
echo "For ideal weak scaling, simulation time should be CONSTANT"
echo "across all configurations (since agents-per-rank is constant)."
