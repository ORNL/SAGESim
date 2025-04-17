#!/bin/bash
#SBATCH -A csc536
#SBATCH -J sagesim_sir
#SBATCH -o logs/sagesim_sir_%j.o
#SBATCH -e logs/sagesim_sir_%j.e
#SBATCH -t 01:00:00
#SBATCH -p batch
#SBATCH -N 10

# Only necessary if submitting like: sbatch --export=NONE ... (recommended)
# Do NOT include this line when submitting without --export=NONE
unset SLURM_EXPORT_ENV


# Load modules
module load PrgEnv-gnu/8.6.0
module load miniforge3/23.11.0-0
module load rocm/5.7.1
module load craype-accel-amd-gfx90a


# Activate your environment
source activate /lustre/orion/proj-shared/csc536/envs/sagesimenv_cg

# Point to source
export SRC_DIR=/lustre/orion/proj-shared/csc536/gunaratnecs/SAGESim/examples/sir/
export PYTHONPATH=/lustre/orion/proj-shared/csc536/gunaratnecs/SAGESim/:$PYTHONPATH

# Make run dir if not exists per job id
RUN_DIR=/lustre/orion/proj-shared/csc536/SAGESim/examples/sir
if [ ! -d "$RUN_DIR" ]
then
        mkdir -p $RUN_DIR
fi
cd $RUN_DIR

for num_nodes in 2 4 6 8 10
do
    for num_agents in 1000 10000 100000 1000000
    do
        for num_init_connections in 10 20 30
        do
            num_mpi_ranks=$((8 * ${num_nodes}))
            # Run script
            echo Running Python Script with ${num_nodes} nodes, ${num_agents} agents, and ${percent_init_connections} percent init connections.
            time srun -N${num_nodes} -n${num_mpi_ranks} -c7 --gpus-per-task=1 --gpu-bind=closest python3 -u ${SRC_DIR}/run.py --num_agents ${num_agents} --num_init_connections ${num_init_connections} --num_nodes ${num_nodes}
            echo Run Finished
            date
        done
    done
done

echo All runs Finished
date


