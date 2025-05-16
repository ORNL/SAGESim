#!/bin/bash
#SBATCH -A csc536
#SBATCH -J sagesim_sir
#SBATCH -o logs/sagesim_sir_%j.o
#SBATCH -e logs/sagesim_sir_%j.e
#SBATCH -t 00:30:00
#SBATCH -p batch
#SBATCH -q debug
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
export PYTHONPATH=/lustre/orion/proj-shared/csc536/gunaratnecs/mpiasync/SAGESim/:$PYTHONPATH

# Make run dir if not exists per job id
RUN_DIR=/lustre/orion/proj-shared/csc536/gunaratnecs/mpiasync/SAGESim/examples/sir
if [ ! -d "$RUN_DIR" ]
then
        mkdir -p $RUN_DIR
fi
cd $RUN_DIR


for num_agents in 100000
do
    for num_nodes in 10
    do
        for num_init_connections in 20
        do
            num_mpi_ranks=$((8 * ${num_nodes}))
            # Run script
            echo Running. Python Script with ${num_nodes} nodes, ${num_agents} agents, and ${percent_init_connections} percent init connections.
            time srun -N${num_nodes} -n${num_mpi_ranks} -c7 --ntasks-per-gpu=1 --gpu-bind=closest python3 -u ./run.py --num_agents ${num_agents} --num_init_connections ${num_init_connections} --num_nodes ${num_nodes}
            #time srun -N${num_nodes} -n${num_mpi_ranks} -c7 --gpus-per-task=1 --gpu-bind=closest /bin/bash -c 'echo $(hostname) $(grep Cpus_allowed_list /proc/self/status) GPUS: $ROCR_VISIBLE_DEVICES' | sort
            echo Run finished. Python Script with ${num_nodes} nodes, ${num_agents} agents, and ${percent_init_connections} percent init connections.
            date
        done
    done
done

echo All runs Finished
date
