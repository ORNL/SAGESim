#!/bin/bash
#
# Batch submission script for a Dask workflow to run SAGESim simulations on Summit
#
#SBATCH -A LRN047
#SBATCH -t 0:10:00
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -J sagesim_debug


export SRC_DIR=/ccs/home/gunaratnecs/sagesim/examples/sir

RUN_DIR=/gpfs/alpine/proj-shared/lrn047/chathika/runs/sagesim_debug/${LSB_JOBID}

if [ ! -d "$RUN_DIR" ]
then
	mkdir -p $RUN_DIR
fi
cd $RUN_DIR

# location of dask scheduler file
export SCHEDULER_FILE=${RUN_DIR}/scheduler_file.json

module load ums
module load ums-gen119
module load nvidia-rapids/21.08

conda activate /gpfs/alpine/csc505/proj-shared/conda_sagesim_hpc

# Ensure Summit uses the right Python
export PATH=/gpfs/alpine/csc505/proj-shared/conda_sagesim_hpc/bin:$PATH

# Ensure Python can find the source
PYTHONPATH=${SRC_DIR}:$PYTHONPATH

# Launch dask scheduler 
# Command options at https://docs.dask.org/en/stable/deploying-cli.html
jsrun  --smpiargs="-disable_gpu_hooks"  --nrs 1 --tasks_per_rs 1 --cpu_per_rs 1  dask-scheduler  --interface ib0 --no-dashboard --no-show \
  --scheduler-file $SCHEDULER_FILE > dask-scheduler.out 2>&1 &

# Wait for the dask-scheduler to spin up
sleep 10

# Start the dask workers and ask them to use the same scheduler file to find our scheduler
Start_Workers() {
    for gpu in $(seq 0 5); do
        echo Setting up for GPU rank $gpu on $(hostname) ;
        (env -v CUDA_VISIBLE_DEVICES=${gpu} \
            dask worker \
            --scheduler-file $SCHEDULER_FILE --local-directory /tmp \--name worker-$(hostname)-gpu${gpu} --nthreads 2 --nworkers 1 \
            --no-dashboard --no-nanny --death-timeout 600) &
        sleep 2 ;
    done
}

jsrun -h $RUN_DIR \
  --smpiargs="-disable_gpu_hooks" \
  --tasks_per_rs 1 --cpu_per_rs 2 --gpu_per_rs 1 --rs_per_host 6 \
    Start_Workers &

sleep 5

# kick off the sagesim model
python -u $SRC_DIR/run.py

wait
# echo "finish  running python script"

#clean DASK files
rm -fr $SCHEDULER_DIR $SCHEDULER_FILE

echo "Done!"
bkill $LSB_JOBID
