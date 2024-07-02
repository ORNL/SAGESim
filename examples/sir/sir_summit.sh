#!/bin/bash
#
# Batch submission script for a Dask workflow to run SAGESim simulations on Summit
#
#BSUB -P LRN047
#BSUB -W 00:10
#BSUB -nnodes 2
#BSUB -J sagesim_debug
#BSUB -q debug

date

# For wider logging
export COLUMNS=132

export NUM_NODES=$(cat ${LSB_DJOB_HOSTFILE} | sort | uniq | grep -v login | grep -v batch | wc -l)
let "NUM_WORKERS=NUM_NODES*6"
export NUM_WORKERS=$NUM_WORKERS

export OMP_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=64


export SRC_DIR=/ccs/home/gunaratnecs/sagesim

RUN_DIR=/gpfs/alpine2/proj-shared/lrn047/chathika/runs/sagesim_debug/${LSB_JOBID}

if [ ! -d "$RUN_DIR" ]
then
	mkdir -p $RUN_DIR
fi
cd $RUN_DIR

# location of dask scheduler file
export SCHEDULER_FILE=${RUN_DIR}/scheduler_file.json

module load DefApps-2023
module load ums
module load ums-gen119
module load nvidia-rapids/21.08
module load python/3.10-miniforge3

conda init
source /ccs/home/gunaratnecs/.bashrc
conda activate /ccs/home/gunaratnecs/sagesimenv

# Ensure Summit uses the right Python
export PATH=/ccs/home/gunaratnecs/sagesimenv/bin:$PATH

# Ensure Python can find the source
PYTHONPATH=${SRC_DIR}:$PYTHONPATH
# Ensure Python can find the generated step function code
PYTHONPATH=${RUN_DIR}:$PYTHONPATH

# Copy over the hosts allocated for this job so that we can later verify
# that all the allocated nodes were busy with the correct worker allocation.
# Catches both the batch and compute nodes.
cat $LSB_DJOB_HOSTFILE | sort | uniq > $LSB_JOBID.hosts


# Just echo stuff for reality check
echo "################################################################################"
echo "Using python: " `which python3`
echo "PYTHONPATH: " $PYTHONPATH
echo "Source dir: $SRC_DIR"
echo "Scripts dir: $SCRIPTS_DIR"
echo "Run dir: $RUN_DIR"
echo "Dask scheduler file:" $SCHEDULER_FILE
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "NUMEXPR_MAX_THREADS: $NUMEXPR_MAX_THREADS"
echo "Number of nodes: $NUM_NODES"
echo "Number of workers: $NUM_WORKERS"
echo "################################################################################"



# Launch dask scheduler 
# Command options at https://docs.dask.org/en/stable/deploying-cli.html
echo "Starting scheduler"
jsrun  --gpu_per_rs 0 --nrs 1 --tasks_per_rs 1 --cpu_per_rs 2 --rs_per_host 1 dask scheduler --interface ib0 --idle-timeout 600 --no-jupyter --no-dashboard --no-show \
  --scheduler-file $SCHEDULER_FILE > dask-scheduler.out 2>&1 &


# Wait for the dask-scheduler to spin up
sleep 10

echo "Starting workers"
jsrun --nrs $NUM_NODES --tasks_per_rs 1 --cpu_per_rs 36 --gpu_per_rs 6 --rs_per_host 1 -b none --latency_priority gpu-cpu  --launch_distribution cyclic ${SRC_DIR}/examples/sir/start_workers.sh 2>&1 &

sleep 5

# kick off the sagesim model
python -u ${SRC_DIR}/examples/sir/run.py $SCHEDULER_DIR

wait
# echo "finish  running python script"

#clean DASK files
rm -fr $SCHEDULER_DIR

echo "Done!"
bkill $LSB_JOBID
