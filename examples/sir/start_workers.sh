#!/bin/bash
#
# For spinning up the dask workers per GPU for a node.  This will have been
# invoked via a top-level LSF script via a jsrun that allocates a resource
# set for an entire node for this script to then allocate workers per GPU.
#

export COLUMNS=132

export OMP_NUM_THREADS=4
export NUMEXPR_MAX_THREADS=64

export SRC_DIR=/ccs/home/gunaratnecs/sagesim

RUN_DIR=/gpfs/alpine2/proj-shared/lrn047/chathika/runs/sagesim_debug/${LSB_JOBID}
cd $RUN_DIR

export SCHEDULER_FILE=${RUN_DIR}/scheduler_file.json

module load python/3.10-miniforge3

conda activate /ccs/home/gunaratnecs/sagesimenv

# Ensure Summit uses the right Python
export PATH=/ccs/home/gunaratnecs/sagesimenv/bin:$PATH

# Make sure gremlin can find our stuff
PYTHONPATH=${SRC_DIR}:$PYTHONPATH

echo "################################################################################"
echo $0
echo "Using python: " `which python3`
echo "PYTHONPATH: " $PYTHONPATH
echo "Source dir: $SRC_DIR"
echo "Scripts dir: $SCRIPTS_DIR"
echo "Run dir: $RUN_DIR"
echo "SCHEDULER_FILE: " $SCHEDULER_FILE
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "NUMEXPR_MAX_THREADS: $NUMEXPR_MAX_THREADS"
echo "################################################################################"

for gpu in $(seq 0 5); do
	echo Setting up for GPU rank $gpu on $(hostname) ;
	(env -v CUDA_VISIBLE_DEVICES=${gpu} \
		dask worker \
		--scheduler-file $SCHEDULER_FILE --local-directory /tmp \--name worker-$(hostname)-gpu${gpu} --nthreads 1 --nworkers 1 \
		--no-dashboard --no-nanny --death-timeout 600) &
	sleep 2 ;
done

# If we don't wait, then this script exits, killing off the workers.
wait
