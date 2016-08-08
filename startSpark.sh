#!/bin/bash

# Define SPARK_ROOT variable to point to the root folder of Spark (that where
# the folder 'bin' resides in), e.g. add that variable in your .bashrc.
# Source this script, possibly also in your .bashrc, and then you can use simply
#
#   startSpark [Slurm Parameters]
#
# E.g.:
#
#   startSpark --time=11:00:00 --nodes=8 --partition=gpu2 --cpus-per-task=4 --gres='gpu:4'
#
# Don't choose anything else than --ntasks-per-node=1, because each SLURM
# task will start one Spark Executor. If nothing specified 1 is the default.
#
# Each Spark Executor will run --cpus-per-task threads. This is the parallelism
# availabe by using Spark Partitions.
#
# So if you want each partition to work on another GPU you will have to watch
# out that --cpus-per-task=4 and --gres='gpu:4' have the same number!
# But maybe you want only 1 thread per node where each thread distributes it's
# work manually to GPU, then you can experiment with --cpus-per-task=1 and
# --gres='gpu:4'

export START_SPARK_SLURM_PATH=$HOME/scaromare/common/start_spark_slurm.sh

function startSpark() {
    export SPARK_LOGS=$HOME/spark/logs
    mkdir -p "$SPARK_LOGS"
    if [ ! -d "$SPARK_LOGS" ]; then return 1; fi
    jobid=$(sbatch "$@" --output="$SPARK_LOGS/%j.out" --error="$SPARK_LOGS/%j.err" $START_SPARK_SLURM_PATH)
    jobid=${jobid##Submitted batch job }
    echo "Job ID : $jobid"
    # looks like: 16/05/13 20:44:59 INFO MasterWebUI: Started MasterWebUI at http://123.123.123.123:8080
    echo -n "Waiting for Job to run and Spark to start.."
    MASTER_WEBUI=''
    while [ -z "$MASTER_WEBUI" ]; do
        echo -n "."
        sleep 1s
        if [ -f $HOME/spark/logs/$jobid.err ]; then
            MASTER_WEBUI=$(sed -nE 's|.*Started MasterWebUI at (http://[0-9.:]*)|\1|p' $HOME/spark/logs/$jobid.err)
        fi
    done
    echo "OK"
    export MASTER_WEBUI
    export MASTER_ADDRESS=$(cat "$HOME/${jobid}_spark_master")
    function sparkSubmit() {
        "$SPARK_ROOT"/bin/spark-submit --master $MASTER_ADDRESS $@
    }
    cat "$SPARK_LOGS"/$jobid.*
    echo "MASTER_WEBUI   : $MASTER_WEBUI"
    echo "MASTER_ADDRESS : $MASTER_ADDRESS"
}
export -f startSpark
