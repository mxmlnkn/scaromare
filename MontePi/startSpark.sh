#!/bin/bash

function startSpark() {
    export SPARK_LOGS=$HOME/spark/logs
    mkdir -p "$SPARK_LOGS"
    if [ ! -d "$SPARK_LOGS" ]; then return 1; fi
    jobid=$(sbatch "$@" --output="$SPARK_LOGS/%j.out" --error="$SPARK_LOGS/%j.err" $HOME/scaromare/start_spark_slurm.sh)
    jobid=${jobid##Submitted batch job }
    echo "Job ID : $jobid"
    # looks like: 16/05/13 20:44:59 INFO MasterWebUI: Started MasterWebUI at http://172.24.36.19:8080
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
    export MASTER_ADDRESS=$(cat ~/spark/logs/${jobid}_spark_master)
    function sparkSubmit() {
        ~/spark-1.5.2-bin-hadoop2.6/bin/spark-submit --master $MASTER_ADDRESS $@
    }
    cat "$SPARK_LOGS"/$jobid.*
    echo "MASTER_WEBUI   : $MASTER_WEBUI"
    echo "MASTER_ADDRESS : $MASTER_ADDRESS"
}
