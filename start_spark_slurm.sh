#!/bin/bash

# E.g. use it like this to run MontePi.jar with 8 slices (slice count is
# specified in the program itself, but it was written to accept arguments):
#   jobid=$(sbatch ./start_spark_slurm.sh) &&
#   jobid=${jobid##Submitted batch job } &&
#   MASTER_WEB_UI='' &&
#   # looks like: 16/05/13 20:44:59 INFO MasterWebUI: Started MasterWebUI at http://172.24.36.19:8080
#   while [ -z "$MASTER_WEB_UI" ]; do MASTER_WEB_UI=$(sed -nE 's|.*Started MasterWebUI at (http://[0-9.:]*)|\1|p' $HOME/spark/logs/start-spark-slurm_$jobid.err); sleep 1s; done &&
#   cat ~/spark/logs/start-spark-slurm_$jobid.* &&
#   MASTER_ADDRESS=$(cat ~/spark/logs/${jobid}_spark_master) &&
#   ~/spark-1.5.2-bin-hadoop2.6/bin/spark-submit --master $MASTER_ADDRESS \
#       ~/scaromare/MontePi/multiNode/multiCore/MontePi.jar 1234567890 8 2>/dev/null
# Output could be:
#   Rolling the dice 1234567890 times resulted in pi ~ 3.1416070527180278 and took 7.370468868 seconds
# and when running with only 1 slice( ontePi.jar 1234567890 1 ):
#   Rolling the dice 1234567890 times resulted in pi ~ 3.141646073428979 and took 27.902197481 seconds

#SBATCH --account=p_scads
#SBATCH --partition=gpu1
#SBATCH --nodes=3
# ntasks per node MUST be one, because multiple slaves per work does not work with slurm + spark in this script
#SBATCH --ntasks-per-node=1
# CPUs per Task must be equal to gres:gpu or else too few or too much GPUs will
# be used. Partition 'gpu1' on taurus has 2 K20x GPUs per node and 'gpu2' has
# 4 K80 GPUs per node
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=500
# Beware! $HOME will not be expanded and invalid output-URIs will result
# slum jobs hanging indefinitely.
#SBATCH --output="/home/s3495379/spark/logs/start-spark-slurm_%j.out"
#SBATCH --error="/home/s3495379/spark/logs/start-spark-slurm_%j.err"
#SBATCH --time=01:00:00

realpath() { echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }

# This section will be run when started by sbatch
if [ "$1" != 'sran:D' ]; then
{
    # Get path of this script
    this=$0
    if [ ! -f "$this" ]; then
    {
        echo "[Note] Can't find calling argument path '$this', trying another method"
        this=$(scontrol show jobid $SLURM_JOBID | grep -i command)
        if [ -z "$this" ]; then
            echo "[Warning] Couldn't get path from slurm job info!"
        elif [ ${this:0:1} != '/' ]; then
            this=$SLURM_SUBMIT_DIR/$this
        fi
        if [ ! -f "$this" ]; then
            echo "[Note] Can't find SLURM job argument path '$this', trying another method"
            this=$SLURM_SUBMIT_DIR/$(basename "$0")
            if [ ! -f "$this" ]; then
                echo "[Error] Couldn't find path of this script. All methods exhausted"
                exit 1
            fi
        fi
    }
    fi
    # I experienced random problems with the second thread not finding the script:
    #   slurmstepd: execve(): /var/spool/slurm/job6924681/slurm_script: No such file or directory
    #   srun: error: taurusi2029: task 1: Exited with exit code 2
    if [ ! -d "/scratch/$USER/" ]; then
        echo "[ERROR] Couldn't find shared directory '/scratch/$USER/'"
        exit 1
    fi
    script=/scratch/$USER/${SLURM_JOBID}_$(basename "$0")
    cp "$this" "$script"
    echo "[Father] Working directory  : '$(pwd)'"
    echo "[Father] Path of this script: '$this'"

    module load scala/2.10.4 java/jdk1.7.0_25 cuda/7.0.28

    export sparkLogs=$HOME/spark/logs
    export sparkTmp=$HOME/spark/tmp
    mkdir -p "$sparkLogs" "$sparkTmp"

    # these variables must be set!
    export SPARK_ROOT=$HOME/spark-1.5.2-bin-hadoop2.6/
    export SPARK_JAVA_OPTS+="-Dspark.local.dir=$sparkTmp -XX:+UseParallelGC -XX:MaxPermSize=5G"
    export SPARK_DAEMON_MEMORY=$(( $SLURM_MEM_PER_CPU * $SLURM_CPUS_PER_TASK / 2 ))m
    export SPARK_MEM=$SPARK_DAEMON_MEMORY
    export SPARK_WORKER_DIR=$sparkLogs
    export SPARK_LOCAL_DIRS=$sparkLogs
    export SPARK_MASTER_PORT=7077
    export SPARK_MASTER_WEBUI_PORT=8080
    export SPARK_WORKER_CORES=$SLURM_CPUS_PER_TASK

    echo "[Father] srun $script 'sran:D' $@"
    srun "$script" 'sran:D' "$@"
    echo "[Father] srun finished, exiting now"
    exit 0
}
# If run by srun, then decide by $SLURM_PROCID whether we are master or worker
else
    #echo "SLURM_PROCID = $SLURM_PROCID"
    if [ -z "$SLURM_PROCID" ]; then
        echo "[Process $SLURM_PROCID] [Error] $SLURM_PROCID is not set, maybe srun failed somehow?"
        exit 1
    elif [ ! "$SLURM_PROCID" -eq "$SLURM_PROCID" ] 2>/dev/null; then
        echo "[Process $SLURM_PROCID] [Error] SLURM_PROCID=$SLURM_PROCID is not a number!"
        exit 1
    elif [ $SLURM_PROCID -eq 0 ]; then
    {
        # This does similar things as vanilla $SPARK_ROOT/sbin/start-master.sh
        # but slurm compatible, e.g. not in daemon-mode

        . "$SPARK_ROOT/sbin/spark-config.sh"
        . "$SPARK_ROOT/bin/load-spark-env.sh"

        export SPARK_MASTER_IP=$(hostname)
        MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
        if [ "$MASTER_NODE" != "$SPARK_MASTER_IP" ]; then
            echo "[Process $SLURM_PROCID] [Error] The method to get the master hostname won't work for the worker nodes! (This process is the master and is on $(hostname), but method will find '$MASTER_NODE' to be the master.)"
            exit 1
        fi

        # This can be used for debugging purposed and/or to find out the WebUI address
        # Furthermore this is necessary to submit jobs to the spark instance!
        echo "spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT" > "$sparkLogs/${SLURM_JOBID}_spark_master"

        echo "[Process $SLURM_PROCID] Starting Master at spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT (WebUI: $SPARK_MASTER_WEBUI_PORT)"
        "$SPARK_ROOT/bin/spark-class" org.apache.spark.deploy.master.Master \
            --ip $SPARK_MASTER_IP                                           \
            --port $SPARK_MASTER_PORT                                       \
            --webui-port $SPARK_MASTER_WEBUI_PORT
        echo "[Process $SLURM_PROCID] spark master finished, exiting now!"
    }
    else
    {
        # This does similar things as vanilla $SPARK_ROOT/sbin/start-slave.sh but slurm compatible
        # scontrol show hostname is used to convert host20[39-40] to host2039
        MASTER_NODE=spark://$(scontrol show hostname $SLURM_NODELIST | head -n 1):7077
        echo "[Process $SLURM_PROCID] Process $SLURM_PROCID starting slave at $(hostname) linked to $MASTER_NODE"
        "$SPARK_ROOT/bin/spark-class" org.apache.spark.deploy.worker.Worker $MASTER_NODE
        echo "[Process $SLURM_PROCID] spark slave finished, exiting now!"
    }
    fi
fi
