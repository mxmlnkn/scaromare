#!/bin/bash

# These settings can be overwritten by specifying 'fresh' command line parameters to startSpark / sbatch
# ntasks per node (per node) MUST be one, because multiple slaves per work does not work with slurm + spark in this script
#SBATCH --ntasks-per-node=1
# CPUs per Task should be equal to gres:gpu or else too few or too much GPUs will
# be used. Partition 'gpu1' on taurus has 2 K20x GPUs per node and 'gpu2' has
# 4 K80 GPUs per node
#SBATCH --mem-per-cpu=1000
# Beware! $HOME will not be expanded in --output option iuf given here in this script and invalid output-URIs will result Slurm jobs hanging indefinitely.

# E.g. use it like this to run MontePi.jar with 8 slices (slice count is
# specified in the program itself, but it was written to accept arguments):
#   startSpark
#   sparkSubmit ~/scaromare/MontePi/multiNode/multiCore/MontePi.jar 1234567890 8 2>/dev/null
# Output could be:
#   Rolling the dice 1234567890 times resulted in pi ~ 3.1416070527180278 and took 7.370468868 seconds
# and when running with only 1 slice( ontePi.jar 1234567890 1 ):
#   Rolling the dice 1234567890 times resulted in pi ~ 3.141646073428979 and took 27.902197481 seconds

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

    # Exported Variables are available after srun! You can test this out with
    # salloc -N 2
    #   mimi=momo
    #   srun bash -c 'echo [$(hostname)] mimi = $mimi'
    #     [taurusi1052] mimi =
    #     [taurusi1053] mimi =
    #   export mimi
    #   srun bash -c 'echo mimi = $mimi'
    #     [taurusi1052] mimi = momo
    #     [taurusi1053] mimi = momo
    export SPARK_DAEMON_MEMORY=$(( $SLURM_MEM_PER_CPU * $SLURM_CPUS_PER_TASK / 2 ))m
    export SPARK_MEM=$SPARK_DAEMON_MEMORY
    export SPARK_WORKER_CORES=$SLURM_CPUS_PER_TASK

    echo "[Father] srun $script 'sran:D' $@"
    srun "$script" 'sran:D' "$@"
    echo "[Father] srun finished, exiting now"
    exit 0
}
# If run by srun, then decide by $SLURM_PROCID whether we are master or worker
else
    # mktemp -d must run for each host, that's why this is only done if
    # srun was called on this script! Else the temporary directory would be
    # created on tauruslogin
    sparkTmp=$(mktemp -d)   # using $HOME/spark/tmp is not a good idea as it is slow and shared
    echo "[$(hostname)] Spark Working Directory (Logs, Distributed Jar, ...): $sparkTmp"

    # these variables must be set!
    # http://spark.apache.org/docs/latest/configuration.html#application-properties
    export SPARK_ROOT=$HOME/spark-1.5.2-bin-hadoop2.6/
    # SPARK_JAVA_OPTS is interpreted by
    # spark-1.5.2/core/src/main/scala/org/apache/spark/SparkConf.scala
    # and then used in
    # spark-1.5.2/launcher/src/main/java/org/apache/spark/launcher/SparkClassCommandBuilder.java
    export SPARK_JAVA_OPTS+=" -XX:+UseParallelGC "
    # This folder will contain the distributed jar and the output logs ...
    # This is a tad sad, because that means if I want to store the jar locally,
    # then I also have to store the logs locally and therefore not really
    # accessible from tauruslogin, only from the spark WebUI.
    # -> Well you cann use scp and ssh directly ... (and thereby circumvent SLURM)
    export SPARK_WORKER_DIR=$sparkTmp  # $sparkLogs
    # Not sure what is stored in here: "Directory to use for "scratch" space in Spark"
    # it is saved to the variable 'workDir' in
    # spark-1.5.2/core/src/main/scala/org/apache/spark/deploy/worker/WorkerArguments.scala
    # It is suggest, that SPARK_LOCAL_DIRS overrides spark.local.dir in
    # spark-1.5.2/core/src/test/scala/org/apache/spark/storage/LocalDirsSuite.scala:
    export SPARK_LOCAL_DIRS=$sparkTmp
    export SPARK_MASTER_PORT=7077
    export SPARK_MASTER_WEBUI_PORT=8080

    #echo "SLURM_PROCID = $SLURM_PROCID"

    module load scala/2.10.4 java/jdk1.7.0_25 cuda/7.0.28
    nvidia-smi

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
        echo "spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT" > "$HOME/${SLURM_JOBID}_spark_master"

        echo "[Process $SLURM_PROCID] Starting Master at spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT (WebUI: $SPARK_MASTER_WEBUI_PORT)"

        "$SPARK_ROOT/bin/spark-class" org.apache.spark.deploy.master.Master \
            --ip $SPARK_MASTER_IP                                           \
            --port $SPARK_MASTER_PORT                                       \
            --webui-port $SPARK_MASTER_WEBUI_PORT &
        echo "[Process $SLURM_PROCID] spark master finished, trying to start slave now!"

        # For some reason there was a bug with creating a temporary directory:
        # java.io.IOException: Failed to create a temp directory (under ) after 10 attempts!
        #   https://groups.google.com/forum/#!topic/spark-users/aWva61WAnMc
        #   https://issues.apache.org/jira/browse/SPARK-2325
        # My guess is that the automatically chosen folder name only depends
        # on things like the hostname, but not the process ID, therefore
        # clashing if trying to run Master and Executor on one node ...
        # But I also did many other things wrong, like using mktemp instead
        # of mktemp -d and so on.
        sparkTmp=$(mktemp -d)
        echo "[$(hostname)] Spark Working Directory (Logs, Distributed Jar, ...): $sparkTmp"
        export SPARK_JAVA_OPTS+=" -Dspark.local.dir=$sparkTmp "
        export SPARK_LOCAL_DIRS=$sparkTmp
        export SPARK_WORKER_DIR=$sparkTmp

        MASTER_NODE=spark://$SPARK_MASTER_IP:$SPARK_MASTER_PORT
        "$SPARK_ROOT/bin/spark-class" org.apache.spark.deploy.worker.Worker $MASTER_NODE
        echo "[Process $SLURM_PROCID] spark master + slave finished, exiting now!"
    }
    else
    {
        # This does similar things as vanilla $SPARK_ROOT/sbin/start-slave.sh but slurm compatible
        # scontrol show hostname is used to convert host20[39-40] to host2039
        MASTER_NODE=spark://$(scontrol show hostname $SLURM_NODELIST | head -n 1):$SPARK_MASTER_PORT
        echo "[Process $SLURM_PROCID] Process $SLURM_PROCID starting slave at $(hostname) linked to $MASTER_NODE"

        "$SPARK_ROOT/bin/spark-class" org.apache.spark.deploy.worker.Worker $MASTER_NODE

        echo "[Process $SLURM_PROCID] spark slave finished, exiting now!"
    }
    fi
fi
