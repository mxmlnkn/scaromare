# About this Repository

The goal of this repository are first tests into using Spark with GPUs by using Rootbeer.

The more long term goal is to hide the antics of the GPU task, so that a user can provide Rootbeer kernels and after that can use Spark like normal plus the defined kernels.


# Starting MontePi Spark Multi GPU Test

    startSpark --time=15:00:00 --nodes=2 --partition=gpu2 --cpus-per-task=2 --gres='gpu:2'
    squeue -u $USER
         JOBID   PARTITION NAME     ST      TIME  NODES NODELIST(REASON)
         7660143 gpu2      start_sp R   14:49:09      2 taurusi[2093-2094]
    makeOptsSpark=(                                                      \
        "SPARK_ROOT=$HOME/spark-1.5.2-bin-hadoop2.6"                     \
        "SPARK_JAR=$SPARK_ROOT/lib/spark-assembly-1.5.2-hadoop2.6.0.jar" \
        "SPARKCORE_JAR=$SPARK_JAR"                                       \
        "SCALA_ROOT=$(dirname $(which scala))/../lib"                    \
    )
    ( cd "$HOME/scaromare/rootbeer1/" && ( cd csrc && ./compile_linux_x64 ) && ant clean && 'rm' -f dist/Rootbeer1.jar Rootbeer.jar && ant jar && ./pack-rootbeer ) && make clean && make -B && java -jar Count.jar
    folder=$HOME/scaromare/MontePi/multiNode/multiGpu/scala
    make -C "$folder" -B ${makeOptsSpark[@]} MontePi.jar
    sparkSubmit "$folder/MontePi.jar" 268435456 2 2>&1 | sed '/ INFO /d'


# Known Bugs

  - 
  
        Exception in thread "main" java.lang.NoClassDefFoundError: org/trifort/rootbeer/runtime/Kernel

    It seems Rootbeer.jar wasn't merged into the fat JAR or specified in the class path. This may happen after the Rootbeer commit where it doesn't do that automatically anymore.

  - 
 
        16/05/23 12:01:39 ERROR FatalExceptionHandler: Exception processing: 1 org.trifort.rootbeer.runtime.GpuEvent@5a425c28
        java.lang.OutOfMemoryError: [FixedMemory.java]
           currentHeapEnd / bytes currently in use: 2670592 B
           Bytes requested to allocate            : 64 B
           total size available in FixedMemory    : 2670592 B
        (This happens if createContext(size) was called with an insufficient size, or CUDAContext.java:findMemorySize failed to determine the correct size automatically)
           at org.trifort.rootbeer.runtime.FixedMemory$MemPointer.mallocWithSize(FixedMemory.java:451)
           at org.trifort.rootbeer.runtime.FixedMemory.mallocWithSize(FixedMemory.java:306)
           at org.trifort.rootbeer.runtime.Serializer.checkWriteCache(Serializer.java:125)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:148)
           at org.trifort.rootbeer.runtime.Serializer.writeToHeap(Serializer.java:49)
           at org.trifort.rootbeer.runtime.CUDAContext.writeBlocksList(CUDAContext.java:473)
           at org.trifort.rootbeer.runtime.CUDAContext.access$1500(CUDAContext.java:19)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:408)
           at org.trifort.rootbeer.runtime.CUDAContext$GpuEventHandler.onEvent(CUDAContext.java:369)
           at com.lmax.disruptor.BatchEventProcessor.run(BatchEventProcessor.java:128)
           at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1145)
           at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:615)
           at java.lang.Thread.run(Thread.java:724)

   => This seems to be a critical bug in the automatic memory management, especially `findMemorySize`. Try to use a manual estimate for the needed context size instead. Unfortunately I can't yet give a formula or even some estimates. It shouldn't be much more than all the member variables per Kernel times the kernel count plus some constant for the kernel code.


# Compiling on Taurus

Currently the example is using raw make, without CMake or a configure or similar, so you will have to configure some paths manually, e.g. on my Taurus:

    makeOptsSpark=(
        "SPARK_ROOT=$HOME/spark-1.5.2-bin-hadoop2.6"
        "SPARK_JAR=$HOME/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar"
        "SPARKCORE_JAR=$SPARK_JAR"
        "SCALA_ROOT=$(dirname $(which scala))/../lib"
    )

Also you need to have `zipmerge` and for Rootbeer Compilation `ant` in your `PATH` variable. E.g. I installed zipmerge manually to `~/programs/bin`.

Now you can compile e.g. the example using Spark und multiple GPUs with:

    make ${makeOptsSpark[@]} -C multiNode/multiGpu/scala/


# Benchmarking

## Weak Scaling

If a Spark instance is already running and the address of the master is saved in the variable `MASTER_ADDRESS`, then you can simply run:

    nGpusAvailable=32
    jarFile=$HOME/scaromare/MontePi/multiNode/multiGpu/scala/MontePi.jar
    fname=weak-scaling_$(date +%Y-%m-%d_%H-%M-%S)
    echo '' | tee -- "$fname-cpu.log" "$fname-gpu.log"
    printf "# Total Elements   Elements Per Thread    Partitions   Time / s   Calculated Pi\n" | tee -- "$fname-cpu.dat" "$fname-gpu.dat"
    for (( nPerPartition=2**37; nPerPartition<=2**41; nPerPartition*=4 )); do
      for (( iRepetition=0; iRepetition<3; iRepetition++ )); do
        # Spark + GPU
        for (( nPartitions=1; nPartitions <= $nGpusAvailable; nPartitions++ )) do
            sparkSubmit "$jarFile" $((nPerPartition*nPartitions)) $nPartitions 2>&1 |
                tee tmp.log |             # log file to extract times in seconds from
                tee -a "$fname-gpu.log" | # complete log file for human to view
                sed '/ INFO /d'           # delete numerous Spark messages
            seconds=$(sed -nr '
                s/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
            pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)

            printf "%16i %16i %4i %10.5f %s\n" $((nPerPartition*nPartitions)) \
                $nPerPartition $nPartitions $seconds $pi >> "$fname-gpu.dat"
        done
      done
    done

![Weak Scaling Plot (run times) with up to 32 K20x graphic cards](/MontePi/benchmark/weak-scaling-time-gpu.png?raw=true)

![ibid. Speedup](/MontePi/benchmark/weak-scaling-speedup-gpu.png?raw=true)

![ibid. Parallel Efficiency](/MontePi/benchmark/weak-scaling-efficiency-gpu.png?raw=true)

### About the Performance Decrease at roughly the half the GPUs

Let's try to find out, if maybe some all the slower runs have a certain host in common which wasn't used in one of the faster rans. First let's extract the times and corresponding used hostnames using a sed script:

    sed -nr '
    /INFO/d;                    # delete spark output (more a performance thing)
    /\(taurusi[0-9]*,/{
        :findhosts
            /\) : 0[ \t]*$/{    # skip hosts which are not using GPUs
                s|\n[^\n]*$||   # delete last appended line
                b findhosts
            }
            s|[\^n]*\((taurusi[0-9]*),.*|\1|    # extract hostname
            N                                   # append next line with \n
            /\(taurusi[0-9]*,/b findhosts       # repeat if the new line contains host
        s|\n[^\n]*$||           # delete last appended line
        s|[ \t]*\n[ \t]*|,|g    # delete newlines and whitespaces between hostnames
        :pisearch
            N
            / pi ~ /!{          # if not line with pi value found
                s|\n[^\n]*$||   # delete last appended line
                b pisearch
            }
        # extract seconds needed
        s/([^\n]*).* pi ~ ([0-9.]+) and took ([0-9.]+) seconds.*/\3 \1/
        p
    }
    ' weak-scaling_2016-08-03_22-52-00-gpu.log | column -t > time-hosts.dat

Manually delete the only half-done longest run with 222s run time and then try to sort the results in fast and slow runs.

 - fast runs: `7.0 +- 0.3`, `17.3 +- 0.3`, `58.4 +- 0.3`
 - slow runs: `8.0 +- 0.3`, `18.3 +- 0.3`, `59.4 +- 0.3`

There seems to be no correlation with one straggling host.

Note that `taurusi2001`-`taurusi2009` and `taurusi2010`-`taurusi2018` respectively share one chassis switch (Gigabit Ethernet and Infiniband).

This also doesn't seem to be the reason, because almost all runs have hosts from both chassis, especially also fast runs.

Idea: Maybe the switch gets saturated for a certain amount of connections? E.g. only if more than 8 nodes per chassis are used.
 - Also does not seem to be the case.


## Strong Scaling

    nGpusAvailable=32
    jarFile=$HOME/scaromare/MontePi/multiNode/multiGpu/scala/MontePi.jar
    fname=strong-scaling_$(date +%Y-%m-%d_%H-%M-%S)
    echo '' | tee -- "$fname-cpu.log" "$fname-gpu.log"
    printf "# Total Elements    Partitions   Time / s   Calculated Pi\n" | tee -- "$fname-cpu.dat" "$fname-gpu.dat"
    for (( n=2**37; n<=2**43; n*=4 )); do
      for (( iRepetition=0; iRepetition<5; iRepetition++ )); do
        # Spark + GPU
        for (( nPartitions=1; nPartitions <= $nGpusAvailable; nPartitions++ )) do
            sparkSubmit "$jarFile" $n $nPartitions 2>&1 |
                tee tmp.log |             # log file to extract times in seconds from
                tee -a "$fname-gpu.log" | # complete log file for human to view
                sed '/ INFO /d'           # delete numerous Spark messages
            seconds=$(sed -nr '
                s/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
            pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)

            printf "%16i %4i %10.5f %s\n" $n $nPartitions $seconds $pi >> "$fname-gpu.dat"
        done
      done
    done
    
![Strong Scaling Plot (run times) with up to 32 K20x graphic cards](/MontePi/benchmark/strong-scaling-time-logscale-gpu.png?raw=true)

![ibid. Speedup](/MontePi/benchmark/strong-scaling-speedup-gpu.png?raw=true)

![ibid. Parallel Efficiency](/MontePi/benchmark/strong-scaling-efficiency-gpu.png?raw=true)


## Plots

To make the plots `MontePi/benchmark/plot.py` can be used. Try also `python plot.py --help`.

    python plot.py -s strong-scaling_2016-08-04_03-40-03 -k weak-scaling_2016-08-03_22-52-00


## Profiling

    salloc --time=02:00:00 --nodes=1 --partition=gpu2-interactive --cpus-per-task=4 --gres='gpu:4' --mem-per-cpu=1000M
    make ${makeOptsSpark[@]} -C $HOME/scaromare/MontePi/singleNode/multiGpu/scala/ MontePi.jar
    srun java -jar $HOME/scaromare/MontePi/singleNode/multiGpu/scala/MontePi.jar 32435123451 4
    which java
        /sw/global/tools/java/jdk1.7.0_25/bin/java
    sbatch --time=00:30:00 --nodes=1 --partition=gpu2 --cpus-per-task=4 --gres='gpu:4' <<EOF
    #!/bin/bash
    nvprof --analysis-metrics --metrics all -o $HOME/scaromare/MontePi/profilingDataMultiGpuScala.nvp%p /sw/global/tools/java/jdk1.7.0_25/bin/java -jar $HOME/scaromare/MontePi/singleNode/multiGpu/scala/MontePi.jar $((4*14351234510)) 4
    EOF

Output:
    
    ==26439== NVPROF is profiling process 26439, command: /sw/global/tools/java/jdk1.7.0_25/bin/java -jar $HOME/scaromare/MontePi/singleNode/multiGpu/scala/MontePi.jar 1435123451 4
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 0 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 6.281092309 seconds
    0
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:0] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 1 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 7.513381237 seconds
    0
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 2 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 7.538209838 seconds
    0
    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.748453808 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 5.4038E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.265656465 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:1] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:2] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads
    [MonteCarloPi.scala:<constructor>] Using the following GPUs : 3 
    [MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took 7.580257042 seconds
    0
    [MonteCarloPi.scala:runOnDevice] [Host:taurusi2051,GPU:3] Total Thread Count = 26624, KernelConfig = (104,1,1) blocks with each (256,1,1) threads

    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.965182109 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 4.1006E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.017713559 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.974981803 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 3.4606E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 1.004266163 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).
    [MonteCarloPi.scala:runOnDevice] runOnDevice configuration took 0.960264386 seconds
    [MonteCarloPi.scala:runOnDevice] context.run( work ) took 2.3148E-5 seconds
    [MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took 0.976993471 seconds
    [MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks).



    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.673575898 seconds
    Join thread after 9.409900961 seconds
    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.296328742 seconds
    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.3390092 seconds
    [MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took 1.326177838 seconds
    Join thread after 10.060772521 seconds
    Join thread after 10.25184158 seconds
    Join thread after 10.253567369 seconds
    Rolling the dice 1435123451 times resulted in pi ~ 3.141675075310326 and took 10.253777488 seconds
    ==26439== Generated result file: $HOME/scaromare/MontePi/profilingDataMultiGpuScala.nvp26439


# ToDo

- Benchmark Multi-GPU
  - Implement Multi-GPU using runAsync
    - try to convert MontePi.java to MontePi.scala so that it is easer to program
      - Try it out on singleNode/singleGpu/scala first

