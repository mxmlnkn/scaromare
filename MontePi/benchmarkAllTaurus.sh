#!/bin/bash


make -C singleNode/multiGpu/java
if [ -f benchmarks.log ]; then
    mv benchmarks.log benchmarks-$(date +%Y-%m-%d_%H-%M-%S).log
fi

i0=256
imax=0 #26843545600

allDiceRolls=( )
timesSingleCoreJava=( 0)
timesSingleCoreScala=( 0 )
timesSingleGpuCpp=( )
timesSingleGpuJava=( )
timesSingleGpuScala=( )

# Make all if changed
#make -C singleNode/singleCore/java/
#make -C singleNode/singleCore/scala/
make -C singleNode/singleGpu/cpp/
make -C singleNode/multiGpu/java/
make -C singleNode/multiGpu/scala/ SCALA_ROOT=/sw/global/compilers/scala/2.10.4/lib/

getTime() {
    minutes=$(sed -nr 's/real[[:space:]]*([0-9.]+)m([0-9.]+)s.*/\1/p' $1)
    seconds=$(sed -nr 's/real[[:space:]]*([0-9.]+)m([0-9.]+)s.*/\2/p' $1)
    echo "$minutes*60+$seconds" | bc
}

benchmarksFile="benchmarks-$(date +%Y-%m-%d_%H-%M-%S)"
resultsFile="results-$(date +%Y-%m-%d_%H-%M-%S).log"
printf "#nDiceRolls  |  1 Core (java)  |  1 Core (scala)  | 1 GPU (C++)  | 1 GPU (Java)  | 1 GPU (Scala)\n" | tee $resultsFile

# benchmark scaling with different work loads
for (( i=i0; i<imax; i=3*i/2  )); do
    nDiceRolls=$((i-i%256))
    allDiceRolls+=( $nDiceRolls )

    #echo "nDiceRolls = $nDiceRolls"
    # single core java
    #(time java -jar singleNode/singleCore/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleCore-java.log"
    timesSingleCoreJava+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )
    #timesSingleCoreJava+=( $(getTime tmp.log) )

    # single core scala
    #(time java -jar singleNode/singleCore/scala/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleCore-scala.log"
    #timesSingleCoreScala+=( $(getTime tmp.log) )
    timesSingleCoreScala+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )

    # single Gpu C++
    srun --nodes=1 --ntasks=1 bash -c "(time singleNode/singleGpu/cpp/TestMonteCarloPiV2.exe $nDiceRolls 0) > tmp.log 2>&1"
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-Cpp.log"
    #timesSingleGpuCpp+=( $(getTime tmp.log) )
    timesSingleGpuCpp+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )

    # single Gpu java
    srun --nodes=1 --ntasks=1 bash -c "(time java -jar singleNode/multiGpu/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1"
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-java.log"
    #timesSingleGpuJava+=( $(getTime tmp.log) )
    timesSingleGpuJava+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )

    # single Gpu scala
    srun --nodes=1 --ntasks=1 bash -c "(time scala singleNode/multiGpu/scala/MontePiGPU.jar $nDiceRolls 0) > tmp.log 2>&1"
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-scala.log"
    #timesSingleGpuScala+=( $(getTime tmp.log) )
    timesSingleGpuScala+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )

    printf "%15i%10.5f%10.5f%10.5f%10.5f%10.5f\n" \
    ${allDiceRolls[         ${#allDiceRolls[*]}        -1 ]} \
    ${timesSingleCoreJava[  ${#timesSingleCoreJava[*]} -1 ]} \
    ${timesSingleCoreScala[ ${#timesSingleCoreScala[*]}-1 ]} \
    ${timesSingleGpuCpp[    ${#timesSingleGpuCpp[*]}   -1 ]} \
    ${timesSingleGpuJava[   ${#timesSingleGpuJava[*]}  -1 ]} \
    ${timesSingleGpuScala[  ${#timesSingleGpuScala[*]} -1 ]} | tee -a $resultsFile
done

#sed -nr 's/Rolling the dice ([0-9]+) times resulted in pi ~ ([0-9.]+) and took (.*) seconds.*/\1   \2/p' benchmarks-2016-02-22_01-34-52.log > error-scaling.log

#echo "" > results.log
#for (( i=0; i<${#timesSingleGpuCpp[*]}; i=i+1 )); do
#    printf "%10i%10.5f%10.5f%10.5f%10.5f%10.5f\n" ${allDiceRolls[i]} ${timesSingleCoreJava[i]} ${timesSingleCoreScala[i]} ${timesSingleGpuCpp[i]} ${timesSingleGpuJava[i]} ${timesSingleGpuScala[i]} >> results.log
#done
#cat results.log
#
#if [ -f benchmarks.log ]; then
#    mv benchmarks.log benchmarks-$(date +%Y-%m-%d_%H-%M-%S).log
#fi
#if [ -f results.log ]; then
#    mv results.log results-$(date +%Y-%m-%d_%H-%M-%S).log
#fi

benchmarksFile="benchmarks-gpu-scaling-$(date +%Y-%m-%d_%H-%M-%S)"
resultsFile="results-gpu-scaling-$(date +%Y-%m-%d_%H-%M-%S).log"
printf "#nDiceRolls  |  n GPUs (C++)  | n GPUs (Java)  | n GPUs (Scala)\n" | tee $resultsFile

timesSingleGpuCpp=( )
timesSingleGpuJava=( )
timesSingleGpuScala=( )

# benchmark scaling with GPU numbers without spark
for (( i=$SLURM_NPROCS; i>0; i-=1  )); do
    # single Gpu C++
    cat > tmp.slurm <<"EOF"
#!/bin/bash
nGpusPerNode=4
iGpuToUse=$((SLURM_PROCID % nGpusPerNode ))
echo $(hostname) Process ID: $SLURM_PROCID
echo GPU device to use: $iGpuToUse
time singleNode/singleGpu/cpp/TestMonteCarloPiV2.exe 2684354560 $iGpuToUse
EOF
    chmod u+x tmp.slurm
    srun --ntasks=$i tmp.slurm > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-Cpp.log"
    #timesSingleGpuCpp+=( $(getTime tmp.log) )
    timesSingleGpuCpp+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )

    # single Gpu java
    cat > tmp.slurm <<"EOF"
#!/bin/bash
nGpusPerNode=4
iGpuToUse=$((SLURM_PROCID % nGpusPerNode ))
echo $(hostname) Process ID: $SLURM_PROCID
echo GPU device to use: $iGpuToUse
java -jar singleNode/multiGpu/java/MontePi.jar 2684354560 $iGpuToUse
EOF
    srun --ntasks=$i tmp.slurm > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-java.log"
    #timesSingleGpuJava+=( $(getTime tmp.log) )
    timesSingleGpuJava+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )

    # single Gpu scala
    cat > tmp.slurm <<"EOF"
#!/bin/bash
nGpusPerNode=4
iGpuToUse=$((SLURM_PROCID % nGpusPerNode ))
echo $(hostname) Process ID: $SLURM_PROCID
echo GPU device to use: $iGpuToUse
cd singleNode/multiGpu/scala/
scala MontePiGPU.jar 2684354560 $iGpuToUse
EOF
    srun --ntasks=$i tmp.slurm > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-scala.log"
    #timesSingleGpuScala+=( $(getTime tmp.log) )
    timesSingleGpuScala+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log) )

    printf "%4i%10.5f%10.5f%10.5f\n" \
    $i \
    ${timesSingleGpuCpp[    ${#timesSingleGpuCpp[*]}   -1 ]} \
    ${timesSingleGpuJava[   ${#timesSingleGpuJava[*]}  -1 ]} \
    ${timesSingleGpuScala[  ${#timesSingleGpuScala[*]} -1 ]} | tee -a $resultsFile
done
