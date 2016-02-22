#!/bin/bash


make -C singleNode/multiGpu/java
if [ -f benchmarks.log ]; then
    mv benchmarks.log benchmarks-$(date +%Y-%m-%d_%H-%M-%S).log
fi

i0=256
imax=26843545600

allDiceRolls=( )
timesSingleCoreJava=( )
timesSingleCoreScala=( )
timesSingleGpuCpp=( )
timesSingleGpuJava=( )
timesSingleGpuScala=( )

# Make all if changed
make -C singleNode/singleCore/java/
make -C singleNode/singleCore/scala/
make -C singleNode/singleGpu/cpp/
make -C singleNode/multiGpu/java/
make -C singleNode/multiGpu/scala/

getTime() {
    minutes=$(sed -nr 's/real[[:space:]]*([0-9.]+)m([0-9.]+)s.*/\1/p' $1)
    seconds=$(sed -nr 's/real[[:space:]]*([0-9.]+)m([0-9.]+)s.*/\2/p' $1)
    echo "$minutes*60+$seconds" | bc
}

benchmarksFile="benchmarks-$(date +%Y-%m-%d_%H-%M-%S)"
resultsFile="results-$(date +%Y-%m-%d_%H-%M-%S).log"
printf "nDiceRolls  |  1 Core (java)  |  1 Core (scala)  | 1 GPU (C++)  | 1 GPU (Java)  | 1 GPU (Scala)\n" | tee $resultsFile

# benchmark scaling with different work loads
for (( i=i0; i<imax; i=3*i/2  )); do
    nDiceRolls=$((i-i%256))
    allDiceRolls+=( $nDiceRolls )

    #echo "nDiceRolls = $nDiceRolls"
    # single core java
    (time java -jar singleNode/singleCore/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleCore-java.log"
    #timesSingleCoreJava+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' benchmarks.log) )
    timesSingleCoreJava+=( $(getTime tmp.log) )

    # single core scala
    (time java -jar singleNode/singleCore/scala/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleCore-scala.log"
    timesSingleCoreScala+=( $(getTime tmp.log) )

    # single Gpu C++
    #(time singleNode/singleGpu/cpp/TestMonteCarloPiV2.exe $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-Cpp.log"
    timesSingleGpuCpp+=( $(getTime tmp.log) )

    # single Gpu java
    #(time java -jar singleNode/multiGpu/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-java.log"
    timesSingleGpuJava+=( $(getTime tmp.log) )

    # single Gpu scala
    #(time scala singleNode/multiGpu/scala/MontePiGPU.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$benchmarksFile-singleNode-singleGpu-scala.log"
    timesSingleGpuScala+=( $(getTime tmp.log) )

    printf "%15i%10.5f%10.5f%10.5f%10.5f%10.5f\n" ${allDiceRolls[-1]} ${timesSingleCoreJava[-1]} ${timesSingleCoreScala[-1]} ${timesSingleGpuCpp[-1]} ${timesSingleGpuJava[-1]} ${timesSingleGpuScala[-1]} | tee -a $resultsFile
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
