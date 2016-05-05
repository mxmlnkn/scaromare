#!/bin/bash



i0=256
imax=268435456000
tmaxCpu=100 # seconds
tmaxGpu=10

allDiceRolls=( )
timesSingleCoreJava=( )
timesSingleCoreScala=( )
timesSingleGpuCpp=( )
timesSingleGpuJava=( )
timesSingleGpuScala=( )

trap "rm tmp.log" EXIT

# Make all if changed
#make -C singleNode/singleCore/java/
#make -C singleNode/singleCore/scala/
#make -C singleNode/singleGpu/cpp/
#make -C singleNode/multiGpu/java/
#make -C singleNode/multiGpu/scala/

getTime() {
    minutes=( $(sed -nr 's/real[[:space:]]*([0-9.]+)m([0-9.]+)s.*/\1/p' $1) )
    seconds=( $(sed -nr 's/real[[:space:]]*([0-9.]+)m([0-9.]+)s.*/\2/p' $1) )
    for (( i=0; i<${#seconds[@]}; i++ )); do
        echo "${minutes[i]}*60+${seconds[i]}" | bc
    done
}

logFolder="./raw-logs/benchmarks-$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$logFolder"
resultsFile="$logFolder/results.log"
printf "nDiceRolls  |  1 Core (java)  |  1 Core (scala)  | 1 GPU (C++)  | 1 GPU (Java)  | 1 GPU (Scala)\n" | tee $resultsFile

nRollsList=( $(python <<EOF
from numpy import *
values=(10**linspace( log10(26624), log10(123456789000), 500 )).astype(int64)
values=array([ ( (x + 26624 - 1 ) / 26624 ) * 26624 for x in values ])
print ' '.join( str(x) for x in unique( values ) )
EOF
) )
# command for taurus interactive
#    echo '' > tmp.log && cat > tmp.sh <<EOF
#    #!/bin/bash
#    for nRolls in ${nRollsList[@]}; do
#        ./TestMonteCarloPiV2.exe \$nRolls 0 | tee tmp.log
#    done
#    EOF
#    chmod u+x tmp.sh
#    srun ./tmp.sh

# benchmark scaling with different work loads until imax, but only to maximum
# time so that we can still benchmark GPU versions while CPU is long unfeasible
for (( i=i0; i<imax; i=3*i/2  )); do
    nDiceRolls=$((i-i%256))
    allDiceRolls+=( $nDiceRolls )

    #echo "nDiceRolls = $nDiceRolls"
    # single core java
    if [ ${#timesSingleCoreJava[@]} -eq 0 ] ||
       ( [ $( echo "${timesSingleCoreJava[-1]} < $tmaxCpu" | bc ) -eq 1 ] && # 1 means true!
         [ ! ${timesSingleCoreJava[-1]} == -1 ] )
    then
        (time java -jar singleNode/singleCore/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
        cat tmp.log >> "$logFolder/singleNode-singleCore-java.log"
        timesSingleCoreJava+=( $(getTime tmp.log) )
    else
        timesSingleCoreJava+=( -1 )
    fi

    # single core scala
    if [ ${#timesSingleCoreScala[@]} -eq 0 ] ||
       ( [ $( echo "${timesSingleCoreScala[-1]} < $tmaxCpu" | bc ) -eq 1 ] && # 1 means true!
         [ ! ${timesSingleCoreScala[-1]} == -1 ] )
    then
        (time java -jar singleNode/singleCore/scala/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
        cat tmp.log >> "$logFolder/singleNode-singleCore-scala.log"
        timesSingleCoreScala+=( $(getTime tmp.log) )
    else
        timesSingleCoreScala+=( -1 )
    fi

    # single Gpu C++
    if [ ${#timesSingleGpuCpp[@]} -eq 0 ] ||
       ( [ $( echo "${timesSingleGpuCpp[-1]} < $tmaxGpu" | bc ) -eq 1 ] && # 1 means true!
         [ ! ${timesSingleGpuCpp[-1]} == -1 ] )
    then
        (time singleNode/singleGpu/cpp/TestMonteCarloPiV2.exe $nDiceRolls 0) > tmp.log 2>&1
        cat tmp.log >> "$logFolder/singleNode-singleGpu-Cpp.log"
        timesSingleGpuCpp+=( $(getTime tmp.log) )
    else
        timesSingleGpuCpp+=( -1 )
    fi

    # single Gpu java
    if [ ${#timesSingleGpuJava[@]} -eq 0 ] ||
       ( [ $( echo "${timesSingleGpuJava[-1]} < $tmaxGpu" | bc ) -eq 1 ] && # 1 means true!
         [ ! ${timesSingleGpuJava[-1]} == -1 ] )
    then
        (time java -jar singleNode/multiGpu/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
        cat tmp.log >> "$logFolder/singleNode-singleGpu-java.log"
        timesSingleGpuJava+=( $(getTime tmp.log) )
    else
        timesSingleGpuJava+=( -1 )
    fi

    # single Gpu scala
    if [ ${#timesSingleGpuScala[@]} -eq 0 ] ||
       ( [ $( echo "${timesSingleGpuScala[-1]} < $tmaxGpu" | bc ) -eq 1 ] && # 1 means true!
         [ ! ${timesSingleGpuScala[-1]} == -1 ] )
    then
        (time java -jar singleNode/multiGpu/scala/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
        cat tmp.log >> "$logFolder/singleNode-singleGpu-scala.log"
        timesSingleGpuScala+=( $(getTime tmp.log) )
    else
        timesSingleGpuScala+=( -1 )
    fi

    printf "%15i%10.5f%10.5f%10.5f%10.5f%10.5f\n" ${allDiceRolls[-1]} ${timesSingleCoreJava[-1]} ${timesSingleCoreScala[-1]} ${timesSingleGpuCpp[-1]} ${timesSingleGpuJava[-1]} ${timesSingleGpuScala[-1]} | tee -a $resultsFile
done

sed -nr 's/Rolling the dice ([0-9]+) times resulted in pi ~ ([0-9.]+) and took (.*) seconds.*/\1   \2/p' "$logFolder/singleNode-singleGpu-java.log" > "$logFolder/error-scaling.log"

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

exit

# benchmark scaling with different work loads
for (( i=i0; i<imax; i=3*i/2  )); do
    nDiceRolls=$((i-i%256))
    allDiceRolls+=( $nDiceRolls )

    #echo "nDiceRolls = $nDiceRolls"
    # single core java
    (time java -jar singleNode/singleCore/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$logFolder/singleNode-singleCore-java.log"
    #timesSingleCoreJava+=( $(sed -nr 's/Rolling the dice.*and took (.*) seconds.*/\1/p' benchmarks.log) )
    timesSingleCoreJava+=( $(getTime tmp.log) )

    # single core scala
    (time java -jar singleNode/singleCore/scala/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$logFolder/singleNode-singleCore-scala.log"
    timesSingleCoreScala+=( $(getTime tmp.log) )

    # single Gpu C++
    #(time singleNode/singleGpu/cpp/TestMonteCarloPiV2.exe $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$logFolder/singleNode-singleGpu-Cpp.log"
    timesSingleGpuCpp+=( $(getTime tmp.log) )

    # single Gpu java
    #(time java -jar singleNode/multiGpu/java/MontePi.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$logFolder/singleNode-singleGpu-java.log"
    timesSingleGpuJava+=( $(getTime tmp.log) )

    # single Gpu scala
    #(time scala singleNode/multiGpu/scala/MontePiGPU.jar $nDiceRolls 0) > tmp.log 2>&1
    cat tmp.log >> "$logFolder/singleNode-singleGpu-scala.log"
    timesSingleGpuScala+=( $(getTime tmp.log) )

    printf "%15i%10.5f%10.5f%10.5f%10.5f%10.5f\n" ${allDiceRolls[-1]} ${timesSingleCoreJava[-1]} ${timesSingleCoreScala[-1]} ${timesSingleGpuCpp[-1]} ${timesSingleGpuJava[-1]} ${timesSingleGpuScala[-1]} | tee -a $resultsFile
done
    cat > tmp.slurm <<"EOF"
#!/bin/bash
nGpusPerNode=4
iGpuToUse=$((SLURM_PROCID % nGpusPerNode ))
echo $(hostname) Process ID: $SLURM_PROCID
echo GPU device to use: $iGpuToUse
./TestMonteCarloPiV2.exe 2684354560 $iGpuToUse
EOF
    chmod u+x tmp.slurm
    srun tmp.slurm

        taurusi2094 Process ID = 6
        taurusi2094 Process ID = 5
        taurusi2094 Process ID = 4
        taurusi2094 Process ID = 7
        taurusi2093 Process ID = 0
        taurusi2093 Process ID = 1
        taurusi2093 Process ID = 2
        taurusi2093 Process ID = 3

    cat > tmp.slurm <<"EOF"
#!/bin/bash
nGpusPerNode=4
iGpuToUse=$((SLURM_PROCID % nGpusPerNode ))
echo $(hostname) Process ID: $SLURM_PROCID
echo GPU device to use: $iGpuToUse
java -jar MontePi.jar 2684354560 $iGpuToUse
EOF

    srun tmp.slurm
