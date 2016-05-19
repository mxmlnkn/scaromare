#!/bin/bash

# E.g. use this in
#   salloc -p gpu2-interactive --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 --time=1:00:00
# or
#   sbatch -p gpu2 --nodes=1 --ntasks-per-node=1 --cpus-per-task=1 --gres=gpu:1 --time=01:00:00 ./benchmarkAll.sh 1
taurus=$1
if [ -z "$taurus" ]; then
    taurus=0
    gpusPerNode=1
else
    taurus=1
    gpusPerNode=4
fi
RUN="time -p"
if [ "$taurus" -eq 1 ]; then RUN="srun $RUN"; fi

i0=256
imax=268435456000
tmaxCpu=100 # seconds
tmaxGpu=7

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
    # POSIX format has no 's' suffix but it is in seconds. In contrast
    # to the time built-in of bash there also is no 1m2s format
    # 'time' -p sleep 122s
    #   real 122.00
    #   user 0.00
    #   sys 0.00
    sed -nr 's/real[[:space:]]*([0-9.]+).*/\1/p' $1
}

logFolder="./raw-logs/benchmarks-impl-$HOSTNAME-$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$logFolder"
resultsFile="$logFolder/results.log"
echo "You can find the output in '$resultsFile'"
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

    # because a[-1] does not work in older bash versions and also to
    # cope with indirections
    # after 1st eval: echo ${s[${#s[@]}-1]}
    # Note that inside this script one eval could be saved by using "
    # instead of ', but that wouldn't work when copy-pasting into a shell!
    function last() {
        if [ $(eval eval 'echo \${#$1[@]}') -gt 0 ]; then
            eval eval 'echo \${$1[\${#$1[@]}-1]}'
        fi
    }
    function append() {
        x=$1
        shift
        while [ ! -z "$1" ]; do
            eval eval '$x[ \${#$x[@]} ]=\$1'
            shift
        done
    }

    function measure() {
        # $1: array name where to append result, also covers as log-name
        # $2: command to run, e.g. 'java -jar ...'
        # $3: maximum time before stopping benchmark for good
        # $4: GPU device to use
        lastElem=$(last $1)
        if [ -z $lastElem ] ||
           ( [ $( echo "$lastElem < $3" | bc ) -eq 1 ] && # 1 means true!
             [ ! $lastElem == -1 ] )
        then
            # without eval instead of the built-in time command /usr/bin/time
            # would be used! -> works bash 4.3, but not in 4.1 -.-
            # Therefore just use 'time' but with -p (posix) which has the
            # same output as bash
            ( $RUN $2 $nDiceRolls $4 ) > tmp.log 2>&1
            cat tmp.log >> "$logFolder/$1.log"
            append $1 $(getTime tmp.log)
        else
            append $1 -1
        fi
    }

    #echo "nDiceRolls = $nDiceRolls"
    #measure timesSingleCoreJava  'java -jar singleNode/singleCore/java/MontePi.jar'  $tmaxCpu
    #measure timesSingleCoreScala 'java -jar singleNode/singleCore/scala/MontePi.jar' $tmaxCpu
    measure timesSingleGpuCpp    'singleNode/singleGpu/cpp/TestMonteCarloPiV2.exe'   $tmaxGpu 0
    measure timesSingleGpuJava   'java -jar singleNode/multiGpu/java/MontePi.jar'    $tmaxGpu 0
    measure timesSingleGpuScala  'java -jar singleNode/multiGpu/scala/MontePi.jar'   $tmaxGpu 0

    # use last instead of [-1] because latter doesn't work on taurus -.-
    printf "%15i%10.5f%10.5f%10.5f%10.5f%10.5f\n" \
        $(last allDiceRolls)         \
        $(last timesSingleCoreJava)  \
        $(last timesSingleCoreScala) \
        $(last timesSingleGpuCpp)    \
        $(last timesSingleGpuJava)   \
        $(last timesSingleGpuScala) | tee -a $resultsFile
done
