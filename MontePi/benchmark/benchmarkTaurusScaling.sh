#!/bin/bash
#
# e.g. for Taurus gpu2 queue (K80)
# ./benchmarkTaurusScaling.sh -c 1 -g 4 -n 1
# e.g. for Taurus gpu1 queue (K20x)
# ./benchmarkTaurusScaling.sh -p gpu1 -c 1 -g 2 -n 1 2
#
############# Strong Scaling #############

gpusPerNode=4
nodeCounts=(1 2 4)
partition=gpu2
#nodeCounts=(1 2 4 8 16 24 32)
#nodeCounts=(32 24 16 8 4 2 1)
#nodeCounts=(24 16 8 4 2 1)
dryrun=
while [ ! -z "$1" ]; do
    case "$1" in
        '-g'|'--gpus-per-node')
            if [ "$2" -eq "$2" ] 2>/dev/null; then gpusPerNode=$2; shift; fi
            ;;
        '-p'|'--partition')
            partition=$2
            shift
            ;;
        '-n'|'--node-counts')
            nodeCounts=()
            # while argument is a number
            while [ "$2" -eq "$2" ] 2>/dev/null; do
                nodeCounts+=( "$2" )
                shift
            done
            ;;
        '-r'|'--dry-run')
            dryrun='echo'
            ;;
    esac
    shift
done
jarFile=$(pwd)/multiNode/multiGpu/scala/MontePi.jar
if [ ! -f "$jarFile" ]; then
    echo "Couldn't find '$jarFile', please check path!"
    exit 1
fi
echo "Run with following node configurations : ${nodeCounts[@]}"
echo "GPUs  per node                         : $gpusPerNode"
echo "Using '$jarFile' <nTotalIterations> <nSlices i.e. parallelization> <gpusPerNode (@todo: let process get this metric itself, necessary for heterogenous clusters)>"

fname=strong-scaling_$(date +%Y-%m-%d_%H-%M-%S)
echo '' > $fname-cpu.log
echo '' > $fname-gpu.log
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s\n" > "$fname-cpu.dat"
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s\n" > "$fname-gpu.dat"
for nodes in ${nodeCounts[@]}; do   # 32 nodes equals 128 GPUs
{
    # sets alias for sparkSubmit which includes --master
    $dryrun startSpark                  \
        --time=04:00:00                 \
        --nodes=$((nodes+1))            \
        --partition="$partition"        \
        --gres=gpu:"$gpusPerNode"       \
        --cpus-per-task="$gpusPerNode"

    for (( workPerNode=3*2**36; workPerNode<=3*2**40; workPerNode*=4 )); do
    {
        # Spark + GPU
        nPerSlice=$((workPerNode/gpusPerNode))
        for nSlices in $((nodes*gpusPerNode-gpusPerNode/2)) $((nodes*gpusPerNode)); do
        {
            # start job and wait for it to finish
            $dryrun sparkSubmit "$jarFile" \
                    $((nPerSlice*nSlices)) \
                    $nSlices               \
                    $gpusPerNode           \
                2>&1 |
                tee tmp.log |  # log file to extract times in seconds from
                tee -a "$fname-gpu.log" # complete log file for human to view

            seconds=$(sed -nr '
                        s/.*Rolling the dice.*and took (.*) seconds.*/\1/p
                      ' tmp.log)
            pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)

            printf "%16i %16i %4i %4i %10.5f %s\n" $((nPerSlice*nSlices)) \
                $nPerSlice $nSlices $nodes $seconds $pi >> "$fname-gpu.dat"
        }
        done

        # Spark + CPU
        #nPerSlice=$((workPerNode/coresPerNode))
        #if [ $nPerSlice -gt $((2**35)) ]; then continue; fi
        #for nSlices in $((nodes*coresPerNode-coresPerNode/2)) $((nodes*coresPerNode)); do
        #    echo sparkSubmit ~/scaromare/MontePi/multiNode/multiCore/MontePi.jar \
        #        $((nPerSlice*nSlices)) $nSlices 2>/dev/null |
        #        tee tmp.log | tee -a "$fname-cpu.log"
        #    seconds=$(sed -nr 's/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
        #    pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)
        #    printf "%16i %16i %4i %4i %10.5f %s\n" $((nPerSlice*nSlices)) \
        #        $nPerSlice $nSlices $nodes $seconds $pi >> "$fname-cpu.dat"
        #done
    }
    done
    $dryrun scancel $jobid
}
done
