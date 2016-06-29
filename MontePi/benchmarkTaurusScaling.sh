#!/bin/bash
# ./benchmarkTaurusScaling.sh -c 12 -g 2 -n 1
############# Strong Scaling #############

gpusPerNode=4
coresPerNode=1
nodeCounts=(1 2 4)
#nodeCounts=(1 2 4 8 16 24 32)
#nodeCounts=(32 24 16 8 4 2 1)
#nodeCounts=(24 16 8 4 2 1)
while [ ! -z "$1" ]; do
    case "$1" in
        '-g'|'--gpus-per-node')
            if [ "$2" -eq "$2" ] 2>/dev/null; then gpusPerNode=$2; shift; fi
            ;;
        '-c'|'--cores-per-node')
            if [ "$2" -eq "$2" ] 2>/dev/null; then coresPerNode=$2; shift; fi
            ;;
        '-n'|'--node-counts')
            nodeCounts=()
            while [ "$2" -eq "$2" ] 2>/dev/null; do
                nodeCounts+=( "$2" )
                shift
            done
            ;;
    esac
    shift
done
echo "Run with following node configurations : ${nodeCounts[@]}"
echo "Cores per node                         : $coresPerNode"
echo "GPUs  per node                         : $gpusPerNode"

fname=strong-scaling_$(date +%Y-%m-%d_%H-%M-%S)
echo '' > $fname-cpu.log
echo '' > $fname-gpu.log
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s\n" > "$fname-cpu.dat"
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s\n" > "$fname-gpu.dat"
for nodes in ${nodeCounts[@]}; do   # 32 nodes equals 128 GPUs
    # sets alias for sparkSubmit which includes --master
    startSpark --time=04:00:00 --nodes=$((nodes+1)) --partition=gpu2 --gres=gpu:$gpusPerNode --cpus-per-task=$coresPerNode
    for (( workPerNode=3*2**36; workPerNode<=3*2**41; workPerNode*=2 )); do
    {
        # Spark + GPU
        nPerSlice=$((workPerNode/gpusPerNode))
        for nSlices in $((nodes*gpusPerNode-gpusPerNode/2)) $((nodes*gpusPerNode)); do
            sparkSubmit ~/scaromare/MontePi/singleNode/singleGpu/multiGpuTestSpark/MontePi.jar \
                $((nPerSlice*nSlices)) $nSlices $gpusPerNode 2>/dev/null |
                tee tmp.log | tee -a "$fname-gpu.log"
            seconds=$(sed -nr 's/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
            pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)
            printf "%16i %16i %4i %4i %10.5f %s\n" $((nPerSlice*nSlices)) \
                $nPerSlice $nSlices $nodes $seconds $pi >> "$fname-gpu.dat"
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
    scancel $jobid
done
