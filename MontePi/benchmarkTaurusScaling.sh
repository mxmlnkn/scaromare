#!/bin/bash

############# Strong Scaling #############

fname=strong-scaling_$(date +%Y-%m-%d_%H-%M-%S)
echo '' > $fname-cpu.log
echo '' > $fname-gpu.log
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s\n" > "$fname-cpu.dat"
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s\n" > "$fname-gpu.dat"
for nodes in 1 2 4 8 16 24 32; do   # 32 nodes equals 128 GPUs
    gpusPerNode=4
    startSpark --time=04:00:00 --nodes=$((nodes+1)) --partition=gpu2 --gres=gpu:$gpusPerNode --cpus-per-task=$gpusPerNode
    for (( nPerSlice=2147483648/16; nPerSlice<=2147483648*4; nPerSlice*=2 )); do
        for (( nSlices = (nodes-1)*gpusPerNode; nSlices <= nodes*gpusPerNode+2; nSlices++ )); do
            # Spark + GPU
            sparkSubmit ~/scaromare/MontePi/multiNode/multiGpu/scala/MontePi.jar $((nPerSlice*nSlices)) $nSlices $gpusPerNode 2>/dev/null | tee tmp.log | tee -a "$fname-gpu.log"
            seconds=$(sed -nr 's/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
            printf "%16i %16i %4i %4i %10.5f\n" $((nPerSlice*nSlices)) $nPerSlice $nSlices $nodes $seconds >> "$fname-gpu.dat"
            # Spark + CPU
            sparkSubmit ~/scaromare/MontePi/multiNode/multiCore/MontePi.jar $((nPerSlice*nSlices)) $nSlices 2>/dev/null | tee tmp.log | tee -a "$fname-cpu.log"
            seconds=$(sed -nr 's/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
            printf "%16i %16i %4i %4i %10.5f\n" $((nPerSlice*nSlices)) $nPerSlice $nSlices $nodes $seconds >> "$fname-cpu.dat"
        done
    done
    scancel $jobid
done
