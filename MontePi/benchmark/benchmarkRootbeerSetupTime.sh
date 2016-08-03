#!/bin/bash
# Let the work be constant, using 1 GPU per node for 4 nodes.
# Now double the number slices and double and double and see the runtime
# increase (for no setup cost, e.g for CPU we expect no increase)


gpusPerNode=1
coresPerNode=6
nodes=1
work=$((2**35))

fname=rootbeer-setup_$(date +%Y-%m-%d_%H-%M-%S)
echo '' > $fname-cpu.log
echo '' > $fname-gpu.log
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s   Pi\n" > "$fname-cpu.dat"
printf "# Total Elements   Elements Per Thread    Slices   Nodes   Time / s   Pi\n" > "$fname-gpu.dat"

# sets alias for sparkSubmit which includes --master
startSpark --time=01:00:00 --nodes=$((nodes+1)) --partition=gpu2 --gres=gpu:$gpusPerNode --cpus-per-task=$coresPerNode -A p_scads
for nSlices in 24 12 6 4 3 2 1; do
{
    # Spark + GPU
    sparkSubmit ~/scaromare/MontePi/multiNode/multiGpu/scala/MontePi.jar \
        $work $nSlices $gpusPerNode 2>/dev/null |
        tee tmp.log | tee -a "$fname-gpu.log"
    seconds=$(sed -nr 's/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
    pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)
    printf "%16i %16i %4i %4i %10.5f %s\n" $work $((work/nSlices)) $nSlices
        $nodes $seconds $pi >> "$fname-gpu.dat"

    # Spark + CPU
    sparkSubmit ~/scaromare/MontePi/multiNode/multiCore/MontePi.jar \
        $work $nSlices 2>/dev/null |
        tee tmp.log | tee -a "$fname-cpu.log"
    seconds=$(sed -nr 's/.*Rolling the dice.*and took (.*) seconds.*/\1/p' tmp.log)
    pi=$(sed -nr 's/.*pi ~ ([0-9.]+).*/\1/p' tmp.log)
    printf "%16i %16i %4i %4i %10.5f %s\n" $work $((work/nSlices)) $nSlices
        $nodes $seconds $pi >> "$fname-cpu.dat"
}
done
scancel $jobid
echo "Saved results to '$fname-{g,c}pu.log'"
