
object TestMonteCarloPi
{
    def main( args : Array[String] ) =
    {
        /* args[0] is not the path as can be shwon with the code below.
         * This behavior is different from C++ */
        //for ( i <- 0 until args.length )
        //    println( "args["+i+"] = "+args(i).toString );
        val nDiceRolls = if ( args.length > 0 ) args(0).toLong else 0l
        val iGpuToUse  = if ( args.length > 1 ) args(1).toInt else 0
        var piCalculator = new MonteCarloPi( iGpuToUse )

        /* execute and time pi calculation */
        val t0 = System.nanoTime()
            /* Max Threads per Device which can run: 12288 on GTX 760.
             * Because of oversubscription even though we have no memory
             * latency only 12288 / 32 (warp size) = 384 are actually running.
             * That's true, but even arithmetic operations have latency,
             * meaning running the full possible 12288 gets an improvement of
             *   3.70s down to 0.65s for 2684354560 rolls.
             *   0.50s down to 0.41s for 268435456 rolls.
             * The problem is, how to choose this automatically ... Future
             * GPUs will possibly support more, but more than allowed is
             * also not bad until the overhead e.g. for the seed generator
             * gets too large.
             * The number of threads which are actually usefull could
             * reduced by reducing data dependency inside the kernels, so
             * that instructions inside on thread could be pipelined instead
             * of instructions only from different threads.
             * The C++ version takes
             *   0.124191 for 268435456 rolls
             *   1.120284 for 2684354560 rolls
             *   1.971345 for 4684354560 rolls
             *   3.623924 for 8684354560 rolls
             * So while C++ is faster for few rolls by factor 4 it's slower
             * for higher rolls by factor 3. I can't explain neither !!!

             * The above problems did appear, because of this annoying unfixed
             * bug: https://github.com/pcpratts/rootbeer1/issues/168
             * 384  threads 2684354560 rolls: 3.806618433
             * 384  threads 4684354560 rolls: 6.525701724
             * 8192 threads 2684354560 rolls: 0.792929928
             * 8192 threads 4684354560 rolls: 1.133301676
             * 8192 threads 8684354560 rolls: 1.731940062
             * 8192 threads 46843545600 rolls: 7.985081616
             *
             * The timescaling is roughly equivalent to pi/4 * dice rolls
             * in both teh C++ and the Rootbeer version.
             * I can't explain why the Rootbeer version is almost twice as
             * fast ... it sounds like quite some optimization. Would have
             * to look at the PTX code if possible ...
             * NVVP shows almost 100% utilization of which almost 95% are
             * arithmetic instructions of which ~65% are integer instructions
             * there are some Misc. instructions not explained, but it seems
             * to me like there is no improvement for a factor 2 Oo ???!!!
             *
             * After declaring DEVICE_RAND_MAX constexpr or just inlining that
             * value we get a proper runtime with C++:
             *    2684354560  took 0.358401
             *    4684354560  took 0.597786
             *    8684354560  took 1.088308
             *    26843545600 took 3.344195
             *    46843545600 took 5.789622
             *
             * meaning the C++ version is roughly 1.5 times faster. This can
             * be almost exactly attributed to the higher possible occupancy.
             * Because of that bug in Rootbeer I can only reach 66% occupancy.
             * Meaning it should take roughly 1.5 the time ..
             * Fewer stalls with submaximum occupancy could be reached by
             * reducing the data dependency on the random number genertor.
             * E.g. by giving the kernel 2 seeds so that two random number
             * generators could be run in parallel.
             **/ // 12288
            val pi = piCalculator.calc( nDiceRolls, -1 /* threads per device */ );
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9;

        println( "Rolling the dice " + nDiceRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds" );
    }
}
