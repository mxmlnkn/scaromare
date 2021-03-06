
import scala.collection.JavaConversions._
import org.trifort.rootbeer.runtime.{Kernel, Rootbeer, GpuDevice, Context, ThreadConfig, GpuFuture}
import org.trifort.rootbeer.configuration.RootbeerPaths;
import java.net.InetAddress


/**
 * This class starts the actual CUDA-Kernels and also calculates random seeds
 * for those kernels
 **/
class MonteCarloPi( iGpusToUse : Array[Int] = null )
{
    private val t0 = System.nanoTime

    private val mRootbeerContext  = new Rootbeer()
    private val mAvailableDevices = mRootbeerContext.getDevices()
    private var miGpusToUse       = iGpusToUse
    if ( miGpusToUse == null || miGpusToUse.size == 0 )
        miGpusToUse = List.range( 0, mAvailableDevices.size ).toArray
    else
    {
        assert( miGpusToUse.size <= mAvailableDevices.size )
        for ( iGpu <- miGpusToUse )
            assert( iGpu < mAvailableDevices.size )
    }

    var output = ""

    output += "[MonteCarloPi.scala:<constructor>] Using the following GPUs : " +
              miGpusToUse.map( x => x + " " ).reduce( _ + _ ) + "\n"
    private val t1 = System.nanoTime
    output += "[MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took " +
              ((t1-t0)/1e9) + " seconds\n"
    print( output )

    /******* End of Constructor *******/

    /**
     * Partitions a given interval into subintervals and returns some for
     * riKernel.
     * Note that MonteCarloPiKernel doesn't yet support a 64-bit
     * random number generator, meaning 64-bit seeds are useless if not
     * negative, because some kernels could be started with the same seeds.
     * Of course for a pure performance benchmark it doesn't really matter,
     * but for the error scaling it may very well matter.
     */
    def calcRandomSeed(
        rnKernels : Long,
        riKernel  : Long,
        rSeed0    : Long,
        rSeed1    : Long
    ) : Long =
    {
        assert( riKernel < rnKernels )
        var dSeed = rSeed1 - rSeed0
        if ( dSeed < 0 )
            dSeed += Long.MaxValue
        var x = rSeed0 + ( dSeed.toDouble / rnKernels * riKernel ).toLong
        if (x < 0)
            x += Long.MaxValue
        return x
    }

    /**
     * This routine automatically chooses a GPU device and manually sets the
     * Kernel configuration.
     */
    def runOnDevice(
        device : GpuDevice,
        work   : List[Kernel]
    ) : Tuple2[ Context, GpuFuture ] =
    {
        val t00 = System.nanoTime

        val context = device.createContext(
            ( work.size * 2 /* nIteration and nHits List */ * 8 /* sizeof(Long) */ +
              work.size * 4 /* sizeof(exception) ??? */ ) * 4 /* empirical factor */ +
              64*1024*1024 /* safety padding */
        )
        context.useCheckedMemory
        /* After some bisection on a node with two K80 GPUs (26624 max. threads)
         * I found 2129920 B to be too few and 2129920+1024 sufficient.
         * The later means
         *     safety padding is : 1598464 B
         *      calculated size  :  532480 B = max.Threads * ( 2 * 8 + 4 )
         * The total size needed is therefore almost exactly 4 times the size
         * calculated!
         * Further tests show that max.Threads * ( 2 * 8 + 4 ) * 4 + pad
         * works for pad = 192, but not for 191
         * K20x (28672 max. threads) failed with the pad of 192
         * (total size: 2293952 B) , a pad of 1024*1024 worked, though.
         **/

        /* Do our own configuration, because BlockShaper has some serious
         * flaws, at least performance-wise */
        val threadsPerBlock = 256;
        val thread_config = new ThreadConfig(
            threadsPerBlock, /* threadCountX */
            1,               /* threadCountY */
            1,               /* threadCountZ */
            ( work.size + threadsPerBlock - 1 ) / threadsPerBlock, /* blockCountX */
            1,               /* blockCountY */
            work.size        /* numThreads */
        );
        assert( thread_config.getThreadCountX() * thread_config.getBlockCountX() >= work.size() );

        println( "[MonteCarloPi.scala:runOnDevice] " +
                 "[Host:" + InetAddress.getLocalHost.getHostName +
                 ",GPU:" + context.getDevice.getDeviceId + "] " +
                 "Total Thread Count = " + thread_config.getNumThreads + ", " +
                 "KernelConfig = (" +
                     thread_config.getBlockCountX + "," +
                     thread_config.getBlockCountY + ",1) blocks " +
                    /* for some reason only getBlockCountZ is missing in Rootbeer -.- */
                 "with each (" +
                    thread_config.getThreadCountX + "," +
                    thread_config.getThreadCountY + "," +
                    thread_config.getThreadCountZ + ") threads" );

        /* This part was taken from Rootbeer.java:run */
        context.setThreadConfig( thread_config )
        context.setKernel( work.get(0) )
        context.setUsingHandles( true )
        context.buildState()

        val t01 = System.nanoTime
        println( "[MonteCarloPi.scala:runOnDevice] runOnDevice configuration took " +
                ((t01-t00)/1e9) + " seconds" )

        val t10 = System.nanoTime
        val runWaitEvent = context.runAsync( work )
        val t11 = System.nanoTime
        println( "[MonteCarloPi.scala:runOnDevice] context.run( work ) took " +
                ((t11-t10)/1e9) + " seconds" )

        return ( context, runWaitEvent )
    }

    /**
     * prepares Rootbeer kernels and starts them. Only seeds in the range
     * [rSeed0,rSeed1) will be used for the kernels. By specifying different
     * ranges for different processes this function can be used in a thread-
     * safe manner
     **/
    def calc( nDiceRolls : Long, rSeed0 : Long, rSeed1 : Long ) : Double =
    {
        val distributor = new Distribute

        val t00 = System.nanoTime

        /* start as many threads as possible. (More are possible, but
         * they wouldn't be hardware multithreaded anymore, but
         * they would be executed in serial after the first maxThreads
         * batch finished */
        /* { 0, ..., nGpusToUse-1 } -> ( GpuDevice, GpuDevice, ... }
         * prefix l stands for list */
        val lGpuDevices = miGpusToUse.map( mAvailableDevices.get(_) )
        val lnKernelsPerGpu = lGpuDevices.map( device =>
                                 device.getMultiProcessorCount *
                                 device.getMaxThreadsPerMultiprocessor
                             )
        val nKernelsTotal = lnKernelsPerGpu.sum

        //long[] nHitsA = new long[nKernels];
        //long[] nHitsB = new long[nKernels];
        /* Scala arrays correspond one-to-one to Java arrays. That is, a Scala
         * array Array[Int] is represented as a Java int[] */
        var lnHits       = lnKernelsPerGpu.map( List.fill[Long](_)(0).toArray )
        var lnIterations = lnKernelsPerGpu.map( List.fill[Long](_)(0).toArray )

        /* first distribute work to each GPU accordingly. Then distribute
         * on each GPU the work to the Kernel configuration and calculate
         * corresponding seeds for each kernel */
        val lnWorkPerGpu = distributor.distributeWeighted( nDiceRolls,
                                            lnKernelsPerGpu.map( _.toDouble ) )

        /**************************** Debug Output ****************************/
        // var output = ""
        // output += "[MonteCarloPi.scala:calc] Running MonteCarlo on " +
        //           miGpusToUse.size + " GPUs with these maximum kernel " +
        //           "configurations : \n" +
        //           "| " + lnKernelsPerGpu.map( x => "" + x + " " ).reduce(_+_) + "\n"
        //           "| with each these workloads / number of iterations :\n" +
        //           "|     " + lnWorkPerGpu.map( x => "" + x + " " ).reduce(_+_) + "\n"
        // print( output );
        // val nKernelsToShow = 4
        // output = "[MonteCarloPi.scala:calc] These are the seeds for the first " +
        //          nKernelsToShow + " kernels:\n"

        /* for each device create Rootbeer-Kernels */
        var runStates = List[ Tuple2[ Context, GpuFuture ] ]()
        /*   iGpu              nRollsPerKernel
         * {  0, 1, 2,... } -> { {4,3,3},      {4,3,3}, ... } */
        for ( iGpu <- 0 until lGpuDevices.size )
        {
            // output += "|   GPU " + iGpu + " which runs " +
            //           lnKernelsPerGpu(iGpu) + " kernels and a total of " +
            //           lnWorkPerGpu(iGpu) + " iterations: "

            /* for each GPU distributed work to each kernel */
            /* distributes e.g. 10 on n=3 to (4,3,3) */
            val lnWorkPerKernel = distributor.distribute(
                                      lnWorkPerGpu(iGpu),
                                      lnKernelsPerGpu(iGpu)
                                  )
            /* map each kernel to a seed (use zipWithIndex only to be almost
             * exact like the non-debug loop following which needs the index
             * i.e. the kernel rank to call MonteCarloPiKernel */
            val tasks = lnWorkPerKernel.zipWithIndex.map( x => {
                val nWorkPerKernel = x._1
                val iKernel        = x._2
                assert( iKernel < lnKernelsPerGpu(iGpu) )
                val nPreviousKernels = lnKernelsPerGpu.slice(0,iGpu).sum
                val kernelSeed = calcRandomSeed(
                                     lnKernelsPerGpu.sum,
                                     nPreviousKernels + iKernel,
                                     rSeed0, rSeed1
                                 )
                /* Debug output */
                // if ( iKernel < (nKernelsToShow+1)/2 ||
                //      ( lnKernelsPerGpu(iGpu)-1 - iKernel ) < nKernelsToShow/2 )
                // {
                //     /* show toInt seed, because the seed will be cast to int
                //      * in MonteCarloPiKernel */
                //     val seed = Math.abs( kernelSeed.toInt )
                //
                //     output += "|    MonteCarloPiKernel( " +
                //         //lnHits(iGpu)       + ", "
                //         //lnIterations(iGpu) + ", "
                //         "iKernel:"     + iKernel        + ", " +
                //         "seed:"        + seed           + ", " +
                //         "nIterations:" + nWorkPerKernel + " )\n"
                // }
                // if ( iKernel == 0 )
                // {
                //     println( "+---- MonteCarloPiKernel(0):\n" +
                //              "| long[] mnHits           = " + lnHits(iGpu)       + "\n" +
                //              "| long[] mnIterations     = " + lnIterations(iGpu) + "\n" +
                //              "| int    miLinearThreadId = " + iKernel            + "\n" +
                //              "| long   mRandomSeed      = " + kernelSeed         + "\n" +
                //              "| long   mnDiceRolls      = " + nWorkPerKernel     + "\n" );
                // }

                /* return */
                new MonteCarloPiKernel(
                    lnHits(iGpu)      ,
                    lnIterations(iGpu),
                    iKernel           , /* rank on GPU */
                    kernelSeed        ,
                    nWorkPerKernel      /* iterations to do */
                )
            } )

            runStates +:= runOnDevice( lGpuDevices(iGpu), tasks )
        }
        val t01 = System.nanoTime
        println( "[MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took " + ((t01-t00)/1e9) + " seconds" )

        // /* Test if random seed and global iRank is unique:
        //  *   iGpu -> kernels -> kernel+GPU rank
        //  * rank = sum of kernels on lower GPUs + kernel rank on this GPU */
        // val testKernelHostRanks = List.range( 0, lGpuDevices.size ).map( iGpu => {
        //     distributor.distribute( lnWorkPerGpu(iGpu), lnKernelsPerGpu(iGpu) ).
        //     zipWithIndex.map( z => {
        //         lnKernelsPerGpu.slice(0,iGpu).sum + z._2
        //     } )
        // } )
        // println( "[MonteCarloPi.scala:calc] This is the list of kernel ranks for this host (one line per GPU) : " )
        // testKernelHostRanks.foreach( gpu => {
        //     print( "[MonteCarloPi.scala:calc]     " )
        //     gpu.slice(0,10).foreach( rank => print( rank+" " ) )
        //     println( "..." )
        // } )
        // assert( testKernelHostRanks.distinct.size == lGpuDevices.size )

        val t10 = System.nanoTime
        println( "[MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks)." )
        for ( x <- runStates )
            x._2.take
        val t11 = System.nanoTime
        println( "[MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took " + ((t11-t10)/1e9) + " seconds" )

        /* Cumulate all the hits from the kernels. Divide each hit count by
         * number of rolls first and work with double then. This evades
         * integer overflows by maybe being less exact */
        val quarterPi = lnHits.flatten.map( _.toDouble / nDiceRolls ).sum

        /* Count and check iterations done in total by kernels */
        // println( "[MonteCarloPi.scala:calc] iterations actually done : " + lnIterations.flatten.sum )
        // println( "[MonteCarloPi.scala:calc] iterations done per kernel : " )
        // lnIterations.foreach( x => {
        //     print( x.slice(0,20).map( "  " + _ + "\n" ).reduce(_+_) )
        //     println( "..." )
        //     print( x.slice( x.length-20, x.length ).map( "  " + _ + "\n" ).reduce(_+_) )
        // } )
        // if ( ! ( lnIterations.flatten.sum == nDiceRolls ) )
        // {
        //     val lnWorkPerKernel = distributor.distribute(
        //                               lnWorkPerGpu(0),
        //                               lnKernelsPerGpu(0)
        //                           )
        //     throw new RuntimeException(
        //         "[MonteCarloPi] This thread working on GPUs " +
        //         miGpusToUse.mkString(" ") +
        //         " should do " + nDiceRolls + "(nWorkPerKernel(iGpu=0).sum = " +
        //         lnWorkPerKernel.sum + ")" +
        //         " iterations, but actually did " + lnIterations.flatten.sum +
        //         " (seed range " + rSeed0 + " -> " + rSeed1 + ")"
        //     )
        // }
        //
        // /* Note in contrast to Java Scala does not allow access to static
        //  * methods over objecst Oo. I.e.
        //  * Legacy RootbeerPaths.v().getRootbeerHome() does not work,
        //  * but RootbeerPaths.getRootbeerHome() does!
        //  */
        // println( "[MonteCarloPi.scala:calc] RootbeerPaths.v().getRootbeerHome() = " + RootbeerPaths.getRootbeerHome() )
        // println( "[MonteCarloPi.scala:calc] Closing contexts now." )
        for ( x <- runStates )
            x._1.close

        return 4.0*quarterPi;
    }

}
