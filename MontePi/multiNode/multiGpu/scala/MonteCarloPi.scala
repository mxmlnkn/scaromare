
import scala.collection.JavaConversions._
import org.trifort.rootbeer.runtime.{Kernel, Rootbeer, GpuDevice, Context, ThreadConfig, GpuFuture}
import java.net.InetAddress

/**
 * This class starts the actual CUDA-Kernels and also calculates random seeds
 * for those kernels
 **/
class MonteCarloPi( iGpusToUse : Array[Int] = null )
{
    private val t0 = System.nanoTime

    private var mRootbeerContext  = new Rootbeer()
    private val mAvailableDevices = mRootbeerContext.getDevices()
    private var miGpusToUse       = iGpusToUse
    if ( iGpusToUse == null || iGpusToUse.size == 0 )
    {
        miGpusToUse = List.range( 0, mAvailableDevices.size ).toArray
    }
    else
    {
        assert( miGpusToUse.size <= mAvailableDevices.size )
        for ( iGpu <- miGpusToUse )
            assert( iGpu < mAvailableDevices.size )
    }
    print( "[MonteCarloPi.scala:<constructor>] Using the following GPUs : " )
        miGpusToUse.foreach( x => print( x+", " ) )
        println
    private val t1 = System.nanoTime
    println( "[MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took " + ((t1-t0)/1e9) + " seconds" )

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
        riDevice         : Integer,
        work             : List[Kernel]
    ) : Tuple2[ Context, GpuFuture ] =
    {
        val t00 = System.nanoTime

        assert( riDevice < mAvailableDevices.size() )
        val context = mAvailableDevices.get( riDevice ).createContext(
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
        println( "[MonteCarloPi.scala:runOnDevice] runOnDevice configuration took " + ((t01-t00)/1e9) + " seconds" )

        val t10 = System.nanoTime
        val runWaitEvent = context.runAsync( work )
        val t11 = System.nanoTime
        println( "[MonteCarloPi.scala:runOnDevice] context.run( work ) took " + ((t11-t10)/1e9) + " seconds" )

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

        if ( mAvailableDevices.size <= iGpusToUse.size ) {
            throw new RuntimeException( "Not enough GPU devices found!" )
        }

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

        var lnHits       = lnKernelsPerGpu.map( List.fill[Long](_)(0).toArray )
        var lnIterations = lnKernelsPerGpu.map( List.fill[Long](_)(0).toArray )

        /* first distribute work to each GPU accordingly. Then distribute
         * on each GPU the work to the Kernel configuration and calculate
         * corresponding seeds for each kernel */
        val lnWorkPerGpu = distributor.distributeWeighted( nDiceRolls,
                                            lnKernelsPerGpu.map( _.toDouble ) )

        /**************************** Debug Output ****************************/
        println( "[MonteCarloPi.scala:calc] Running MonteCarlo on " +
                 miGpusToUse.size + " GPUs with these maximum kernel " +
                 "configurations : " )
        print( "[MonteCarloPi.scala:calc]     " )
        lnKernelsPerGpu.foreach( x => print( x.toString + " " ) )
        println

        println( "[MonteCarloPi.scala:calc] with each these workloads / number of iterations :" )
        print( "[MonteCarloPi.scala:calc]     " )
        lnWorkPerGpu.foreach( x => print( x.toString + " " ) )
        println
        /**********************************************************************/

        /* debug output the first 4 kernel constructor calls */
        val nKernelsToShow = 4
        println( "[MonteCarloPi.scala:calc] These are the seeds for the first " +
                 nKernelsToShow + " kernels:" )

        /* for each device create Rootbeer-Kernels */
        var runStates = List[ Tuple2[ Context, GpuFuture ] ]()
        /*   iGpu              nRollsPerKernel
         * {  0, 1, 2,... } -> { {4,3,3},      {4,3,3}, ... } */
        for ( iGpu <- 0 until lGpuDevices.size )
        {
            println( "[MonteCarloPi.scala:calc]   GPU " + iGpu + " which runs " +
                     lnKernelsPerGpu(iGpu) + " kernels and a total of " +
                     lnWorkPerGpu(iGpu) + " iterations: " )
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
                val nPreviousKernels = lnKernelsPerGpu.slice(0,iGpu).sum
                val kernelSeed = calcRandomSeed(
                                     lnKernelsPerGpu.sum,
                                     nPreviousKernels + iKernel,
                                     rSeed0, rSeed1
                                 )
                /* Debug output */
                if ( iKernel < (nKernelsToShow+1)/2 ||
                     ( lnKernelsPerGpu(iGpu)-1 - iKernel ) < nKernelsToShow/2 )
                {
                    /* show toInt seed, because the seed will be cast to int
                     * in MonteCarloPiKernel */
                    val seed = Math.abs( kernelSeed.toInt )
                    println( "[MonteCarloPi.scala:calc] " +
                        "    MonteCarloPiKernel( " +
                        //lnHits(iGpu)       + ", "
                        //lnIterations(iGpu) + ", "
                        "iKernel:"     + iKernel        + ", " +
                        "seed:"        + seed           + ", " +
                        "nIterations:" + nWorkPerKernel + " )"
                    )
                }
                /* return */
                new MonteCarloPiKernel(
                    lnHits(iGpu)      ,
                    lnIterations(iGpu),
                    iKernel           , /* rank on GPU */
                    kernelSeed        ,
                    nWorkPerKernel      /* iterations to do */
                )
            } )
            runStates +:= runOnDevice( iGpu, tasks )
        }
        val t01 = System.nanoTime
        println( "[MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took " + ((t01-t00)/1e9) + " seconds" )

        /* Test if random seed and global iRank is unique:
         *   iGpu -> kernels -> kernel+GPU rank
         * rank = sum of kernels on lower GPUs + kernel rank on this GPU */
        val testKernelHostRanks = List.range( 0, lGpuDevices.size ).map( iGpu => {
            distributor.distribute( lnWorkPerGpu(iGpu), lnKernelsPerGpu(iGpu) ).
            zipWithIndex.map( z => {
                lnKernelsPerGpu.slice(0,iGpu).sum + z._2
            } )
        } )
        println( "[MonteCarloPi.scala:calc] This is the list of kernel ranks for this host (one line per GPU) : " )
        testKernelHostRanks.foreach( gpu => {
            print( "[MonteCarloPi.scala:calc]     " )
            gpu.slice(0,10).foreach( rank => print( rank+" " ) )
            println( "..." )
        } )
        assert( testKernelHostRanks.distinct.size == lGpuDevices.size )

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
        println( "[MonteCarloPi.scala:calc] iterations actually done : " + lnIterations.flatten.sum )
        assert( lnIterations.flatten.sum == nDiceRolls )

        println( "[MonteCarloPi.scala:calc] Closing contexts now." )
        for ( x <- runStates )
            x._1.close

        return 4.0*quarterPi;
    }

}
