
import scala.collection.JavaConversions._
import org.trifort.rootbeer.runtime.{Kernel, Rootbeer, GpuDevice, Context, ThreadConfig, GpuFuture}
import java.net.InetAddress


/**
 * This class starts the actual CUDA-Kernels and also calculates random seeds
 * for those kernels
 **/
class MonteCarloPi( gpusToUse : Array[Int] = null )
{
    private val t0 = System.nanoTime

    private var mRootbeerContext = new Rootbeer()
    private val mDevices = mRootbeerContext.getDevices()
    private var mGpusToUse = gpusToUse
    if ( gpusToUse == null || gpusToUse.size == 0 )
    {
        mGpusToUse = List.range( 0, mDevices.size ).toArray
    }
    else
    {
        assert( mGpusToUse.size < mDevices.size )
        for ( iGpu <- mGpusToUse )
            assert( iGpu < mDevices.size )
    }
    print( "[MonteCarloPi.scala:<constructor>] Using the following GPUs : " )
        mGpusToUse.foreach( x => print( x+", " ) )
        println
    private val t1 = System.nanoTime
    println( "[MonteCarloPi.scala:<constructor>] MonteCarloPi constructor took " + ((t1-t0)/1e9) + " seconds" )

    /******* End of Constructor *******/

    def calcRandomSeed( rnKernels : Integer, riKernel : Integer ) : Long =
    {
        assert( riKernel < rnKernels )
        var x = 17138123l + ( Long.MaxValue.toDouble / rnKernels * riKernel ).toLong
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

        assert( riDevice < mDevices.size() )
        val context = mDevices.get( riDevice ).createContext(
            ( work.size * 2 /* nIteration and nHits List */ * 8 /* sizeof(Long) */ +
              work.size * 4 /* sizeof(exception) ??? */ ) * 4 /* empirical factor */ +
              2*1024*1024 /* safety padding */
        )
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

    /* distributes e.g. 10 on n=3 to (4,3,3) */
    def distribute( x : Long, n : Int ) = {
        val res = List.range(0,n).map( i => {
            if ( i < x % n ) x/n + 1 else x/n
        } )
        assert( res.sum == x )
        res
    }

    /* distributes e.g. 10 on n=3 if weighted as 1,2,2 to: 2,4,4 */
    def distributeWeighted( x : Long, w : Array[Double] ) = {
        assert( w != null )
        val n = w.size
        assert( n >= 1 )
        val tmp = w.map( weight => { ( weight / w.sum * x ).toLong } )
        val missing = x - tmp.sum
        println(missing)
        val res = tmp.
            zip( List.range( 0, tmp.size ) ).
            /* missing may be > n when rounding errors happen while being
             * casted to double in the formula above.
             * missing may be negative in some weird cases, that's why we
             * use signum here instead of +1 */
            map( zipped => {
                val x = zipped._1
                val i = zipped._2
                if ( i < missing % n )
                    x + missing / n + missing.signum
                else
                    x + missing / n
            } )
        if ( res.sum != x )
        {
            print( "distributeWeighted( x="+x+", w=Array(" )
            w.foreach( x => print(x+" ") )
            println( ") ) failed." )
            println( "Sum of distributed is not equal to original count x." )
            println( "  n = "+w.size )
            print( "  tmp = " ); tmp.foreach( x => print(x+" ") ); println
            print( "  missing = "+missing )
            assert( res.sum == x )
        }
        res.foreach( x => assert( x >= 0 ) )
        res
    }

    /**
     * prepares Rootbeer kernels and starts them. Only seeds in the range
     * [rSeed0,rSeed1) will be used for the kernels. By specifying different
     * ranges for different processes this function can be used in a thread-
     * safe manner
     **/
    def calc( nDiceRolls : Long, rSeed0 : Long, rSeed1 : Long ) : Double =
    {
        val t00 = System.nanoTime

        if ( mDevices.size <= 0 ) {
            throw new RuntimeException( "No GPU devices found!" )
        }

        /* start as many threads as possible. (More are possible, but
         * they wouldn't be hardware multithreaded anymore, but
         * they would be executed in serial after the first maxThreads
         * batch finished */
        val nKernelsPerGpu = mGpusToUse.map( x => 5*1024 /* mDevices.get(_) ).
            map( x => x.getMultiProcessorCount * x.getMaxThreadsPerMultiprocessor */ )

        var nHits       = nKernelsPerGpu.map( List.fill[Long](_)(0).toArray )
        var nIterations = nKernelsPerGpu.map( List.fill[Long](_)(0).toArray )

        /* first distribute work to each GPU accordingly. Then distribute
         * on each GPU the work to the Kernel configuration and calculate
         * corresponding seeds for each kernel */
        val nWorkPerGpu = distributeWeighted( nDiceRolls, nKernelsPerGpu.map( _.toDouble ) )

        println( "[MonteCarloPi.scala:calc] Running MonteCarlo on " + mGpusToUse.size +
                 " GPUs with these maximum kernel configurations : " )
        print( "[MonteCarloPi.scala:calc]     " )
        nKernelsPerGpu.foreach( x => print( x+" " ) )
        println

        println( "[MonteCarloPi.scala:calc] with each these workloads / number of iterations :" )
        print( "[MonteCarloPi.scala:calc] " )
        nWorkPerGpu.foreach( x => print( x+" " ) )
        println

        println( "[MonteCarloPi.scala:calc] These are the seed ranges for each partition of MonteCarloPi:" )
        List.range( 0, mDevices.size ).map( iGpu => {
            distribute( nWorkPerGpu(iGpu), nKernelsPerGpu(iGpu) ).
            zipWithIndex.map( z => { calcRandomSeed( nKernelsPerGpu.sum,
                                nKernelsPerGpu.slice(0,iGpu).sum + z._2 ) } )
        } ).foreach( x => { x.slice(0,4).foreach(
            x => print( "    "+x+" " ) )
            println
        } )

        var runStates = List[ Tuple2[ Context, GpuFuture ] ]()
        for ( iGpu <- 0 until mDevices.size  )
        {
            val tasks =
                distribute( nWorkPerGpu(iGpu), nKernelsPerGpu(iGpu) ).
                zipWithIndex.map( z => {
                    new MonteCarloPiKernel(
                            nHits(iGpu),
                            nIterations(iGpu),
                            z._2,   /* rank on GPU */
                            calcRandomSeed( nKernelsPerGpu.sum,
                                nKernelsPerGpu.slice(0,iGpu).sum + z._2 ),
                            z._1    /* iterations to do */
                    )
                } )
            runStates +:= runOnDevice( iGpu, tasks )
        }
        val t01 = System.nanoTime
        println( "[MonteCarloPi.scala:calc] Ran Kernels asynchronously on all GPUs. Took " + ((t01-t00)/1e9) + " seconds" )

        /* Test if random see global iRank is unique:
         *   iGpu -> kernels -> kernel+GPU rank
         * rank = sum of kernels on lower GPUs + kernel rank on this GPU */
        val testKernelHostRanks = List.range( 0, mDevices.size ).map( iGpu => {
            distribute( nWorkPerGpu(iGpu), nKernelsPerGpu(iGpu) ).
            zipWithIndex.map( z => {
                nKernelsPerGpu.slice(0,iGpu).sum + z._2
            } )
        } )
        println( "[MonteCarloPi.scala:calc] This is the list of kernel ranks for this host (one line per GPU) : " )
        testKernelHostRanks.foreach( gpu => {
            print( "[MonteCarloPi.scala:calc]     " )
            gpu.slice(0,10).foreach( rank => print( rank+" " ) )
            println( "..." )
        } )
        assert( testKernelHostRanks.distinct.size == mDevices.size )

        val t10 = System.nanoTime
        println( "[MonteCarloPi.scala:calc] Taking from GpuFuture now (Wait for asynchronous tasks)." )
        for ( x <- runStates )
            x._2.take
        val t11 = System.nanoTime
        println( "[MonteCarloPi.scala:calc] synchronize (take) i.e. kernels took " + ((t11-t10)/1e9) + " seconds" )

        /* Cumulate all the hits from the kernels. Divide each hit count by
         * number of rolls first and work with double then. This evades
         * integer overflows by maybe being less exact */
        val quarterPi = nHits.flatten.map( _.toDouble / nDiceRolls ).sum

        /* Count and check iterations done in total by kernels */
        println( "[MonteCarloPi.scala:calc] iterations actually done : " + nIterations.flatten.sum )
        assert( nIterations.flatten.sum == nDiceRolls )

        println( "[MonteCarloPi.scala:calc] Closing contexts now." )
        for ( x <- runStates )
            x._1.close

        return 4.0*quarterPi;
    }

}
