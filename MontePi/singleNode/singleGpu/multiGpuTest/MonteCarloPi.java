
import java.io.*;
import java.lang.Long;  // MAX_VALUE
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.ThreadConfig;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.GpuFuture;
import org.trifort.rootbeer.runtime.CacheConfig;

/**
 * This class starts the actual CUDA-Kernels and also calculates random seeds
 * for those kernels
 **/
public class MonteCarloPi
{
    private Rootbeer  mRootbeerContext;
    private int       miGpuDeviceToUse;
    private GpuDevice mDevice;

    /* creates rootbeer context and chooses device */
    MonteCarloPi( int riGpuDeviceToUse )
    {
        long t0, t1;
        t0 = System.nanoTime();

        System.out.print( "Creating Rootbeer Context..." );
        mRootbeerContext = new Rootbeer();
        System.out.println( "OK" );
        miGpuDeviceToUse = riGpuDeviceToUse;
        System.out.println( "Get Device List" );
        List<GpuDevice> devices = mRootbeerContext.getDevices();
        assert( miGpuDeviceToUse < devices.size() );
        System.out.println( "Get device "+miGpuDeviceToUse+" from list of length "+devices.size() );
        mDevice = devices.get( miGpuDeviceToUse );

        t1 = System.nanoTime();
        System.out.println( "MonteCarloPi constructor took " + ((t1-t0)/1e9) + " seconds" );
    }

    private long calcRandomSeed( int rnKernels, int riKernel )
    {
        assert( riKernel < rnKernels );
        return 17138123l + (long)( (double)Long.MAX_VALUE/rnKernels * riKernel );
    }

    /**
     * This routine automatically chooses a GPU device and manually sets the
     * Kernel configuration.
     */
    static public void runOnDevice
    (
        Rootbeer     rRootbeerContext,
        int          riDevice,
        List<Kernel> work
    )
    {
        List<GpuDevice> devices = rRootbeerContext.getDevices();
        assert( riDevice < devices.size() );
        GpuDevice device = devices.get( riDevice );
        Context context = device.createContext( -1 /* auto choose memory size. Not sure what this is about or what units -.- */ );
        /* this is more or less copy-past from Rootbeer.run because an easy
         * API for multi-GPU seems to be missing */
        //Context context = createDefaultContext();
        /* Debug output */
        /* Do our own configuration, because BlockShaper has some serious
         * flaws, at least performance-wise */
        final int threadsPerBlock = 256;
        ThreadConfig thread_config = new ThreadConfig(
                                threadsPerBlock, /* threadCountX */
                                1,               /* threadCountY */
                                1,               /* threadCountZ */
                                ( work.size() + threadsPerBlock - 1 ) / threadsPerBlock, /* blockCountX */
                                1,               /* blockCountY */
                                work.size()      /* numThreads */
                             );
        assert( thread_config.getThreadCountX() * thread_config.getBlockCountX() >= work.size() );

        System.out.println( "Run a total of " + thread_config.getNumThreads() +
                            " threads in (" +
                            thread_config.getBlockCountX() + "," +
                            thread_config.getBlockCountY() + ",1) blocks " +
                            "with each (" +
                            thread_config.getThreadCountX() + "," +
                            thread_config.getThreadCountY() + "," +
                            thread_config.getThreadCountZ() + ") threads " +
                            "on GPU device " + context.getDevice().getDeviceId() );

        try
        {
            context.setThreadConfig( thread_config );
            context.setKernel( work.get(0) );
            context.setUsingHandles( true );
            context.buildState();

            long t0, t1;
            t0 = System.nanoTime();
            context.run( work );
            t1 = System.nanoTime();
            System.out.println( "context.run( work ) took " + ((t1-t0)/1e9) + " seconds" );
        }
        finally
        {
            context.close();
        }
    }

    /**
     * prepares Rootbeer kernels and starts them
     **/
    public double calc( long nDiceRolls, int nKernels )
    {
        if ( nKernels <= 0 )
        {
            /* start as many threads as possible. (More are possible, but
             * they wouldn't be hardware multithreaded anymore, but
             * they would be executed in serial after the first maxThreads
             * batch finished */
            nKernels = mDevice.getMultiProcessorCount()
                     * mDevice.getMaxThreadsPerMultiprocessor();
        }
        System.out.println( "Starting kernels with each "+nKernels+" threads in total" );

        long[] nHits0       = new long[nKernels];
        long[] nIterations0 = new long[nKernels];
        long[] nHits1       = new long[nKernels];
        long[] nIterations1 = new long[nKernels];

        Rootbeer rootbeer = new Rootbeer();

        List<GpuDevice> devices = rootbeer.getDevices();
        GpuDevice device0 = devices.get(0);
        GpuDevice device1 = devices.get(1);

        Context context0 = device0.createContext( 4*1024*1024 /* 4 MiB with no basis whatsoever ... */ );
        Context context1 = device1.createContext( 4*1024*1024 /* 4 MiB with no basis whatsoever ... */ );

        //context0.setCacheConfig(CacheConfig.PREFER_SHARED);
        //context1.setCacheConfig(CacheConfig.PREFER_SHARED);

        context0.setThreadConfig( new ThreadConfig( 1,1,1, 1,1, 1 ) );
        context1.setThreadConfig( new ThreadConfig( 1,1,1, 1,1, 1 ) );

        /* Note that we need to different arrays, because even if kernel on
         * GPU 0 only writes to the first element and kernel on GPU 1 in the
         * second, the whole array will be serialized to and from the GPU! */
        context0.setKernel( new MonteCarloPiKernel( nHits0, nIterations0, 0, 87461487, nDiceRolls ) );
        context1.setKernel( new MonteCarloPiKernel( nHits1, nIterations1, 1, 17461487, nDiceRolls ) );

        context0.buildState();
        context1.buildState();

        /* run using two gpus without blocking the current thread */
        System.out.println( "Start asynchronous kernel on GPU 0" );
        GpuFuture future0 = context0.runAsync();
        System.out.println( "Start asynchronous kernel on GPU 1" );
        GpuFuture future1 = context1.runAsync();
        System.out.println( "Wait for asynchronous kernels on GPU 0 to finish" );
        future0.take();
        context0.close();
        System.out.println( "Wait for asynchronous kernels on GPU 1 to finish" );
        future1.take();
        context1.close();
        System.out.println( "All finished" );

        System.out.println( "GPU 0 did "+nIterations0[0]+" out of which "+nHits0[0]+" were inside the circle." );
        System.out.println( "GPU 1 did "+nIterations1[1]+" out of which "+nHits1[1]+" were inside the circle." );

        /* Cumulate all the hits from the kernels. Divide each hit count by
         * number of rolls first and work with double then. This evades
         * integer overflows by maybe being less exact */
        double quarterPi = 0;
        for ( int i = 0; i < nKernels; ++i )
            quarterPi += (double) nHits0[i] / (double) nDiceRolls;

        long N = 0;
        for ( int i = 0; i < nKernels; ++i )
            N += nIterations0[i];
        System.out.println( "Total iterations done by all kernels: " + N );
        assert( N == nDiceRolls );

        return 4.0*quarterPi;
    }

}
