
import java.io.*;
import java.lang.Long;  // MAX_VALUE
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
import org.trifort.rootbeer.runtime.GpuDevice;
import org.trifort.rootbeer.runtime.Context;
import org.trifort.rootbeer.runtime.ThreadConfig;


/**
 * This class starts the actual CUDA-Kernels and also calculates random seeds
 * for those kernels
 **/
public class MonteCarloPi
{
    private Rootbeer mRootbeerContext;
    private int miGpuDeviceToUse;
    private GpuDevice mDevice;

    MonteCarloPi( int riGpuDeviceToUse )
    {
        mRootbeerContext = new Rootbeer();
        miGpuDeviceToUse = riGpuDeviceToUse;
        List<GpuDevice> devices = mRootbeerContext.getDevices();
        assert( miGpuDeviceToUse < devices.size() );
        mDevice = devices.get( miGpuDeviceToUse );
    }

    private long calcRandomSeed( int rnKernels, int riKernel )
    {
        assert( riKernel < rnKernels );
        return 17138123l + (long)( (double)Long.MAX_VALUE/rnKernels * riKernel );
    }

    static public void gpuDeviceInfo( int riDevice )
    {
        Rootbeer rootbeerContext = new Rootbeer();
        List<GpuDevice> devices = rootbeerContext.getDevices();
        GpuDevice device = devices.get( riDevice );

        System.out.println( "\n================== Device Number " + device.getDeviceId() + " ==================" );
        System.out.println( "| Device name              : " + device.getDeviceName() );
        System.out.println( "| Device type              : " + device.getDeviceType() );
        System.out.println( "| Computability            : " + device.getMajorVersion() + "." + device.getMinorVersion() );
		System.out.println( "|------------------- Architecture -------------------" );
        System.out.println( "| Number of SM             : " + device.getMultiProcessorCount() );
        System.out.println( "| Max Threads per SM       : " + device.getMaxThreadsPerMultiprocessor() );
        System.out.println( "| Max Threads per Block    : " + device.getMaxThreadsPerBlock() );
        System.out.println( "| Warp Size                : " + device.getWarpSize() );
        System.out.println( "| Clock Rate               : " + (device.getClockRateHz() / 1e6) + " GHz" );
        System.out.println( "| Max Block Size           : (" + device.getMaxBlockDimX() + "," + device.getMaxBlockDimY() + "," + device.getMaxBlockDimZ() + ")" );
        System.out.println( "| Max Grid Size            : (" + device.getMaxGridDimX() + "," + device.getMaxGridDimY() + "," + device.getMaxGridDimZ() + ")" );
		System.out.println( "|---------------------- Memory ----------------------" );
        System.out.println( "| Total Global Memory      : " + device.getTotalGlobalMemoryBytes() + " Bytes" );
        System.out.println( "| Free Global Memory       : " + device.getFreeGlobalMemoryBytes() + " Bytes" );
        System.out.println( "| Total Constant Memory    : " + device.getTotalConstantMemoryBytes() + " Bytes" );
        System.out.println( "| Shared Memory per Block  : " + device.getMaxSharedMemoryPerBlock() + " Bytes" );
        System.out.println( "| Registers per Block      : " + device.getMaxRegistersPerBlock() );
        System.out.println( "| Memory Clock Rate        : " + (device.getMemoryClockRateHz() / 1e9 ) + " GHz" );
        System.out.println( "| Memory Pitch             : " + device.getMaxPitch() );
        System.out.println( "| Device is Integrated     : " + device.getIntegrated() );
        System.out.println( "=====================================================" );
    }

    static public void runOnDevice
    (
        Rootbeer rRootbeerContext,
        int riDevice,
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
                                1, /* threadCountY */
                                1, /* threadCountZ */
                                ( work.size() + threadsPerBlock - 1 ) / threadsPerBlock, /* blockCountX */
                                1, /* blockCountY */
                                work.size() /* numThreads */
                             );
        assert( thread_config.getThreadCountX() * thread_config.getBlockCountX() >= work.size() );

        System.out.println( "Run a total of " + thread_config.getNumThreads() + " threads in (" + thread_config.getBlockCountX() + "," + thread_config.getBlockCountY() + ",1) blocks with each (" + thread_config.getThreadCountX() + "," + thread_config.getThreadCountY() + "," + thread_config.getThreadCountZ() + ") threads on GPU device " + context.getDevice().getDeviceId() );

        try
        {
            context.setThreadConfig( thread_config );
            context.setKernel( work.get(0) );
            context.setUsingHandles( true );
            context.buildState();
            context.run( work );
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

        long nRollsPerThreads = nDiceRolls / (long) nKernels;
        int  nRollsRemainder  = (int)( nDiceRolls % (long) nKernels );
        /* The first nRollsRemainder threads will work on 1 roll more. The
         * rest of the threads will roll the dice nRollsPerThreads */
        long[] nHits = new long[nKernels];

        /* List of kernels / threads we want to run in this Level */
        List<Kernel> tasks = new ArrayList<Kernel>();
        for (int i = 0; i < nRollsRemainder; ++i )
        {
            nHits[i] = 0;
            final long seed = calcRandomSeed( nKernels,i );
            tasks.add( new MonteCarloPiKernel( nHits,i, seed, nRollsPerThreads+1 ) );
            //System.out.println( "Kernel " + i + " has seed: " + seed );
        }
        for (int i = nRollsRemainder; i < nKernels; ++i )
        {
            nHits[i] = 0;
            final long seed = calcRandomSeed(nKernels,i);
            tasks.add( new MonteCarloPiKernel( nHits,i, seed, nRollsPerThreads ) );
            //System.out.println( "Kernel " + i + " has seed: " + seed );
        }

        System.out.println( "Do " + (nRollsPerThreads+1) + " dice rolls in " + nRollsRemainder + " threads and " + nRollsPerThreads + " dice rolls in " + (nKernels-nRollsRemainder) + " threads" );
        //mRootbeerContext.run( tasks ); // kernel in order out-of-order ?
        runOnDevice( mRootbeerContext, miGpuDeviceToUse, tasks ); // kernel in order out-of-order ?

        /* Cumulate all the hits from the kernels. Divide each hit count by
         * number of rolls first and work with double then. This evades
         * integer overflows by maybe being less exact */
        double quarterPi = 0;
        for ( int i=0; i<nKernels; ++i )
            quarterPi += (double) nHits[i] / (double) nDiceRolls;

        return 4.0*quarterPi;
    }

}
