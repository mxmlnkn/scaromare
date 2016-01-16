
import java.io.*;
import java.lang.Long;  // MAX_VALUE
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;

/**
 * This class starts the actual CUDA-Kernels and also calculates random seeds
 * for those kernels
 **/
public class MonteCarloPi
{
    private long calcRandomSeed( int riKernel, int rnKernels )
    {
        return 17138123l + (long)( (double)Long.MAX_VALUE/rnKernels * riKernel );
    }

    /**
     * prepares Rootbeer kernels and starts them
     **/
    public double calc(long nDiceRolls, int nKernels )
    {
        long nRollsPerThreads = nDiceRolls / (long) nKernels;
        int  nRollsRemainder  = (int)( nDiceRolls % (long) nKernels );
        /* The first nRollsRemainder threads will work on 1 roll more. The
         * rest of the threads will roll the dice nRollsPerThreads */
        long[] nHits = new long[nKernels];

        /* List of kernels / threads we want to run in this Level */
        List<Kernel> tasks = new ArrayList<Kernel>();
        for (int i = 0; i < nRollsRemainder; ++i )
        {
            final long seed = calcRandomSeed(nKernels,i);
            tasks.add( new MonteCarloPiKernel(nHits,i, seed,nRollsPerThreads+1) );
        }
        for (int i = nRollsRemainder; i < nKernels; ++i )
        {
            final long seed = calcRandomSeed(nKernels,i);
            tasks.add( new MonteCarloPiKernel(nHits,i, seed,nRollsPerThreads ) );
        }

        System.out.println( "Run tasks with length "+tasks.size() );
        Rootbeer rootbeer = new Rootbeer();
        rootbeer.run(tasks); // kernel in order out-of-order ?

        /* Cumulate all the hits from the kernels. Divide each hit count by
         * number of rolls first and work with double then. This evades
         * integer overflows by maybe being less exact */
        double quarterPi = 0;
        for ( int i=0; i<nKernels; ++i )
            quarterPi += (double) nHits[i] / (double) nDiceRolls;

        return 4.0*quarterPi;
    }

}
