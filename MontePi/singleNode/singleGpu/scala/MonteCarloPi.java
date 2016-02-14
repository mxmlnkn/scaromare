
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
    private int calcRandomSeed( int rnKernels, int riKernel )
    {
        assert( riKernel < rnKernels );
        //return  (int)( 17138123l + (long)( (double)Integer.MAX_VALUE/rnKernels * riKernel ) );
        return  (int)( ( 17138123l * riKernel ) % 0x7FFFFFFF );
    }

    /**
     * prepares Rootbeer kernels and starts them
     **/
    public double calc(long nDiceRolls, int nKernels )
    {
        long nRollsPerThreads = nDiceRolls / (long) nKernels;
        int  nRollsRemainder  = (int)( nDiceRolls % (long) nKernels );
        System.out.println( "remainder="+nRollsRemainder );

        /* The first nRollsRemainder threads will work on 1 roll more. The
         * rest of the threads will roll the dice nRollsPerThreads */
        long[] nHits = new long[nKernels];

        /* List of kernels / threads we want to run in this Level */
        List<Kernel> tasks = new ArrayList<Kernel>();
        for (int i = 0; i < nRollsRemainder; ++i )
        {
            final long seed = calcRandomSeed(nKernels,i);
            tasks.add( new MonteCarloPiKernel(nHits,i, seed,nRollsPerThreads+1) );
            //if ( i % 100 < 2 )
            //    System.out.println( "i="+i+", seed="+seed+", rolls="+(nRollsPerThreads+1) );
        }
        for (int i = nRollsRemainder; i < nKernels; ++i )
        {
            final long seed = calcRandomSeed(nKernels,i);
            tasks.add( new MonteCarloPiKernel(nHits,i, seed,nRollsPerThreads ) );
            //if ( i % 100 < 2 )
            //    System.out.println( "i="+i+", seed="+seed+", rolls="+nRollsPerThreads );
        }

        System.out.println( "Run "+tasks.size()+" tasks with length " );
        Rootbeer rootbeer = new Rootbeer();
        rootbeer.run(tasks); // kernel in order out-of-order ?

        /* Cumulate all the hits from the kernels. Divide each hit count by
         * number of rolls first and work with double then. This evades
         * integer overflows by maybe being less exact */
        double quarterPi = 0;
        for ( int i=0; i<nKernels; ++i )
        {
            quarterPi += (double) nHits[i] / (double) nDiceRolls;
            //if ( i % 100 < 2 )
            //    System.out.println( "i="+i+", nHits="+nHits[i] );
        }

        return 4.0*quarterPi;
    }

}
