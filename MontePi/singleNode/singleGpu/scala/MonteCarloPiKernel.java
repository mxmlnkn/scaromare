
import org.trifort.rootbeer.runtime.Kernel;
/* using a selfmade pseudo random number generator reduces the time from 9.5s
 * down to 4.65s or with all the other optimizations on using Random increases
 * the time from 0.58s for 268435456 rolls to 1.66s for 2684354 rolls.
 * This is a 300 fold increase. Meaning the kernels seem to be serialized,
 * although it's interesting that it isn't even slower. It seems like there
 * is no host-device communication, only the serialization.
 * The rolls had to be decreased because of watchdog timer */
//import java.util.Random;


public class MonteCarloPiKernel implements Kernel
{
    private long[] mnHits;
    private long[] mnIterations;
    private int    miLinearThreadId;
    private long   mRandomSeed;
    private long   mnDiceRolls;

    /**
     * Constructor which stores thread arguments: seed, diceRolls
     * @param[in] rnHits           stores the reduced result, i.e. the number
     *                             of random numbers inside the circle
     * @param[in] rnIterations     array storing the actually done number of
     *                             iterations i.e. dice rolls (was added later
     *                             for debugging)
     * @param[in] riLinearThreadId a thread id used to access the correct
     *                             entry in rnHits and rnIterations
     * @param[in] rRandomSeed      random seed to use, should be different for
     *                             each kernel!
     * @param[in] rnDiceRolls      Could also be called rnIterations, i.e.
     *                             number of Monte-Carlo iterations this kernel
     *                             shall do, not the total of all kernels!
     */
    public MonteCarloPiKernel
    (
        long[] rnHits          ,
        long[] rnIterations    ,
        int    riLinearThreadId,
        long   rRandomSeed     ,
        long   rnDiceRolls
    )
    {
        mnHits           = rnHits;
        miLinearThreadId = riLinearThreadId;
        mRandomSeed      = rRandomSeed;
        mnDiceRolls      = rnDiceRolls;
        mnIterations     = rnIterations;
    }

    /**
     * Creates mnDiceRolls 2D-Vectors and checks if inside upper right quarter
     * circle countint the hits.
     **/
    public void gpuMethod()
    {
        /* why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed
         * ... I strongly think this algorithm was used for signed integers,
         * that's why */
        final int  randMax   = 0x7FFFFFFF;
        final long randMagic = 950706376;
        /* copy parameter to gpu. Without this the program takes 4.65s instead
         * of 2.25s! */
        int dRandomSeed = Math.abs( (int) mRandomSeed );
        assert( mnDiceRolls <= Integer.MAX_VALUE );
        final int dnDiceRolls = (int) mnDiceRolls;
        /* using nHits += 1 instead of mnHits[ miLinearThreadId ] += 1
         * reduces 2.25s down to 0.58s ! */
        /* using int instead of long reduces 0.52s down to 0.5s. This is
         * because 64-bit integer arithmetic is not supported natively on
         * most GPUs. The speedup was quite a bit higher in the C++ version
         * though*/
        long nHits = 0;

        //Random uniRand = new Random( mRandomSeed );
        /* using int here only makes sense, because  one kernel needs to do
         * e.g. for GTX 760 12288 times less work, than a single core CPU
         * would need to do. For single core CPU long was needed to get
         * to a number which makes sense to benchmark on the GPU. Also on CPU
         * long doesn't has such a large penalty than on GPU. But that penalty
         * on GPU could be reduced by increasing loop unrolling.
         * Also if really more dice rolls are needed, just start additional
         * kernels */
        for ( int i = 0; i < dnDiceRolls; ++i )
        {
            /* create random 2D vector with coordinates ranging from 0 to 1,  *
             * meaning the length ranges from 0 to sqrt(2)                    */
            //double x = uniRand.nextDouble();
            //double y = uniRand.nextDouble();

            /* using float here instead of double decreased the running time
             * from 2.9s down to 2.25s */
            dRandomSeed = (int)( (randMagic*dRandomSeed) % randMax );
            float x = (float) dRandomSeed / randMax;
            dRandomSeed = (int)( (randMagic*dRandomSeed) % randMax );
            float y = (float) dRandomSeed / randMax;

            /* if random vector inside circle, then increase hits */
            if ( x*x + y*y < 1.0 )
                nHits += 1;
        }
        mnHits[ miLinearThreadId ] = nHits;
        // no measurable slowdown by this extra check, but could be removed for benchmarking anyway
        mnIterations[ miLinearThreadId ] = dnDiceRolls;
    }
}
