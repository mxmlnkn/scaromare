
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
    private int    miLinearThreadId;
    private long   mRandomSeed;
    private long   mnDiceRolls;

    /* Constructor which stores thread arguments: seed, diceRolls */
    public MonteCarloPiKernel
    ( long[] rnHits, int riLinearThreadId, long rRandomSeed, long rnDiceRolls )
    {
        mnHits           = rnHits;
        miLinearThreadId = riLinearThreadId;
        mRandomSeed      = rRandomSeed;
        mnDiceRolls      = rnDiceRolls;
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
        final int randMax = 0x7FFFFFFF;
        final long randMagic = 950706376;
        /* copy parameter to gpu. Without this the program takes 4.65s instead
         * of 2.25s! */
        int dRandomSeed = (int) mRandomSeed;
        assert( mnDiceRolls <= Integer.MAX_VALUE );
        final int dnDiceRolls = (int) mnDiceRolls;
        /* using nHits += 1 instead of mnHits[ miLinearThreadId ] += 1
         * reduces 2.25s down to 0.58s ! */
        /* using int instead of long reduces 0.52s down to 0.5s. This is
         * because 64-bit integer arithmetic is not supported natively on
         * most GPUs. The speedup was quite a bit higher in the C++ version
         * though*/
        //if ( miLinearThreadId == 8192 )
        //    System.out.println( "[i="+miLinearThreadId+"] mnDiceRolls = "+mnDiceRolls+", seed="+dRandomSeed );
        long nHits = 0;

        //Random uniRand = new Random( mRandomSeed );
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
            /*
            if ( i % 10000 == 0 )
            {
                System.out.println( "  seed="+dRandomSeed+", x="+x+", y="+y );
            }*/
        }
        /*if ( miLinearThreadId == 0 )
            System.out.println( "[i=0] nHits = "+nHits);*/
        mnHits[ miLinearThreadId ] = nHits;
    }
}
