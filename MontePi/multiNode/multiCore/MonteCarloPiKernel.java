
import java.util.Random;


public class MonteCarloPiKernel
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
        // why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed ...
        final long randMax   = 0x7FFFFFFFl;
        final long randMagic = 950706376l;
        /* copy parameter to gpu. Without this the program takes 4.65s instead
         * of 2.25s! */
        long dRandomSeed = mRandomSeed;
        final long dnDiceRolls = mnDiceRolls;
        /* using nHits += 1 instead of mnHits[ miLinearThreadId ] += 1
         * reduces 2.25s down to 0.58s ! */
        long nHits = 0;

        //Random uniRand = new Random( mRandomSeed );
        for ( long i = 0; i < dnDiceRolls; ++i )
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
    }
}
