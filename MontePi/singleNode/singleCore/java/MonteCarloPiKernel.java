
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
        mnHits[miLinearThreadId] = 0;
        Random uniRand = new Random( mRandomSeed );
        for ( long i = 0; i < mnDiceRolls; ++i )
        {
            /* create random 2D vector with coordinates ranging from 0 to 1,  *
             * meaning the length ranges from 0 to sqrt(2)                    */
            double x = uniRand.nextDouble();
            double y = uniRand.nextDouble();
            /* if random vector inside circle, then increase hits */
            if ( x*x + y*y < 1.0 )
                mnHits[miLinearThreadId] += 1;
        }
    }
}
