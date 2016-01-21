
import org.trifort.rootbeer.runtime.Kernel;
import java.util.Random;


class MonteCarloPiKernel
(
    mnHits           : Array[Long],
    miLinearThreadId : Integer,
    mRandomSeed      : Long,
    mnDiceRolls      : Long
) extends Kernel {
    /**
     * Creates mnDiceRolls 2D-Vectors and checks if inside upper right quarter
     * circle countint the hits.
     **/
    def gpuMethod() =
    {
        mnHits(miLinearThreadId) = 0;
        var uniRand = new Random( mRandomSeed );
        for ( i <- 0l to mnDiceRolls )
        {
            /* create random 2D vector with coordinates ranging from 0 to 1,  *
             * meaning the length ranges from 0 to sqrt(2)                    */
            val x = uniRand.nextDouble();
            val y = uniRand.nextDouble();
            /* if random vector inside circle, then increase hits */
            if ( x*x + y*y < 1.0 )
                mnHits(miLinearThreadId) += 1;
        }
    }
}
