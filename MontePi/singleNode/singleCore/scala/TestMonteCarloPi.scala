
import scala.util.Random

class MonteCarloPiKernel
(
    mnHits           : Array[Long],
    miLinearThreadId : Integer,
    mRandomSeed      : Long,
    mnDiceRolls      : Long
) {
    /**
     * Creates mnDiceRolls 2D-Vectors and checks if inside upper right quarter
     * circle countint the hits.
     **/
    def gpuMethod() =
    {
        var uniRand = new Random( mRandomSeed )
        // -.- http://stackoverflow.com/questions/17965448/a-bigger-loop-in-scala
        var i = 0l
        while ( i < mnDiceRolls )
        {
            i += 1
            /* create random 2D vector with coordinates ranging from 0 to 1,  *
             * meaning the length ranges from 0 to sqrt(2)                    */
            val x = uniRand.nextDouble()
            val y = uniRand.nextDouble()
            /* if random vector inside circle, then increase hits */
            if ( x*x + y*y < 1.0 )
                mnHits(miLinearThreadId) += 1
        }
    }
}


class MonteCarloPi
{
    def calc( nDiceRolls : Long ) : Double =
    {
        var nHits = new Array[Long](1)
        nHits(0) = 0
        var kernel = new MonteCarloPiKernel( nHits, 0, 17138123l, nDiceRolls )
        kernel.gpuMethod()
        return 4.0*nHits(0) / nDiceRolls
    }

}



object TestMonteCarloPi
{
    def main( args : Array[String] ) =
    {

        val nDiceRolls = if ( args.length > 0 ) args(0).toLong else 0l;
        var piCalculator = new MonteCarloPi();

        /* execute and time pi calculation */
        val t0 = System.nanoTime();
        val pi = piCalculator.calc( nDiceRolls );
        val t1 = System.nanoTime();
        val duration = (t1-t0).toDouble / 1e9;

        System.out.println( "Rolling the dice " + nDiceRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds" );
    }
}

