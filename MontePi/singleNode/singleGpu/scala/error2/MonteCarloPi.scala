
import scala.collection.JavaConversions._
import org.trifort.rootbeer.runtime.Kernel
import org.trifort.rootbeer.runtime.Rootbeer


/**
 * This class starts the actual CUDA-Kernels and also calculates random seeds
 * for those kernels
 **/
class MonteCarloPi
{
    def calcRandomSeed( rnKernels : Integer, riKernel : Integer ) : Long =
    {
        assert( riKernel < rnKernels )
        ( 17138123l + Long.MaxValue.toDouble / rnKernels * riKernel ).toLong
    }

    /**
     * prepares Rootbeer kernels and starts them
     **/
    def calc( nDiceRolls : Long, nKernels : Integer ) : Double =
    {
        val nRollsPerThreads = nDiceRolls / nKernels.toLong
        val nRollsRemainder  = ( nDiceRolls % nKernels.toLong ).toInt
        /* The first nRollsRemainder threads will work on 1 roll more. The
         * rest of the threads will roll the dice nRollsPerThreads */
        var nHits = new Array[Long](nKernels)

        /* List of kernels / threads we want to run in this Level */
        var tasks = List[Kernel]()
        for ( i <- 0 to nRollsRemainder )
        {
            val seed = calcRandomSeed(nKernels,i);
            tasks.add( new MonteCarloPiKernel( nHits,i, seed, nRollsPerThreads+1 ) )
        }
        for ( i <- nRollsRemainder to nKernels )
        {
            val seed = calcRandomSeed(nKernels,i);
            tasks.add( new MonteCarloPiKernel( nHits,i, seed, nRollsPerThreads ) )
        }

        println( "Run tasks with length "+tasks.size() )
        val rootbeer = new Rootbeer()
        rootbeer.run( tasks ); // kernel in order out-of-order ?

        /* Cumulate all the hits from the kernels. Divide each hit count by
         * number of rolls first and work with double then. This evades
         * integer overflows by maybe being less exact */
        var quarterPi = 0.0;
        for ( i <- 0 to nKernels )
            quarterPi += nHits(i).toDouble / nDiceRolls;

        return 4.0*quarterPi;
    }

}
