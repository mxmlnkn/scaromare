
import org.apache.spark._
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer


object TestMonteCarloPi
{
    /**
     * @param args command line arguments:
     *               [ <rolls> [ <number of slices> ] ]
     **/
    def main( args : Array[String] ) =
    {
        val sparkConf = new SparkConf().setAppName("MontePi")
        var sc = new SparkContext( sparkConf )

        val nRolls  = if ( args.length > 0 ) args(0).toLong else 100000l
        val nSlices = if ( args.length > 1 ) args(1).toInt  else 1
        val nRollsPerSlice = nRolls / nSlices;

        var sliceParams = new ArrayBuffer[ Array[Long] ]()
        for ( i <- 0 until nSlices )
        {
            var tmp = new ArrayBuffer[Long]()
            tmp += nSlices         // number of processes
            tmp += i               // rank
            tmp += nRollsPerSlice  // how many to do
            sliceParams += tmp.toArray
        }

        var dataSet = sc.parallelize( sliceParams, nSlices )

        val t0 = System.nanoTime()
            val piSum = dataSet.map( ( params : Array[Long] ) =>
                {
                    val nRanks = params(0)
                    val iRank  = params(1)
                    val nRolls = params(2)

                    val seed = 71210l + Long.MaxValue / nRanks * iRank

                    var piCalculator = new MonteCarloPi()
                    piCalculator.calc( nRollsPerSlice, seed )
                } ).reduce( _+_ );
            val pi = piSum / nSlices
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9

        println( "\nUsing "+nSlices+" slice / cores. Rolling the dice " + nRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds\n" )

        sc.stop();
    }
}

