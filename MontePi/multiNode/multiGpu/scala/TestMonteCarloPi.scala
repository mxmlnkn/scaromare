/*
make -B SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6 SPARKCORE_JAR=~/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar SCALA_ROOT=$(dirname $(which scala))/../lib
*/

import org.apache.spark._
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer


object TestMonteCarloPi
{
    /**
     * @param args command line arguments:
     *               [ <rolls> [ <number of slices> [ <GPUs per node> ] ] ]
     **/
    def main( args : Array[String] ) =
    {
        val sparkConf = new SparkConf().setAppName("MontePi")
        var sc = new SparkContext( sparkConf )

        val nRolls  = if ( args.length > 0 ) args(0).toLong else 100000l
        val nSlices = if ( args.length > 1 ) args(1).toInt  else 1
        val nGpusPerNode = if ( args.length > 2 ) args(2).toInt  else 2
        val nRollsPerSlice = nRolls / nSlices;

        var sliceParams = new ArrayBuffer[ Array[Long] ]()
        for ( i <- 0 until nSlices )
        {
            var tmp = new ArrayBuffer[Long]()
            tmp += nSlices         // number of processes
            tmp += i               // rank
            tmp += nRollsPerSlice  // how many to do
            tmp += nGpusPerNode
            sliceParams += tmp.toArray
        }

        var dataSet = sc.parallelize( sliceParams, nSlices )

        val t0 = System.nanoTime()
        val piSum = dataSet.map( ( params : Array[Long] ) =>
            {
                val nRanks = params(0)
                val iRank  = params(1)
                val nRolls = params(2)

                val seed0 = 71210l + Long.MaxValue / nRanks *  iRank
                val seed1 = 71210l + Long.MaxValue / nRanks * (iRank+1)

                var piCalculator = new MonteCarloPi( iRank.toInt % nGpusPerNode )
                /* Max Threads per Device which can run: 12288 on GTX 760.
                 * Because of oversubscription even though we have no
                 * memory latency only 12288 / 32 (warp size) = 384 are
                 * actually running */
                piCalculator.calc( nRolls, 384 /* threads per device */, seed0, seed1 )
            } ).reduce( _+_ );
        val pi = piSum / nSlices
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9

        println( "\nUsing "+nSlices+" slices. Rolling the dice " + nRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds\n" )

        sc.stop();
    }
}

