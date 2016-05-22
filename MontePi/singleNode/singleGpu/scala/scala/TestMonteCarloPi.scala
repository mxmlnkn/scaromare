/*
make -B SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6 SPARKCORE_JAR=~/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar SCALA_ROOT=$(dirname $(which scala))/../lib
*/

import org.apache.spark._
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer
import java.net.InetAddress


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

        /**** Eval Command Line Arguments ****/
        val nRolls  = if ( args.length > 0 ) args(0).toLong else 100000l
        val nSlices = if ( args.length > 1 ) args(1).toInt  else 1
        val nGpusPerNode = if ( args.length > 2 ) args(2).toInt  else 2
        val nRollsPerSlice = nRolls / nSlices;

        /**** Initialize Kernel + Parameter List ****/
        val sliceParams = List.range( 0, nSlices ).
            map( ( _, ( nSlices, nRollsPerSlice ) ) )

        /* This partitioner is for some reason more exact than the standard RangePartitioner -.- */
        class ExactPartitioner[V]( partitions: Int, elements: Int) extends Partitioner {
            def numPartitions() : Int = partitions
            def getPartition(key: Any): Int = key.asInstanceOf[Int] % partitions
        }
        var dataSet = sc.parallelize( sliceParams, sliceParams.size ).
                         partitionBy( new ExactPartitioner( nSlices, nSlices ) )

        /*** Check if every partition really goes to exactly one host ***/
        assert( dataSet.map( x => InetAddress.getLocalHost.getHostName ).
                distinct.collect.size == nSlices )
        /* Would maybe be better to include this in the real map as the
         * distribution may change in the two runs */

        println( "Calling MonteCarloPi.calc() on "+nSlices+" distinct nodes" )
        val t0 = System.nanoTime()
        val piSum = dataSet.map( x =>
            {
                val iRank  = x._1
                val nRanks = x._2._1
                val nRolls = x._2._2

                var seed0 = 71210l + Long.MaxValue / nRanks *  iRank
                var seed1 = 71210l + Long.MaxValue / nRanks * (iRank+1)
                if ( seed0 < 0 ) seed0 += Long.MaxValue
                if ( seed1 < 0 ) seed1 += Long.MaxValue

                var piCalculator = new MonteCarloPi()
                piCalculator.calc( nRolls, seed0, seed1 )
            } ).reduce( _+_ );
        val pi = piSum / nSlices
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9

        println( "\nUsing "+nSlices+" slice / GPUs. Rolling the dice " + nRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds\n" )

        sc.stop();
    }
}

