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

        /* This partitioner is for some reason more exact than the standard RangePartitioner -.- */
        class ExactPartitioner[V]( partitions: Int, elements: Int) extends Partitioner {
            def numPartitions() : Int = partitions
            def getPartition(key: Any): Int = key.asInstanceOf[Int] % partitions
        }
        var dataSet = sc.
            parallelize( ( 0 until nSlices ).zipWithIndex ).
            partitionBy( new ExactPartitioner( nSlices, nSlices ) )

        /*** Check if every partition really goes to exactly one host ***/
        val hosts = dataSet.map( x => InetAddress.getLocalHost.getHostName )
        print( "The "+nSlices+" partitions / slices will be mapped to these hosts : \n    " )
        hosts.collect.foreach( x => print( x+" " ) )
        assert( hosts.distinct.count == nSlices )
        /* Would maybe be better to include this in the real map as the
         * distribution may change in the two runs */

        val seeds = dataSet.map( x => {
            val iRank  = x._1
            var seed0 = 71210l + (Long.MaxValue.toDouble / nSlices *  iRank   ).toLong
            var seed1 = 71210l + (Long.MaxValue.toDouble / nSlices * (iRank+1)).toLong
            if ( seed0 < 0 ) seed0 += Long.MaxValue
            if ( seed1 < 0 ) seed1 += Long.MaxValue
            ( seed0, seed1 )
        } )
        println( "These are the seed ranges for each partition of MonteCarloPi:" )
        seeds.collect.foreach( x => println( "    "+x._1+" -> "+x._2) )

        val t0 = System.nanoTime()
        val piSum = seeds.map( x => {
            var piCalculator = new MonteCarloPi()
            piCalculator.calc( nRolls, x._1, x._2 )
        } ).reduce( _+_ )
        val pi = piSum / nSlices
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9

        println( "\nUsing "+nSlices+" slice / GPUs. Rolling the dice " + nRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds\n" )

        sc.stop();
    }
}
