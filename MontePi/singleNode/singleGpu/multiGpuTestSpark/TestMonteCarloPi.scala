/*
make -B SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6 SPARKCORE_JAR=~/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar SCALA_ROOT=$(dirname $(which scala))/../lib
*/

import org.apache.spark._
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer
import java.net.InetAddress
import org.trifort.rootbeer.runtime.Rootbeer


object TestMonteCarloPi
{
    /**
     * @param args command line arguments:
     *               [ <rolls> [ <total number of GPUs to use> ] ]
     **/
    def main( args : Array[String] ) =
    {
        val sparkConf = new SparkConf().setAppName("MontePi")
        var sc = new SparkContext( sparkConf )

        /**** Eval Command Line Arguments ****/
        val nRolls  = if ( args.length > 0 ) args(0).toLong else 100000l
        val nSlices = if ( args.length > 1 ) args(1).toInt  else 1
        val nRollsPerSlice = nRolls / nSlices;

        /* This partitioner is for some reason more exact than the standard RangePartitioner -.- */
        class ExactPartitioner[V]( partitions: Int, elements: Int) extends Partitioner {
            def numPartitions() : Int = partitions
            def getPartition(key: Any): Int = key.asInstanceOf[Int] % partitions
        }

        /**
         * Find out the number of executors by doubling up the number as
         * long as the resulting unique hostnames changes.
         * This only works, if the distributor doesn't first fill all cores
         * on one executor and then all other on the next node.
         */
        var nExecutors = 1 // sc.getExecutorMemoryStatus.size - 1 // For some reason this only works in the shell -.- ???
        var nExesNew   = 0
        var nParts     = 1
        do
        {
            nExecutors = nExesNew
            nExesNew = sc.parallelize( 0 to nParts, nParts ).map( x=>(x,x) ).
                partitionBy( new ExactPartitioner( nParts, nParts ) ).
                map( x => ( InetAddress.getLocalHost().getHostName(), 1 ) ).
                reduceByKey(_+_).collect.size
            nParts = nParts * 2
        } while( nExecutors != nExesNew )
        println( "Found there to be "+nExecutors+" executors available" )

        /**
         * First start as many partitions as possible and count number of GPUs
         * per host.
         * In the worst case this means every node may only have one GPU.
         */
        val dataSet = sc.
            parallelize( ( 0 until nExecutors ).zipWithIndex ).
            partitionBy( new ExactPartitioner( nExecutors, nExecutors ) )
        val hostGpus = dataSet.
            map( x => {
                ( InetAddress.getLocalHost.getHostName,
                  (new Rootbeer()).getDevices.size    )
            } ).
            reduceByKey( (x,y) => { assert( x == y ); x } ).
            collect
        println( "Found these hosts with each these number of GPUs : " )
        hostGpus.foreach( x => println( "    "+x._1+" : "+x._2 ) )
        if ( hostGpus.size != nExecutors )
        {
            throw new RuntimeException( "Anticipated mapping seems to be wrong. Some partitions didn't map to different hosts! This could, but also shouldn't happen if the Spark cluster has some workers with multiple cores." )
        }
        val totalGpusAvailable = hostGpus.map( x => x._2 ).sum
        println( "Found "+totalGpusAvailable+" GPUs in total." )
        if ( totalGpusAvailable < nSlices )
        {
            throw new RuntimeException( "More parallelism specified in command line argument than GPUs found!" );
        }

        val distributor = new Distribute
        val hostGpusToUse = distributor.distributeRessources( nSlices, hostGpus )
        println( "GPUs per host actually to be used : " )
        hostGpusToUse.foreach( x => println( "    "+x._1+" : "+x._2 ) )
        val hostWork = distributor.distributeWeighted( nRolls, hostGpusToUse.map( _._2.toDouble ) )

        /*** generate random seeds ***/
        val seeds = dataSet.map( x => {
            val iRank  = x._1
            /* Assuming nSlices ~< Long.MaxValue / 100 or else the div
             * discretisation error may become too large. Add another seed */
            var seed0 = Long.MaxValue / nSlices *  iRank + 7135068l
            var seed1 = seed0 + Long.MaxValue / nSlices
            if ( seed0 < 0 ) seed0 += Long.MaxValue
            if ( seed1 < 0 ) seed1 += Long.MaxValue
            ( seed0, seed1 )
        } )
        println( "These are the seed ranges for each partition of MonteCarloPi:" )
        seeds.collect.foreach( x => println( "    "+x._1+" -> "+x._2) )

        /*** Finally launch Spark ***/
        val t0 = System.nanoTime()
        val pis = seeds.map( x => {
            val hostname = InetAddress.getLocalHost.getHostName
            val rank = hostGpusToUse.map( _._1 ).indexWhere( _ == hostname )
            var piCalculator = new MonteCarloPi( (0 until hostGpusToUse(rank)._2).toArray /* list of GPUs to use */ )
            piCalculator.calc( hostWork( rank ),
                               x._1, /* seed range start */
                               x._2  /* seed range end */
            )
        } ).collect
        print( "Individual PIs are : " )
        pis.foreach( x => print( x+" " ) )

        val pi = pis.sum / nSlices
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9

        println( "\nUsing "+nSlices+" slice / GPUs. Rolling the dice " + nRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds\n" )

        sc.stop();
    }
}
