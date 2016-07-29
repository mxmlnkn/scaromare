/*
make -B SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6 SPARKCORE_JAR=~/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar SCALA_ROOT=$(dirname $(which scala))/../lib
*/

import org.apache.spark._
import org.apache.spark.Partitioner
import org.apache.spark.SparkContext._
import scala.collection.mutable.ArrayBuffer
import java.net.InetAddress
import org.trifort.rootbeer.runtime.Rootbeer
import scala.collection.JavaConversions._   // toList

object TestMonteCarloPi
{
    /**
     * Find out the number of executors by doubling up the number as
     * long as the resulting unique hostnames changes.
     * This only works, if the distributor doesn't first fill all cores
     * on one executor and then all other on the next node, but empirical
     * tests showed that it is not the case, although it may change for
     * for other versions than spark 1.5.2 or even other cluster
     * configurations. Would need a look at the source code of spark
     */
    def getNumberOfExecutors( sc : SparkContext ) : Int =
    {
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
        }
        while( nExecutors != nExesNew )
        return nExecutors
    }

    /* This partitioner is for some reason more exact than the standard
     * RangePartitioner -.-.
     * Actually not really necessary anymore in this version, because each
     * partition spawns multiple GPUs. Normally there should be one
     * partition per Host i.e. one executor per Host.
     * It is needed for the first step or else the spawned partitions may
     * not end up using alle the nodes */
    class ExactPartitioner[V]( partitions: Int, elements: Int) extends Partitioner {
        def numPartitions() : Int = partitions
        def getPartition(key: Any): Int = key.asInstanceOf[Int] % partitions
    }

    /**
     * @param args command line arguments:
     *               [ <rolls> [ <total number of GPUs to use> ] ]
     **/
    def main( args : Array[String] ) =
    {
        val sparkConf = new SparkConf().setAppName("MontePi")
        var sc = new SparkContext( sparkConf )

        /**** Eval Command Line Arguments ****/
        val nRolls          = if ( args.length > 0 ) args(0).toLong else 100000l
        val nGpusToUse      = if ( args.length > 1 ) args(1).toInt  else 1
        val nCoresPerWorker = if ( args.length > 2 ) args(2).toInt  else 1

        /**
         * sc.parallelize( ( 0 until nSlices ).zipWithIndex ).
         * partitionBy( new ExactPartitioner( nSlices, nSlices ) ).
         * map( (x) => {
         *     Thread.sleep(10);
         *     x + " : " + InetAddress.getLocalHost().getHostName() + " -> iGPU = " + (x._1/4)
         * } ).
         * collect().
         * foreach( println )
         *
         * ( 0, 0) : taurusi2066 -> iGPU = 0
         * ( 1, 1) : taurusi2064 -> iGPU = 0
         * ( 2, 2) : taurusi2065 -> iGPU = 0
         * ( 3, 3) : taurusi2068 -> iGPU = 0
         * ( 4, 4) : taurusi2066 -> iGPU = 1
         * ( 5, 5) : taurusi2064 -> iGPU = 1
         * ( 6, 6) : taurusi2065 -> iGPU = 1
         * ( 7, 7) : taurusi2068 -> iGPU = 1
         * ( 8, 8) : taurusi2066 -> iGPU = 2
         * ( 9, 9) : taurusi2064 -> iGPU = 2
         * (10,10) : taurusi2065 -> iGPU = 2
         * (11,11) : taurusi2068 -> iGPU = 2
         * (12,12) : taurusi2066 -> iGPU = 3
         * (13,13) : taurusi2064 -> iGPU = 3
         * (14,14) : taurusi2065 -> iGPU = 3
         * (15,15) : taurusi2068 -> iGPU = 3
         *
         * nSlices-2:
         *
         * ( 0, 0) : taurusi2066 -> iGPU = 0
         * ( 1, 1) : taurusi2064 -> iGPU = 0
         * ( 2, 2) : taurusi2068 -> iGPU = 0
         * ( 3, 3) : taurusi2065 -> iGPU = 0
         * ( 4, 4) : taurusi2066 -> iGPU = 1
         * ( 5, 5) : taurusi2064 -> iGPU = 1
         * ( 6, 6) : taurusi2068 -> iGPU = 1
         * ( 7, 7) : taurusi2065 -> iGPU = 1
         * ( 8, 8) : taurusi2066 -> iGPU = 2
         * ( 9, 9) : taurusi2064 -> iGPU = 2
         * (10,10) : taurusi2068 -> iGPU = 2
         * (11,11) : taurusi2065 -> iGPU = 2
         * (12,12) : taurusi2066 -> iGPU = 3
         * (13,13) : taurusi2064 -> iGPU = 3
         *
         * => iGpu = iSlice / 4
         */

        val nExecutors = getNumberOfExecutors(sc)
        if ( ! ( nGpusToUse <= nExecutors * nCoresPerWorker ) )
        {
            throw new RuntimeException(
                "Found there to be " + nExecutors + " executors available " +
                "with each " + nCoresPerWorker + " cores / threads onto " +
                "which " + nGpusToUse + " GPUs need to be distribted"
            )
        }
        val nSlices = nGpusToUse

        /* assign each spark process an ID and cache it! */
        val dataSet = sc.
            parallelize( ( 0 until nSlices ).zipWithIndex ).
            partitionBy( new ExactPartitioner( nSlices, nSlices ) ).
            cache()

        /**
         * First start as many partitions as possible and count number of GPUs
         * per host: slices -> ( hostname, nGPUs, peak flops )
         * In the worst case this means every node may only have one GPU.
         * partitionBy needs RDD of a tuple, that's why zipWithIndex is
         * necessary
         */
        val hostGpus = dataSet.
            map( x => {
                val devices = (new Rootbeer()).getDevices
                val totalPeakFlops = devices.toList.map( x => {
                    x.getMultiProcessorCount.toDouble *
                    x.getMaxThreadsPerMultiprocessor.toDouble *
                    x.getClockRateHz.toDouble
                } ).sum
                /* return */
                ( ( InetAddress.getLocalHost.getHostName,
                      totalPeakFlops ),
                  devices.size
                )
            } ).
            reduceByKey( (x,y) => { assert( x == y ); x } ).
            collect

        /************ all this are only tests, not really necessary ***********/
        println( "Found these hosts with each these number of GPUs : " )
        hostGpus.foreach( x => println( "    " + x._1._1 + " : " + x._2 +
                          " (" + x._1._2/1e9 + "GFlops)" ) )

        if ( hostGpus.size != Math.min( nSlices, nExecutors ) )
        {
            throw new RuntimeException( "Anticipated mapping seems to be wrong. Some partitions didn't map to different hosts! This could, but also shouldn't happen if the Spark cluster has some workers with multiple cores." )
        }

        val totalGpusAvailable = hostGpus.map( x => x._2 ).sum
        println( "Found "+totalGpusAvailable+" GPUs in total." )
        if ( totalGpusAvailable < nGpusToUse )
        {
            throw new RuntimeException( "More parallelism specified in command line argument than GPUs found!" );
        }

        /* test if cluster is homogenous in gpus per node */
        if ( hostGpus.map( _._2 ).min != hostGpus.map( _._2 ).max )
        {
            throw new RuntimeException( "Currently this hack only works if every node has the same amount of GPUs!" );
        }
        val nGpusPerWorker = hostGpus.map( _._2 ).min
        if ( ! ( nCoresPerWorker == nGpusPerWorker ) )
        {
            throw new RuntimeException( "Specified number of cores per worker (" + nCoresPerWorker + ") should be equal to the number of GPUs per Worker/Host (" + nGpusPerWorker + ")" );
        }

        /* { ( "taurusi2095", 4 ), ( "taurusi2095", 4 ), ( "taurusi2095", 4 ) }
         * -> { ( "taurusi2095", 4 ) } */
        val lnGpusPerHost = hostGpus.
                            map( x => ( x._1._1, x._2 ) ).
                            distinct

        /**********************************************************************/
        /*** Distribute nGpusToUse to the available GPUs / hosts */
        val distributor = new Distribute
        /* E.g. dist( 10,{4,4,4} ) -> { 4,3,3 } */
        val hostGpusToUse = distributor.distributeRessources( nGpusToUse, hostGpus )
        println( "GPUs per host actually to be used : " )
        hostGpusToUse.foreach( x => println( "    "+x._1+" : "+x._2 ) )
        val hostWork = distributor.distributeWeighted(
                           nRolls,
                           hostGpusToUse.map( _._1._2.toDouble /* peak flops */ )
                       )
        val sliceWork = distributor.distribute( nRolls, nSlices )

        /*** generate random seeds ***/
        val seeds = dataSet.map( x => {
            val iRank  = x._1
            /* Assuming nSlices ~< Long.MaxValue / 100 or else the div
             * discretisation error may become too large. Add another seed */
            var seed0 = Long.MaxValue / nSlices *  iRank + 7135068l
            var seed1 = seed0 + Long.MaxValue / nSlices
            if ( seed0 < 0 ) seed0 += Long.MaxValue
            if ( seed1 < 0 ) seed1 += Long.MaxValue
            ( iRank, ( seed0, seed1 ) )
        } )
        println( "These are the seed ranges for each partition of MonteCarloPi:" )
        seeds.collect.foreach( x => println( "    " + x._1 + " -> " + x._2 ) )

        /**** Finally launch Spark i.e. one worker per node who sends work ****
         **** to all available GPUs                                        ****/
        val t0 = System.nanoTime()
        val piTripels = seeds.map( x => {
            val iRank     = x._1
            val seedStart = x._2._1
            val seedEnd   = x._2._2

            val hostname = InetAddress.getLocalHost.getHostName
            val iGpu = iRank / nCoresPerWorker

            var piCalculator = new MonteCarloPi(
                Array( iGpu ) /* Array of GPU IDs to use */ )
            val pi = piCalculator.calc( sliceWork( iRank ), seedStart, seedEnd )

            /* return */
            ( pi, ( hostname, iRank, iGpu ) )
        } ).collect

        println( "Results per Slice : " )
        piTripels.foreach( x => {
            val pi    = x._1
            val host  = x._2._1
            val iRank = x._2._2
            val iGpu  = x._2._3
            println( "    [" + host + ", Rank " + iRank + ", GPU " + iGpu + "] " +
                     sliceWork( iRank ) + " iterations -> pi = " + pi )
        } )

        val pis = piTripels.map( _._1 )
        val pi = pis.sum / pis.size
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9

        println( "\nUsing " + nSlices + " slices and " + nGpusToUse +
                 " GPUs. Rolling the dice " + nRolls + " times resulted " +
                 "in pi ~ " + pi + " and took " + duration + " seconds\n" )

        sc.stop();
    }
}
