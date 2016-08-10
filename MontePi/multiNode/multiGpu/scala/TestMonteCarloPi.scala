/*
make -B SPARK_ROOT=~/spark-1.5.2-bin-hadoop2.6 SPARKCORE_JAR=~/spark-1.5.2-bin-hadoop2.6/lib/spark-assembly-1.5.2-hadoop2.6.0.jar SCALA_ROOT=$(dirname $(which scala))/../lib

Try out interactively with:
    startSpark --time=11:00:00 --nodes=3 --partition=gpu1 --cpus-per-task=2 --gres='gpu:2'
    spark-shell --master=$MASTER_ADDRESS --jars "$HOME/scaromare/rootbeer1/Rootbeer.jar"
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
     * Find out the number of Spark executors (assuming one executor per host)
     *
     * Finds out the number of different hosts by doubling up the amount of
     * partitions as long as the resulting number of unique hostnames change.
     *
     * This only works, if the distributor doesn't first fill all cores
     * on one executor and then all other on the next node, but empirical
     * tests showed that it is not the case, although it may change for
     * for other versions than spark 1.5.2 or even other cluster
     * configurations. Would need a look at the source code of spark
     */
    def getNumberOfExecutors( sc : SparkContext ) : Int =
    {
        var nExecutors = 1 // sc.getExecutorMemoryStatus.size - 1 // For some reason this only works in the shell -.- ???
        var nExecutorsNew   = 0
        var nPartitions     = 1
        do
        {
            nExecutors = nExecutorsNew
            nExecutorsNew = sc.parallelize( 0 to nPartitions, nPartitions ).
                map( x=>(x,x) ).
                partitionBy( new ExactPartitioner( nPartitions, nPartitions ) ).
                map( x => ( InetAddress.getLocalHost().getHostName(), 1 ) ).
                reduceByKey(_+_).
                collect.
                size
            nPartitions = nPartitions * 2
        }
        while( nExecutors != nExecutorsNew )
        return nExecutors
    }

    /**
     * This partitioner is for some reason more exact than the standard
     * RangePartitioner. I.e. partitions will first get assigned
     * each to a different "slot"
     * I.e. if 4 executors are running each using 4 threads per host, then
     * starting 16 partitions will distribute them filling all available 16
     * threads.
     * It is needed for the first step or else the spawned partitions may
     * not end up using alle the nodes
     */
    class ExactPartitioner[V]( partitions: Int, elements: Int) extends Partitioner {
        def numPartitions() : Int = partitions
        def getPartition(key: Any): Int = key.asInstanceOf[Int] % partitions
    }

    /**
     * Returns an RDD of hosts followed by the number of GPUs available and
     * the total peak flop performance for possible weighted distribution
     *
     * Example Output:
     *   Array((taurusi2041,2,4.1975808E13), (taurusi2040,2,4.1975808E13))
     */
    def getClusterGpuConfiguration( sc : SparkContext, nPartitions : Int ) = {
        /**
         * Start as many partitions as possible and count number of GPUs
         * per host: partitions -> ( hostname, nGPUs, peak flops )
         *
         * In the worst case this means every node may only have one GPU.
         *
         * partitionBy needs RDD of a tuple, that's why zipWithIndex is
         * necessary
         *
         * Returns e.g. Array((taurusi2041,100000.0,2), (taurusi2040,100000.0,2))
         */
        val ret = sc.
            parallelize( (0 until nPartitions).zipWithIndex ).
            partitionBy( new ExactPartitioner( nPartitions, nPartitions ) ).
            map( x => {
                val devices = (new Rootbeer()).getDevices
                val totalPeakFlops = devices.toList.map( x => {
                    x.getMultiProcessorCount.toDouble *
                    x.getMaxThreadsPerMultiprocessor.toDouble *
                    x.getClockRateHz.toDouble
                } ).sum
                /* return */
                ( /* key */ InetAddress.getLocalHost.getHostName,
                  /* val */ ( devices.size, totalPeakFlops ) )
            } ).
            cache /* ! */
        ret
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

        val nAvailableCores = sc.defaultParallelism;
        /**
         * Start as many partitions as there are cores available in the
         * cluster.
         * @see https://spark.apache.org/docs/1.5.2/configuration.html
         *      spark.default.parallelism
         * But I'm still not 100% sure, whether that also applies to
         * SparkContext.defaultParallelism
         */
        //val nPartitions = sc.defaultParallelism
        val nPartitions = nGpusToUse               /*< actual partitions to be used */
        println( "nPartitions           = ", nPartitions )
        println( "sc.defaultParallelism = ", sc.defaultParallelism )

        /* the returned RDD is cached, which is very important! */
        val cluster  = getClusterGpuConfiguration( sc, nPartitions )
        cluster.collect.foreach( x => println( x._1 ) )
        val hostGpus = cluster.
            /* delete double keys, but only if the queried GPU metrics
             * (x and y) per host are the same. */
            reduceByKey( (x,y) => { assert( x == y ); x } ).
            collect

        /************ all this are only tests, not really necessary ***********/
        println( "Found these hosts with each these number of GPUs : " )
        hostGpus.foreach( x => println( "    " + x._1 + " : " + x._2._1 +
                          " (" + x._2._2/1e9 + "GFlops)" ) )

        val totalGpusAvailable = hostGpus.map( x => x._2._1 ).sum
        println( "Found " + totalGpusAvailable + " GPUs in total." )
        if ( totalGpusAvailable < nGpusToUse )
        {
            throw new RuntimeException( "More parallelism specified in command line argument than GPUs found!" );
        }
        /**********************************************************************/

        /*** Distribute nGpusToUse to the available GPUs / hosts ***/
        val distributor = new Distribute
        val nHostGpusToUse = distributor.distributeRessources( nGpusToUse,
                                /* host  flops       nGpus */
            hostGpus.map( x => ( ( x._1, x._2._2 ), x._2._1 ) )
        )
        println( "GPUs per host actually to be used : " )
        nHostGpusToUse.foreach( x => println( "    "+x._1+" : "+x._2 ) )
        /* @todo use distributeWeighted to distribute the rolls to do
         *       proportional to the computing power of each GPU */

        /**
         * nHostGpusToUse            dataSet              iRank, iGpuToUse
         * ( "host0", 2 )   ( "host0", ( 2, 4.198e13 ) )    ( 0, 0 )
         * ( "host1", 1 )   ( "host0", ( 2, 4.198e13 ) )    ( 1, 1 )
         * ( "host2", 0 )   ( "host1", ( 2, 4.198e13 ) )    ( 2, 0 )
         * ( "host3", 0 )   ( "host1", ( 2, 4.198e13 ) ) -> ( 3, 1 )
         *                  ( "host2", ( 2, 4.198e13 ) )    ( 4, 0 )
         *                  ( "host2", ( 2, 4.198e13 ) )    ( 5, 1 )
         *                  ( "host3", ( 2, 4.198e13 ) )    ( 6, 0 )
         *                  ( "host3", ( 2, 4.198e13 ) )    ( 7, 1 )
         */
        /* http://stackoverflow.com/questions/8016750/convert-list-of-tuple-to-map-and-deal-with-duplicate-key */
        var gpusPerHostAlreadyUsed = scala.collection.mutable.Map(
            hostGpus.
            map( x => ( x._1 /* host */, 0 ) ).
            toSeq: _*
        )
        val rankToGpuMapping = ArrayBuffer.empty[Int]
        val clusterHostnamesInOrder = cluster.map( x => x._1 ).collect
        for ( i <- 0 until clusterHostnamesInOrder.size )
        {
            val host = clusterHostnamesInOrder(i)
            val iGpu = gpusPerHostAlreadyUsed( host )
            rankToGpuMapping += iGpu
            gpusPerHostAlreadyUsed.update( host, iGpu + 1 )
        }

        /********************* distribute work and seeds *********************/
        val nRollsPerPartition = distributor.distribute( nRolls, nPartitions )
        val rankToNGpusTouse   = nHostGpusToUse.map( x => ( x._1._1, x._2 ) ).toMap

        println( "Split the total work of " + nRolls + " iterations into:" )
        nRollsPerPartition.zipWithIndex.foreach( x => println( "    " + x._2 + " : " + x._1 ) )

        /************** generate random seeds for each partition **************/
        val rankToSeedMapping = ( 0 until nPartitions ).map( iRank => {
            /* Assuming nPartitions ~< Long.MaxValue / 100 or else the div
             * discretisation error may become too large. Add another seed */
            var seed0 = Long.MaxValue / nPartitions *  iRank + 7135068l
            var seed1 = seed0 + Long.MaxValue / nPartitions
            if ( seed0 < 0 ) seed0 += Long.MaxValue
            if ( seed1 < 0 ) seed1 += Long.MaxValue
            ( seed0, seed1 )
        } )
        println( "These are the seed ranges for each partition of MonteCarloPi:" )
        rankToSeedMapping.foreach( x => println( "    " + x._1 + " -> " + x._2 ) )

        /**** Finally launch Spark i.e. one worker per node who sends work ****
         **** to all available GPUs                                        ****/
        val t0 = System.nanoTime()
        val piTripels = cluster.zipWithIndex.map( x => {
            /* Task not serializable -> try to interleave all these values into the cached RDD */
            val host            = x._1._1
            val nGpusAvailable  = x._1._2._1
            val iRank           = x._2.toInt
            val seedStart       = rankToSeedMapping(iRank)._1
            val seedEnd         = rankToSeedMapping(iRank)._2
            val iGpuToUse       = rankToGpuMapping(iRank)
            val nGpusToUse      = rankToNGpusTouse(host)
            val hostname        = InetAddress.getLocalHost.getHostName

            assert( host == hostname ) /* if not, cluster RDD not cached ? */
            // @todo unneded partitions should be filtered out:
            // assert( iGpuToUse < nGpusToUse )
            // assert( nGpusToUse > 0 )

            var pi = -1.0
            if ( iGpuToUse < nGpusToUse )
            {
                var piCalculator = new MonteCarloPi(
                    Array( iGpuToUse ) /* Array of GPU IDs to use */ )
                pi = piCalculator.calc( nRollsPerPartition( iRank ), seedStart, seedEnd )
            }
            /* return */
            ( pi, ( hostname, iRank, iGpuToUse ) )
        } ).filter( _._1 != -1.0 ).collect
        val t1 = System.nanoTime()

        /* Assert that no host calculated on one GPU twice! */
        piTripels.
            map( x => ( x._2._1 /* hostname */, x._2._3 /* iGpu */ ) ).
            /* piTripels is not an RDD, but a simple Scala Array which does
             * not have groupByKey, but groupby(_._1) is the same */
            groupBy( _._1 ).
            foreach( x => {
                if ( x._2.toList.length != x._2.toList.distinct.length )
                    throw new RuntimeException( "Host " + x._1 +
                        " calculated on a GPU twice! (" +
                        x._2.toList.map( ""+_+" " ).reduce(_+_) + ")" )
            } )

        println( "Results per Partition : " )
        piTripels.foreach( x => {
            val pi    = x._1
            val host  = x._2._1
            val iRank = x._2._2
            val iGpu  = x._2._3
            println( "    [" + host + ", Rank " + iRank + ", GPU " + iGpu + "] " +
                     nRollsPerPartition( iRank ) + " iterations -> pi = " + pi )
        } )

        val pis = piTripels.map( _._1 )
        val pi = pis.sum / pis.size
        val duration = (t1-t0).toDouble / 1e9

        println( "\nUsing " + nPartitions + " partitions and " + nGpusToUse +
                 " GPUs. Rolling the dice " + nRolls + " times resulted " +
                 "in pi ~ " + pi + " and took " + duration + " seconds\n" )

        sc.stop();
    }
}
