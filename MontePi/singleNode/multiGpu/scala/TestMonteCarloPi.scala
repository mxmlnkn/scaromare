
object TestMonteCarloPi
{
    def main( args : Array[String] ) =
    {
        /* args[0] is not the path as can be shwon with the code below.
         * This behavior is different from C++ */
        //for ( i <- 0 until args.length )
        //    println( "args["+i+"] = "+args(i).toString );
        val nDiceRolls = if ( args.length > 0 ) args(0).toLong else 0l
        val nGpusToUse = if ( args.length > 1 ) args(1).toInt else 0

        /* execute and time pi calculation */
        val t0 = System.nanoTime()

        val threads = new Array[Thread]( nGpusToUse )
        val pis     = new Array[Double]( nGpusToUse )
        for ( iGpuToUse <- 0 until nGpusToUse )
        {
            val t = new Thread( new Runnable {
                def run() {
                    pis( iGpuToUse ) = new MonteCarloPi( Array(iGpuToUse) ).
                                       calc( nDiceRolls / nGpusToUse, /* seed range */ 12435, 12434 );
                } } )
            t.start
            threads( iGpuToUse ) = t
        }
        for ( t <- threads )
        {
            t.join
            val t1 = System.nanoTime()
            val duration = (t1-t0).toDouble / 1e9;
            println( "Join thread after " + duration + " seconds" );
        }
        val pi = pis.sum / pis.size

        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9;

        println( "Rolling the dice " + nDiceRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds" );
    }
}
