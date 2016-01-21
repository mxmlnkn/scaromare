
object TestMonteCarloPi
{
    def main( args : Array[String] ) =
    {
        val nDiceRolls = if ( args.length > 0 ) args(0).toLong else 0l
        var piCalculator = new MonteCarloPi()

        /* execute and time pi calculation */
        val t0 = System.nanoTime()
            /* Max Threads per Device which can run: 12288 on GTX 760.
             * Because of oversubscription even though we have no memory
             * latency only 12288 / 32 (warp size) = 384 are actually running */
            val pi = piCalculator.calc( nDiceRolls, 384 /* threads per device */ );
        val t1 = System.nanoTime()
        val duration = (t1-t0).toDouble / 1e9;

        println( "Rolling the dice " + nDiceRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds" );

    }
}

