
import java.io.*;          // System.out.println
import java.util.Arrays;
import java.util.Scanner;  // nextLong
import java.lang.Long;


public class TestMonteCarloPi
{
    public static void main ( String[] args )
    {
        long nDiceRolls = args.length > 0 ? Long.parseLong( args[0], 10 /* decimal system */ ) : 0l;
        MonteCarloPi piCalculator = new MonteCarloPi();

        /* execute and time pi calculation */
        long t0 = System.nanoTime();
            /* Max Threads per Device which can run: 12288 on GTX 760.
             * Because of oversubscription even though we have no memory
             * latency only 12288 / 32 (warp size) = 384 are actually running */
            double pi = piCalculator.calc( nDiceRolls, 384 /* threads per device */ );
        long t1 = System.nanoTime();
        double duration = (double) (t1-t0) / 1e9;

        System.out.println( "Rolling the dice " + nDiceRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds" );

    }
}

