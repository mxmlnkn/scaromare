
import java.io.*;          // System.out.println
import java.util.Arrays;
import java.lang.Long;
import java.lang.System;   // System.nanoTime


public class TestMonteCarloPi
{
    public static void main ( String[] args )
    {
        long nDiceRolls = args.length > 0 ? Long.parseLong( args[0], 10 /* decimal system */ ) : 0l;
        System.out.print( "Rolling the dice " + nDiceRolls + " times " );
        MonteCarloPi piCalculator = new MonteCarloPi();

        /* execute and time pi calculation */
        long t0 = System.nanoTime();
        double pi = piCalculator.calc( nDiceRolls );
        long t1 = System.nanoTime();
        double duration = (double) (t1-t0) / 1e9;

        System.out.println( "resulted in pi ~ " + pi + " and took " + duration + " seconds" );
    }
}

