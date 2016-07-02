
import java.io.*;          // System.out.println
import java.util.Arrays;
import java.util.Scanner;  // nextLong
import java.lang.Long;


public class TestMonteCarloPi
{

    public static void main ( String[] args )
    {
        /* args[0] is not the path as can be shwon with the code below.
         * This behavior is different from C++ */
        //for ( int i = 0; i < args.length; ++i )
        //    System.out.println( "args["+i+"] = "+args[i] );
        long nDiceRolls     = args.length < 1 ? 0l : Long.parseLong  ( args[0], 10 /* decimal system */ );
        int iGpuDeviceToUse = args.length < 2 ? 0  : Integer.parseInt( args[1], 10 /* decimal system */ );

        MonteCarloPi piCalculator = new MonteCarloPi( iGpuDeviceToUse );
        piCalculator.gpuDeviceInfo( iGpuDeviceToUse );

        /* execute and time pi calculation */
        long t0 = System.nanoTime();
            double pi = piCalculator.calc( nDiceRolls, -1 /* threads per device. -1: auto */ );
        long t1 = System.nanoTime();
        double duration = (double) (t1-t0) / 1e9;

        System.out.println( "Rolling the dice " + nDiceRolls + " times " +
            "resulted in pi ~ " + pi + " and took " + duration + " seconds" );

    }
}

