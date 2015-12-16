/*
rm *.class *.jar; javac *.java -classpath ../rootbeer1/Rootbeer.jar && jar -cvfm TestMontePi.jar manifest.txt *.class && java -jar ../rootbeer1/Rootbeer.jar TestMontePi.jar TestMontePiGPU.jar -64bit -computecapability=sm_30 && rm TestMontePi.jar
*/

import java.io.*;          // System.out.println
import java.util.Arrays;
import java.util.Scanner;  // nextLong

public class TestMonteCarloPi
{
    public static void main ( String[] args)
    {
        System.out.println("How many times should I roll the dice? (Long in double format) (Tested up to 10 million)");
        Scanner inputScanner = new Scanner(System.in);
        long rolls = (long) inputScanner.nextDouble();

        MonteCarloPi piCalculator = new MonteCarloPi();
        /* 1152 threads: value for GTX 760, used 1024, because X-Server is
         * is running */
        double pi = piCalculator.calc( rolls, 1024 /*threads*/ );
        System.out.println("Calculated Pi: "+pi);
    }
}

