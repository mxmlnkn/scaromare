
import java.io.*;          // System.out.println
import java.util.Arrays;
import java.util.Scanner;  // nextLong
import java.lang.Long;


public class TestMonteCarloPi
{
    public static void main ( String[] args )
    {
        final long maxThreads = 12288;

        Rootbeer rootbeer = new Rootbeer();
        List<Kernel> tasks = new ArrayList<Kernel>();
        for ( int nThreads = 1; nThreads <= maxThreads; ++nThreads )
        {
            tasks.add( new TestKernel( nHits,i, seed, nRollsPerThreads+1 ) );
            rootbeer.run( tasks );

        }
    }
}

