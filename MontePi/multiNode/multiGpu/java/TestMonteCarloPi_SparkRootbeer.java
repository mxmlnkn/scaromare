
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public final class TestMonteCarloPi_SparkRootbeer
{
    /**
     * @param args command line arguments:
     *               [ <rolls per slice> [ <number of slices> ] ]
     **/
    public static void main( String[] args )
    {
        SparkConf sparkConf  = new SparkConf().setAppName("JavaSparkPi");
        JavaSparkContext sc = new JavaSparkContext( sparkConf );

        final int nRollsPerSlice = ( args.length == 1 ) ? Integer.parseInt( args[0] ) : 100000;
        final int nSlices        = ( args.length == 2 ) ? Integer.parseInt( args[1] ) : 1;

        List< ArrayList<Integer> > sliceParams = new ArrayList< ArrayList<Integer> >();
        for ( int i = 0; i < nSlices; i++ )
        {
            ArrayList<Integer> tmp = new ArrayList<Integer>();
            tmp.add( nSlices       );  // number of processes
            tmp.add( i             );  // rank
            tmp.add( nRollsPerSlice);  // how many to do
            sliceParams.add( tmp );
        }

        JavaRDD< ArrayList<Integer> > dataSet = sc.parallelize( sliceParams, nSlices );

        final int count = dataSet.map( new Function< ArrayList<Integer>, Integer >()
            {
                public Integer call( ArrayList<Integer> params )
                {
                    final int nRanks = params.get(0);
                    final int iRank  = params.get(1);
                    final int nRolls = params.get(2);

                    MonteCarloPi piCalculator = new MonteCarloPi();
                    /* 1152 threads: value for GTX 760, used 1024, because X-Server is
                     * is running */
                    double pi = piCalculator.calc( nRolls, -1 /*threads*/ );

                    //final long seed = (long)( (double) Long.MAX_VALUE / (double) nRanks * (double) iRank );
                    //Random uniRand = new Random( seed );
                    return (int)( pi * nRolls );
                }
            } ).reduce( new Function2<Integer,Integer,Integer>() // _+_
            {
                public Integer call( Integer a, Integer b )
                {
                    return a + b;
                }
            } );

        System.out.println( "\ncount = " + count );
        System.out.println( "Pi is roughly " + count / (double) ( nSlices * nRollsPerSlice ) + "\n" );

        sc.stop();
    }
}
