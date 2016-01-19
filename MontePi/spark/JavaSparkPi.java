/*
javac -cp /opt/spark-1.5.2/core/target/spark-core_2.10-1.5.2.jar:. JavaSparkPi.java
java -cp /opt/spark-1.5.2/core/target/spark-core_2.10-1.5.2.jar:. JavaSparkPi
*/


import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public final class JavaSparkPi
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
        final int nSlices        = ( args.length == 2 ) ? Integer.parseInt( args[1] ) : 2;

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

                    final long seed = (long)( (double) Long.MAX_VALUE / (double) nRanks * (double) iRank );
                    Random uniRand = new Random( seed );

                    int nHits = 0;
                    for ( int i = 0; i < nRolls; ++i )
                    {
                        final double x = uniRand.nextDouble();
                        final double y = uniRand.nextDouble();
                        nHits += ( x*x + y*y < 1.0 ) ? 1 : 0;
                    }
                    return nHits;
                }
            } ).reduce( new Function2<Integer,Integer,Integer>() // _+_
            {
                public Integer call( Integer a, Integer b )
                {
                    return a + b;
                }
            } );

        System.out.println( "\ncount = " + count );
        System.out.println( "Pi is roughly " + 4.0 * count / ( nSlices * nRollsPerSlice ) + "\n" );

        sc.stop();
    }
}
