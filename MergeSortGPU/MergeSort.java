import java.util.Arrays;

import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;
// import org.trifort.rootbeer.runtime.util.Stopwatch;


public class MergeSort
{
    private double[] mValues;
    private double[] mSorted;

    /**
     *
     * @return number of groups in level, meaning number of maximum
     * threads possible when parallelizing the sorting
     **/
    private int getGroupsInLevel(int rLevel, int rLength)
    {
        int groupLength = (int) Math.pow(2,rLevel);
        int nGroups     = rLength / groupLength;
        int mod         = rLength % groupLength;
        /* basically return ceil( len/2**lvl ), but integer div is like floor */
        return (mod == 0 ? nGroups : nGroups + 1);
    }

    /**
     * prepares Rootbeer kernels and starts them
     **/
    public void sort(double[] rValues)
    {
        mValues = rValues;
        mSorted = new double[rValues.length];
        /* level 1 only needs to be sorted, if we have more than 1 value
         * level 2 only needs to be sorted, if n/2 > 1
         * level l only needs to be sorted, if n/2**(l-1) > 1 <=> n > 2**(l-1)
         */
        for ( int level = 1; mValues.length > (int) Math.pow(2,level-1); ++level )
        {

            final int nGroups = getGroupsInLevel( level, rValues.length );

            /* List of kernels / threads we want to run in this Level */
            System.out.println("Collecting tasks for level "+level+" which has "+nGroups+" groups");
            List<Kernel> tasks = new ArrayList<Kernel>();
            for (int iGroup = 0; iGroup < nGroups; ++iGroup )
                tasks.add( new MergeSortKernel(level, iGroup, mValues, mSorted) );

            System.out.println( "Run tasks with length "+tasks.size() );
            Rootbeer rootbeer = new Rootbeer();
            rootbeer.run(tasks); // kernel in order out-of-order ?

            /* no deep swap, just swap 'pointers' */
            double[] tmp = mSorted;
            mSorted = mValues;
            mValues = tmp;
            System.out.println("mValues: "+Arrays.toString(mValues));
        }
        /* Pointer swap won't work in return type except for the case where
         * the number of levels is even, so that mValues == rValues */
        for ( int i = 0; i < rValues.length; ++i )
            rValues[i] = mValues[i];
    }

}
