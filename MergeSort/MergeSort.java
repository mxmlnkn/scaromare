import java.util.Arrays;

public class MergeSort
{
    private double[] mValues;
    private double[] mSorted;

    /**
     *
     * @return number of groups in level, meaning number of maximum
     * threads possible when parallelizing the sorting
     **/
    private int getGroupsInLevel(int rLevel)
    {
        int groupLength = (int) Math.pow(2,rLevel);
        int nGroups     = mValues.length / groupLength;
        int mod         = mValues.length % groupLength;
        /* basically return ceil( len/2**lvl ), but integer div is like floor */
        return (mod == 0 ? nGroups : nGroups + 1);
    }

    /**
     * returns number of elements in group for specified level. Meaning level 0
     * are the separate elements we want to sort. Level 1 are groups of 2 of
     * these elements and so on
     **/
    private int getElementsPerGroup(int rLevel)
    {
        return (int) Math.pow(2,rLevel);
    }

    /**
     * sorts values in mValues into array mSorted.
     * \param[in] rLevel  Merge sort level. Level 1 means i sort the elements
     *                    inside groups of 2 by calling the comparison operator
     *                    and iterating over elements in level 0, which are
     *                    just single elements
     **/
    private void sortLevel(int rLevel)
    {
        assert( rLevel > 0 );
        /* this loop can be parallized with thread ID */
        final int nGroups   = getGroupsInLevel(rLevel);
        final int nElements = getElementsPerGroup(rLevel);
        System.out.println( "Groups in lvl "+rLevel+": "+nGroups+" with each " + nElements+" elements" );
        for (int i = 0; i < nGroups; ++i )
        {
            System.out.println("Group: "+i);
            /* Sort and merge from two groups into a new group twice as large
             * in mSorted */
            int iLeft   = i*nElements;
            int iRight  = iLeft+nElements/2;
            int iTarget = iLeft;

            final int iLeftEnd   = Math.min( iLeft   +nElements/2, mValues.length );
            final int iRightEnd  = Math.min( iLeftEnd+nElements/2, mValues.length );
            final int iTargetEnd = Math.min( iTarget +nElements  , mValues.length );

            System.out.println(" iLeftEnd="+iLeftEnd+", iRightEnd="+iRightEnd+", iTargetEnd="+iTargetEnd);

            do {
                System.out.print(" iTarget="+iTarget+", iLeft="+iLeft+", iRight="+iRight);
                /* increment iLeft and iRight depending on where from the
                 * element will be copied */
                assert( iLeft < iLeftEnd || iRight < iRightEnd );
                if ( iLeft >= iLeftEnd ) {
                    mSorted[iTarget++] = mValues[iRight++];
                } else if ( iRight >= iRightEnd ) {
                    mSorted[iTarget++] = mValues[iLeft++];
                } else if ( mValues[iLeft] < mValues[iRight] ) {
                    mSorted[iTarget++] = mValues[iLeft++];
                } else {
                    mSorted[iTarget++] = mValues[iRight++];
                }
                System.out.println(" -> iLeft="+iLeft+", iRight="+iRight);
            } while( iTarget < iTargetEnd );
            //System.out.print("Sorted Level 1 Array: ");
            //System.out.println( Arrays.toString(mSorted) );
        }
    }

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
            sortLevel(level);
            // goddamn Java, I miss std::swap and simple pointers
            for (int i=0; i<mValues.length; ++i)
            {
                double tmp = mSorted[i];
                mSorted[i] = mValues[i];
                mValues[i] = tmp;
            }
            System.out.println("mValues: "+Arrays.toString(mValues));
        }
    }

    public void test()
    {

    }
}
