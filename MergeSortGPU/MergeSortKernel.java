
import java.util.Arrays;

import org.trifort.rootbeer.runtime.Kernel;

public class MergeSortKernel implements Kernel
{
    /* a group is always thought of as the target group. Meaning the algorithm
     * begins with nElements/2 groups in the first step */
    private int mLevel;        // The recursion level specifies the size and count of groups / partitions
    private int miGroup;       // group number == thread ID. A Group is a subarray of elements of which we want to merge two each
    private double[] mValues;  // holds values to sort
    private double[] mSorted;  // holds sorted values

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
     * Constructor which stores kernel parameters. In CUDA these arguments
     * could be given directly to the Kernel.
     *
     * \param[in] rLevel  Merge sort level. Level 1 means i sort the elements
     *                    inside groups of 2 by calling the comparison operator
     *                    and iterating over elements in level 0, which are
     *                    just single elements
     **/
    MergeSortKernel( int rLevel, int riGroup, double[] rValues, double[] rSorted )
    {
        assert( rLevel > 0 );
        assert( riGroup * getElementsPerGroup(rLevel) < rValues.length );
        mLevel  = rLevel;
        miGroup = riGroup;
        mValues = rValues;
        mSorted = rSorted;
    }

    public void gpuMethod()
    {
        final int nElements = getElementsPerGroup(mLevel);

        System.out.println("Group: "+miGroup);
        /* Sort and merge from two groups into a new group twice as large
         * in mSorted */
        int iLeft   = miGroup*nElements;
        int iRight  = iLeft+nElements/2;
        int iTarget = iLeft;

        final int iLeftEnd   = Math.min( iLeft   +nElements/2, mValues.length );
        final int iRightEnd  = Math.min( iLeftEnd+nElements/2, mValues.length );
        final int iTargetEnd = Math.min( iTarget +nElements  , mValues.length );

        System.out.println(" iLeftEnd="+iLeftEnd+", iRightEnd="+iRightEnd+", iTargetEnd="+iTargetEnd);

        /**
         * - level=3 => nElementsPerGroup=8
         * - array length: 10
         * - || are only there to suggest the half of groups in mValues and
         *   groups in mSorted
         * - groups in level 2: 4 (mValues)
         * - groups in level 3: 8 (mSorted)
         *
         *             iLeftEnd
         * iLeft        iRight       iRightEnd
         *  |             |             |
         * +--+--+--+--+ +--+--+--+--+ +--+--+
         * |  |  |  |  | |  |  |  |  | |  |  |  mValues
         * +--+--+--+--+ +--+--+--+--+ +--+--+
         *
         * iLeft                   iTargetEnd
         *  |                          |
         * +--+--+--+--+--+--+--+--+   +--+--+
         * |  |  |  |  |  |  |  |  |   |  |  |  mSorted
         * +--+--+--+--+--+--+--+--+   +--+--+
         **/

        do {
            System.out.print(" iTarget="+iTarget+", iLeft="+iLeft+", iRight="+iRight);
            /* increment iLeft and iRight depending on where from the
             * element will be copied */
            assert( iLeft < iLeftEnd || iRight < iRightEnd );

            /* if left mValues subarray is empty, then just stream from right */
            if ( iLeft >= iLeftEnd ) {
                mSorted[iTarget++] = mValues[iRight++];
            /* if right mValues subarray is empty, then just stream from left */
            } else if ( iRight >= iRightEnd ) {
                mSorted[iTarget++] = mValues[iLeft++];
            /* if both subarrays/groups still have elements, compare which of *
             * the next element is smaller and move that to target            */
            } else if ( mValues[iLeft] < mValues[iRight] ) {
                mSorted[iTarget++] = mValues[iLeft++];
            } else {
                mSorted[iTarget++] = mValues[iRight++];
            }
            System.out.println(" -> iLeft="+iLeft+", iRight="+iRight);
        } while( iTarget < iTargetEnd );
    }
}
