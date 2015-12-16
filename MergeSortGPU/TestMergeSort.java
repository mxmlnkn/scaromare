// rm *.class; javac HelloWorld.java 2>&1 | grep -C 5 'error'; java HelloWorld

import java.util.Arrays;

public class TestMergeSort
{
    public static void main ( String[] args)
    {
        System.out.println("Hello World");
        MergeSort sorter = new MergeSort();
        //double[] values = new double[] {7,5,4,1,3,1,9,2,0,2};
        double[] values = {686, 360, 383, 579, 316, 529, 197, 594, 801, 495, 610, 554, 591, 693, 631, 384, 341, 240, 449, 814, 393, 707, 247, 844, 280, 15, 745, 58, 586, 341, 987, 485, 611, 122, 893, 823, 311, 149, 235, 6, 216, 158, 510, 57, 618, 711, 259, 269, 183, 61, 670, 212, 611, 119, 431, 838, 675, 801, 408, 918, 168, 144, 262, 505, 243, 307, 66, 420, 601, 186, 482, 662, 982, 68, 374, 519, 713, 910, 695, 179, 90, 214, 265, 525, 269, 431, 307, 906, 381, 68, 405, 317, 565, 731, 896, 930, 339, 336, 353, 152};

        System.out.print("Unsorted Array: ");
        System.out.println( Arrays.toString(values) );

        sorter.sort(values);
        System.out.print("Sorted   Array: ");
        System.out.println( Arrays.toString(values) );

        int monotonicallyIncreasing = 1;
        for ( int i=0; i<values.length-1; ++i )
            if ( values[i] > values[i+1] )
                monotonicallyIncreasing = 0;

        if ( monotonicallyIncreasing == 1 )
            System.out.println("Sorted Array seems OK");
        else
            System.out.println("Array is not sorted!");
    }
}
