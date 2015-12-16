// rm *.class; javac HelloWorld.java 2>&1 | grep -C 5 'error'; java HelloWorld

import java.util.Arrays;

public class TestMergeSort
{
    public static void main ( String[] args)
    {
        System.out.println("Hello World");
        MergeSort sorter = new MergeSort();
        double[] values = new double[] {7,5,4,1,3,1,9,2,0,2};
        System.out.print("Unsorted Array: ");
        System.out.println( Arrays.toString(values) );

        sorter.sort(values);
        System.out.print("Sorted   Array: ");
        System.out.println( Arrays.toString(values) );
    }
}

