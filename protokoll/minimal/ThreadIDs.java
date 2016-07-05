import java.io.*;
import java.util.List;
import java.util.ArrayList;
import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.Rootbeer;

public class ThreadIDs
{
    /* prepares Rootbeer kernels and starts them */
    public static void main( String[] args )
    {
        final int nKernels = 3000;
        long[] results = new long[nKernels];
        /* List of kernels / threads we want to run in this Level */
        List<Kernel> tasks = new ArrayList<Kernel>();
        for ( int i = 0; i < nKernels; ++i )
        {
            results[i] = 0;
            tasks.add( new ThreadIDsKernel( i, results ) );
        }
        Rootbeer rootbeer = new Rootbeer();
        rootbeer.run(tasks);
        System.out.println( results[0] );
    }
}
