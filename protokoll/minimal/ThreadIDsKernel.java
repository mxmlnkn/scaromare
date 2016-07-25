import org.trifort.rootbeer.runtime.Kernel;
import org.trifort.rootbeer.runtime.RootbeerGpu;

public class ThreadIDsKernel implements Kernel
{
    private int miLinearThreadId;
    private long[] mResults;
    /* Constructor which stores thread arguments: seed, diceRolls */
    public ThreadIDsKernel( int riLinearThreadId, long[] rResults )
    {
        miLinearThreadId = riLinearThreadId;
        mResults         = rResults;
    }
    public void gpuMethod()
    {
        mResults[ miLinearThreadId ] = RootbeerGpu.getThreadId();
    }
}
