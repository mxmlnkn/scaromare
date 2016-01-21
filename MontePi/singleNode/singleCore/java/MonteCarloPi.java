
import java.io.*;
import java.lang.Long;  // MAX_VALUE
import java.util.Arrays;
import java.util.List;
import java.util.ArrayList;

public class MonteCarloPi
{
    public double calc( long nDiceRolls )
    {
        long[] nHits = new long[1];
        nHits[0] = 0;
        MonteCarloPiKernel kernel = new MonteCarloPiKernel( nHits, 0, 17138123l, nDiceRolls );
        kernel.gpuMethod();
        return 4.0*nHits[0] / nDiceRolls;
    }

}
