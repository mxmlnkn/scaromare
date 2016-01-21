
import scala.io.StdIn._  // readInt

object TestMonteCarloPi
{
    def main( args : Array[String] ) =
    {
        System.out.println("How many times should I roll the dice? (Long in double format) (Tested up to 10 million)");
        val rolls = readInt();

        var piCalculator = new MonteCarloPi();
        /* 1152 threads: value for GTX 760, used 1024, because X-Server is
         * is running */
        val pi = piCalculator.calc( rolls, 1024 /*threads*/ );
        System.out.println("Calculated Pi: "+pi);
    }
}

