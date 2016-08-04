
object TestMonteCarloPi
{
    def main( args : Array[String] ) =
    {
        val nRolls = if ( args.length > 0 ) args(0).toLong else 100000l

        var piCalculator = new MonteCarloPi();
        val pi = piCalculator.calc( nRolls, 12288 /*threads*/ );
        System.out.println("Calculated Pi: "+pi);
    }
}

