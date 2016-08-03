
class Distribute {

    /* distributes e.g. 10 on n=3 to (4,3,3) */
    def distribute( x : Long, n : Long ) : List[Long] = {
        val res = List.range( 0l, n ).map( i => {
            if ( i < x % n )
                x/n + 1l
            else
                x/n
        } )
        assert( res.sum == x )
        /* return */ res
    }

    /* distributes e.g. 10 on n=3 if weighted as 1,2,2 to: 2,4,4 */
    def distributeWeighted( x : Long, w : Array[Double] ) = {
        assert( w != null )
        val n = w.size
        assert( n >= 1 )
        val tmp = w.map( weight => { ( weight.toDouble / w.sum * x ).toLong } )
        val missing = x - tmp.sum
        println(missing)
        val res = tmp.
            zip( List.range( 0, tmp.size ) ).
            /* missing may be > n when rounding errors happen while being
             * casted to double in the formula above.
             * missing may be negative in some weird cases, that's why we
             * use signum here instead of +1 */
            map( zipped => {
                val x = zipped._1
                val i = zipped._2
                if ( i < missing % n )
                    x + missing / n + missing.signum
                else
                    x + missing / n
            } )
        if ( res.sum != x )
        {
            print( "distributeWeighted( x="+x+", w=Array(" )
            w.foreach( x => print(x+" ") )
            println( ") ) failed." )
            println( "Sum of distributed is not equal to original count x." )
            println( "  n = "+w.size )
            print( "  tmp = " ); tmp.foreach( x => print(x+" ") ); println
            print( "  missing = "+missing )
            assert( res.sum == x )
        }
        res.foreach( x => assert( x >= 0 ) )
        /* return */ res
    }

    /**
     * Now distribute nSlices / parallelism on the found workers.
     * For that we need to map the map( hostname, nGpusAvailable ) to
     * map( hostname, nGpusToUse )
     */
    def distributeRessources[K](
        work    : Int,
        maxWork : Array[(K,Int)]
    ) : Array[(K,Int)] =
    {
        assert( work >= 0 )
        val totalMaxWork = maxWork.map( _._2 ).sum
        assert( work <= totalMaxWork )
        val cumMaxWork   = maxWork.map( _._2 ).scanLeft(0)(_+_)
        val afterFull    = cumMaxWork.indexWhere( _ > work ) - 1
        /* if no index found, then work is equal to all maxWorks summed.
         * Note that cumSum has 1 element more than the original array! */
        if ( afterFull < 0 )
        {
            assert( totalMaxWork == work )
            return maxWork
        }
        /* set index where maxWork began to exceed work needed to some
         * less work and all after that to 0 */
        var ret = maxWork
        println( "maxWork.size = "+(maxWork.size)+", ret.size = "+(ret.size) )
        ret.update( afterFull, (
            ret( afterFull )._1,
            /* subtract work which is too much */
            ret( afterFull )._2 - ( cumMaxWork( afterFull+1 ) - work )
        ) )
        (afterFull+1 until ret.size).foreach { i => ret.update(i, ( ret(i)._1, 0 ) ) }
        assert( ret.map( _._2 ).sum == work )
        return ret
    }

}
