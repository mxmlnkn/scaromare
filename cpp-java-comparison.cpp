public void gpuMethod()
{
    final int randMax = 0x7FFFFFFF;
    final long randMagic = 950706376;

    long dRandomSeed = mRandomSeed;
    final long dnDiceRolls = mnDiceRolls;
    long nHits = 0;
    assert( dnDiceRolls <= Integer.MAX_VALUE );

    for ( long i = 0; i < dnDiceRolls; ++i )
    {
        dRandomSeed = (int)( (randMagic*dRandomSeed) % randMax );
        float x = (float) dRandomSeed / randMax;
        dRandomSeed = (int)( (randMagic*dRandomSeed) % randMax );
        float y = (float) dRandomSeed / randMax;

        if ( x*x + y*y < 1.0 )
            nHits += 1;
    }
    mnHits[ miLinearThreadId ] = nHits;
}

__device__ static uint32_t DEVICE_RAND_MAX = 0x7FFFFFFFlu;
__device__ inline uint32_t rand( uint32_t rSeed )
{
    // why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed ...
    return ((uint64_t)950706376*rSeed) % DEVICE_RAND_MAX;
}

__global__ void kernelMonteKarloPi( CountType * rnInside, uint32_t nTimes, uint32_t rSeed )
{
    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    int nInsideCircle = 0;

    rSeed = ( (uint64_t) rSeed * linId ) % DEVICE_RAND_MAX;

    #pragma unroll 32
    for ( int i = nTimes; i >= 0; --i )
    {
        rSeed = rand(rSeed);
        float x = float(rSeed) / float(DEVICE_RAND_MAX);
        rSeed = rand(rSeed);
        float y = float(rSeed) / float(DEVICE_RAND_MAX);
        if ( x*x + y*y < 1.0 )
            nInsideCircle++;
    }

    CountType nInsideConverted = nInsideCircle;
    atomicAdd( rnInside, nInsideConverted );
}

/**
 * Notable Differences:
 *   - In the Java version the reduction is on the Host
 *   - In the Java version the random number generator always works in 64-bit
 *     but that should rather slow it down, than speed it up by factor 2 :S
 *     (changing it to 32-bit changes nothing. This makes me suspicious as to
 *      what kind of optimizations are applied)
 *   - the C++ version uses unsigned int where Java doesn't even have unsigned
 *     integers ... -.- Tested: changing that has no significant effect
 *   - even though in C++ the rand function should be inlined, maybe it is not
 *     Tested: changes nothing, meaning it is already correctly inlined
 *   - use 0x7FFFFFFF directly instead of DEVICE_RAND_MAX! WORKS! Speedup of 4!
 *     I'm not sured how I could have spotted that error with nvvp. But after
 *     solving that performance bug the Misc. instructions are much less
 *     significant! Also before FP were almost pure FP-MUL-ADDs, now the
 *     most part is MUL ... But this should rather lead to a speeddown ...
 **/
