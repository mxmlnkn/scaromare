/*
file=MontePiError; nvcc -arch=sm_30 -x cu $file.cpp -o $file.exe -std=c++11 -DNDEBUG -O3 && ./$file.exe 2684354560

In contrast to TestMonteCarloPiV2, this version includes basically a loop over
the number of rolls by calculating a cumulative sum, which makes the timings
worthless but gives back faster an output for the error scaling!
*/

#include <iostream>
#include <cstdlib>   // malloc, free
#include <cuda.h>
//#include <cuda_application_api.h> // cudaMalloc, cudaFree, cudaMemcpy
#include <cstdint>   // uint64_t, ...
#include <cstdlib>   // atoi, atol
#include <climits>   // UINT_MAX
#include <cassert>
#include "../../../common/cudacommon.cpp"

typedef unsigned long long int CountType;
typedef float SampleType;


/* forgetting the const-specifier here leads to a 4 times slower code !!!!! */
__device__ static CountType const DEVICE_RAND_MAX = 0x7FFFFFFFlu;

__device__ inline CountType rand( CountType rSeed )
{
    // why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed ...
    return ((CountType)950706376*rSeed) % DEVICE_RAND_MAX;
}


/**
 * @param[out] rnInside pointer to number of hits inside quarter circle.
 *                      Watch out: Kernel assumes *rnInside to be 0!
 * @param[in]  rnTimes  How often to repeat rolling the dice.
 * @param[in]  rSeed    Seed which will additionally used to the linear ID
 *                      to make two kernel runs independent from each other
 **/
__global__ void kernelMonteKarloPi( CountType * rnInside, uint64_t nTimes, uint32_t rSeed )
{
    static_assert( sizeof(CountType) == sizeof(uint64_t), "" );

    uint64_t const linId    = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t const nThreads = gridDim.x * blockDim.x;
    nTimes = (nTimes-linId-1) / ( gridDim.x * blockDim.x ) + 1;
    //CountType seed = ( (CountType) rSeed * linId ) % DEVICE_RAND_MAX;
    //CountType seed = ( (CountType) rSeed * (linId+1) ) % DEVICE_RAND_MAX;
    CountType seed = ( (CountType) rSeed + linId * DEVICE_RAND_MAX / nThreads ) % DEVICE_RAND_MAX;

    #pragma unroll 128  // 0.128823s -> 0.112251s, higher e.g. 64 unrolling brings no notable speedup
    for ( int i = 0; i < nTimes; ++i )
    {
        // not that double can hold integers up to 2**53 exactly, while rSeed of uint32_t only goes to 2**32-1, so no precision is lost in this conversion
        seed = rand( seed );
        SampleType x = SampleType(seed) / SampleType(DEVICE_RAND_MAX);
        seed = rand( seed );
        SampleType y = SampleType(seed) / SampleType(DEVICE_RAND_MAX);
        if ( x*x + y*y < 1.0 )
            atomicAdd( rnInside, 1 );
    }
}


#include "../../../common/getLogSamples.tpp"
#include <chrono>
#include <thread>

double calcPi( uint64_t nRolls, int iDevice )
{
    /* initialize timer */
    cudaEvent_t start, stop;
    CUDA_ERROR( cudaEventCreate(&start) );
    CUDA_ERROR( cudaEventCreate(&stop) );
    CUDA_ERROR( cudaEventRecord( start ) );

    /* allocate and initialize needed memory */
    CountType nInside;
    CountType * dpnInside = NULL;
    CUDA_ERROR( cudaMalloc( (void**) &dpnInside,    sizeof(dpnInside[0]) ) );
    CUDA_ERROR( cudaMemset( (void*)   dpnInside, 0, sizeof(dpnInside[0]) ) );

    int nBlocks, nThreads;
    calcKernelConfig( iDevice, nRolls, &nBlocks, &nThreads );
    kernelMonteKarloPi<<< nBlocks, nThreads >>>
        ( dpnInside, nRolls, 684168515 /*411965*/ /*237890291*/ /*seed*/ );
    CUDA_ERROR( cudaMemcpy( (void*) &nInside, (void*) dpnInside, sizeof(dpnInside[0]), cudaMemcpyDeviceToHost ) );

    CUDA_ERROR( cudaEventRecord( stop ) );
    CUDA_ERROR( cudaEventSynchronize( stop ) );
    float milliseconds;
    cudaEventElapsedTime( &milliseconds, start, stop );

    double const pi = 4.0 * nInside / nRolls;
    printf( "%lu   %.15f   %.8f\n", nRolls, pi, milliseconds / 1000.0f );
    fflush( stdout );
    /* wait a bit so in order to not freeze the whole system */
    if ( milliseconds > 1000 )
        std::this_thread::sleep_for(std::chrono::milliseconds( 500 ));

    CUDA_ERROR( cudaFree( dpnInside ) );
    return pi;
}


/**
 * Usage is:
 *     program.exe <nTotalRolls> <iGpuDeviceToUse>
 **/
int main( int argc, char** argv )
{
    /* interpet command line arguments */
    long unsigned int nTotalRolls = 0;
    int iGpuDeviceToUse = 0;
    assert( argc > 1 ); /* note that argument 0 is the executable name */
    if ( argc > 1 )
    {
        nTotalRolls = atol( argv[1] );
    }
    if ( argc > 2 )
    {
        iGpuDeviceToUse = atoi( argv[2] );
    }

    for ( auto nRolls : getLogSamples( 1, nTotalRolls, 1000 ) )
        calcPi( nRolls, iGpuDeviceToUse );

    return 0;
}
