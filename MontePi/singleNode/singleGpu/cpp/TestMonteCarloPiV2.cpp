/*
file=TestMonteCarloPiV2; rm $file.exe; nvcc -arch=sm_30 -x cu $file.cpp -o $file.exe -std=c++11 -DNDEBUG -O3 --keep; ./$file.exe 268435456
*/

#include <iostream>
#include <cstdlib>   // malloc, free
#include <cuda.h>
//#include <cuda_application_api.h> // cudaMalloc, cudaFree, cudaMemcpy
#include <cstdint>   // uint64_t, ...
#include <cstdlib>   // atoi, atol
#include <climits>   // UINT_MAX
#include <cassert>
#include "../../../../common/cudacommon.cpp"

using CalcType = float;

typedef unsigned long long int uint64;
typedef float SampleType;


/* forgetting the const-specifier here leads to a 4 times slower code !!!!! */
__device__ static uint64 const DEVICE_RAND_MAX = 0x7FFFFFFFlu;

__device__ inline uint64 rand( uint64 rSeed )
{
    // why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed ...
    return ((uint64)950706376*rSeed) % DEVICE_RAND_MAX;
}


/**
 * @param[out] rnInside pointer to number of hits inside quarter circle.
 *                      Watch out: Kernel assumes *rnInside to be 0!
 * @param[in]  rnTimes  How often to repeat rolling the dice.
 * @param[in]  rSeed    Seed which will additionally used to the linear ID
 *                      to make two kernel runs independent from each other
 **/
__global__ void kernelMonteKarloPi
(
    uint64 * const rnInside,
#ifndef NDEBUG
    uint64 * const nIterationsDone,
#endif
    uint64   const nTimesTotal,
    uint32_t       rSeed
)
{
    static_assert( sizeof(uint64) == sizeof(uint64_t), "" );

    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    int nInsideCircle = 0;  // local variable
    int nTimes = (nTimesTotal-linId-1) / ( gridDim.x * blockDim.x ) + 1;

    uint64 seed = ( (uint64) rSeed * linId ) % DEVICE_RAND_MAX;

    //#pragma unroll 16  // 0.128823s -> 0.112251s, higher e.g. 64 unrolling brings no notable speedup
    for ( int i = nTimes; i >= 0; --i )
    {
        // not that double can hold integers up to 2**53 exactly, while rSeed of uint32_t only goes to 2**32-1, so no precision is lost in this conversion
        seed = rand( seed );
        SampleType x = SampleType(seed) / SampleType(DEVICE_RAND_MAX);
        seed = rand( seed );
        SampleType y = SampleType(seed) / SampleType(DEVICE_RAND_MAX);
        if ( x*x + y*y < 1.0 )
            nInsideCircle++; // no atomic needed, because local variable
    }

    uint64 nInsideConverted = nInsideCircle;
    atomicAdd( rnInside, nInsideConverted );  // shfl_down brings no speedup
#ifndef NDEBUG
    atomicAdd( nIterationsDone, nTimes );
#endif
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

    /* Debug output of configuration */
    int nBlocks, nThreadsPerBlock;
    calcKernelConfig( iGpuDeviceToUse, nTotalRolls, &nBlocks, &nThreadsPerBlock ); // this also sets the device!
    float nTimesPerThread = float(nTotalRolls) / ( nBlocks * nThreadsPerBlock );
    int currentDevice;
    CUDA_ERROR( cudaGetDevice( &currentDevice ) );
    std::cout << "Launch " << nBlocks << " blocks with " << nThreadsPerBlock
              << " threads each on GPU device " << currentDevice
              << " (parameter: " << iGpuDeviceToUse << ") each doing "
              << nTimesPerThread << " dice rolls each" << std::endl;

    /* initialize timer */
    cudaEvent_t start, stop;
    CUDA_ERROR( cudaEventCreate(&start) );
    CUDA_ERROR( cudaEventCreate(&stop) );

        CUDA_ERROR( cudaEventRecord( start ) );

    /* allocate and initialize needed memory */
    double pi = 1.0;

    GpuArray<uint64> nInside;
    nInside.host[0] = 0;
    nInside.up();
    #ifndef NDEBUG
        GpuArray<uint64> nIterationsDone;
        nIterationsDone.host[0] = 0;
        nIterationsDone.up();
    #endif


    kernelMonteKarloPi<<<nBlocks,nThreadsPerBlock>>>(
        nInside.gpu,
        #ifndef NDEBUG
            nIterationsDone.gpu,
        #endif
        nTotalRolls /*nRepeat*/, 237890291 /*seed*/);
    /* as both kernel and memcpy are sent to standardstream, they are executed
     * sequentially on the deviec */
    nInside.down();
    #ifndef NDEBUG
        nIterationsDone.down();
        printf( "nIterationsDone = %llu\n", nIterationsDone.host[0] );
        assert( nIterationsDone.host[0] == nTotalRolls );
    #endif
    CUDA_ERROR( cudaDeviceSynchronize() );
    pi = 4.0 * double( nInside.host[0] ) / ( nTotalRolls );

        CUDA_ERROR( cudaEventRecord( stop ) );

    CUDA_ERROR( cudaEventSynchronize( stop ) );
    float milliseconds;
    cudaEventElapsedTime( &milliseconds, start, stop );

    printf( "Rolling the dice %lu times resulted in pi ~ %.15f and took %f seconds\n",
        nTotalRolls, pi, milliseconds / 1000.0f );

    return 0;
}
