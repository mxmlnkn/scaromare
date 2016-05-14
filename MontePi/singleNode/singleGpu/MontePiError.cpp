/*
file=MontePi; nvcc -arch=sm_30 -x cu $file.cpp -o $file.exe -std=c++11 -DNDEBUG -O3 && ./$file.exe 2684354560

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


typedef unsigned long long int CountType;
typedef float SampleType;


void checkCudaError(const cudaError_t rValue, const char * file, int line )
{
    if ( (rValue) != cudaSuccess )
    std::cout << "CUDA error in " << file
              << " line:" << line << " : "
              << cudaGetErrorString(rValue) << "\n";
}
#define CUDA_ERROR(X) checkCudaError(X,__FILE__,__LINE__);


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

    #pragma unroll 32  // 0.128823s -> 0.112251s, higher e.g. 64 unrolling brings no notable speedup
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

template< typename T, typename S >
inline T ceilDiv( T a, S b ) { return (a+b-1)/b; }

/**
 * Chooses an optimal configuration for number of blocks and number of threads
 * Note that every kernel may have to calculate on a different amount of
 * elements, so this needs to be calculated inside the kernel with:
 *    for ( i = linid; i < nElements; i += nBlocks * nThreads )
 * which yields the following number of iterations:
 *    nIterations = (nElements-1 - linid) / ( nBlocks * nThreads ) + 1
 * derivation:
 *    search for highest n which satisfies i + n*s <= m-1
 *    note that we used <= m-1 instead of < m to work with floor later on
 *    <=> search highest n: n <= (m-1-i)/s
 *    which is n = floor[ (m-1-i)/s ]. Note that floor wouldn't be possible
 *    for < m, because it wouldn't account for the edge case for (m-1-i)/s == n
 *    the highest n means the for loop will iterate with i, i+s, i+2*s, i+...n*s
 *    => nIterations = n+1 = floor[ (m-1-i)/s ] + 1
 */
void calcKernelConfig( int iDevice, uint64_t n, int * nBlocks, int * nThreads )
{
    int const nMaxThreads  = 256;
    int const nMinElements = 32; /* The assumption: one kernel with nMinElements work won't be much slower than nMinElements kernels with each 1 work element. Of course this is workload / kernel dependent, so the fixed value may not be the best idea */

    /* set current device and get device infos */
    int nDevices;
    CUDA_ERROR( cudaGetDeviceCount( &nDevices ) );
    assert( iDevice < nDevices );
    CUDA_ERROR( cudaSetDevice( iDevice ) );

    // for GTX 760 this is 12288 threads per device and 384 real cores
    cudaDeviceProp deviceProperties;
    CUDA_ERROR( cudaGetDeviceProperties( &deviceProperties, iDevice) );

    int const nMaxThreadsGpu = deviceProperties.maxThreadsPerMultiProcessor
                             * deviceProperties.multiProcessorCount;
    if ( n < (uint64_t) nMaxThreadsGpu * nMinElements )
    {
        auto nThreadsNeeded = ceilDiv( n, nMinElements );
        *nBlocks  = ceilDiv( nThreadsNeeded, nMaxThreads );
        *nThreads = nMaxThreads;
        if ( *nBlocks == 1 )
        {
            assert( nThreadsNeeded <= nMaxThreads );
            *nThreads = nThreadsNeeded;
        }
    }
    else
    {
        *nBlocks  = nMaxThreadsGpu / nMaxThreads;
        *nThreads = nMaxThreads;
    }
    assert( *nBlocks > 0 );
    assert( *nThreads > 0 );
    uint64_t nIterations = 0;
    for ( uint64_t linid = 0; linid < (uint64_t) *nBlocks * *nThreads; ++linid )
    {
        /* note that this only works if linid < n */
        assert( linid < n );
        nIterations += (n-linid-1) / ( *nBlocks * *nThreads ) + 1;
        //printf( "[thread %i] %i elements\n", linid, (n-linid) / ( *nBlocks * *nThreads ) );
    }
    //printf( "Total %i elements out of %i wanted\n", nIterations, n );
    assert( nIterations == n );
}

#include "../../../getLogSamples.tpp"
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
