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
__global__ void kernelMonteKarloPi( CountType * rnInside, uint32_t nTimes, uint32_t rSeed )
{
    static_assert( sizeof(CountType) == sizeof(uint64_t), "" );

    uint64_t const linId = blockIdx.x * blockDim.x + threadIdx.x;

    CountType seed = ( (CountType) rSeed * linId ) % DEVICE_RAND_MAX;

    #pragma unroll 32  // 0.128823s -> 0.112251s, higher e.g. 64 unrolling brings no notable speedup
    for ( int i = nTimes; i >= 0; --i )
    {
        // not that double can hold integers up to 2**53 exactly, while rSeed of uint32_t only goes to 2**32-1, so no precision is lost in this conversion
        seed = rand( seed );
        SampleType x = SampleType(seed) / SampleType(DEVICE_RAND_MAX);
        seed = rand( seed );
        SampleType y = SampleType(seed) / SampleType(DEVICE_RAND_MAX);
        if ( x*x + y*y < 1.0 )
            atomicAdd( &rnInside[i], 1 );
    }
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

    /* set current device and get device infos */
    int nDevices;
    CUDA_ERROR( cudaGetDeviceCount( &nDevices ) );
    assert( iGpuDeviceToUse < nDevices );
    CUDA_ERROR( cudaSetDevice( iGpuDeviceToUse ) );

    // for GTX 760 this is 12288 threads per device and 384 real cores
    cudaDeviceProp deviceProperties;
    CUDA_ERROR( cudaGetDeviceProperties( &deviceProperties, iGpuDeviceToUse) );
    int const nThreadsPerBlock  = 256;
    int const nBlocks           = deviceProperties.maxThreadsPerMultiProcessor
                                * deviceProperties.multiProcessorCount
                                / nThreadsPerBlock;

    /* Caclulate rolls per thread */
    long unsigned int nTimesPerThread = nTotalRolls / ( nBlocks * nThreadsPerBlock );
    assert( nTimesPerThread <= UINT_MAX );

    /* Debug output of configuration */
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
    CountType * nInside = NULL;
    CountType * dpnInside = NULL;

    CUDA_ERROR( cudaMallocHost( (void**) &nInside, nTimesPerThread * sizeof(nInside[0]) ) );
    CUDA_ERROR( cudaMalloc( (void**) &dpnInside, nTimesPerThread * sizeof(dpnInside[0]) ) );
    CUDA_ERROR( cudaMemset( (void*) dpnInside, 0, nTimesPerThread * sizeof(dpnInside[0]) ) );

    kernelMonteKarloPi<<< nBlocks, nThreadsPerBlock >>>
        ( dpnInside, nTimesPerThread /*nRepeat*/, 237890291 /*seed*/ );

    CUDA_ERROR( cudaMemcpy( (void*) nInside, (void*) dpnInside, nTimesPerThread * sizeof(dpnInside[0]), cudaMemcpyDeviceToHost ) );
    /* cum sum */
    uint64_t sum = 0;
    for ( auto i = 0u; i < nTimesPerThread; ++i )
    {
        sum += nInside[i];
        ((double*) nInside)[i] = 4.0 * sum / ( (double) nBlocks*nThreadsPerBlock*i );
    }

        CUDA_ERROR( cudaEventRecord( stop ) );

    CUDA_ERROR( cudaEventSynchronize( stop ) );
    float milliseconds;
    cudaEventElapsedTime( &milliseconds, start, stop );

    printf( "# Run took %.8f seconds\n", milliseconds / 1000.0f );
    for ( auto i = 1u; i < nTimesPerThread; i+=100 )
        printf( "%lu   %.15f\n", (uint64_t)nBlocks*nThreadsPerBlock*i, ((double*) nInside)[i] );

    /* free all allocated memory */
    CUDA_ERROR( cudaFreeHost( nInside ) );
    CUDA_ERROR( cudaFree( dpnInside ) );

    return 0;
}
