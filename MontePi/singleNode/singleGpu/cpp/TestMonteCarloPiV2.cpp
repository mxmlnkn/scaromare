/*
file=TestMonteCarloPiV2; rm $file.exe; nvcc -arch=sm_30 -x cu $file.cpp -o $file.exe -std=c++11; ./$file.exe 268435456
*/

#include <iostream>
#include <cstdlib>   // malloc, free
#include <cuda.h>
//#include <cuda_application_api.h> // cudaMalloc, cudaFree, cudaMemcpy
#include <cstdint>   // uint64_t, ...
#include <cstdlib>   // atoi, atol
#include <climits>   // UINT_MAX
#include <cassert>


void checkCudaError(const cudaError_t rValue, const char * file, int line )
{
    if ( (rValue) != cudaSuccess )
    std::cout << "CUDA error in " << file
              << " line:" << line << " : "
              << cudaGetErrorString(rValue) << "\n";
}
#define CUDA_ERROR(X) checkCudaError(X,__FILE__,__LINE__);

/* forgetting the const-specifier here leads to a 4 times slower code !!!!! */
__device__ static int const DEVICE_RAND_MAX = 0x7FFFFFFFlu;

__device__ inline int rand( int rSeed )
{
    // why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed ...
    return ((uint64_t)950706376*rSeed) % DEVICE_RAND_MAX;
}

typedef unsigned long long int CountType;
/**
 * @param[out] rnInside pointer to number of hits inside quarter circle.
 *                      Watch out: Kernel assumes *rnInside to be 0!
 * @param[in]  rnTimes  How often to repeat rolling the dice.
 * @param[in]  rSeed    Seed which will additionally used to the linear ID
 *                      to make two kernel runs independent from each other
 **/
__global__ void kernelMonteKarloPi( CountType * rnInside, uint32_t nTimes, uint32_t rSeed )
{
    int const linId = blockIdx.x * blockDim.x + threadIdx.x;
    int nInsideCircle = 0;  // local variable

    int seed = ( (uint64_t) rSeed * linId ) % DEVICE_RAND_MAX;

    #pragma unroll 32  // 0.128823s -> 0.112251s, higher e.g. 64 unrolling brings no notable speedup
    for ( int i = nTimes; i >= 0; --i )
    {
        // not that double can hold integers up to 2**53 exactly, while rSeed of uint32_t only goes to 2**32-1, so no precision is lost in this conversion
        seed = rand( seed );
        float x = float(seed) / float(DEVICE_RAND_MAX);
        seed = rand( seed );
        float y = float(seed) / float(DEVICE_RAND_MAX);
        if ( x*x + y*y < 1.0 )
            nInsideCircle++; // no atomic needed, because local variable
    }

    CountType nInsideConverted = nInsideCircle;
    atomicAdd( rnInside, nInsideConverted );  // shfl_down brings no speedup
}


/**
 * Usage is:
 *     program.exe <nTotalRolls> <iGpuDeviceToUse>
 **/
int main( int argc, char** argv )
{
    /* Debug output */
    /*
    std::cout << "Parameters: ";
    for ( auto i = 0; i < argc; ++i )
        std::cout << argv[i] << " ";
    std::cout << std::endl;
    */

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
                                / nThreadsPerBlock
                                * deviceProperties.multiProcessorCount;

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
    double pi = 1.0;
    CountType nInside = 0;
    CountType * dpnInside = NULL;

    CUDA_ERROR( cudaMalloc( (void**) &dpnInside,    sizeof(CountType) ) );
    CUDA_ERROR( cudaMemset( (void*)   dpnInside, 0, sizeof(CountType) ) );
    CUDA_ERROR( cudaMemcpy( &nInside, dpnInside, sizeof(CountType), cudaMemcpyDeviceToHost ) );

    kernelMonteKarloPi<<<nBlocks,nThreadsPerBlock>>>(dpnInside, nTimesPerThread /*nRepeat*/, 237890291 /*seed*/);
    /* as both kernel and memcpy are sent to standardstream, they are executed
     * sequentially on the deviec */
    CUDA_ERROR( cudaMemcpy( &nInside, dpnInside, sizeof(CountType), cudaMemcpyDeviceToHost ) );
    CUDA_ERROR( cudaDeviceSynchronize() );
    pi = 4.0 * double(nInside) / ( (double)nBlocks*nThreadsPerBlock*nTimesPerThread );

        CUDA_ERROR( cudaEventRecord( stop ) );

    CUDA_ERROR( cudaEventSynchronize( stop ) );
    float milliseconds;
    cudaEventElapsedTime( &milliseconds, start, stop );

    printf( "Rolling the dice %lu times resulted in pi ~ %f and took %f seconds\n",
        nTimesPerThread * nThreadsPerBlock * nBlocks, pi, milliseconds / 1000.0f );

    /* free all allocated memory */
    CUDA_ERROR( cudaFree( dpnInside ) );

    return 0;
}
