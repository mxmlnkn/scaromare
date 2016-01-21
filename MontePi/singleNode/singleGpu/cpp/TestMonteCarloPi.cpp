
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

__device__ inline uint64_t getLinearThreadId(void)
{
    return threadIdx.x +
           threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
}

__device__ inline uint64_t getLinearId(void)
{
    uint64_t linId = threadIdx.x;
    uint64_t maxSize = blockDim.x;
    linId += maxSize*threadIdx.y;
    maxSize *= blockDim.y;
    linId += maxSize*threadIdx.z;
    maxSize *= blockDim.z;
    linId += maxSize*blockIdx.x;
    maxSize *= gridDim.x;
    linId += maxSize*blockIdx.y;
    maxSize *= gridDim.y;
    linId += maxSize*blockIdx.z;
    // maxSize *= gridDim.z;
    return linId;
}

__device__ static uint32_t DEVICE_RAND_MAX = 0x7FFFFFFFlu;
__device__ inline uint32_t rand( uint32_t rSeed )
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
    uint64_t linThreadId = getLinearThreadId();
    uint64_t linId = getLinearId();
    CountType nInsideCircle = 0;  // local variable

    // only long long, because atomicAdd didn't allow long int -.- WHY?!
    __shared__ CountType snInsideCircle;
    __syncthreads();
    if ( linThreadId == 0 )   // master thread of every block shall set shared mem variable to 0
        snInsideCircle = 0;
    // make seed unique by multiplying with linear ID
    rSeed = ((uint64_t)rSeed*linId) % DEVICE_RAND_MAX;
    __syncthreads();

    for ( int i = nTimes; i >= 0; --i )
    {
        // not that double can hold integers up to 2**53 exactly, while rSeed of uint32_t only goes to 2**32-1, so no precision is lost in this conversion
        rSeed = rand(rSeed);
        float x = float(rSeed) / float(DEVICE_RAND_MAX);
        rSeed = rand(rSeed);
        float y = float(rSeed) / float(DEVICE_RAND_MAX);
        if ( x*x + y*y < 1.0 )
            nInsideCircle++; // no atomic needed, because local variable
    }
    atomicAdd( &snInsideCircle, nInsideCircle );
    __syncthreads();

#ifndef NDEBUG
    // master thread of every block shall add shared memory result to global memory end result
    if ( linThreadId == 0 )
    {
        printf("block:%i found %llu points inside circle.\n",blockIdx.x,snInsideCircle);
        atomicAdd( rnInside, snInsideCircle );
    }
#endif
}


int main( int argc, char** argv )
{
    // for GTX 760 this is 12288 i.e. 384 real cores
    int nBlocks           = 6;
    int nThreadsPerBlock  = 64;

    long nTimesPerThread = 0;
    assert( argc > 1 );
    if ( argc > 1 )
    {
        nTimesPerThread = atol( argv[1] ) / ( nBlocks * nThreadsPerBlock );
        assert( nTimesPerThread <= UINT_MAX );
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

        cudaEventRecord( start );

    /* allocate and initialize needed memory */
    double pi = 1.0;
    CountType nInside = 0;
    CountType * dpnInside = NULL;

    CUDA_ERROR( cudaMalloc( (void**) &dpnInside,    sizeof(CountType) ) );
    CUDA_ERROR( cudaMemset( (void*)   dpnInside, 0, sizeof(CountType) ) );
    cudaMemcpy( &nInside, dpnInside, sizeof(CountType), cudaMemcpyDeviceToHost );

    kernelMonteKarloPi<<<nBlocks,nThreadsPerBlock>>>(dpnInside, nTimesPerThread /*nRepeat*/, 237890291 /*seed*/);
    /* as both kernel and memcpy are sent to standardstream, they are executed
     * sequentially on the deviec */
    cudaMemcpy( &nInside, dpnInside, sizeof(CountType), cudaMemcpyDeviceToHost );
    cudaDeviceSynchronize();
    pi = 4.0 * double(nInside) / ( (double)nBlocks*nThreadsPerBlock*nTimesPerThread );

        cudaEventRecord( stop );

    cudaEventSynchronize( stop );
    float milliseconds;
    cudaEventElapsedTime( &milliseconds, start, stop );

    printf( "Rolling the dice %lu times resulted in pi ~ %f and took %f seconds\n",
        nTimesPerThread * nThreadsPerBlock * nBlocks, pi, milliseconds / 1000.0f );

    /* free all allocated memory */
    CUDA_ERROR( cudaFree( dpnInside ) );

    return 0;
}
