/*
file=PrngPeriodLength; nvcc -arch=sm_30 -x cu $file.cpp -o $file.exe -std=c++11 -DNDEBUG -O3 && ./$file.exe
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
                  CountType const   HOST_RAND_MAX = 0x7FFFFFFFlu;

__device__ inline CountType rand( CountType rSeed )
{
    // why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed ...
    return ((CountType)950706376*rSeed) % DEVICE_RAND_MAX;
}


__global__ void kernelMonteKarloPi(
    uint32_t * const startSeeds,
    uint32_t * const curSeeds,
    uint64_t * const counts,
    uint32_t   const nTriesPerThread
)
{
    uint32_t const linId = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t const seed = startSeeds[linId];
    uint64_t       x    = curSeeds  [linId];
    if ( counts[linId] == 0 )
        x = rand(x);
    /* 1e6 tries takes 0.8s, meaning to reach 2e9 for int max it would take
     * ~ 2000s ~ 33min. This is feasible, but in order to not trigger the
     * watchdog timer it would be necessary to run shorter kernels e.g.
     * each only 100ms long, i.e. ~ 1e5 elements. Therefore we need
     * to save the current seed and the initial seed and the count in three
     * arrays for each kernel thread */
    int i;
    #pragma unroll 32
    for ( i = 0; i < nTriesPerThread; ++i )
    {
        if ( x == seed ) break;
        x = rand(x);
    }

    curSeeds[linId]  = x;
    counts  [linId] += i;

    if ( x == seed )
        printf( "Period length for PRNG with seed %lu (curSeed %lu) was found to be %lu\n",
                seed, curSeeds[linId], counts[linId] );
}

template< class T >
struct GpuArray {
    T * host, * gpu;
    uint64_t const nBytes;
    GpuArray( uint64_t nElements = 1 )
    : nBytes( nElements * sizeof(T) )
    {
        host = (T*) malloc( nBytes );
        CUDA_ERROR( cudaMalloc( (void**) &gpu, nBytes ) );
        assert( host != NULL );
        assert( gpu  != NULL );
    }
    ~GpuArray()
    {
        CUDA_ERROR( cudaFree( gpu ) );
        free( host );
    }
    void down( void )
    {
        CUDA_ERROR( cudaMemcpy( (void*) host, (void*) gpu, nBytes, cudaMemcpyDeviceToHost ) );
    }
    void up( void )
    {
        CUDA_ERROR( cudaMemcpy( (void*) gpu, (void*) host, nBytes, cudaMemcpyHostToDevice ) );
    }
};

int main( void )
{
    /* initialize timer */
    cudaEvent_t start, stop;
    CUDA_ERROR( cudaEventCreate(&start) );
    CUDA_ERROR( cudaEventCreate(&stop) );
    CUDA_ERROR( cudaEventRecord( start ) );

    int const nBlocks = 9, nThreadsPerBlock = 128;
    int const nThreads = nBlocks * nThreadsPerBlock;
    GpuArray<uint32_t> startSeeds(nThreads), curSeeds(nThreads);
    GpuArray<uint64_t> counts(nThreads);

    CUDA_ERROR( cudaMemset( (void**) counts.gpu, 0, counts.nBytes ) );
    startSeeds.host[0] = 684168515 /*411965*/ /*237890291*/;
    for ( int i = 0; i < nThreads; ++i )
        startSeeds.host[i] = startSeeds.host[0] * (i+1) % HOST_RAND_MAX;
    startSeeds.up();
    memcpy( (void*) curSeeds.host, (void*) startSeeds.host, curSeeds.nBytes );
    curSeeds.up();

    for ( auto i = 0u; i < 1e8; ++i )
    {
        kernelMonteKarloPi<<< 1, 1 >>>( startSeeds.gpu, curSeeds.gpu, counts.gpu, 5e4 );
        CUDA_ERROR( cudaDeviceSynchronize() );
        curSeeds.down();
        if ( curSeeds.host[0] == startSeeds.host[0] )
            break;
    }

    CUDA_ERROR( cudaEventRecord( stop ) );
    CUDA_ERROR( cudaEventSynchronize( stop ) );
    float milliseconds;
    cudaEventElapsedTime( &milliseconds, start, stop );

    printf( "Kernel took %.8f seconds.\n", milliseconds / 1000.0f );

    return 0;
}
