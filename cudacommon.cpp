
#include <cstdint>
#include <cassert>
#include <iostream>
#include <cstdlib>  // malloc, free
#include <cuda.h>


void checkCudaError(const cudaError_t rValue, const char * file, int line )
{
    if ( (rValue) != cudaSuccess )
    std::cout << "CUDA error in " << file
              << " line:" << line << " : "
              << cudaGetErrorString(rValue) << "\n";
}
#define CUDA_ERROR(X) checkCudaError(X,__FILE__,__LINE__);

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


template< class T >
struct GpuArray
{
    T * host, * gpu;
    unsigned long long int const nBytes;
    cudaStream_t mStream;

    inline GpuArray
    (
        unsigned long long int const nElements = 1,
        cudaStream_t rStream = 0
    )
    : nBytes( nElements * sizeof(T) ),
      mStream( rStream )
    {
        host = (T*) malloc( nBytes );
        CUDA_ERROR( cudaMalloc( (void**) &gpu, nBytes ) );
        assert( host != NULL );
        assert( gpu  != NULL );
    }
    inline ~GpuArray()
    {
        CUDA_ERROR( cudaFree( gpu ) );
        free( host );
    }
    inline void down( void )
    {
        CUDA_ERROR( cudaMemcpyAsync( (void*) host, (void*) gpu, nBytes, cudaMemcpyDeviceToHost ) );
        CUDA_ERROR( cudaPeekAtLastError() );
    }
    inline void up( void )
    {
        CUDA_ERROR( cudaMemcpyAsync( (void*) gpu, (void*) host, nBytes, cudaMemcpyHostToDevice ) );
        CUDA_ERROR( cudaPeekAtLastError() );
    }
};
