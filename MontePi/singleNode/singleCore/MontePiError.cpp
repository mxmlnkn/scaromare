/*
g++ MontePiError.cpp -std=c++11 -O3 && ./a.out | tee error.dat && python ../../plot.py -e error.dat

In contrast to TestMonteCarloPiV2, this version does not benchmark the time
needed and the random number generator is sequential which is not the case for
singleNode/singleCpu/MontePiError.cpp!
*/

#include <iostream>
#include <cstdlib>   // malloc, free
//#include <cuda_application_api.h> // cudaMalloc, cudaFree, cudaMemcpy
#include <cstdint>   // uint64_t, ...
#include <cstdlib>   // atoi, atol
#include <climits>   // UINT_MAX
#include <cassert>
#include "../../../getLogSamples.tpp"


typedef unsigned long long int CountType;
typedef float SampleType;

static uint64_t const DEVICE_RAND_MAX = 0x7FFFFFFFlu;

inline uint32_t rand( CountType rSeed )
{
    // why 0x7F... instead 0xFF ??? ... I mean it is unsigned, not signed ...
    return ((uint64_t)950706376*rSeed) % DEVICE_RAND_MAX;
}

#include <fstream>
std::vector<uint32_t> readRandSeeds( std::string const fname )
{
    using namespace std;
    fstream dat( fname.c_str(), ios_base::in );
    std::vector<uint32_t> results;
    uint32_t tmp = 0;
    uint8_t nTmp = 0;
    int number;
    while ( dat >> number )
    {
        assert( number >= 0 && number <= 255 );
        if ( ++nTmp >= 4 )
        {
            nTmp = 0;
            results.push_back( tmp % DEVICE_RAND_MAX );
            tmp = 0;
        }
        else
        {
            tmp = ( tmp << 8 ) + number;
        }
    }
    return results;
}

#include <chrono>
double calcPi( uint32_t nRolls, uint32_t seed )
{
    auto now = [](){ return std::chrono::high_resolution_clock::now(); };
    auto t0 = now();

    uint64_t nInside = 0;
    for ( uint32_t i = 0; i < nRolls; ++i )
    {
        // not that double can hold integers up to 2**53 exactly, while rSeed of uint32_t only goes to 2**32-1, so no precision is lost in this conversion
        seed = rand( seed );
        float x = float(seed) / float(DEVICE_RAND_MAX);
        seed = rand( seed );
        float y = float(seed) / float(DEVICE_RAND_MAX);
        if ( x*x + y*y < 1.0 )
            ++nInside;
    }

    double const pi = 4.0 * nInside / nRolls;
    printf( "%lu   %.15f   %.8f\n", nRolls, pi,
        std::chrono::duration_cast<std::chrono::duration<double>>( now()-t0 ) );
    fflush( stdout );

    return pi;
}

double calcPiParallel( uint32_t nRolls, uint32_t const seed0 )
{
    auto now = [](){ return std::chrono::high_resolution_clock::now(); };
    auto t0 = now();

    uint64_t nInside = 0;
    int nThreads = 4;
    int nOddThreads = nRolls % nThreads;
    for ( int linId = 0; linId < nThreads; ++linId )
    {
        uint32_t nRollsPerThread = nRolls / nThreads;
        if ( linId < nOddThreads )
            ++nRollsPerThread;
        auto seed = ( (uint64_t) seed0 + linId * DEVICE_RAND_MAX / nThreads ) % DEVICE_RAND_MAX;

        for ( uint32_t i = 0; i < nRollsPerThread; ++i )
        {
            // not that double can hold integers up to 2**53 exactly, while rSeed of uint32_t only goes to 2**32-1, so no precision is lost in this conversion
            seed = rand( seed );
            float x = float(seed) / float(DEVICE_RAND_MAX);
            seed = rand( seed );
            float y = float(seed) / float(DEVICE_RAND_MAX);
            if ( x*x + y*y < 1.0 )
                ++nInside;
        }
    }

    double const pi = 4.0 * nInside / nRolls;
    printf( "%lu   %.15f   %.8f\n", nRolls, pi,
        std::chrono::duration_cast<std::chrono::duration<double>>( now()-t0 ) );
    fflush( stdout );

    return pi;
}

template <class T>
float covariance( std::vector<T> a, std::vector<T> b )
{
    float meanA = 0, meanB = 0;
    for ( auto x : a ) meanA += x;
    for ( auto x : b ) meanB += x;
    meanA /= a.size();
    meanB /= b.size();

    float cov = 0;
    assert( a.size() == b.size() );
    for ( auto i = 0u; i < a.size(); ++i )
        cov += ( a[i]-meanA ) * ( b[i] - meanB );
    cov /= a.size();

    return cov;
}

template <class T>
float covariance( T* a, int nA, T * b, int nB )
{
    int N = fmin( nA, nB );

    float meanA = 0, meanB = 0;
    for ( int i = 0; i < N; ++i )
    {
        meanA += a[i];
        meanB += b[i];
    }
    meanA /= N;
    meanB /= N;

    float cov = 0;
    for ( int i = 0; i < N; ++i )
        cov += ( a[i]-meanA ) * ( b[i] - meanB );
    cov /= N;

    return cov;
}

#include <cmath>
#include <iomanip>
int main( void )
{
    auto seeds = readRandSeeds( "../../../random.dat" );

    /* for 1e8 the covariance explodes basically Oo ... really correlated, or
     * maybe some kind of precision overflow? (because it also happens to
     * the sequential version */
    unsigned int nPerThread = 4e7;
    /* check correlation between differing kind of subsequences in order
     * to find a scheme which works well in parallel. In order to simulate
     * large amounts of rolls, just use a 16-bit PRNG */
    auto seed = seeds[0];
    std::vector<float> x;
    for ( int i = 0; i < nPerThread; ++i )
    {
        uint64_t const DEVICE_RAND_MAX = 0x7FFFFFFFlu;
        seed = ((uint64_t)950706376*seed) % DEVICE_RAND_MAX;
        x.push_back( (float) seed / DEVICE_RAND_MAX );
    }
    for ( int i = 0; i < 10; ++i )
        std::cout << x[i] << " ";
    std::cout << std::endl;
    auto cov = covariance( std::vector<float>( x.begin(), x.begin() + x.size()/2 ),
                           std::vector<float>( x.begin() + x.size()/2, x.end() ) );
    std::cout << "cov = " << cov << std::endl;

    /* now in parallel scheme */
    unsigned int nThreads   = 8;
    auto y = new float[ nPerThread * nThreads ];
    for ( auto i = 0u; i < nThreads;   ++i )
    {
        /* take randomly spaced values */
        /* adding random numbers from the same generator only should result
         * in 100% correlated sequences, if additive constant is 0 */
        //auto seed = ( (uint64_t) seeds[0] + seeds[i] ) % DEVICE_RAND_MAX;
        /* take perfect lattice distributed seed values */
        auto seed = ( (uint64_t) seeds[0] + i* DEVICE_RAND_MAX / nThreads ) % DEVICE_RAND_MAX;

        for ( auto j = 0u; j < nPerThread; ++j )
        {
            uint64_t const DEVICE_RAND_MAX = 0x7FFFFFFFlu;
            seed = ((uint64_t)950706376*seed) % DEVICE_RAND_MAX;

            y[ i*nPerThread + j ] = (float) seed / DEVICE_RAND_MAX;
        }
    }
    for ( int i = 0; i < nThreads; ++i )
    {
        std::cout << std::setw(3) << i << ": ";
        for ( int j = 0; j < i; ++j )
        {
            std::cout << covariance( y + i*nPerThread, nPerThread,
                                     y + j*nPerThread, nPerThread ) << " ";
        }
        std::cout << std::endl;
    }
    return 0;


    /* check how Monte-Carlo Integration scales, i.e. if the parallel pseudo
     * random numbers are sufficiently random */
    uint32_t nTotalRolls = 1e7;
    auto nSeeds = 10u;

    for ( auto i = 0u; i < fmin( nSeeds, seeds.size() ); ++i )
    {
        for ( auto nRolls : getLogSamples( 1, nTotalRolls, 500 ) )
        {
            calcPi( nRolls, seeds[i] );
            //calcPiParallel( nRolls, seeds[i] );
        }
    }

    return 0;
}
