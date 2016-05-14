#pragma once

#include <cassert>
#include <cstdint>  // uint64_t
#include <vector>
#include <cmath>    // logf, exp, pow, ceil

std::vector<uint64_t> getLogSamples
(
    uint64_t const riStartPoint,
    uint64_t const riEndPoint,
    uint64_t const rnPoints,
    bool     const endPoint = true
)
{
    assert( riStartPoint > 0 );
    assert( riEndPoint > riStartPoint );
    assert( rnPoints > 0 );
    assert( rnPoints <= riEndPoint-riStartPoint+1 );

    std::vector<uint64_t> tmpPoints( rnPoints );
    std::vector<uint64_t> points( rnPoints );

    /* Naively create logspaced points rounding float to int */
    const double dx = ( logf(riEndPoint) - logf(riStartPoint) ) / (rnPoints-endPoint);
    const double factor = exp(dx);

    tmpPoints[0] = riStartPoint;
    uint64_t nWritten = 1;
    for ( unsigned int i = 1u; i < rnPoints; ++i )
    {
        assert( nWritten < rnPoints );
        tmpPoints[nWritten] = ceil( riStartPoint * pow( factor, i) ); // 0.4 -> 1
        /* sift out values which appear twice */
        if ( nWritten == 0 || tmpPoints[nWritten] != tmpPoints[nWritten-1] )
            ++nWritten;
    }
    /* set last point manually because of rounding errors */
    if ( endPoint == true )
        tmpPoints[ nWritten-1 ] = riEndPoint;

    /* fill in as many values as deleted */
    int nToInsert = rnPoints - nWritten;
    points[0] = tmpPoints[0];
    uint64_t iTarget = 1;
    for ( unsigned int iSrc = 1u; iSrc < rnPoints; ++iSrc )
    {
        for ( ; nToInsert > 0 and points[iTarget-1] < tmpPoints[iSrc]-1;
              ++iTarget, --nToInsert )
        {
            assert( iTarget >= 1 && iTarget < points.size() );
            points[iTarget] = points[iTarget-1] + 1;
        }
        assert( iSrc    < tmpPoints.size() );
        //assert( iTarget < points.size() );
        /* this condition being false should only happen very rarely, if
         * rnPoints == riEndPoint - riStartPoint + 1 */
        if ( iTarget < points.size() )
            points[iTarget++] = tmpPoints[iSrc];
    }

    for ( unsigned int i = 1u; i < points.size(); ++i )
        assert( points[i-1] < points[i] );

    return points;
}
