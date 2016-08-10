#!/usr/bin/python
# -*- coding: utf-8 -*-
# python plotAll.py results-2016-02-22_08-09-37.log taurus/results-2016-02-22_10-09-10.log error-scaling.log


from numpy import *
import matplotlib.pyplot as plt
import sys
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument("rundir", help="Directory which contains 'simOutput'")
parser.add_argument("-e", "--error-log", dest="errorlog", help="File with data from error scaling", type=str, nargs="*" )
parser.add_argument("-w", "--workload-log", dest="workloadlog", help="File with data from benchmark", type=str, nargs=1 )
parser.add_argument("-k", "--weakscaling", dest="weakscaling", help="File with data from benchmark", type=str, nargs=1 )
parser.add_argument("-s", "--strongscaling", dest="strongscaling", help="File with data from benchmark", type=str, nargs=1 )
parser.add_argument("-r", "--rootbeer-setup-time", dest="rootbeersetup", help="File with data from benchmark", type=str, nargs=1 )
args = parser.parse_args()


def bisectionExtrema( f,a,b,nIterations=16,debug=False ):
    """
    " finds the extremum of f(x) in the interval (a,b)
    " assumes that b > a and that only one extremum exists in [a,b]
    """
    """
    " ^
    " |           .´
    " |  .'',   .'
    " |.'    ´'`
    " +--------------------
    "  a  c  b
    """
    extremaWasInside = True
    for i in range(nIterations):
        if b < a:
            a,b = b,a
        c  = 0.5 *(a+b)
        # everything smaller than interval width / 6 should basically be enough
        # if the factor is too small it may result in problems like the
        # bisection quitting too early, thereby giving back wrong maxima!
        dx = 1e-2*(b-a)
        # if floating point precision exhausted then these differences will be 0
        # for a and b do only onesided, because else we would leave the given interval!
        left   = f(a+dx) > f(a   )
        middle = f(c+dx) > f(c-dx)
        right  = f(b   ) > f(b-dx)
        if left == middle and middle == right and i == 0:
            extremaWasInside = False

        if debug:
            print "f at x=",a,"going up?",left  ," ( f(x+dx)=",f(a+dx),", f(x   =",f(a   )
            print "f at x=",c,"going up?",middle," ( f(x+dx)=",f(c+dx),", f(x-dx)=",f(c-dx)
            print "f at x=",b,"going up?",right ," ( f(x   )=",f(b   ),", f(x-dx)=",f(b-dx)

        # If the sign of the derivative is the same for all, then
        # the maximum is not inside the specified interval!
        if ( left == middle and middle == right ):
            if extremaWasInside:
                break   # this can also happen if dx is too small to resolve, so we break the search
            else:
                raise Exception(
                    "Specified Interval seems to not contain any extremum!\n" +
                    "  ["+str(a)+","+str(b)+"]: f(a)=" + str(f(a)) +
                    ", f((a+b)/2)=" + str(f(c)) + "f(b)=" + str(f(b))
                )
                return None, None, None # unreasonable result, therefore error code
        elif left == middle:
            a = c
        elif middle == right:
            b = c
        else:
            # This happens if there are two extrema inside interval
            raise Exception( "Specified Interval has more than one extremum!" )
            return None, None, None

    c = 0.5*(a+b)
    return f(c), c, 0.5*(b-a)

def bisectionNExtrema( f,a,b,nExtrema,nIterations=16,maxLevel=8,debug=False ):
    """
    " finds at least nExtrema extrema of f(x) in the interval (a,b)
    " assumes that b > a and that only one extremum exists in (a,b)
    " maxLevel ... specifies maximum iterations. In every iteration the
    "              interval count will be doubled in order to find more
    "              extrema
    """
    assert( nExtrema != 0 )
    nExtremaFound      = 0
    nIntervals         = nExtrema+1
    for curLevel in range(maxLevel):
        if nExtremaFound >= nExtrema:
            break
        xi = np.linspace( a,b,nIntervals )
        # not simply *2, because we don't want half of the xi be on the same
        # positions like in the last iteration as that could be unwanted if
        # one such x is exactly on an extrema
        dx = 1e-7*(b-a)
        assert( dx < 0.5*(xi[1]-xi[0]) )
        fIncreasing =  f(xi+dx) > f(xi-dx)
        nExtremaFound = np.sum( np.logical_xor( fIncreasing[:-1], fIncreasing[1:] ) )
        if debug:
            print "nIntervals = ",nIntervals," with ",nExtremaFound," extrema"
        nIntervals = nIntervals*2+1

    if nExtremaFound < nExtrema:
        return np.zeros(0)  # error code

    extrema = np.empty( nExtremaFound )
    curExtremum = 0
    extremumInInterval = np.logical_xor( fIncreasing[:-1], fIncreasing[1:] )
    for i in range(len(extremumInInterval)):
        if not extremumInInterval[i]:
            continue
        if debug:
            sys.stdout.write( "Find extremum in ["+str(xi[i])+","+str(xi[i+1])+"] : " )
        xmax = bisectionExtrema( f, xi[i], xi[i+1], nIterations, debug )
        if (xi[i] <= xmax) and (xmax <= xi[i+1]):  # check for error code of bisectionExtrema
            extrema[curExtremum] = xmax
            curExtremum += 1

        if debug:
            if (xi[i] <= xmax) and (xmax <= xi[i+1]):  # check for error code of bisectionExtrema
                print "found at ",xmax
            else:
                print "not found!"

    return extrema

def relErr( x, y ):
    from numpy import zeros
    assert len(x) == len(y)
    non0 = abs(y) > 1e-16
    #non0 = abs(y) != 0
    tmp = ( x[non0] - y[non0] ) / y[non0]
    res = zeros( len(y) )
    res[non0] = tmp
    return y[non0], abs(res[non0])

def p(x0,y0,x):
    """
    This function approximates f so that f(x0)=y0 and returns f(x)
    """
    assert( len(x0) == len(y0) )
    # number of values. Approximating polynomial is of degree n-1
    n = len(x0)
    assert( n > 1 )
    res = 0
    for k in range(n):
        prod = y0[k]
        for j in range(n):
            if j != k:
                prod *= (x0[j]-x)/(x0[j]-x0[k])
        res += prod
    return res


def finishPlot( fig, ax, fname, loc='best' ):
    if not isinstance( ax, list):
        ax = [ ax ]
    for a in ax:
        l = a.legend( loc=loc, prop={'size':10}, labelspacing=0.2, # fontsize=10 also works
                      fancybox=True, framealpha=0.5 )
    #if l != None:
    #    l.set_zorder(0)  # alternative to transparency
    fig.tight_layout()
    fig.savefig( fname+".pdf" )
    print "[Saved '"+fname+".pdf']"
    fig.savefig( fname+".png" )
    print "[Saved '"+fname+".png']"
    plt.close( fig )

def axisIsLog( ax, axis ):
    assert ( axis == 'x' or axis == 'y' ) # axis neither 'x' nor 'y'!
    if axis == 'x':
        return ax.get_xscale() == 'log'
    elif axis == 'y':
        return ax.get_yscale() == 'log'

def axisMin( ax, axis ):
    xmin=float('+inf')
    isLog = axisIsLog( ax, axis )
    for line in ax.get_lines():
        if axis == 'x':
            x = line.get_xdata()
        else:
            x = line.get_ydata()
        if isLog:
            x = x[ x>0 ]
        xmin = min( xmin, min(x) )
    return xmin

def axisMax( ax, axis ):
    xmax=float('-inf')
    isLog = axisIsLog( ax, axis )
    for line in ax.get_lines():
        if axis == 'x':
            x = line.get_xdata()
        else:
            x = line.get_ydata()
        if isLog:
            x = x[ x>0 ]
        xmax = max( xmax, max(x) )
    return xmax

def autoRange( ax, axis, lb, rb = None ):
    if rb == None:
        rb = lb
    isLog = axisIsLog( ax, axis )

    xmin=axisMin( ax, axis )
    xmax=axisMax( ax, axis )

    from math import log,exp

    if isLog:
        dx   = log(xmax) - log(xmin)
        xmin /= exp( lb*( dx ) )
        xmax *= exp( rb*( dx ) )
    else:
        dx = xmax - xmin
        xmin -= lb*dx
        xmax += rb*dx

    if axis == 'x':
        ax.set_xlim( [xmin,xmax] )
    else:
        ax.set_ylim( [xmin,xmax] )

def autoRangeXY( ax, lb = 0.1, rb = None, bb = None, tb = None ):
    if rb == None:
        rb = lb
    if tb == None:
        tb = lb
    if bb == None:
        bb = lb

    autoRange( ax, 'x', lb, rb )
    autoRange( ax, 'y', bb, tb )

def autoLabel( ax, axis, nbins=5, roundFunc=ceil ):
    """
    This functions is a workaround for ticks being too many or too few in
    log scale.
    https://github.com/matplotlib/matplotlib/issues/6549
    """
    from math import log10,ceil,floor
    xmin  = axisMin( ax, axis )
    xmax  = axisMax( ax, axis )
    isLog = axisIsLog( ax, axis )
    assert isLog # Autolabeling only implemented for log scale yet
    if isLog:
        dx   = roundFunc( ( log10(xmax) - log10(xmin) ) / nbins )

    from numpy import arange
    n0 = int( floor( log10( xmin ) ) )
    n1 = int( ceil ( log10( xmax ) ) )
    #print "n0 =",n0,", n1 =",n1,", dx =",dx
    xpos = 10.**( n0 + arange(nbins+2)*dx )
    ax.set_xticks( xpos )
    #print "set xlabels at : ", xpos

def calcStatisticsFromDupes( x, y ):
    # If x contains duplicates, then those elements will be merged and the
    # corresponding elements in y will be calculated to mean + standard deviation
    # x,y must be numpy arrays
    assert( x.size == y.size )
    todo = ones( x.size, dtype=bool )
    i = 0
    xres = empty( x.size )
    yres = empty( y.size )
    yerr = empty( y.size )
    while todo.sum() > 0:
        assert( i < x.size )
        dupes = x == x[todo][0]
        xres[i] = x[dupes][0]
        yres[i] = mean( y[dupes] )
        yerr[i] = std( y[dupes] ) if y[dupes].size >= 3 else  0.
        i += 1
        # no found dupes should already be found!
        nextTodo = logical_and( todo, logical_not( dupes ) )
        assert( dupes.sum() == todo.sum() - nextTodo.sum() )
        todo = nextTodo
    return xres[:i], yres[:i], yerr[:i]

def plotBenchmark( ax, rx,ry,rz=None, label=None, color=None ):
    """
    " Creates multiple lines for each different z-Values (so there shouldn't
    " be too may different z values or the plot will get chaotic.
    " ^ y             +-----+
    " |           .´  |..z=1|
    " |  .'',   .'.´  |..z=2|
    " |.'.'',´'`.'.´  |..z=3|
    " |.'.'',´'`.'    +-----+
    " |.'    ´'`            x
    " +--------------------->
    " If there are multiple duplicate x-values to one z value, then the
    " corresponding st of y values will get merged to mean y +- std y
    """
    assert len(rx) == len(ry)
    if rz != None:
        assert len(ry) == len(rz)
    else:
        rz = zeros( len(rx) )
        params = array( [0] )
    # Sort z-values for correct coloring and overlay z-order
    params = sort( unique(rz) )
    colors = linspace( 0,1, len( params ) )
    for i in range( len(params) ):
        filter = rz == params[i]
        x = array( rx[filter] )
        y = array( ry[filter] )

        x,y,sy = calcStatisticsFromDupes( x, y )

        # sort by x-value or else the connecting line will make zigzags
        sorted = argsort( x )
        x  = x [sorted]
        y  = y [sorted]
        sy = sy[sorted]

        ax.errorbar( x, y, sy, linestyle='-', marker='.',
                     label=None if label == None else (
                        label if isinstance( label, basestring ) else
                        label( params[i] )
                     ), color=color if color != None or len(params) == 1 else
                            plt.get_cmap( 'cool' )(colors[i]) )

    autoRangeXY( ax, 0,0, 0.1,0.1 )

########################### workload scaling ###########################

if args.workloadlog != None:
    data = genfromtxt( args.workloadlog[0] );
    print "data (from "+args.workloadlog[0]+") = ",data

    fig = plt.figure( figsize=(6,4) )
    ax = fig.add_subplot( 111,
        xlabel = r"Anzahl an Monte-Carlo-Iterationen $N$",
        ylabel = u"Ausführungszeit / s",
        xscale = 'log',
        yscale = 'log'
    )
    plotBenchmark( ax, data[ data[:,1]>0, 0 ], data[ data[:,1]>0, 1 ], label="single core (Java)"  )
    plotBenchmark( ax, data[ data[:,2]>0, 0 ], data[ data[:,2]>0, 2 ], label="single core (Scala)" )
    plotBenchmark( ax, data[ data[:,3]>0, 0 ], data[ data[:,3]>0, 3 ], label="single GPU (C++)"    )
    plotBenchmark( ax, data[ data[:,4]>0, 0 ], data[ data[:,4]>0, 4 ], label="single GPU (Java)"   )
    plotBenchmark( ax, data[ data[:,5]>0, 0 ], data[ data[:,5]>0, 5 ], label="single GPU (Scala)"  )
    #ax.plot( data[:,0], data[:,6] , 'o-', label="Spark local" )
    #ax.plot( data[:,0], data[:,7] , 'o-', label="Spark + GPUs" )
    #ax.plot( data[:,0], data[:,8] , 'o-', label="Spark 2 nodes" )
    #ax.plot( data[:,0], data[:,9] , 'o-', label="Spark 2 nodes 10x workload" )
    #ax.plot( data[:,0], data[:,10] / data[:,10], 'o-', label="Spark 2 nodes 100x workload" )
    #data = genfromtxt( sys.argv[2] );
    #ax.plot( data[:,0], data[:,3], 'o-', label="Tesla K80: single GPU (C++)" )
    #ax.plot( data[:,0], data[:,4], 'o-', label="Tesla K80: single GPU (Java)" )
    #ax.plot( data[:,0], data[:,5], 'o-', label="Tesla K80: single GPU (Scala)" )

    # scaling comparison
    x = linspace( data[0,0], data[-1,0], 50 )
    y = 1e-7*x**1
    ax.plot( x, y, 'k--', lw=1.5, label=r"$\sim N$" )            # O(h^1)
    #ax.text( 1e-4, 1e-2, r"$\sim h^{-0.5}$", fontsize=18, color="b", label="Scaling ~N" )

    Npos = data[ any( data > 0, axis=1 ), 0 ]
    ax.set_xlim( [ Npos[0], Npos[-1] ] )
    tpos = data[:,1:5][ data[:,1:5] > 0 ]
    a = tpos.min(); b = tpos.max(); d=0.1*log(b-a)
    ax.set_ylim( [ a/10**d, b*10**d ] )

    finishPlot( fig, ax, "benchmarks-workload-scaling" )


########################### error scaling ###########################

import os.path

if args.errorlog:
    fig = plt.figure( figsize=(6,3) )
    ax = fig.add_subplot( 111,
        xlabel = r"Anzahl an Monte-Carlo-Iterationen $N$",
        ylabel = r"$\frac{ \left| \tilde{\pi} - \pi \right| }{\pi}$",
        xscale = 'log',
        yscale = 'log'
    )

    xmin  = uint64(-1)  # UINT64_MAX
    xmax  = 0
    iArg  = -1
    iPlot = 0
    markers = [ '.', '.' ]
    while iArg < len( args.errorlog )-1:
        iArg += 1
        print iArg
        file = args.errorlog[iArg]
        if not os.path.exists( file ):
            print file,"not found!"
            continue
        else:
            data = genfromtxt( file );

        label = file
        if len( args.errorlog ) < iArg+1 and not os.path.exists( args.errorlog[iArg+1] ):
            label = args.errorlog[iArg+1]
            print "When trying to plot",file,"the next argument encountered was found to not be a valid file, therefore interpret '"+label+"'as the label to use!"
        ax.yaxis.label.set_size(18)
        ax.plot( data[:,0], abs( data[:,1] - pi ) / pi, markers[ iPlot % len(markers) ], markersize=4, label=label.decode("utf-8"))
        iPlot += 1

        xmin = min( xmin, data[ 0,0] )
        xmax = max( xmax, data[-1,0] )
        print "xmin=",xmin,", xmax=",xmax

    # scaling comparison
    x = linspace( xmin, xmax, 50 )
    y = xmin * x**(-0.5) / xmin**(0.5)
    ax.plot( x, y, 'k--', lw=1.5, label=r"$\sim \frac{1}{\sqrt{N}}$" )            # O(h^-0.5)
    #ax.text( 1e-4*h[0], 1e-2, r"$\sim \frac{1}{\sqrt{N}}$", fontsize=18, color="b")

    finishPlot( fig, ax, "monte-carlo-pi-error-scaling" )

    ax.set_xlim( [ xmin, xmax ] )


########################### Weak Scaling ###########################

if args.weakscaling != None:
    ####### Weak Scaling ( work per Partition constant -> Partitions, work ^ ) #######
    def plotWeakScaling( fileprefix, cgpu, paramName, colorMap, lineStyle, ylims, labelGenerator ):
        figs, axes = [], []
        nPlots = 3  # run time, speedup, parallel efficiency
        for i in range(nPlots):
            figs.append( plt.figure( figsize=(6,4) ) )
            axes.append( figs[-1].add_subplot( 111,
                xlabel = "Anzahl Grafikkarten",
                title  = "Weak Scaling\nTaurus, " + (
                            "Intel Xeon E5-2680" if cgpu == "cpu" else
                            "NVIDIA Tesla K20x" ) # K80
            ) )

        axes[0].set_ylabel( "Laufzeit / s"        )
        axes[1].set_ylabel( "Speedup"             )
        axes[2].set_ylabel( "Parallele Effizienz" )

        if not ( cgpu == "cpu" or cgpu == "gpu" ):
            return

        data = genfromtxt( fileprefix+"-"+cgpu+".dat" )
        # Total Elements   Elements Per Thread    Partitions   Time / s   Calculated Pi
        assert len( data.shape ) == 2
        assert data.shape[1] == 5
        data = { "t"             : data[:,3],
                 "N"             : data[:,0],
                 "nPerPartition" : data[:,1],
                 "nPartitions"   : data[:,2]  }

        params = unique( data[ paramName ] )[::-1]
        colors = linspace( 0,1, len( params ) )
        xmax = 0
        xmin = float('inf')
        for i in range( len(params) ):
            filter = data["nPerPartition"] == params[i]
            x = array( data["nPartitions"][filter] )
            y = array( data["t"][filter] )
            x,y,sy = calcStatisticsFromDupes( x, y )
            sorted = argsort( x )
            x  = x [sorted]
            y  = y [sorted]
            sy = sy[sorted]

            axes[0].errorbar( x, y, sy, linestyle='-', marker='.',
                         label=labelGenerator( params[i] ),
                         color=plt.get_cmap( colorMap )(colors[i]) )

            # Speedup in respect to n=1
            if i == 0:  # add ideal speedup
                axes[1].plot( x,x, '--', color='gray', label='Ideal' )
            Sp = y[0] / ( y / x )
            # Watch out! As this is weakscaling the work is scaled with x,
            # i.e. to get the speedup we also need to scale the time needed
            # down by x
            SpErr = Sp * sqrt( (sy/y)**2 + (sy[0]/y[0])**2 )
            axes[1].errorbar( x, Sp, SpErr, linestyle='-', marker='.',
                         label=labelGenerator( params[i] ),
                         color=plt.get_cmap( colorMap )(colors[i]) )

            # Parallel Efficiency in respect to n=1
            if i == 0:  # add ideal speedup
                axes[2].plot( x, ones( x.size ), '--', color='gray', label='Ideal' )
            P    = Sp / x
            PErr = SpErr / x
            axes[2].errorbar( x, P, PErr, linestyle='-', marker='.',
                         label=labelGenerator( params[i] ),
                         color=plt.get_cmap( colorMap )(colors[i]) )

            xmin = min( xmin, data["nPartitions"][filter].min() )
            xmax = max( xmax, data["nPartitions"][filter].max() )

        for i in range(nPlots):
            axes[i].set_xlim( [ xmin, xmax ] )
        axes[0].set_ylim( ylims )
        #axes[1].set_ylim( ylims )
        #axes[2].set_ylim( ylims )

        finishPlot( figs[0], axes[0], "weak-scaling-time-"       + cgpu )
        finishPlot( figs[1], axes[1], "weak-scaling-speedup-"    + cgpu )
        finishPlot( figs[2], axes[2], "weak-scaling-efficiency-" + cgpu )

    #plotWeakScaling( args.weakscaling[0], "cpu", "nPerPartition", "cool_r", ".-", [1,200],
    #    lambda p : r"$2^{"+str(int( log(p)/log(2) ))+"}$ pro Partition" )
    plotWeakScaling( args.weakscaling[0], "gpu", "nPerPartition", "cool_r", ".-", [0,66],
        lambda p : r"$2^{"+str(int( log(p)/log(2) ))+"}$ Iterationen pro Grafikkarte" )



################################# Strong Scaling #################################

if args.strongscaling != None:

    def plotStrongScaling( fileprefix, cgpu, colorMap, ylims, labelGenerator ):
        # fileprefix: File name without -gpu.dat or -cpu.dat respectively
        figs, axes = [], []
        nPlots = 4  # run time, speedup, parallel efficiency
        for i in range(nPlots):
            figs.append( plt.figure( figsize=(6,4) ) )
            axes.append( figs[-1].add_subplot( 111,
                xlabel = "Anzahl Grafikkarten",
                title  = "Strong Scaling\nTaurus, " + (
                            "Intel Xeon E5-2680" if cgpu == "cpu" else
                            "NVIDIA Tesla K20x" ) # K80
            ) )
        axes[3].set_yscale( 'log' )
        axes[0].set_ylabel( "Laufzeit / s"        )
        axes[3].set_ylabel( "Laufzeit / s"        )
        axes[1].set_ylabel( "Speedup"             )
        axes[2].set_ylabel( "Parallele Effizienz" )

        if not ( cgpu == "cpu" or cgpu == "gpu" ):
            return

        data = genfromtxt( fileprefix+"-"+cgpu+".dat" )
        # Total Elements    Partitions   Time / s   Calculated Pi
        assert len( data.shape ) == 2
        assert data.shape[1] == 4
        data = { "t"             : data[:,2],
                 "N"             : data[:,0],
                 "nPartitions"   : data[:,1]  }

        params = unique( data["N"] )[::-1]
        colors = linspace( 0,1, len( params ) )
        xmax = 0
        xmin = float('inf')
        for i in range( len(params) ):
            filter = data["N"] == params[i]
            x = array( data["nPartitions"][filter] )
            y = array( data["t"][filter] )
            x,y,sy = calcStatisticsFromDupes( x, y )
            sorted = argsort( x )
            x  = x [sorted]
            y  = y [sorted]
            sy = sy[sorted]

            axes[0].errorbar( x, y, sy, linestyle='-', marker='.',
                         label=labelGenerator( params[i] ),
                         color=plt.get_cmap( colorMap )(colors[i]) )
            axes[3].errorbar( x, y, sy, linestyle='-', marker='.',
                         label=labelGenerator( params[i] ),
                         color=plt.get_cmap( colorMap )(colors[i]) )

            # Speedup in respect to n=1
            if i == 0:  # add ideal speedup
                axes[1].plot( x,x, '--', color='gray', label='Ideal' )
            Sp = y[0] / y
            SpErr = Sp * sqrt( (sy/y)**2 + (sy[0]/y[0])**2 )
            axes[1].errorbar( x, Sp, SpErr, linestyle='-', marker='.',
                         label=labelGenerator( params[i] ),
                         color=plt.get_cmap( colorMap )(colors[i]) )

            # Parallel Efficiency in respect to n=1
            if i == 0:  # add ideal speedup
                axes[2].plot( x, ones( x.size ), '--', color='gray', label='Ideal' )
            P    = Sp / x
            PErr = SpErr / x
            axes[2].errorbar( x, P, PErr, linestyle='-', marker='.',
                         label=labelGenerator( params[i] ),
                         color=plt.get_cmap( colorMap )(colors[i]) )

            xmin = min( xmin, data["nPartitions"][filter].min() )
            xmax = max( xmax, data["nPartitions"][filter].max() )

        for i in range(nPlots):
            axes[i].set_xlim( [ xmin, xmax ] )
        #axes[0].set_ylim( ylims )
        #axes[1].set_ylim( ylims )
        #axes[2].set_ylim( ylims )

        finishPlot( figs[0], axes[0], "strong-scaling-time-"          + cgpu )
        finishPlot( figs[3], axes[3], "strong-scaling-time-logscale-" + cgpu )
        finishPlot( figs[1], axes[1], "strong-scaling-speedup-"       + cgpu )
        finishPlot( figs[2], axes[2], "strong-scaling-efficiency-"    + cgpu )

    #plotStrongScaling( args.strongscaling[0], "cpu", "cool_r", ".-", [1,200],
    #    lambda p : r"$2^{"+str(int( log(p)/log(2) ))+"}$ pro Partition" )
    plotStrongScaling( args.strongscaling[0], "gpu", "cool_r", [0,66],
        lambda p : r"$2^{"+str(int( log(p)/log(2) ))+"}$ Iterationen gesamt" )


########################### Rootbeer Setup Time ###########################

if args.rootbeersetup != None:
    fig = plt.figure( figsize=(6,4) )
    ax = fig.add_subplot( 111,
        xlabel = "Anzahl Partitions / Partitionen",
        ylabel = "Laufzeit / s"
    )
    # Total Elements   Elements Per Thread    Partitions   Nodes   Time / s
    data = genfromtxt( args.rootbeersetup[0]+"-cpu.dat" )
    ax.plot( data[:,2], data[:,4], 'ro-', label="Intel Xeon E5-2680" )
    data = genfromtxt( args.rootbeersetup[0]+"-gpu.dat" )
    ax.plot( data[:,2], data[:,4], 'ro-', label="NVIDIA Tesla K80" )

    finishPlot( fig, ax, "rootbeer-setup-time" )

plt.show()
exit()
