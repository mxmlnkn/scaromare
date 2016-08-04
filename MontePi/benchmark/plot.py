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

def finishPlot( fig, ax, fname ):
    l = ax.legend( loc='best', prop={'size':10}, labelspacing=0.2, # fontsize=10 also works
                   fancybox=True, framealpha=0.5 )
    #if l != None:
    #    l.set_zorder(0)  # alternative to transparency
    fig.tight_layout()
    fig.savefig( fname+".pdf" )
    print "[Saved '"+fname+".pdf']"
    fig.savefig( fname+".png" )
    print "[Saved '"+fname+".png']"
    plt.close( fig )

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
    ax.plot( data[ data[:,1]>0, 0 ], data[ data[:,1]>0, 1 ], 'o-', label="single core (Java)" )
    ax.plot( data[ data[:,2]>0, 0 ], data[ data[:,2]>0, 2 ], 'o-', label="single core (Scala)" )
    ax.plot( data[ data[:,3]>0, 0 ], data[ data[:,3]>0, 3 ], 'o-', label="single GPU (C++)" )
    ax.plot( data[ data[:,4]>0, 0 ], data[ data[:,4]>0, 4 ], 'o-', label="single GPU (Java)" )
    ax.plot( data[ data[:,5]>0, 0 ], data[ data[:,5]>0, 5 ], 'o-', label="single GPU (Scala)" )
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
