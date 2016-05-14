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
parser.add_argument("-c", "--workload-cluster", dest="workloadcluster", help="File with data from benchmark", type=str, nargs=1 )
args = parser.parse_args()

def finishPlot( fig, ax, fname ):
    l = ax.legend( loc='best', prop={'size':10}, labelspacing=0.2, # fontsize=10 also works
                   fancybox=True, framealpha=0.5 )
    #if l != None:
    #    l.set_zorder(0)  # alternative to transparency
    fig.tight_layout()
    fig.savefig( fname+".pdf" )
    print "[Saved '"+fname+".pdf']"
    plt.close( fig )


########################### workload scaling ###########################

if args.workloadlog != None:
    data = genfromtxt( args.workloadlog );

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


########################### workload scaling ###########################

if args.workloadcluster != None:
    data = genfromtxt( args.workloadcluster[0] )
    print args.workloadcluster
    print data.shape
    print data[0,0]

    # No.  | singleCore | singleCore | singleGPU | singleGPU | singleGPU |    Spark    |             |  Spark    |    Spark     |    Spark
    # res. |  (Java)    |  (Scala)   |   (CPP)   |   (Java)  |   (Scala) |   (local)   | Spark + GPU | (network) |  (network)   |  (network)
    #      |            |            |           |           |           |             |             |           | 10x workload | 100x workload

    fig = plt.figure( figsize=(8,5) )
    ax = fig.add_subplot( 111,
        xlabel = "Number of threads/cores/GPUs",
        ylabel = "Speedup",
        xscale = "log",
        yscale = "log"
    )

    x = linspace( 1, 8, 100 )
    ax.plot( x, x, '--', color='gray', label="ideal speedup" )
    ax.plot( data[:,0], data[0,1] / data[:,1], 'o-', label="single core (Java)" )
    ax.plot( data[:,0], data[0,2] / data[:,2], 'o-', label="single core (Scala)" )
    ax.plot( data[:,0], data[0,3] / data[:,3], 'o-', label="single GPU (C++)" )
    ax.plot( data[:,0], data[0,4] / data[:,4], 'o-', label="single GPU (Java)" )
    ax.plot( data[:,0], data[0,5] / data[:,5], 'o-', label="single GPU (Scala)" )
    ax.plot( data[:,0], data[0,6] / data[:,6], 'o-', label="Spark local" )
    ax.plot( data[:,0], data[0,7] / data[:,7], 'o-', label="Spark + GPUs" )
    ax.plot( data[:,0], data[0,8] / data[:,8], 'o-', label="Spark 2 nodes" )
    ax.plot( data[:,0], data[0,9] / data[:,9], 'o-', label="Spark 2 nodes 10x workload" )
    ax.plot( data[:,0], data[0,10] / data[:,10], 'o-', label="Spark 2 nodes 100x workload" )

    finishPlot( fig, ax, "benchmarks-scaling-cluster" )



plt.show()
exit()
