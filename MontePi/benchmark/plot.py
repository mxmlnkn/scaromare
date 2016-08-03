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
    plt.close( fig )


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


########################### workload cluster ###########################

if args.workloadcluster != None:
    ####### Strong Scaling ( work per slice constant -> slices, work ^ ) #######
    def plotData( cgpu, paramName, colorMap, lineStyle, ylims, labelGenerator ):
        fig = plt.figure( figsize=(6,4) )
        ax = fig.add_subplot( 111,
            xlabel = "Parallelisierung",
            ylabel = "Laufzeit / s",
            xscale = "log",
            yscale = "log"
        )

        if not ( cgpu == "cpu" or cgpu == "gpu" ):
            return

        data = genfromtxt( args.workloadcluster[0]+"-"+cgpu+".dat" )
        # Total Elements   Elements Per Thread    Slices   Nodes   Time / s
        assert len( data.shape ) == 2
        assert data.shape[1] == 5
        data = { "t"         : data[:,4],
                 "N"         : data[:,0],
                 "nPerSlice" : data[:,1],
                 "nSlices"   : data[:,2],
                 "nodes"     : data[:,3] }

        if cgpu == "cpu":
            ax.set_title( "Intel Xeon E5-2680" )
        else:
            ax.set_title( "NVIDIA Tesla K80" )

        params = unique( data[ paramName ] )[::-1]
        colors = linspace( 0,1, len( params ) )
        xmax = 0
        xmin = 1e9
        gpusPerNode=1 # 4
        for i in range( len(params) ):
            filter = all( [
                        data["nPerSlice"] == params[i],
                        data["nSlices"] != 0,
                        data["t"] != 0,
                        data["nSlices"] >= gpusPerNode*(data["nodes"]-1),
                        data["nSlices"] <= gpusPerNode*data["nodes"]
                     ], axis=0 )
            ax.plot( data["nSlices"][filter], data["t"][filter], lineStyle,
                     label=labelGenerator( params[i] ),
                     color=plt.get_cmap( colorMap )(colors[i]) )
            xmin = min( xmin, data["nSlices"][filter].min() )
            xmax = max( xmax, data["nSlices"][filter].max() )
        ax.set_xlim( [ xmin/1.1, xmax*1.1 ] )
        ax.set_ylim( ylims )

        finishPlot( fig, ax, "cluster-strong-scaling-"+cgpu )


    plotData( "cpu", "nPerSlice", "copper", ".-", [1,200],
              lambda p : r"$2^{"+str(int( log(p)/log(2) ))+"}$ pro slice" )
    plotData( "gpu", "nPerSlice", "cool_r", ".-", [1,200],
              lambda p : r"$2^{"+str(int( log(p)/log(2) ))+"}$ pro slice" )

    iTime=-1
    iPerSlice=1
    iSlices=2
    iTime=-1

    ####### GPU ######
    #data = genfromtxt( args.workloadcluster[0]+"-gpu.dat" )
    ##x = linspace( 1, 8, 100 )
    ##ax.plot( x, x, '--', color='gray', label="ideal speedup" )
    #nPerSlice = unique( data[:,iPerSlice] )
    #colors = linspace( 0,1, len(nPerSlice) )
    #iC = 0
    #for perSlice in nPerSlice:
    #    iHits = logical_and( data[:,iPerSlice] == perSlice, data[:,iSlices] != 0 )
    #    ax.plot( data[iHits,iSlices] , data[iHits,iTime], 'x--', label=r"GPU, $2^{"+str(int( log(perSlice)/log(2) ))+"}$ pro slice", color=plt.get_cmap("cool")(colors[iC]) )
    #    iC += 1
    #
    #finishPlot( fig, ax, "cluster-strong-scaling" )

    ########## Weak Scaling ( work constant -> slices ^ ) ##########

    fig = plt.figure( figsize=(8,5) )
    ax = fig.add_subplot( 111,
        xlabel = "Parallelisierung",
        ylabel = "Laufzeit / s",
        xscale = "log",
        yscale = "log"
    )
    # Total Elements   Elements Per Thread    Slices   Nodes   Time / s
    iTime=-1
    iWork=0
    iPerSlice=1
    iSlices=2

    ###### CPU ######
    data = genfromtxt( args.workloadcluster[0]+"-cpu.dat" )
    #x = linspace( 1, 8, 100 )
    #ax.plot( x, x, '--', color='gray', label="ideal speedup" )
    nWork = unique( data[:,iWork] )
    colors = linspace( 0,1, len(nWork) )
    iC = 0
    for work in nWork:
        if work == 0:
            continue
        iHits = data[:,iSlices] != 0
        #print data[iHits,iTime]
        iHits = logical_and( data[:,iWork] == work, data[:,iSlices] != 0 )
        ax.plot( data[iHits,iSlices] , data[iHits,iTime], '.-', label=r"CPU, "+str(work)+" Iterationen", color=plt.get_cmap("copper")(colors[iC]) )
        iC += 1

    ###### GPU ######
    #data = genfromtxt( args.workloadcluster[0]+"-gpu.dat" )
    ##x = linspace( 1, 8, 100 )
    ##ax.plot( x, x, '--', color='gray', label="ideal speedup" )
    #nPerSlice = unique( data[:,iPerSlice] )
    #colors = linspace( 0.5  , len(nPerSlice) )
    #iC = 0
    #for perSlice in nPerSlice:
    #    iHits = logical_and( data[:,iPerSlice] == perSlice, data[:,iSlices] != 0 )
    #    ax.plot( data[iHits,iSlices] , data[iHits,iTime], 'x--', label=r"GPU, $2^{"+str(int( log(perSlice)/log(2) ))+"}$ pro slice", color=plt.get_cmap("cool")(colors[iC]) )
    #    iC += 1
    #
    finishPlot( fig, ax, "cluster-weak-scaling" )


########################### Rootbeer Setup Time ###########################

if args.rootbeersetup != None:
    fig = plt.figure( figsize=(6,4) )
    ax = fig.add_subplot( 111,
        xlabel = "Anzahl Slices / Partitionen",
        ylabel = "Laufzeit / s"
    )
    # Total Elements   Elements Per Thread    Slices   Nodes   Time / s
    data = genfromtxt( args.rootbeersetup[0]+"-cpu.dat" )
    ax.plot( data[:,2], data[:,4], 'ro-', label="Intel Xeon E5-2680" )
    data = genfromtxt( args.rootbeersetup[0]+"-gpu.dat" )
    ax.plot( data[:,2], data[:,4], 'ro-', label="NVIDIA Tesla K80" )

    finishPlot( fig, ax, "rootbeer-setup-time" )

plt.show()
exit()
