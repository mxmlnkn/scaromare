#!/usr/bin/python
# python plotAll.py results-2016-02-22_08-09-37.log error-scaling.log


from numpy import *
from matplotlib.pyplot import *
import sys


########################### workload scaling ###########################

data = genfromtxt( sys.argv[1] );

figure( figsize=(8,5) )
xlabel( "Number of dice rolls" )
ylabel( "Time / s" )
xscale( "log" )
yscale( "log" )

plot( data[:,0], data[:,1], 'o-', label="single core (Java)" )
plot( data[:,0], data[:,2], 'o-', label="single core (Scala)" )
plot( data[:,0], data[:,3], 'o-', label="single GPU (C++)" )
plot( data[:,0], data[:,4], 'o-', label="single GPU (Java)" )
plot( data[:,0], data[:,5], 'o-', label="single GPU (Scala)" )
#plot( data[:,0], data[:,6] , 'o-', label="Spark local" )
#plot( data[:,0], data[:,7] , 'o-', label="Spark + GPUs" )
#plot( data[:,0], data[:,8] , 'o-', label="Spark 2 nodes" )
#plot( data[:,0], data[:,9] , 'o-', label="Spark 2 nodes 10x workload" )
#plot( data[:,0], data[:,10] / data[:,10], 'o-', label="Spark 2 nodes 100x workload" )

legend( loc='best', fontsize=10 )
tight_layout()
savefig( "benchmarks-workload-scaling.pdf" )


########################### error scaling ###########################

data = genfromtxt( sys.argv[2] );

figure( figsize=(8,5) )
xlabel( "Number of dice rolls" )
ylabel( "relative error" )
xscale( "log" )
yscale( "log" )

plot( data[:,0], abs( data[:,1] - pi ) / pi, '+' )

# scaling comparison
h = linspace( data[0,0], data[-1,0], 50 )
plot( h, data[0,1]*h**(-0.5), 'k--', lw=1.5)            # O(h^-0.5)
text( 1e-4, 1e-2, r"$\sim h^{-0.5}$", fontsize=18, color="b")

legend( loc='best', fontsize=10 )
tight_layout()
savefig( "monte-carlo-pi-error-scaling.pdf" )


show()
exit()

########################### workload scaling ###########################

data = genfromtxt( sys.argv[1] );

# No.  | singleCore | singleCore | singleGPU | singleGPU | singleGPU |    Spark    |             |  Spark    |    Spark     |    Spark
# res. |  (Java)    |  (Scala)   |   (CPP)   |   (Java)  |   (Scala) |   (local)   | Spark + GPU | (network) |  (network)   |  (network)
#      |            |            |           |           |           |             |             |           | 10x workload | 100x workload

figure( figsize=(8,5) )
xlabel( "Number of threads/cores/GPUs" )
ylabel( "Speedup" )
xscale( "log" )
yscale( "log" )

x = linspace( 1, 8, 100 )
plot( x, x, '--', color='gray', label="ideal speedup" )
plot( data[:,0], data[0,1] / data[:,1], 'o-', label="single core (Java)" )
plot( data[:,0], data[0,2] / data[:,2], 'o-', label="single core (Scala)" )
plot( data[:,0], data[0,3] / data[:,3], 'o-', label="single GPU (C++)" )
plot( data[:,0], data[0,4] / data[:,4], 'o-', label="single GPU (Java)" )
plot( data[:,0], data[0,5] / data[:,5], 'o-', label="single GPU (Scala)" )
#plot( data[:,0], data[0,6] / data[:,6], 'o-', label="Spark local" )
#plot( data[:,0], data[0,7] / data[:,7], 'o-', label="Spark + GPUs" )
#plot( data[:,0], data[0,8] / data[:,8], 'o-', label="Spark 2 nodes" )
#plot( data[:,0], data[0,9] / data[:,9], 'o-', label="Spark 2 nodes 10x workload" )
#plot( data[:,0], data[0,10] / data[:,10], 'o-', label="Spark 2 nodes 100x workload" )

legend( loc='best', fontsize=10 )
tight_layout()
savefig( "benchmarks-scaling.pdf" )



show()
