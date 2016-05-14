from numpy import *
from matplotlib.pyplot import *

data = genfromtxt( "benchmarks-scaling-2016-02-08.txt" );


# No.  | singleCore | singleCore | singleGPU | singleGPU | singleGPU |    Spark    |             |  Spark    |    Spark     |    Spark
# res. |  (Java)    |  (Scala)   |   (CPP)   |   (Java)  |   (Scala) |   (local)   | Spark + GPU | (network) |  (network)   |  (network)
#      |            |            |           |           |           |             |             |           | 10x workload | 100x workload

figure( figsize=(8,5) )
xlabel( "Number of threads/cores/GPUs" )
ylabel( "Speedup" )

x = linspace( 1, 8, 100 )
plot( x, x, '--', color='gray' )
plot( data[:,0], data[0,1] / data[:,1], 'o-', label="single core (Java)" )
plot( data[:,0], data[0,2] / data[:,2], 'o-', label="single core (Scala)" )
plot( data[:,0], data[0,3] / data[:,3], 'o-', label="single GPU (C++)" )
plot( data[:,0], data[0,4] / data[:,4], 'o-', label="single GPU (Java)" )
plot( data[:,0], data[0,5] / data[:,5], 'o-', label="single GPU (Scala)" )
plot( data[:,0], data[0,6] / data[:,6], 'o-', label="Spark local" )
plot( data[:,0], data[0,7] / data[:,7], 'o-', label="Spark + GPUs" )
plot( data[:,0], data[0,8] / data[:,8], 'o-', label="Spark 2 nodes" )
plot( data[:,0], data[0,9] / data[:,9], 'o-', label="Spark 2 nodes 10x workload" )
plot( data[:,0], data[0,10] / data[:,10], 'o-', label="Spark 2 nodes 100x workload" )

legend( loc='best', fontsize=10 )
tight_layout()
savefig( "benchmarks-scaling.pdf" )
show()
