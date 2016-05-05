#!/usr/bin/python

from numpy import *
from matplotlib.pyplot import *

N=arange(10000000)+1
x=random.rand( len(N) )
y=random.rand( len(N) )
z = cumsum( x*x + y*y < 1 )
i = unique( (10**( linspace(0,log10(len(N)-1),1000) )).astype( int64 ) )
print i
N = N[i]
p = 4.0*z[i]/N
relErr = abs(p - pi)/pi

fig = figure()
ax = fig.add_subplot( 111,
    yscale = 'log',
    xscale = 'log',
    xlim   = [ N[0], N[-1] ]
)

ax.plot( N, relErr )
ax.plot( N, N**(-0.5)/10.**0.5 )
savetxt( "python-pin.dat", array([N, p]).transpose() )
show()
