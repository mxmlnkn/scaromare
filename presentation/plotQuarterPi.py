#!/usr/bin/python

from matplotlib.pyplot import *
from numpy import *

fig = figure( figsize=(6,6) )
ax = fig.add_subplot( 111,
    aspect='equal',

    xlim=[0,1],
    ylim=[0,1]
)
x = random.rand(100)
y = random.rand(100)
inside = x*x + y*y < 1.0
ax.plot( x[inside], y[inside], 'r.', markersize=16 )
ax.plot( x[logical_not( inside )], y[logical_not( inside )], 'b.', markersize=16 )
x = linspace(0,1,1000)
y = sqrt( 1 - x*x )
ax.plot (x, y, 'k-', linewidth=3 )
ax.fill_between( x, 0, y, facecolor='gray', linewidth=0, alpha=0.5 )

for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(18)
for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(18)
tight_layout()
fig.savefig( "monte-carlo-pi-quarter-sphere.pdf" )

show()
