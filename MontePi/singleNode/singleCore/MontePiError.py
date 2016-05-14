#!/usr/bin/python

from numpy import *
from matplotlib.pyplot import *

N=arange(100000)+1
random.seed(986403)
x=random.rand( len(N) )
y=random.rand( len(N) )
z = cumsum( x*x + y*y < 1 )

#print i
# test statistical error:
p = (x*x + y*y < 1)[::len(N)/10000]
print "n =",len(p)
print "p_i*p_j/n**2 =",1.*(outer(p,p).sum() - outer(p,p).trace())/len(p)**2
print "(pi/4)**2 =",(pi/4.)**2
print "(p_i-mu)*(p_j-mu)/n**2 =",(outer(p-pi/4.,p-pi/4.).sum() - outer(p-pi/4.,p-pi/4.).trace())/len(p)**2
nTest = 100
nMax  = 1e4
base  = log(nMax)
a0 = zeros(nTest)  # n number of samples
a1 = zeros(nTest)  # stddev of mean: <(<x_i> - mu)**2>
a2 = zeros(nTest)  # sum_i (f_i-mu)**2                   # diagonal
a3 = zeros(nTest)  # sum_i sum_{j!=i} (f_i-mu)*(f_j-mu)  # of diagonals
a4 = zeros(nTest)
for i in range(nTest):
    n = int( exp(1.*i*log(nMax)/nTest)+1 )
    #print n
    p = (x*x + y*y < 1)[:n]
    a0[i] = n
    a1[i] = (p.mean() - pi/4.)**2  # = <p-m>**2
    t = p-pi/4
    a2[i] = ( outer(t,t).trace()                    )*1./ len(t)**2
    a3[i] = ( outer(t,t).sum() - outer(t,t).trace() )*1./ len(t)**2

i = unique( (10**( linspace(0,log10(len(N)-1),1000) )).astype( int64 ) )
N = N[i]
p = 4.0*z[i]/N
relErr = abs(p - pi)/pi

fig = figure( figsize=(6,4) )
ax = fig.add_subplot( 111,
    yscale = 'log',
    xscale = 'log',
    xlim   = [ a0[0], a0[-1] ]
)

# test statistical error
ax.set_xlabel("Anzahl an Stichproben N")
ax.plot( a0,a1, 'b.-', label=r"$\sigma_{\mu_N}^2 := \left( \langle x_i \rangle_N - \mu \right)^2$" )
ax.plot( a0,a2, 'r.-', label=r"$\frac{1}{N^2} \sum_i \left( x_i - \mu \right)^2$" )
ax.plot( a0,a3, 'go-', label=r"$\frac{1}{N^2} \sum_i \sum_{i\neq j} \left( x_i - \mu \right)\left( x_j - \mu \right)$" )
ax.plot( a0,abs(a3), 'm.-', label=r"$|\mathrm{ibid.}|$" )

#ax.plot( N, relErr )
#ax.plot( N, N**(-0.5)/10.**0.5, 'k-', label=r"$O\left( \frac{1}{\sqrt{N}} \right)$" )
ax.plot( a0, a0**(-1)*a1[0]/a0[0]**(-1), 'k-', label=r"$O\left( \frac{1}{N} \right)$" )
ax.legend( loc='best', prop={'size':11}, labelspacing=0.2, # fontsize=10 also works
           fancybox=True, framealpha=0.5 )
tight_layout()
savetxt( "meanerror.dat", array([N, p]).transpose() )
savefig( "meanerror.pdf" )

show()
