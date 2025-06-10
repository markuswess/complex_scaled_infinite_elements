import scipy.special as scps
import numpy as np
import matplotlib.pyplot as pl

R=0.5
R0=1
sigmas=[-1+1j,-2+1j,-3+1j,-5+1j]

omega = 1


R1=2
h=1e-3
xs = np.arange(R,R1,h)
ns=[0,1,3,10,100]

#x0 = R0*(1-sigma.real/abs(sigma)**2)
#print(x0)
#m = R0*sigma.imag/abs(sigma)


def sph_hankel(n,x):
    return scps.spherical_jn(n,x)+1j*scps.spherical_yn(n,x)

def xsc(x,R0=R0,sigma=sigmas[0]):
    return x*(x<R0)+(R0+(x-R0)*sigma)*(x>=R0)

pl.figure(1)
for sigma in sigmas:
    gs = abs(xsc(xs,R0,sigma))
    np.savetxt('output/gamma_{}.out'.format(sigma),np.array([xs-1,gs]).T)
    pl.plot(xs,gs,label='$\sigma={}$'.format(sigma))

pl.legend()


pl.show()

