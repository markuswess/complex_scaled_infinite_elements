import scipy.special as scps
import numpy as np
import matplotlib.pyplot as pl

R=1
R0=1
sigma=-3+1j
#sigma = np.exp(1j*(np.pi-np.arcsin(R/R0)))

#omega = -3j+1
omega = 1


R1=4
h=1e-3
xs = np.arange(R,R1,h)
ns=[0,1,3,10,100]

x0 = R0*(1-sigma.real/abs(sigma)**2)
print(x0)
m = R0*sigma.imag/abs(sigma)


def sph_hankel(n,x):
    return scps.spherical_jn(n,x)+1j*scps.spherical_yn(n,x)

def xsc(x,R0=R0,sigma=sigma):
    return x*(x<R0)+(R0+(x-R0)*sigma)*(x>=R0)

pl.figure(1)
#pl.plot(xs,abs(omega*xsc(xs)),label='$|\omega\gamma(x)|$')
for n in ns:
    hs = abs(sph_hankel(n,omega*xsc(xs)))
    np.savetxt('output/scaled_hankel_{}.out'.format(n),np.array([xs,hs/max(hs)]).T)
    pl.plot(xs,hs/max(hs),label='$h_{}(\omega\gamma(x))$ normed'.format(n))
    #pl.plot(xs,hs2/max(hs2),label='asymptotic')
    #pl.plot(R0*(1-sigma.real/abs(sigma)**2),1,'o')


#ann = 2**n*scps.gamma(n+1/2)/np.sqrt(np.pi)
#pl.plot(xs,(sph_hankel(n,omega*xsc(xs))*1j*(omega*xsc(x0))**(n+1)/ann).imag,label='i h1 over asympt')
#pl.plot(xs,(sph_hankel(n,omega*xsc(xs))*1j*(omega*xsc(x0))**(n+1)/ann).real,label='r h1 over asympt')
#pl.plot(xs,abs(sph_hankel(n,omega*xsc(xs))*1j*(omega*xsc(x0))**(n+1)/ann),label='abs h1 over asympt')
#
pl.legend()


pl.show()

