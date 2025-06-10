import numpy as np
import matplotlib.pyplot as pl

ts = np.arange(-10,10,0.1)


mu=1+1j

a=1
b=1
c=1+1j
d=2+1j



def mobius(z):
    return (a*z+b)/(c*z+d)

def line(z):
    z*mu

m1 = mu.conjugate()*abs(c)**2/(d*(c*mu).conjugate()-d.conjugate()*c*mu)
circ1 = 1/(mu*ts+d/c)

circ2 = mobius(mu*ts)
m2 = (b*c-a*d)/c**2*mu.conjugate()*abs(c)**2/(d*(c*mu).conjugate()-d.conjugate()*c*mu)+a/c

print(abs(circ2-m2))
#pl.plot(circ1.real,circ1.imag)
#pl.plot(m1.real,m1.imag,'o')
pl.plot(circ2.real,circ2.imag)
pl.plot(m2.real,m2.imag,'o')
pl.show()

