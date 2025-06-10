import numpy as np
import matplotlib.pyplot as pl

b = 2j
c = 1
d = 1

mu = np.pi/3

ts1 = np.arange(0,100,0.1)*np.exp(1j*(np.pi-mu))
ts2 = np.arange(0,100,0.1)*np.exp(1j*(np.pi+mu))


tsn = -np.arange(0,100,0.1)

tsr = np.arange(-100,100,0.1)

def sigmainv(ts):
    return (b-ts*c)/(d*ts)

def tauinv(ts):
    return (c*ts)/(b-d*ts)

pl.plot(sigmainv(ts1).real,sigmainv(ts1).imag,'x-')
pl.plot(sigmainv(ts2).real,sigmainv(ts2).imag,'x-')
pl.plot(sigmainv(tsn).real,sigmainv(tsn).imag,'x-')
pl.plot(tauinv(tsr).real,tauinv(tsr).imag,'x-')
pl.plot(tauinv(1j).real,tauinv(1j).imag,'o')

pl.show()
