import numpy as np
import matplotlib.pyplot as pl

a = 1+1j
b = 1

mu = np.pi/4

ts1 = np.arange(0,100,0.1)*np.exp(1j*(np.pi-mu))
ts2 = np.arange(0,100,0.1)*np.exp(1j*(np.pi+mu))


tsn = -np.arange(0,100,0.1)

tsr = np.arange(-100,100,0.1)

def sigmainv(ts):
    return a/(ts-b)

def tauinv(ts):
    return (ts-a)/b

pl.plot(sigmainv(ts1).real,sigmainv(ts1).imag,'x-')
pl.plot(sigmainv(ts2).real,sigmainv(ts2).imag,'x-')
pl.plot(sigmainv(tsn).real,sigmainv(tsn).imag,'x-')
pl.plot(tauinv(tsr).real,tauinv(tsr).imag,'x-')
pl.plot(tauinv(1j).real,tauinv(1j).imag,'o')

pl.show()
