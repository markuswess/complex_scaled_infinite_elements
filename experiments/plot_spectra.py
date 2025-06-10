import matplotlib.pyplot as pl
import numpy as np

a = 1j+1
b = 1j
c = 1
d = 1

mu = np.pi/3
ts = np.arange(0,100,0.1)*np.exp(1j*(np.pi-mu))
tsreal = -np.arange(0,100,0.1)
print(ts)
def esssing1(ts):
    return (-(ts*c-b)+np.sqrt((ts*c-b)**2+4*a*ts*d))/(2*ts*d)
def esssing2(ts):
    return (-(ts*c-b)-np.sqrt((ts*c-b)**2+4*a*ts*d))/(2*ts*d)

#pl.plot(test.real,test.imag)
pl.plot(esssing1(ts).real,esssing1(ts).imag,'x-') 
pl.plot(esssing2(ts).real,esssing2(ts).imag,'x-')
pl.plot((-a/b).real,(-a/b).imag,'o')
pl.plot(esssing1(ts)[-1].real,esssing1(ts)[-1].imag,'x-') 
pl.plot(esssing2(ts)[-1].real,esssing2(ts)[-1].imag,'x-')
pl.show()
