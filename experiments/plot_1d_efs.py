import scipy.special as scps
import numpy as np
import matplotlib.pyplot as pl

R=1.2
R0=1
p0=0.1
T=4

n1 = 2
n2 = 15


def sigmastd(w):
    return 1+1j

def sigmafqs(w):
    return (1+10j)/w

def omega(n):
    return 1/(2*R0*(p0+1))*(-1j*np.log((p0+2)/(p0))+2*np.pi*n)

def c(w):
    return np.cos(w*p0*R0)*np.sin(w*R0)-p0*np.cos(w*R0)*np.sin(w*p0*R0)

def d(w):
    return np.cos(w*p0*R0)*np.cos(w*R0)+p0*np.sin(w*R0)*np.sin(w*p0*R0)

def g(w):
    return (c(w)*np.sin(w*R)+d(w)*np.cos(w*R))/np.exp(1j*w*R)

w1 = omega(n1)
w2 = omega(n2)
h=0.5*1e-2
x1 = np.arange(0,R0,h)
x2 = np.arange(R0,R,h/10)
x3 = np.arange(R,T,h)
valsstd1 = np.array(list(np.cos(x1*w1*p0))+list(c(w1)*np.sin(w1*x2)+d(w1)*np.cos(w1*x2))+list(g(w1)*np.exp(1j*w1*(x3+(x3-R)*sigmastd(w1)))))/np.exp(1j*w1*R)/g(w1)
valsfqs1 = np.array(list(np.cos(x1*w1*p0))+list(c(w1)*np.sin(w1*x2)+d(w1)*np.cos(w1*x2))+list(g(w1)*np.exp(1j*w1*(x3+(x3-R)*sigmafqs(w1)))))/np.exp(1j*w1*R)/g(w1)
valsstd2 = np.array(list(np.cos(x1*w2*p0))+list(c(w2)*np.sin(w2*x2)+d(w2)*np.cos(w2*x2))+list(g(w2)*np.exp(1j*w2*(x3+(x3-R)*sigmastd(w2)))))/np.exp(1j*w2*R)/g(w2)
valsfqs2 = np.array(list(np.cos(x1*w2*p0))+list(c(w2)*np.sin(w2*x2)+d(w2)*np.cos(w2*x2))+list(g(w2)*np.exp(1j*w2*(x3+(x3-R)*sigmafqs(w2)))))/np.exp(1j*w2*R)/g(w2)
x = np.array(list(x1)+list(x2)+list(x3))
pl.plot(x,abs(valsstd1))
pl.plot(x,abs(valsstd2))
pl.plot(x,abs(valsfqs1))
pl.plot(x,abs(valsfqs2))
print(omega(n1))
print(omega(n2))

#np.savetxt('output/efs_1d_std_w{}.out'.format(n1),np.array([x,abs(valsstd1)]).T)
#np.savetxt('output/efs_1d_std_w{}.out'.format(n2),np.array([x,abs(valsstd2)]).T)
#np.savetxt('output/efs_1d_fqs_w{}.out'.format(n1),np.array([x,abs(valsfqs1)]).T)
#np.savetxt('output/efs_1d_fqs_w{}.out'.format(n2),np.array([x,abs(valsfqs2)]).T)
pl.show()

