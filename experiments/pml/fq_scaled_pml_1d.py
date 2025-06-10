from ngsolve import *
from saialp import *
from numpy import *
from matplotlib.pyplot import plot,show
from fem1d import *
from ngswaves import *
import numpy as np
#a=10+1j
#b=6j
#c=-1j
#d=1

#standard
#alpha=0
#beta=1+0.5j
#beta=1+1j
#beta=1+5j
#beta=1+10j
#gamma=1
#delta=0

#fqs
#alpha=1+3j
#alpha=1+7j
#alpha=1+10j
#alpha=1+15j
#alpha=1+20j
#beta=0
#gamma=1
#delta=0

#fqs2
alpha=0
beta=10j
gamma=5
delta=1

#fqs3
#alpha=10j
#beta=1
#gamma=1
#delta=0
T=1.3
#for compatibility with diss
a=beta
b=alpha
c=delta
d=gamma




shift=50-5j
K=100
p0=1.1

def correct_res(p0,R):
    return 1/2/p0/R*(-1j*log((p0+1)/(p0-1))+2*arange(100)*pi)


def inv_mobius_r(a,b,c,d):
    ts = arange(-100,100,0.01)
    vs = (d*ts-b)/(-c*ts+a)
    plot(vs.real,vs.imag)
    #plot((-d/c).real,(-d/c).imag,'o')
    #plot((-b/a).real,(-b/a).imag,'s')

def scaling(a,b,c,d,w):
    return (a*w+b)/(c*w+d)

#testw = 50-0.3j
#alpha = scaling(a,b,c,d,testw)/testw
#inv_mobius_r(a,b,c,d)
#alpha=1+1j
#print(scaling(a,b,c,d,testw))

geo = geo1d(0,1,1.2,1.2+T)
geo.SetMaterials('inner','inner','pml')
geo.SetMaxhs(0.01,0.01,0.1)
m = Mesh(geo.GenerateMesh())

fes = H1(m,order=4,complex=True)

Mi = BilinearForm(fes,symmetric=True)
Si = BilinearForm(fes,symmetric=True)

Me = BilinearForm(fes,symmetric=True)
Se = BilinearForm(fes,symmetric=True)

u,v = fes.TnT()

pot=IfPos(x-1,1,p0**2)
Mi+=SymbolicBFI(pot*u*v,definedon=m.Materials('inner'))
Me+=SymbolicBFI(u*v,definedon=m.Materials('pml'))

Si+=SymbolicBFI(grad(u)*grad(v),definedon=m.Materials('inner'))
Se+=SymbolicBFI(grad(u)*grad(v),definedon=m.Materials('pml'))

Mi.Assemble()
Me.Assemble()
Si.Assemble()
Se.Assemble()
Ph=array([[0,0,0,0],[0,0,-1,0],[1,0,0,0],[0,1,0,0]])
Pt=array([[0,0,0,1],[0,0, 0,0],[0,0,0,0],[0,0,0,0]])
Th=array([[0,b,0,0],[0,0,d,0],[0,0,0,1]])
Tt=array([[d,-a,0,c],[b,0,-c,a],[1,0,0,0]])

#M = Mi.mat.CreateMatrix()
#S = Mi.mat.CreateMatrix()
#Z = Mi.mat.CreateMatrix()
#Z.AsVector()[:]=0.

#M.AsVector().Assign(Mi.mat.AsVector(),complex(-1.))
#M.AsVector().Add(Me.mat.AsVector(),complex(-alpha))

#S.AsVector().Assign( Si.mat.AsVector(),complex(1.))
#S.AsVector().Add(Se.mat.AsVector(),complex(1/alpha))

r=random.rand(fes.ndof)

saialp=SaiALP([Mi.mat,Me.mat,Si.mat,Se.mat],Ph,Pt,Th,Tt,shift)
saialp.CalcInverse('sparsecholesky')
saialp.CalcKrylow(3*K,False,r)
lam=saialp.SolveHessenberg(None,K)



#saiapp = PolyArnoldi([S,Z,M],shift)
#saiapp.CalcInverse('sparsecholesky')
#saiapp.CalcKrylow(2*K,False,r)

#lam2 = saiapp.SolveHessenberg(None,K)

#print('{:25}:saiapp, saialp'.format(''))
#for t in saialp.timers:
#    print('{:25}:{:10.5}, {:10.5}'.format(t,saiapp.timers[t],saialp.timers[t]))
plot(lam.real,lam.imag,'x')
#plot(lam2[:100].real,lam2[:100].imag,'o')
ref=correct_res(p0,1)
plot(ref.real,ref.imag,'.k')
np.savetxt('../output/pml_1d_a{}_b{}_c{}_d{}.out'.format(alpha,beta,gamma,delta,T),np.array([lam.real,lam.imag]).T)
show()
