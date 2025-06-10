from ngswaves import *
from ngsolve import *
from numpy.linalg import inv,eigvals
from matplotlib.pyplot import *
from fem1d import *
from numpy import *
from saialp import *
import netgen.gui


def correct_evs():
    from numpy import loadtxt
    loaded=loadtxt('dsph_hankel_1_all_zeros.out')
    lam=loaded[:,0]+1j*loaded[:,1]
    return lam

a=-1+1j
b=0
c=-1j
d=5

def MakeGeometry(R,Rout):
    from netgen.csg import CSGeometry,Sphere,Pnt,OrthoBrick
    geo = CSGeometry()
    spho = Sphere(Pnt(0,0,0),Rout).bc('outer')
    sph = Sphere(Pnt(0,0,0),R).bc('inner')
    b = OrthoBrick(Pnt(0,0,0),Pnt(R+1,R+1,R+1)).bc('sym')
    geo.Add((spho-sph)*b)
    return geo
def inv_mobius_r(a,b,c,d):
    ts = arange(-100,100,0.01)
    vs = (d*ts-b)/(-c*ts+a)
    plot(vs.real,vs.imag)
    plot(((d*1j-b)/(-c*1j+a)).real,((d*1j-b)/(-c*1j+a)).imag,'o')
    #plot((-d/c).real,(-d/c).imag,'o')
    #plot((-b/a).real,(-b/a).imag,'s')
inv_mobius_r(a,b,c,d)

N=60
n=5
shift = 2-1j
nevs =100
R=1
maxh=0.3
order=3

g=MakeGeometry(R/2,R)
m=Mesh(g.GenerateMesh(maxh=maxh))
m.Curve(2*order)
Draw(m)
outbnd = m.Boundaries('outer')
fes = FESpacePlus('h1ho',m,complex=True,order=order)
fes.SetDefinedOn(~m.Materials('.*'))
fes.SetDefinedOn(outbnd)
fes.Update()
hsm=SphericalHSMExterior((0,0,0),R,N,1+0j,order=order,complex=True)
fes = fes.AddExterior(hsm,outbnd)
ue,ve = fes.ExtTrialFunction(), fes.ExtTestFunction()

M = BilinearFormPlus(fes,symmetric=True)
Mx = BilinearFormPlus(fes,symmetric=True)
Mxx = BilinearFormPlus(fes,symmetric=True)
S = BilinearFormPlus(fes,symmetric=True)
Sx = BilinearFormPlus(fes,symmetric=True)
Sxx = BilinearFormPlus(fes,symmetric=True)
Ssurf = BilinearFormPlus(fes,symmetric=True)

M+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (hsm.Tm.T@hsm.Tm).flatten(),definedon=outbnd)
Mx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (hsm.Tm.T@hsm.Diffop@hsm.Tm).flatten(),definedon=outbnd)
Mxx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (hsm.Tm.T@hsm.Diffop@hsm.Diffop@hsm.Tm).flatten(),definedon=outbnd)
S+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (hsm.Tp.T@hsm.Tp).flatten(),definedon=outbnd)
Sx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (hsm.Tp.T@hsm.Diffop@hsm.Tp).flatten(),definedon=outbnd)
Sxx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (hsm.Tp.T@hsm.Diffop@hsm.Diffop@hsm.Tp).flatten(),definedon=outbnd)

Ssurf+=SymbolicExtBFI(
        [u.Trace().Deriv()*v.Trace().Deriv() for u in ue for v in ve],
        (hsm.Tm.T@hsm.Tm).flatten(),definedon=outbnd)

gfu = GridFunction(fes,multidim=nevs)

SetHeapSize(10000000)
M.Assemble()
Mx.Assemble()
Mxx.Assemble()
S.Assemble()
Sx.Assemble()
Sxx.Assemble()
Ssurf.Assemble()


Ph=array([
    [0,0,0,0,2*R,0,0],
    [0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0],
    [0,0,0,R**2,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,-2*R,0,0,0,0,0],
    [0,0,-1,0,0,0,0],
    [0,0,0,0,0,0,0],
    ])

Th=array([
    [-a,d,-b, 0, 0, 0, 0,0, 0],
    [1 ,0, 0, 0, 0, 0, 0,0, 0],
    [d ,0, 0,-a,-b, 0, 0,0, 0],
    [0, 0, 0, 1, 0, 0, 0,0, 0],
    [b ,0, 0, 0, 0,-d, 0,0, 0],
    [0 ,0, 0, 0, 0, b,-d,0, 0],
    [0 ,0, 0, 0, 0, 0,-a,d,-b],
    [0 ,0, 0, 0, 0, 0, 1,0, 0]
    ])
Tt=array([
    [0 ,-c,0, 0, 0, 0, 0,0, 0],
    [0 , 0,1, 0, 0, 0, 0,0, 0],
    [-c, 0,0, 0, 0, 0, 0,0, 0],
    [0,  0,0, 0, 1, 0, 0,0, 0],
    [-a, 0,0, 0, 0, c, 0,0, 0],
    [0 , 0,0, 0, 0,-a, c,0, 0],
    [0 , 0,0, 0, 0, 0, 0,-c, 0],
    [0 , 0,0, 0, 0, 0, 0,0, 1]
    ])
Pt=array([
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [R**2,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0],
    ])

saialp=SaiALP([M.mat,Mx.mat,Mxx.mat,S.mat,Sx.mat,Sxx.mat,Ssurf.mat],Ph.T,Pt.T,Th,Tt,shift,fes.FreeDofs())
saialp.CalcInverse('sparsecholesky')
saialp.CalcKrylow(2*nevs)
lam=saialp.SolveHessenberg(None,nevs)

plot(lam.real,lam.imag,'x')
plot(correct_evs().real,correct_evs().imag,'.k')
show()

