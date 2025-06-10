from ngswaves import *
from ngsolve import *
from netgen.geom2d import *
from numpy.linalg import inv,eigvals
from matplotlib.pyplot import *
from numpy import *
from saialp import *
import netgen.gui


def correct_evs():
    from numpy import loadtxt
    #loaded=loadtxt('dhankel_1_zeros.out')
    loaded=loadtxt('reference_square_trans.out')
    lam=loaded[:,0]+1j*loaded[:,1]
    return lam


#standard
alpha=1+3j
beta=0
gamma=1
delta=0

#fqs
alpha=2+2j
beta=0
gamma=0
delta=1


#fqs 2
#alpha=0
#beta=2j
#gamma=1
#delta=1


#fqs3
#alpha=1j
#beta=1
#gamma=1
#delta=0
#for compatibility with diss
a=beta
b=alpha
c=delta
d=gamma
def MakeCircleGeometry(R1,R2,origin=(0,0)):
    geo = SplineGeometry()
    geo.AddRectangle((-R1,-R1),(R1,R1),leftdomain=2,rightdomain=1, bc='inner')
    #geo.AddCircle(origin,r=R1,leftdomain=0,rightdomain=1,bc='inner')
    geo.AddCircle(origin,r=R2,leftdomain=1,bc='outer')
    geo.SetMaterial(1,'inner')
    geo.SetMaterial(2,'scatterer')
    return geo

def inv_mobius_r(a,b,c,d):
    ts = arange(-100,100,0.01)
    vs = (d*ts-b)/(-c*ts+a)
    plot(vs.real,vs.imag)
    #plot(((d*1j-b)/(-c*1j+a)).real,((d*1j-b)/(-c*1j+a)).imag,'o')
    #plot((-d/c).real,(-d/c).imag,'o')
    #plot((-b/a).real,(-b/a).imag,'s')
#inv_mobius_r(a,b,c,d)

N=15
shift = 55-0.3j
nevs =50
R1=1.5
R=3
maxh=0.1
order=4
pot=CoefficientFunction([0,-0.8])
g=MakeCircleGeometry(R1,R)
m=Mesh(g.GenerateMesh(maxh=maxh))
m.Curve(2*order)
Draw(pot,m,'pot')
Draw(m)
outbnd = m.Boundaries('outer')
fes = FESpacePlus('h1ho',m,complex=True,order=order)
hsm=SphericalHSMExterior((0,0),R,N,1+0j,order=order,complex=True)
fes = fes.AddExterior(hsm,outbnd)
ue,ve = fes.ExtTrialFunction(), fes.ExtTestFunction()

ui,vi = fes.TestFunction(),fes.TrialFunction()

print('total dofs: ', sum(fes.FreeDofs()))
print('ext dofs: ',sum([sum(fes.components[i].FreeDofs()) for i in range(1,len(fes.components))]))
print('int dofs: ',sum(fes.components[0].FreeDofs()))
print('surface dofs: ',sum(fes.components[1].FreeDofs()))

M = BilinearFormPlus(fes,symmetric=True)
Mx = BilinearFormPlus(fes,symmetric=True)
Mxx = BilinearFormPlus(fes,symmetric=True)
S = BilinearFormPlus(fes,symmetric=True)
Sx = BilinearFormPlus(fes,symmetric=True)
Sxx = BilinearFormPlus(fes,symmetric=True)
Ssurf = BilinearFormPlus(fes,symmetric=True)


Mint = BilinearForm(fes,symmetric=True)
Sint = BilinearForm(fes,symmetric=True)

Mint += (1+pot)*(1+pot)*ui*vi*dx

Sint += grad(ui)*grad(vi)*dx

bdmat = zeros(shape(hsm.Tm.T@hsm.Tm))
bdmat[0,0]=-1/4

M+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (R*hsm.Tm.T@hsm.Tm).flatten(),definedon=outbnd)
Mx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (R*hsm.Tm.T@hsm.Diffop@hsm.Tm).flatten(),definedon=outbnd)
Mxx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (R*hsm.Tm.T@hsm.Diffop@hsm.Diffop@hsm.Tm).flatten(),definedon=outbnd)
S+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (1/R*hsm.Tp.T@hsm.Tp).flatten(),definedon=outbnd)
Sx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (1/R*hsm.Tp.T@hsm.Diffop@hsm.Tp+1/R*bdmat).flatten(),definedon=outbnd)
Sxx+=SymbolicExtBFI(
        [u*v for u in ue for v in ve],
        (1/R*hsm.Tp.T@hsm.Diffop@hsm.Diffop@hsm.Tp-1/4/R*hsm.Tm.T@hsm.Tm).flatten(),definedon=outbnd)

Ssurf+=SymbolicExtBFI(
        [u.Trace().Deriv()*v.Trace().Deriv() for u in ue for v in ve],
        (R*hsm.Tm.T@hsm.Tm).flatten(),definedon=outbnd)

gfu = GridFunction(fes,multidim=nevs)

SetHeapSize(10000000)
M.Assemble()
Mx.Assemble()
Mxx.Assemble()
S.Assemble()
Sx.Assemble()
Sxx.Assemble()
Ssurf.Assemble()

Mint.Assemble()
Sint.Assemble()


Ph=array([
    [0,0,0,0,2,0,0,0,1],
    [0,0,0,0,0,1,1,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,-2,0,0,0,0,0,0,0],
    [0,0,-1,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    ])

Th=array([
    [-a,d,-b, 0, 0, 0, 0,0, 0, 0],
    [1 ,0, 0, 0, 0, 0, 0,0, 0, 0],
    [d ,0, 0,-a,-b, 0, 0,0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0,0, 0, 0],
    [b ,0, 0, 0, 0,-d, 0,0, 0, 0],
    [0 ,0, 0, 0, 0, b,-d,0, 0, 0],
    [0 ,0, 0, 0, 0, 0,-a,d,-b, 0],
    [0 ,0, 0, 0, 0, 0, 1,0, 0, 0],
    [0 ,0, 0, 0, 0, 0, 0,0, 0, 1]
    ])
Tt=array([
    [0 ,-c,0, 0, 0, 0, 0,0, 0, 0],
    [0 , 0,1, 0, 0, 0, 0,0, 0, 0],
    [-c, 0,0, 0, 0, 0, 0,0, 0, 0],
    [0,  0,0, 0, 1, 0, 0,0, 0, 0],
    [-a, 0,0, 0, 0, c, 0,0, 0, 0],
    [0 , 0,0, 0, 0,-a, c,0, 0, 0],
    [0 , 0,0, 0, 0, 0, 0,-c, 0, 0],
    [0 , 0,0, 0, 0, 0, 0,0, 1, 0],
    [1 ,0, 0, 0, 0, 0, 0,0, 0, 0]
    ])
Pt=array([
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,1,0],
    ])

saialp=SaiALP([M.mat,Mx.mat,Mxx.mat,S.mat,Sx.mat,Sxx.mat,Ssurf.mat,Mint.mat,Sint.mat],Ph.T,Pt.T,Th,Tt,shift,fes.FreeDofs())
saialp.CalcInverse('sparsecholesky')
saialp.CalcKrylow(2*nevs)
lam=saialp.SolveHessenberg(gfu.vecs,nevs)

Draw(gfu.components[0])

plot(lam.real,lam.imag,'x')
plot((correct_evs()).real,(correct_evs()).imag,'.k')

show()

