import netgen.gui
from ngsolve import *
from matplotlib.pyplot import plot,show,figure
from netgen.geom2d import SplineGeometry
from numpy import array,sqrt,arange
from libpygenvalues import *
from ngswaves import *
ngsglobals.msg_level=10

def MakeEggGeometry(Rx,Ryt,Ryb):
    geo = SplineGeometry()
    pts = [(x,y) for y in [-Ryb,0,Ryt] for x in [-Rx,0,Rx]]
    for pt in pts:
        geo.AppendPoint(*pt)

    inds = [(1,2,5),(5,8,7),(7,6,3),(3,0,1)]
    for i in inds:
        linetype = 'spline3'
        if len(i)==2: linetype='line'
        geo.Append([linetype,*i],bc='inner',leftdomain=1,rightdomain=2)

    geo.SetMaterial(1,'inner')
    geo.SetMaterial(2,'outer')
    return geo

maxh=0.15
order=3
#order=3

hsmorder=25
sigma=1+1j
nevs=80
p0=-0.8
Rx = 2
Ryt = 4
Ryb = 2
shifts=[i-1-0.5j for i in range(32)]+[i-1-1j for i in range(32)]+[i-1-1.5j for i in range(32)]
#shifts=[4]
def egg_fqsc_spherical():
    g = MakeEggGeometry(Rx,Ryt,Ryb)
    g.AddCircle((0,1),3.5,leftdomain=2,rightdomain=0,bc='outer')
    m=Mesh(g.GenerateMesh(maxh=maxh))
    m.Curve(2*order)
    fesint=FESpacePlus('h1ho',m,order=order,complex=True)
    hsm=SphericalHSMExterior((0,1),3.5,hsmorder,sigma,order=order,complex=True)

    fes=fesint.AddExterior(hsm,m.Boundaries('outer'))

    u,v = fes.TrialFunction(), fes.TestFunction()

    a0 = BilinearFormPlus(fes,symmetric=True)
    a1 = BilinearFormPlus(fes,symmetric=True)
    a2 = BilinearFormPlus(fes,symmetric=True)


    #####A0
    a0+=ExtBFI('laplace')
    a0+=SymbolicBFI(grad(u)*grad(v))

    a2+=ExtBFI('mass',m.Boundaries('outer'),-1)
    a2+=SymbolicBFI(-(1+p0)**2*u*v,definedon=m.Materials('inner'))
    a2+=SymbolicBFI(-u*v,definedon=m.Materials('outer'))

    lam=[]

    SetHeapSize(100000000)
    with TaskManager():
        a0.Assemble()
        a1.Assemble()
        a2.Assemble()
    gf = GridFunction(fes,multidim=nevs)

    comp_evs=evs()
    for shift in shifts:
        lam=array(PolyArnoldiSolver([a0.mat,a1.mat,a2.mat],shift,nevs*2,vecs=gf.vecs,nevals=nevs,freedofs=fes.FreeDofs(),inversetype='sparsecholesky'))
        comp_evs.shifts.append(ev_shift(shift,list(lam)))
    Draw(gf.components[0])
    return comp_evs, gf, fes,m

if __name__=='__main__':
    comp_evs,gf,fes,m = egg_fqsc_spherical()
    comp_evs.setuniformrectangleregion(1,1.5)
    comp_evs.killouterevs()
    comp_evs.plot()
    comp_evs.savetofile('../output/egg_spherical_std_s_{}_hsm_{}.out'.format(sigma,hsmorder))


    ref = evs()
    ref.loadfromfile('../output/egg_ref.out',0,900)
    ref.plot(False,1,'.k','ref')
    showplots(True)

