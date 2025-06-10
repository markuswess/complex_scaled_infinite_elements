import netgen.gui
from ngsolve import *
from matplotlib.pyplot import plot,show,figure
from netgen.geom2d import SplineGeometry
from numpy import array,sqrt,arange,shape
from libpygenvalues import *
from ngswaves import *
from saialp import *
ngsglobals.msg_level=10

#standard
alpha=1+0.5j
beta=0
gamma=1
delta=0

#fqs
#alpha=1+1j
#beta=0
#gamma=1
#delta=0


#fqs 2
#alpha=0
#beta=1j
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
maxh=0.15
order=3

hsmorder=20
nevs=80
pot=CoefficientFunction([-0.8,0])
Rx = 2
Ryt = 4
Ryb = 2
R=3.5
#shifts=[i-1-0.5j for i in range(16)]+[i-1-1j for i in range(32)]+[i-1-1.5j for i in range(32)]
#more for sigma_c
#shifts=[-1.5-0.25j,-0.5-0.25j,-0.35-0.7j,0.1-0.1j,-0.3j,-0.8-0.3j,-0.9-0.1j,-0.1-0.38j,-0.9-0.25j]
#more for sigma_v
shifts = [-1-0.1j,-2-0.1j,-0.25j,-1-0.25j]
shifts = [5-0.5j]
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

def egg_fqsc_spherical():
        g = MakeEggGeometry(Rx,Ryt,Ryb)
        g.AddCircle((0,1),R,leftdomain=2,rightdomain=0,bc='outer')
        m=Mesh(g.GenerateMesh(maxh=maxh))
        m.Curve(2*order)
        fes = FESpacePlus('h1ho',m,complex=True,order=order)
        hsm=SphericalHSMExterior((0,1),R,hsmorder,1+0j,order=order,complex=True)


        outbnd = m.Boundaries('outer')
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
        comp_evs=evs()
        for shift in shifts:
            saialp=SaiALP([M.mat,Mx.mat,Mxx.mat,S.mat,Sx.mat,Sxx.mat,Ssurf.mat,Mint.mat,Sint.mat],Ph.T,Pt.T,Th,Tt,shift,fes.FreeDofs())
            saialp.CalcInverse('sparsecholesky')
            saialp.CalcKrylow(2*nevs)
            lam=saialp.SolveHessenberg(gfu.vecs,nevs)
            comp_evs.shifts.append(ev_shift(shift,list(lam)))
        return comp_evs,gfu,fes,m

if __name__=='__main__':
    comp_evs,gf,fes,m = egg_fqsc_spherical()
    comp_evs.setuniformrectangleregion(1,0.5)
    comp_evs.killouterevs()
    comp_evs.plot()
    #comp_evs.savetofile('../output/egg_spherical_mobius_a_{}_b_{}_c_{}_d_{}_hsm_{}.out'.format(alpha,beta,gamma,delta,hsmorder))
    #comp_evs.savetofile('../output/egg_spherical_mobius_a_{}_b_{}_c_{}_d_{}_hsm_{}_more.out'.format(alpha,beta,gamma,delta,hsmorder))


    ref = evs()
    ref.loadfromfile('../output/egg_ref.out',0,900)
    ref.plot(False,1,'.k','ref')
    showplots(True)
