import netgen.gui
from ngsolve import *
from matplotlib.pyplot import plot,show,figure
from netgen.geom2d import SplineGeometry
from ngsolve.internal import visoptions,SnapShot
from numpy import array,sqrt,arange,shape
from libpygenvalues import *
from ngswaves import *
from saialp import *
ngsglobals.msg_level=10

#standard
#alpha=1+0.5j
#beta=0
#gamma=1
#delta=0

#fqs
alpha=1+1j
beta=0
gamma=1
delta=0

#for compatibility with diss
a=beta
b=alpha
c=delta
d=gamma
maxh=0.15
order=3

hsmorder=20
nevs=15
pot=CoefficientFunction([-0.8,0])
Rx = 2
Ryt = 4
Ryb = 2
R=3.5
shifts=[5-0.5j,9-0.5j,12-0.5j]
scalepoints=[(0,0.7),(0.9,0.5),(0,1.5)]
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

def egg_fqsc_spherical(shift):
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
        saialp=SaiALP([M.mat,Mx.mat,Mxx.mat,S.mat,Sx.mat,Sxx.mat,Ssurf.mat,Mint.mat,Sint.mat],Ph.T,Pt.T,Th,Tt,shift,fes.FreeDofs())
        saialp.CalcInverse('sparsecholesky')
        saialp.CalcKrylow(2*nevs)
        lam=saialp.SolveHessenberg(gfu.vecs,nevs)
        return lam,gfu,fes,m

if __name__=='__main__':
    visoptions.autoscale = True
    visoptions.usetexture = True
    visoptions.lineartexture = True
    visoptions.mmaxval = 0.25
    visoptions.mminval = -0.25
    Redraw()
    approx_evs=[]
    input('start')
    for shift,scalepoint in zip(shifts,scalepoints):
        evs,gf,fes,m = egg_fqsc_spherical(shift)
        plot(evs.real,evs.imag,'x')
        plot(evs[0].real,evs[0].imag,'o')
        approx_evs.append(evs[0])
        Draw(gf.components[0]/complex(gf.components[0](m(*scalepoint))),m,'asf')
        Redraw(blocking = True)
        #input('savemesh')
        #SnapShot('../../egg_geo.png')
        #input('save')
        #SnapShot('../../'+'images/egg_res_spherical_{:.3}_{:.3}'.format(evs[0].real,evs[0].imag).replace('.','_')+'.png')
        #input('calc next')
    print(approx_evs)
    show()
