import netgen.gui
from ngsolve import *
from matplotlib.pyplot import plot,show,figure
from netgen.geom2d import SplineGeometry
from numpy import array,arange, loadtxt
from numpy.linalg import inv
from ngswaves import *
from libpygenvalues import *

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
        geo.Append([linetype,*i],bc='outer',leftdomain=1,rightdomain=0)
        #geo.Append([linetype,*i],bc='int',leftdomain=1,rightdomain=2)
    geo.SetMaterial(1,'inner')
    geo.SetMaterial(2,'outer')
    return geo
def MakeEggEggGeometry(Rx,Ryt,Ryb,d):
    geo = SplineGeometry()
    pts = [(x,y) for y in [-Ryb,0,Ryt] for x in [-Rx,0,Rx]]
    for pt in pts:
        geo.AppendPoint(*pt)

    inds = [(1,2,5),(5,8,7),(7,6,3),(3,0,1)]
    for i in inds:
        linetype = 'spline3'
        if len(i)==2: linetype='line'
        #geo.Append([linetype,*i],bc='outer',leftdomain=1,rightdomain=0)
        geo.Append([linetype,*i],bc='int',leftdomain=1,rightdomain=2)
    pts = [(x,y) for y in [-Ryb-d,0,Ryt+d] for x in [-Rx-d,0,Rx+d]]
    ginds=[]
    for pt in pts:
        ginds.append(geo.AppendPoint(*pt))

    inds = [(ginds[1],ginds[2],ginds[5]),(ginds[5],ginds[8],ginds[7]),(ginds[7],ginds[6],ginds[3]),(ginds[3],ginds[0],ginds[1])]
    for i in inds:
        linetype = 'spline3'
        if len(i)==2: linetype='line'
        geo.Append([linetype,*i],bc='outer',leftdomain=2,rightdomain=0)
        #geo.Append([linetype,*i],bc='int',leftdomain=1,rightdomain=2)
    geo.SetMaterial(1,'inner')
    geo.SetMaterial(2,'outer')
    return geo

maxh=0.15
order=3

hsmorder=15
sigma=0.5+0.5j
nevs=100
p0=0.2
Rx = 2
Ryt = 4
Ryb = 2
shifts=[0.5*i-0.25j for i in range(60)]+[0.5*i-0.75j for i in range(60)]+[0.5*i-1.25j for i in range(60)]+[0.5*i-1.75j for i in range(60)]+[0.5*i-2.25j for i in range(60)]+[0.5*i-2.75j for i in range(60)]

#shifts=[1-0.5j]
def egg_std_normal(shifts=shifts,sigma=sigma,nevs=nevs):
    g = MakeEggGeometry(2,4,2)
    #g = MakeEggEggGeometry(2,4,2,0.5)
    #g.AddCircle((0,1),3.5,leftdomain=2,rightdomain=0,bc='outer')
    m=Mesh(g.GenerateMesh(maxh=maxh))
    m.Curve(3*order)
    Draw(m)

    outbnd = m.Boundaries('outer')

    l2=FESpacePlus('l2surf',m,order=order-1,complex=True)
    l2.SetDefinedOn(outbnd)

    l2int=FESpacePlus('VectorL2',m,order=order-1,complex=True)

    h1=FESpacePlus('h1ho',m,order=order,complex=True)

    hsm=RadialHSMExterior((0,0),hsmorder,1+0j,complex = True,order=order)
    hsm1=RadialHSMExterior((0,0),hsmorder,1+0j,complex = True,order=order-1,dirichlet=[])

    l2=l2.AddExterior(hsm1,outbnd)
    h1=h1.AddExterior(hsm,outbnd)

    fescomp=FESpace([h1,l2,l2])

    p,ut,un = fescomp.TrialFunction()
    q,vt,vn = fescomp.TestFunction()


    pint=p[0]
    qint=q[0]


    n = specialcf.normal(2)
    t = CoefficientFunction((-n[1],n[0]))

    #calculate curve
    fes=FESpace('h1ho',m,order=order*2,dim=2)
    fes.SetDefinedOn(~m.Materials('.*'))
    fes.SetDefinedOn(outbnd)
    fes.Update()
    fes.FinalizeUpdate()

    nproj = GridFunction(fes)
    nproj.Set(n,definedon=outbnd)

    fesl2=FESpace('l2surf',m,order=order*2,dim=2)
    fesl2.SetDefinedOn(outbnd)
    fesl2.Update()

    nproj = GridFunction(fes)
    nproj.Set(n,definedon=outbnd)

    kc = GridFunction(fesl2)
    kc.Set(nproj.Deriv()*t,definedon=outbnd)
    k = kc*t


    urad, vrad = hsm.Tm.T, hsm.Tm
    durad, dvrad = hsm.Tp.T, hsm.Tp
    xi = hsm.Diffop

    a0 = BilinearForm(fescomp,symmetric=True)
    a1 = BilinearForm(fescomp,symmetric=True)
    a2 = BilinearForm(fescomp,symmetric=True)

    UnP = [ui*qi                        for ui in un for qi in q  ]
    PUn = [pi*vi                        for pi in p  for vi in vn ]
    UnPk = [ui*qi*k                     for ui in un for qi in q  ]
    PUnk = [pi*vi*k                     for pi in p  for vi in vn ]
    UtdP = [ui*(t*qi.Trace().Deriv())   for ui in ut for qi in q  ]
    PP = [pi*qi                         for pi in p  for qi in q  ] 
    PPk = [pi*qi*k                      for pi in p  for qi in q  ] 
    UnUnk = [ui*vi*k                    for ui in un for vi in vn ]
    UtUtk = [ui*vi*k                    for ui in ut for vi in vt ]
    UnUn = [ui*vi                       for ui in un for vi in vn ]
    UtUt = [ui*vi                       for ui in ut for vi in vt ]
    tdPtU = [(t*pi.Trace().Deriv())*vi  for pi in p  for vi in vt ]


    B000 = (urad@vrad).flatten()
    B001 = (urad@xi@vrad).flatten()

    B100 = (durad@vrad).flatten()
    B010 = (urad@dvrad).flatten()
    B101 = (durad@xi@vrad).flatten()
    B011 = (urad@xi@dvrad).flatten()
    #####A0
    a0+=SymbolicBFI(grad(pint)*grad(qint))
    #####A1
    bfis = (SymbolicExtBFI(UnPk,1j*sigma*B011,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(PUnk,1j*sigma*B101,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(UtdP,1j*sigma*B000,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(tdPtU,1j*sigma*B000,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(UnP,1j*B010,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(PUn,1j*B100,definedon=outbnd).MakeBFIs(fescomp))
    for bf in bfis:
        a1+=bf
    #a1+=SymbolicBFI(1j*u*grad(qint))
    #a1+=SymbolicBFI(1j*grad(pint)*v)

    #####A2
    bfis2 = (SymbolicExtBFI(PPk,-sigma**2*B001,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(UnUnk,sigma**2*B001,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(UtUtk,sigma**2*B001,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(PP,-sigma*B000,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(UnUn,sigma*B000,definedon=outbnd).MakeBFIs(fescomp)+
        SymbolicExtBFI(UtUt,sigma*B000,definedon=outbnd).MakeBFIs(fescomp))
    for bf in bfis2:
        a2+=bf
    a2+=SymbolicBFI(-p0**2*pint*qint,definedon=m.Materials('inner'))
    a2+=SymbolicBFI(-pint*qint,definedon=m.Materials('outer'))
    #a2+=SymbolicBFI(u*v)

    lam=[]

    SetHeapSize(100000000)
    with TaskManager():
        a0.Assemble()
        a1.Assemble()
        a2.Assemble()

    gf = GridFunction(fescomp,multidim=nevs)
    comp_evs=evs()
    for shift in shifts:
        lam=array(PolyArnoldiSolver([a0.mat,a1.mat,a2.mat],shift,nevs*2,vecs=gf.vecs,nevals=nevs,freedofs=fescomp.FreeDofs(),inversetype='sparsecholesky'))
        comp_evs.shifts.append(ev_shift(shift,list(lam)))

    Draw(gf.components[0].components[0])
    return comp_evs, gf, fescomp,m

if __name__=='__main__':
    comp_evs,gf,fes,m = egg_std_normal()
    comp_evs.setuniformrectangleregion(0.5,0.5)
    comp_evs.killouterevs()
    comp_evs.plot()
    comp_evs.savetofile('output/hsm_std_{}.out'.format(sigma))


    ref = evs()
    ref.loadfromfile('output/spherical_hsm_fqsc.out',0,900)
    ref.plot(False,1,'.k','ref')
    showplots(True)
