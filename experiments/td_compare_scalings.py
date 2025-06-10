from ngsolve import *
from netgen.geom2d import unit_square
from ngswaves import time_integrators
from numpy import array,arange,savetxt
from fem1d import geo1d
from matplotlib.pyplot import *

#name = 'works'
#fq=50
#order=3
#Tend=3
#name = 'fails'
#fq=50
#order=6
#Tend=4
#name = 'fails_miserably'
#fq=100
#order=6
#Tend=4

#name = 'works_again'
fq=20
order=10
Tend=3


p0=1

timestep=0.0001


geo = geo1d(-4,-1,1,4)
geo.SetMaterials('pml','inner','pml')
geo.SetMaxhs(0.1,0.01,0.1)
m = Mesh(geo.GenerateMesh())
print(m.GetBoundaries())
input()
fes = H1(m,order=order)

Mi = BilinearForm(fes,symmetric=True)
Si = BilinearForm(fes,symmetric=True)

Me = BilinearForm(fes,symmetric=True)
Se = BilinearForm(fes,symmetric=True)
BD = BilinearForm(fes,symmetric=True)

Source = LinearForm(fes)
u,v = fes.TnT()

Mi+=SymbolicBFI(p0**2*u*v,definedon=m.Materials('inner'))
Me+=SymbolicBFI(u*v,definedon=m.Materials('pml'))

Si+=SymbolicBFI(grad(u)*grad(v),definedon=m.Materials('inner'))
Se+=SymbolicBFI(grad(u)*grad(v),definedon=m.Materials('pml'))

Source += SymbolicLFI(v,definedon=m.Materials('source'))
BD+=SymbolicBFI(u*v,definedon=m.Boundaries('inner1|inner0'))


Mi.Assemble()
Me.Assemble()
Si.Assemble()
Se.Assemble()
Source.Assemble()
BD.Assemble()

gfu = GridFunction(fes)
gfu1 = GridFunction(fes)
gfu2 = GridFunction(fes)
gfu3 = GridFunction(fes)

z = gfu.vec.CreateVector()
z[:]=0.
z1 = gfu.vec.CreateVector()
z1[:]=0.
z2 = gfu.vec.CreateVector()
z2[:]=0.

y = gfu.vec.CreateVector()
y[:]=0.
y1 = gfu.vec.CreateVector()
y1[:]=0.
y2 = gfu.vec.CreateVector()
y2[:]=0.

w = gfu.vec.CreateVector()
w[:]=0.
w1 = gfu.vec.CreateVector()
w1[:]=0.
w2 = gfu.vec.CreateVector()
w2[:]=0.


v = gfu.vec.CreateVector()
v[:]=0.

gfu.Set((cos(fq*x)+1)*exp(-(3*x)**2*10) ,definedon=m.Materials('inner|source'))
gfu1.Set((cos(fq*x)+1)*exp(-(3*x)**2*10),definedon=m.Materials('inner|source'))
gfu2.Set((cos(fq*x)+1)*exp(-(3*x)**2*10),definedon=m.Materials('inner|source'))
gfu3.Set((cos(fq*x)+1)*exp(-(3*x)**2*10),definedon=m.Materials('inner|source'))
fvec = gfu.vec.CreateVector()
def f(t,vec):
    if t<0:
        vec.Assign(Source.vec,10*cos((10*t)))
    else:
        vec[:]=0
    return vec

a=5
b=1
d=1
c=1

#timeint= time_integrators.IELP([Si.mat,None,None,None],[None,Mi.mat,Me.mat,Se.mat],array([[0,-1,0,0],[-b,0,d,0],[-d,0,0,b]]),array([[1,0,0,0],[-a,0,c,0],[-c,0,0,a]]),timestep,[gfu.vec,z,z1,z2],lambda t: f(t,fvec))
timeint= time_integrators.IELP([Si.mat,None,None,None],[None,Mi.mat,Me.mat,Se.mat],array([[0,-1,0,0],[-b,0,d,0],[-d,0,0,b]]),array([[1,0,0,0],[-a,0,c,0],[-c,0,0,a]]),timestep,[gfu.vec,z,z1,z2])

timeint.InvertSchurComp()

a=0
b=5
c=0
d=1
#timeint1= time_integrators.IELP([Si.mat,None,None,None],[None,Mi.mat,Me.mat,Se.mat],array([[0,-1,0,0],[-b,0,d,0],[-d,0,0,b]]),array([[1,0,0,0],[-a,0,c,0],[-c,0,0,a]]),timestep,[gfu1.vec,w,w1,w2],lambda t: f(t,fvec))
timeint1= time_integrators.IELP([Si.mat,None,None,None],[None,Mi.mat,Me.mat,Se.mat],array([[0,-1,0,0],[-b,0,d,0],[-d,0,0,b]]),array([[1,0,0,0],[-a,0,c,0],[-c,0,0,a]]),timestep,[gfu1.vec,w,w1,w2])
timeint1.InvertSchurComp()
a=2
b=8
c=0
d=1
#timeint2= time_integrators.IELP([Si.mat,None,None,None],[None,Mi.mat,Me.mat,Se.mat],array([[0,-1,0,0],[-b,0,d,0],[-d,0,0,b]]),array([[1,0,0,0],[-a,0,c,0],[-c,0,0,a]]),timestep,[gfu2.vec,y,y1,y2],lambda t: f(t,fvec))
timeint2= time_integrators.IELP([Si.mat,None,None,None],[None,Mi.mat,Me.mat,Se.mat],array([[0,-1,0,0],[-b,0,d,0],[-d,0,0,b]]),array([[1,0,0,0],[-a,0,c,0],[-c,0,0,a]]),timestep,[gfu2.vec,y,y1,y2])
timeint2.InvertSchurComp()


#timeint3= time_integrators.IELP([Si.mat,None],[BD.mat,Mi.mat],array([[0,-1]]),array([[1,0]]),timestep,[gfu3.vec,v],lambda t: f(t,fvec))
timeint3= time_integrators.IELP([Si.mat,None],[BD.mat,Mi.mat],array([[0,-1]]),array([[1,0]]),timestep,[gfu3.vec,v])
timeint3.InvertSchurComp('sparsecholesky',fes.GetDofs(m.Materials('source|inner')))

fig = figure()
xs = arange(-2,2,0.005)
plt,=plot(xs,[gfu(m(x)).real for x in xs],label='$\sigma(\omega)=\\frac{\\alpha-\\frac{\\beta}{i\omega}}{-i\omega\gamma+\delta}$')
plt1,=plot(xs,[gfu1(m(x)).real for x in xs],label='$\sigma(\omega)=-\\frac{\\beta}{i\omega}$')
plt2,=plot(xs,[gfu2(m(x)).real for x in xs],label='$\sigma(\omega)=\\alpha-\\frac{\\beta}{i\omega}$')
xs3 = arange(-1,1,0.001)
plt3,=plot(xs3,[gfu3(m(x)).real for x in xs3],'--',label='reference')
legend()
ylim((-0.5,2))
show(block=False)

input('push button to start')
i=0
imgind = 0
while True:
    timeint.Step() 
    timeint1.Step() 
    timeint2.Step() 
    timeint3.Step() 

    if i%3000 == 0:
        print('redrawing')
        plt.set_ydata([gfu(m(x)).real for x in xs])
        plt1.set_ydata([gfu1(m(x)).real for x in xs])
        plt2.set_ydata([gfu2(m(x)).real for x in xs])
        plt3.set_ydata([gfu3(m(x)).real for x in xs3])
        savetxt('output/td_1d_fqs1_{}.out'.format(int(i/3000)),np.array([xs,[gfu(m(x)).real for x in xs]]).T)
        savetxt('output/td_1d_fqs2_{}.out'.format(int(i/3000)),np.array([xs,[gfu1(m(x)).real for x in xs]]).T)
        savetxt('output/td_1d_fqs3_{}.out'.format(int(i/3000)),np.array([xs,[gfu2(m(x)).real for x in xs]]).T)
        fig.canvas.draw()
        #fig.savefig('frames/{}_{:04d}.png'.format(name,imgind))
        imgind +=1
    i+=1

   # if i*timestep>=Tend:
   #     break
