from numpy import array,zeros,exp,shape,atleast_1d,eye,diag,ones

def gen_laguerre_func(n,m,x):
    x = atleast_1d(x)
    res = zeros((n+1,len(x))) 
    if x[0] is complex:
        res+=0j

    res[0,:]=exp(-x/4)
    res[1,:]=exp(-x/4)*(m+1-x)
    for i in range(1,n):
        res[i+1,:]=1/(i+1)*((2*i+m+1-x)*res[i,:]-(i+m)*res[i-1,:])
    return res*exp(-x/4)

def ie_matrices(N):
    Tp=-eye(N+1)
    Tp[:-1,1:]-=eye(N+1-1)
    Tm=eye(N+1)
    Tm[:-1,1:]-=eye(N+1-1)

    Diffop=-1/2*(-diag(range(1,2*N+1+1,2))
                +diag(range(1,N+1),1)+diag(range(1,N+1),-1))

    ie_mass = 1/2*Tm.T@Tm
    ie_laplace = 1/2*Tp.T@Tp
    ie_drift = 1/2*Tm.T@Tp
    ie_mass_x = 1/2*Tm.T@Diffop@Tm
    ie_mass_xx = 1/2*Tm.T@Diffop@Diffop@Tm
    ie_laplace_x = 1/2*Tp.T@Diffop@Tp
    ie_laplace_xx = 1/2*Tp.T@Diffop@Diffop@Tp
    ie_drift_x = 1/2*Tm.T@Diffop@Tp
    return ie_mass,ie_laplace,ie_drift,ie_mass_x, ie_mass_xx, ie_laplace_x, ie_laplace_xx,ie_drift_x

def mod_laguerre_quad(n):
    from numpy.linalg import eig
    mat=diag([(2*i+1) for i in range(n+1)])
    mat[1:,:-1]-=diag(range(1,n+1))
    mat[:-1,1:]-=diag(range(1,n+1))
    points,v = eig(mat)
    
    l=gen_laguerre_func(n,0,points)[-1,:]
    return array(points),array(points/l/l/(n+1)/(n+1))

def ie_mass_quad(coef,N,M):
    points,weights = mod_laguerre_quad(M)
    vals = gen_laguerre_func(N,-1,points)
    coefpoints = coef(points/2)
    mat= zeros((N+1,N+1))
    for i in range(N+1):
        for j in range(N+1):
            mat[i,j]=1/2*(vals[i,:]*coefpoints*vals[j,:])@weights
    return mat
