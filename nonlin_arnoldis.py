# arnoldi solvers for polynomial eigenvalue problems in ngsolve
# m. wess 2018

from ngsolve import *
from numpy import zeros,eye,array,outer,vdot
from numpy.random import random,rand
from numpy.linalg import norm,eig,inv
from time import perf_counter

_MESSAGELEVEL_ = ngsglobals.msg_level

def DBM(level,*varargs):
    if level<=_MESSAGELEVEL_:
        print(*varargs)

class SaiALP(object):
    def __init__(self, Ms: list, Ph, Pt, Th, Tt, shift, freedofs = None):
        DBM(5,'SaiALP:__init__ called')
        time = perf_counter()
        self.n = len(Ms)-1
        self.m = Th.shape[0]
        self.N= Ms[0].height
        ##check dimensions
        if not Th.shape == (self.m,self.m+1):
            raise Exception('Th has wrong shape should be ({},{})'.format(self.m,self.m+1))
        if not Tt.shape == (self.m,self.m+1):
            raise Exception('Tt has wrong shape should be ({},{})'.format(self.m,self.m+1))
        if not Ph.shape == (self.n+1,self.m+1):
            raise Exception('Ph has wrong shape should be ({},{})'.format(self.m+1,self.n+1))
        if not Pt.shape == (self.n+1,self.m+1):
            raise Exception('Pt has wrong shape should be ({},{})'.format(self.m+1,self.n+1))
        self.freedofs = freedofs

        self.Sh = Th[:,1:]
        self.St = Tt[:,1:]
        self.th = Th[:,0]
        self.tt = Tt[:,0]
        
        self.Th = Th
        self.Tt = Tt
        self.shift = shift
        self.Ph = Ph
        self.Pt = Pt
        self.Ms = Ms

        self.A = inv(self.Sh-shift*self.St)@Tt+0j
        self.a = inv(self.Sh-shift*self.St)@(self.th-self.shift*self.tt)+0j
        self.gp = self.Pt@array([-1,*self.a])+(self.Ph-self.shift*Pt)@array([0,*(self.A@array([1,*(-self.a)]))])

        self.Minv = None
        self.B = []
        self.W = []
        for i in range(self.n+1):
            self.W.append([])
        self.F = Pt-(Ph-shift*Pt)[:,1:]@self.A
        self.V=[]
        
        #print('A: ',self.A)
        #print('a: ', self.a)
        #print('F: ',self.F)
        self.timers = {
                'init' : 0.,
                'inverse' : 0.,
                'apply_inverse' : 0.,
                'next_krylow_vec' : 0.,
                'orthogonalize' : 0.,
                'small_vectors' : 0.,
                'build_krylow' : 0.,
                'total' : 0.,
                'hessenberg' : 0.,
                'project': 0.,
                'solve_projected': 0.,
                'residue': 0.,
                'calc_big_vecs' : 0.
                }
        self.timers['init']+=perf_counter()-time
        self.timers['total']+=perf_counter()-time
        DBM(5,'initialized linearizeable EVP with dimensions n={}, m={}, N={} in {} seconds'.format(self.n,self.m,self.N,self.timers['init']))


    def CalcInverse(self, inversetype='sparsecholesky'):
        DBM(5,'called CalcInverse')
        time = perf_counter()
        C = self.Ms[0].CreateMatrix()
        g = (self.Ph-self.shift*self.Pt)@array([1,*(-self.a)])
        C.AsVector().data = complex(g[0])*self.Ms[0].AsVector()
        for i in range(1,len(g)):
            C.AsVector().Add(self.Ms[i].AsVector(),complex(g[i]))

        self.Minv = C.Inverse(inverse=inversetype,freedofs=self.freedofs)
        self.timers['inverse'] += perf_counter()-time
        self.timers['total'] += perf_counter()-time
        DBM(3,'inverted in {} seconds'.format(self.timers['inverse']))

    def CalcKrylow(self, krylowdim, reorthogonalize = False, startvector=None,smallstartvector=None):
        time = perf_counter()
        DBM(5,'CalcKrylow called')
        if not self.Minv:
            raise Exception('no Krylow space without inverse of M(shift)')
        K = min(krylowdim,self.N*(self.m+1))
        DBM(3,'building Krylow space of dimension {}'.format(K))
        
        tmp = self.Ms[0].CreateColVector()
        tmp2 = self.Ms[0].CreateColVector()
        tmp.Cumulate()
        tmp2.Cumulate()
        if startvector is None:
            startvector = rand(self.N)
        if self.freedofs:
            for i in range(self.N):
                if self.freedofs[i]:
                    tmp[i]=complex(startvector[i])
        else: 
            for i in range(self.N):
                tmp[i]=complex(startvector[i])
        
        tmp.Assign(tmp,1/tmp.Norm())
        
        self.H = zeros((K,K))+0j
        self.V.append(zeros((1,self.m+1))+0j)
        if smallstartvector is None:
            self.V[0][0,0] = 1+0j
        elif smallstartvector=='a':
            self.V[0][0,:] = array([1,*(-self.a)])
        else:
            self.V[0][0,:] = smallstartvector + 0j
        DBM(5,'starting iteration')
        for k in range(1,K+1):
            print('{}/{}\r'.format(k,K),end='')
            self.B.append(self.Minv.CreateColVector())
            self.B[-1].Assign(tmp,1)

            #calc next vector
            t1 = perf_counter()
            for i in range(self.n+1):
                self.W[i].append(self.Ms[0].CreateColVector())
                self.Ms[i].Mult(self.B[-1],self.W[i][-1])

            E=self.V[-1]@self.F.T
            #print(E)
            #print(E/E[-1,0])
            for i in range(self.n+1):
                for j in range(k):
                    if i == 0 and j == 0:
                        tmp.Assign(self.W[0][0],complex(E[0,0]))
                    else:
                        tmp.Add(self.W[i][j],complex(E[j,i]))
            t2 = perf_counter()
            self.Minv.Mult(tmp,tmp2)
            self.timers['apply_inverse']+=perf_counter()-t2
            self.timers['next_krylow_vec']+=perf_counter()-t1
            #print(tmp2)
            t1 = perf_counter()
            D=zeros((k+1,self.m+1))+0j
            for j in range(k):
                D[j,0] = tmp2.InnerProduct(self.B[j])
                tmp2.Add(self.B[j],complex(-D[j,0]))
            l = tmp2.Norm()
            tmp.Assign(tmp2,1/l)
            self.timers['orthogonalize']+=perf_counter()-t1
            t1 = perf_counter()
            D[k,0] = l
            #print('D0: ',D[:,0])
            D[:-1,1:]=self.V[-1]@self.A.T
            D[:,1:]-=outer(D[:,0],self.a)
            #print('before ortho: ',D)
            for j in range(k):
                self.H[j,k-1]=self.V[j].flatten().conj().dot(D[:j+1,:].flatten())
                D[:j+1,:]-=self.H[j,k-1]*self.V[j]
            #print('after ortho: ',D)
            if k<K:
                self.H[k,k-1]=norm(D.flatten())
                self.V.append(1/self.H[k,k-1]*D)
                #print('end of step: ', self.V[-1])
            self.timers['small_vectors']+=perf_counter()-t1
        kt = perf_counter()-time
        DBM(3,'Krylowspace built in {} seconds'.format(kt))
        self.timers['build_krylow'] = kt
        self.timers['total'] += kt

    def SolveHessenberg(self, vecs = None, nevals = None, nevecs = None, sort = True):
        time = perf_counter()
        DBM(5,'called SolveHessenberg')
        k = self.H.shape[0]
        if self.H is None:
            raise Exception('Hessenberg not ready')
        if nevals is None:
            nevals = k
        else:
            nevals = min(k, nevals)
        lam,U=eig(self.H)
        
        inds = range(len(lam))
        if sort:
            inds = abs(lam).argsort()[::-1]
            lam = array([lam[i] for i in inds])

        ht = perf_counter()-time
        DBM(3,'solved Hessenberg EVP in {} seconds'.format(ht))
        self.timers['hessenberg']+=ht
        self.timers['total']+=ht
        time = perf_counter()
        if vecs:
            if nevecs is None:
                nevecs = len(vecs)
            n = min(self.H.shape[0],len(vecs),nevecs)
            DBM(3,'calculating {} big vectors'.format(n))
            V = zeros((k,len(self.B)))+0j
            for i in range(len(self.B)):
                V[:i+1,i]=self.V[i][:,0]
            vu=V@U
            for i in range(n):
                vecs[i].Assign(self.B[0],complex(vu[0,inds[i]]))
                for j in range(1,k):
                    vecs[i].Add(self.B[j],complex(vu[j,inds[i]]))
            
            bt = perf_counter()-time
            self.timers['calc_big_vecs']+=bt
            self.timers['total']+=bt
        return (1/lam+self.shift)[:nevals]



class PolyArnoldi(object):
    def __init__(self, mats, shift, freedofs = None):
        self.mats = tuple(mats)
        self.shift = complex(shift)
        self.u = None
        self.H = None
        self.A = None
        self.freedofs = freedofs
        self.Pinv = None
        self.Pmats = None
        self.timers = {
                'inverse' : 0.,
                'apply_inverse' : 0.,
                'next_krylow_vec' : 0.,
                'orthogonalize' : 0.,
                'small_vectors' : 0.,
                'build_krylow' : 0.,
                'total' : 0.,
                'hessenberg' : 0.,
                'project': 0.,
                'solve_projected': 0.,
                'residue': 0.,
                'calc_big_vecs' : 0.
                }
        DBM(5,'initialized PolyArnoldi for EVP of order {} with {} dofs'.format(
            len(mats)-1,mats[0].height))

    def CalcInverse(self, inversetype = ''):
        DBM(5,'called CalcInverse')
        time = perf_counter() 

        createdmat = False
        for i in range(len(self.mats)):
            if self.mats[i]:
                if not createdmat:
                    P = self.mats[i].CreateMatrix()
                    P.AsVector().Assign(self.mats[i].AsVector(),self.shift**i)
                    createdmat = True
                else:
                    P.AsVector().Add(self.mats[i].AsVector(),self.shift**i)
        if not createdmat:
            raise Exception('all matrices zero')

        DBM(3,'inverting P(shift)')
        self.Pinv = P.Inverse(self.freedofs,inversetype)
        self.timers['inverse'] += perf_counter()-time
        self.timers['total'] +=perf_counter()-time
        DBM(3,'inverted in {} seconds'.format(self.timers['inverse']))

    def CalcKrylow(self, krylowdim, reorthogonalize = False):
        time = perf_counter()
        N = len(self.mats)-1
        n = self.mats[0].height
        m = min(krylowdim,n*N)
    
        tmp = self.mats[0].CreateColVector()
        tmp2 = self.mats[0].CreateColVector()
        tmp.Cumulate()
        tmp2.Cumulate()

        if not self.Pinv:
            raise Exception('no Krylow space without inverse...')

        DBM(3,'building Krylow space of dimension {}'.format(m))
        
        rnd = random(n)
        if self.freedofs:
            for i in range(n):
                if self.freedofs[i]:
                    tmp[i]=complex(rnd[i])
        else: 
            for i in range(n):
                tmp[i]=complex(rnd[i])

        #tmp.Assign(tmp,1/Norm(tmp))
        tmp.Assign(tmp,1/tmp.Norm())


        self.H = zeros((m,m))+0j
        self.A = zeros((N*m,m))+0j

        self.A[0,0] = 1+0j
        smalltmp = zeros(m+1)+0j
        medtmp = zeros(N*m)+0j
        self.u = []

        DBM(5,'starting iteration')
        for i in range(m):
            print('{}/{}\r'.format(i+1,m),end='')
            self.u.append(self.Pinv.CreateColVector())
            self.u[-1].Assign(tmp,1)
            tmp2.Assign(self.u[0],complex(self.A[0,i]))

            #calculate next vector
            t1 = perf_counter()
            smalltmp[0] = self.A[0,i]

            for j in range(1,i+1):
                tmp2.Add(self.u[j],complex(self.A[j,i]))
                smalltmp[j] = self.A[j,i]
            if self.mats[1]:
                self.mats[1].Mult(tmp2,tmp)
            else:
                tmp[:] = 0
            
            for j in range(2,N+1):
                smalltmp[0] = self.shift*smalltmp[0]+self.A[(j-1)*m,i]
                tmp2.Assign(self.u[0],complex(smalltmp[0]))
                for l in range(1,i+1):
                    smalltmp[l] = self.shift*smalltmp[l]+self.A[(j-1)*m+l,i]
                    tmp2.Add(self.u[l],complex(smalltmp[l]))
                if self.mats[j]:
                    self.mats[j].MultAdd(1,tmp2,tmp)
            t2 = perf_counter()
            self.Pinv.MultScale(-1,tmp,tmp2)
            self.timers['apply_inverse']+=perf_counter()-t2
            self.timers['next_krylow_vec']+=perf_counter()-t1

            t1 = perf_counter()
            #orthogonalize
            for j in range(i+1):
                #medtmp[j] = InnerProduct(tmp2,self.u[j])
                medtmp[j] = tmp2.InnerProduct(self.u[j])
                tmp2.Add(self.u[j],-complex(medtmp[j]))

            if reorthogonalize:
                for j in range(i+1):
                    #tmpscal = InnerProduct(tmp2,self.u[j])
                    tmpscal = tmp2.InnerProduct(self.u[j])
                    tmp2.Add(self.u[j],-complex(tmpscal))
                    medtmp[j] += tmpscal

            #length = Norm(tmp2)
            length = tmp2.Norm()
            tmp.Assign(tmp2,1/length)
            self.timers['orthogonalize']+=perf_counter()-t1
            
            t1 = perf_counter()
            #small vector stuff
            for k in range(1,N):
                #for l in range(i+1):
                #    medtmp[k*m+l] = self.A[(k-1)*m+l,i]+self.shift*medtmp[(k-1)*m+l]
                medtmp[k*m:k*m+i+1] = self.A[(k-1)*m:(k-1)*m+i+1,i]+self.shift*medtmp[(k-1)*m:(k-1)*m+i+1]
            for k in range(i+1):
                self.H[k,i] = self.A[:,k].conj().dot(medtmp)
                medtmp -= self.H[k,i]*self.A[:,k]
            if reorthogonalize:
                for k in range(i+1):
                    tmpscal = self.A[:,k].conj().dot(medtmp)
                    medtmp -= tmpscal*self.A[:,k]
                    self.H[k,i] += tmpscal
            if i<m-1:
                medtmp[i+1] = length
                for k in range(1,N):
                    medtmp[k*m+i+1] = self.shift*medtmp[(k-1)*m+i+1]
                tmpscal = norm(medtmp)
                self.A[:,i+1] = medtmp/tmpscal
                self.H[i+1,i] = tmpscal
            self.timers['small_vectors']+=perf_counter()-t1

        kt = perf_counter()-time
        DBM(3,'Krylowspace built in {} seconds'.format(kt))
        self.timers['build_krylow'] = kt
        self.timers['total'] += kt


    def SolveHessenberg(self, vecs = None, nevals = None, nevecs = None, sort = True):
        time = perf_counter()
        DBM(5,'called SolveHessenberg')
        k = self.H.shape[0]
        if self.H is None:
            raise Exception('Hessenberg not ready')
        if nevals is None:
            nevals = k
        else:
            nevals = min(k, nevals)
        lam,V=eig(self.H)
        
        inds = range(len(lam))
        if sort:
            inds = abs(lam).argsort()[::-1]
            lam = array([lam[i] for i in inds])

        ht = perf_counter()-time
        DBM(3,'solved Hessenberg EVP in {} seconds'.format(ht))
        self.timers['hessenberg']+=ht
        self.timers['total']+=ht
        time = perf_counter()
        if vecs:
            if nevecs is None:
                nevecs = len(vecs)
            n = min(self.H.shape[0],len(vecs),nevecs)
            DBM(3,'calculating {} big vectors'.format(n))
            va=self.A[:k,:].dot(V)
            for i in range(n):
                vecs[i].Assign(self.u[0],complex(va[0,inds[i]]))
                for j in range(1,k):
                    vecs[i].Add(self.u[j],complex(va[j,inds[i]]))
            
            bt = perf_counter()-time
            self.timers['calc_big_vecs']+=bt
            self.timers['total']+=bt
        return (1/lam+self.shift)[:nevals]

    def Project(self, normalize = False):
        t = perf_counter()
        if self.u is None:
            raise Exception('can\'t project without krylow space base')
        DBM(3,'projecting coefficient matrices')
        k = len(self.u)
        tmp = self.u[0].CreateVector()
        tmp2 = self.u[0].CreateVector()

        self.Pmats = []

        for mat in self.mats:
            self.Pmats.append(zeros((k,k))+0j)
            if mat:
                for j in range(k):
                    if normalize:
                        mat.Mult(self.u[j],tmp)
                        self.Pinv.Mult(tmp,tmp2)
                    else:
                        mat.Mult(self.u[j],tmp2)
                    for l in range(k):
                        #self.Pmats[-1][l,j] = InnerProduct(tmp2,self.u[l])
                        self.Pmats[-1][l,j] = tmp2.InnerProduct(self.u[l])
        
        pt = perf_counter()-t
        self.timers['project']+=pt
        self.timers['total']+=pt
    
    def SolveProjected(self, vecs = None, nevals = None, nevecs = None, sort = True):
        t = perf_counter()
        if self.Pmats is None:
            raise Exception('can\'t solve projected without projected coefficient matrices')
        N=len(self.Pmats)-1
        
        m=self.Pmats[0].shape[0]
        if nevals is None:
            nevals = m*N
        else:
            nevals = min(m*N, nevals)
        

        biga=zeros((N*m,N*m))+0j
        bigb=eye(N*m)+0j

        biga[:(N-1)*m,m:]=eye(m*(N-1))+0j
        bigb[m*(N-1):,m*(N-1):]=-self.Pmats[-1]
        for i in range(N):
            biga[m*(N-1):,i*m:(i+1)*m]=self.Pmats[i]

        lam,V = eig(inv(biga-self.shift*bigb).dot(bigb))
        
        inds = range(len(lam))
        if sort:
            inds = abs(lam).argsort()[::-1]
            lam = array([lam[i] for i in inds])
        
        DBM(3,'returning {} eigenvalues'.format(nevals))

        if vecs:
            if nevecs is None:
                nevecs = len(vecs)
            time = perf_counter()
            n = min(m*N,len(vecs),nevecs)
            DBM(3,'calculating {} big vectors'.format(n))

            for i in range(n):
                vecs[i].Assign(self.u[0],complex(V[0,inds[i]]))
                for j in range(1,m):
                    vecs[i].Add(self.u[j],complex(V[j,inds[i]]))

            bt = perf_counter()-time
            self.timers['calc_big_vecs']+=bt
            self.timers['total']+=bt

        pt = perf_counter()-t
        self.timers['solve_projected']+=pt
        self.timers['total']+=pt

        return (1/lam+self.shift)[:nevals]

    def Residue(self, evecs, evs, res = None):
        t = perf_counter()
        if res is None:
            res = [self.mats[0].CreateColVector() for i in range(min(len(evecs),len(evs)))]

        for i in range(min(len(res), len(evecs), len(evs))):
            self.mats[0].Mult(evecs[i],res[i])
            for j in range(1,len(self.mats)):
                self.mats[j].MultAdd(complex(evs[i]**j),evecs[i],res[i])
        
        pt = perf_counter()-t
        self.timers['residue']+=pt
        self.timers['total']+=pt
        #return [Norm(r) for r in res]
        return [r.Norm() for r in res]

    def PrintTimers(self):
        for t in self.timers:
            if self.timers[t]:
                print(t,': ',self.timers[t])

    def ResetTimers(self,names = None):
        if names is None:
            names = self.timers.keys()
        for t in names:
            self.timers[t] = 0.


def PolyArnoldiSolver(mats: list, shift: complex, krylowdim: int, **kwargs)-> list:
    """Shift-and-invert Arnoldi eigenvalue solver
    problem. Returns list of approximated eigenvalues\n
    keyword arguments:\n
    vecs: list of vectors for eigenvector output\n
    nevals: number of eigenvalues\n
    nevecs: number of eigenvectors\n
    inversetype: type of inverse for shift-and-invert\n
    tol: if tol is given, only eigenvalues and eigenvectors with residue < tol are returned\n
    times: bool, if true timings are printed"""
    args = {
            'vecs' : None,
            'nevals' : None,
            'nevecs' : None,
            'freedofs' : None,
            'inversetype' : '',
            'tol' : None,
            'times' : False
            }
    args.update(kwargs)    

    pa = PolyArnoldi(mats, shift, args['freedofs'])
    pa.CalcInverse(args['inversetype'])
    pa.CalcKrylow(krylowdim)
    lam = pa.SolveHessenberg(args['vecs'], args['nevals'], args['nevecs'])
    if args['tol'] is not None:
        res = pa.Residue(args['vecs'],lam)
        inds = [i for i in range(len(res)) if res[i]<args['tol']]
        args['vecs'] = [args['vecs'][i] for i in inds]
        lam = [lam[i] for i in inds]
        DBM(3,'{} evs have residue < {}'.format(len(lam),args['tol']))
    if args['times']:
        pa.PrintTimers()
        
    DBM(7,'eigenvalues:')   
    for i in range(len(lam)):
        DBM(7,'{} : {:.7}'.format(i,lam[i]))
    return list(lam)

def PolyArnoldiSolverInverted(mats: list, shiftedinverse: object,shift: complex, krylowdim: int, **kwargs)-> list:
    """Shift-and-invert Arnoldi eigenvalue solver
    problem with already given inverse.\n
    Inverse must support method MultScale(fact,invec,outvec).\n
    Returns list of approximated eigenvalues\n
    keyword arguments:\n
    vecs: list of vectors for eigenvector output\n
    nevals: number of eigenvalues\n
    nevecs: number of eigenvectors\n
    tol: if tol is given, only eigenvalues and eigenvectors with residue < tol are returned\n
    times: bool, if true timings are printed"""
    args = {
            'vecs' : None,
            'nevals' : None,
            'nevecs' : None,
            'freedofs' : None,
            'tol' : None,
            'times' : False
            }
    args.update(kwargs)

    pa = PolyArnoldi(mats, shift, args['freedofs'])
    pa.Pinv = shiftedinverse
    pa.CalcKrylow(krylowdim)
    lam = pa.SolveHessenberg(args['vecs'], args['nevals'], args['nevecs'])
    if args['tol'] is not None:
        res = pa.Residue(args['vecs'],lam)
        inds = [i for i in range(len(res)) if res[i]<args['tol']]
        args['vecs'] = [args['vecs'][i] for i in inds]
        lam = [lam[i] for i in inds]
        DBM(3,'{} evs have residue < {}'.format(len(lam),args['tol']))
    if args['times']:
        pa.PrintTimers()
    return list(lam)

def ProjectedPolyArnoldiSolver(mats: list, shift: complex, krylowdim: int, **kwargs) ->  list:
    """Projects coefficient matrices on krylow space and solves linearized small
    problem. Returns list of approximated eigenvalues\n
    keyword arguments:\n
    vecs: list of vectors for eigenvector output\n
    nevals: number of eigenvalues\n
    nevecs: number of eigenvectors\n
    inversetype: type of inverse for shift-and-invert\n
    tol: if tol is given, only eigenvalues and eigenvectors with residue < tol are returned\n
    times: bool, if true timings are printed"""
    args = {
            'vecs' : None,
            'nevals' : None,
            'nevecs' : None,
            'freedofs' : None,
            'inversetype' : '',
            'tol' : None,
            'times' : False
            }
    args.update(kwargs)
    pa = PolyArnoldi(mats, shift, args['freedofs'])
    pa.CalcInverse(args['inversetype'])
    pa.CalcKrylow(krylowdim,True)
    pa.Project()
    lam = pa.SolveProjected(args['vecs'], args['nevals'], args['nevecs'])
    if args['tol'] is not None:
        res = pa.Residue(args['vecs'],lam)
        inds = [i for i in range(len(res)) if res[i]<args['tol']]
        args['vecs'] = [args['vecs'][i] for i in inds]
        lam = [lam[i] for i in inds]
        DBM(3,'{} evs have residue < {}'.format(len(lam),args['tol']))
    if args['times']:
        pa.PrintTimers()
    return list(lam)

