# arnoldi solvers for linearizeable eigenvalue problems in ngsolve
# m. wess 2019
from numpy.linalg import inv,norm,eig
from numpy.random import rand
from numpy import array,zeros,outer,vdot
from ngsolve import *
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

    def CalcKrylowExperimental(self, krylowdim, reorthogonalize = False, startvector=None,smallstartvector=None):
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
        DBM(5,'starting iteration')
        for k in range(1,K+1):
            DBM(5,'{}/{}\r'.format(k,K))
            self.B.append(self.Minv.CreateColVector())
            self.B[-1].Assign(tmp,1)

            #calc next vector
            t1 = perf_counter()
            for i in range(self.n+1):
                self.W[i].append(self.Ms[0].CreateColVector())
                self.Ms[i].Mult(self.B[-1],self.W[i][-1])
            tmp[:]=0.
            for i in range(self.n+1):
                self.Ms[i].MultAdd(complex(self.gp[i]),self.B[-1],tmp)
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
            #D[k,0] = l
            #print('D0: ',D[:,0])
            #D[:-1,1:]=self.V[-1]@self.A.T
            #D[:,1:]-=outer(D[:,0],self.a)
            #print('before ortho: ',D)
            #for j in range(k):
            #    self.H[j,k-1]=self.V[j].flatten().conj().dot(D[:j+1,:].flatten())
            #    D[:j+1,:]-=self.H[j,k-1]*self.V[j]
            #print('after ortho: ',D)
            #if k<K:
            #    self.H[k,k-1]=norm(D.flatten())
            #    self.V.append(1/self.H[k,k-1]*D)
                #print('end of step: ', self.V[-1])
            self.timers['small_vectors']+=perf_counter()-t1
        kt = perf_counter()-time
        DBM(3,'Krylowspace built in {} seconds'.format(kt))
        self.timers['build_krylow'] = kt
        self.timers['total'] += kt
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
        elif smallstartvector is 'a':
            self.V[0][0,:] = array([1,*(-self.a)])
        else:
            self.V[0][0,:] = smallstartvector + 0j
        DBM(5,'starting iteration')
        for k in range(1,K+1):
            DBM(5,'{}/{}\r'.format(k,K))
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
