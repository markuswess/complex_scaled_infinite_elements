from ngsolve import *

class geo1d:
    def __init__(self,*varargs):
        self.points=varargs
        self.ndoms=len(self.points)-1
        self.maxhs=self.ndoms*[1]
        self.materials=[]
        self.bcnames=['left','right']

    def __str__(self):
        return str(self.points)+str(self.maxhs)+str(self.materials)

    def SetMaxhs(self,*varargs):
        if len(varargs)==1:
            self.maxhs=self.ndoms*[varargs[0]]
            return
        if not len(varargs)==self.ndoms:
            raise Exception('wrong number of hs')
        else:
            self.maxhs=varargs

    def SetMaterials(self,*varargs):
        if len(varargs)==1:
            self.materials=self.ndoms*[varargs[0]]
            return
        if not len(varargs)==self.ndoms:
            raise Exception('wrong number of mats')
        else:
            self.materials=varargs

    def SetBCNames(self,left,right):
        self.bcnames=[left,right]


    def GenerateMesh(self,maxh=False):
        if maxh:
            self.SetMaxhs(maxh)
        from netgen.meshing import Mesh,Element1D,Element0D,MeshPoint
        from netgen.csg import Pnt
        m = Mesh()
        m.dim = 1
        pnums = []
        pnums.append(m.Add(MeshPoint(Pnt(self.points[0],0,0))))
        done_el=0
        for idx,(x,x1,h) in enumerate(zip(self.points,self.points[1:],self.maxhs)):
            nel = int((x1-x)/h)
            for i in range(1, nel+1):
                pnums.append (m.Add (MeshPoint (Pnt(x+i*(x1-x)/nel, 0, 0))))
            for i in range(done_el,done_el+nel):
                m.Add (Element1D ([pnums[i],pnums[i+1]], index=idx+1))
            done_el+=nel
            if self.materials:
                m.SetMaterial(idx+1,self.materials[idx])
        m.Add (Element0D (pnums[0], index=1))
        m.Add (Element0D (pnums[-1], index=2))    
        m.SetBCName(0,self.bcnames[0])
        m.SetBCName(1,self.bcnames[1])
        return m


if __name__=='__main__':
    geo=geo1d(0,1,2)
    geo.SetMaterials('asfdd','asddddd')
    geo.SetMaxhs(0.1,0.2)
    print(geo)
    m=geo.GenerateMesh()
