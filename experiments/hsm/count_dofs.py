from ngsolve import *

m = Mesh('ref_geofile.vol.gz')

fes = H1(m,order=4)
print('all dofs', sum(fes.FreeDofs()))
fes.SetDefinedOn(m.Materials('pml'))
fes.Update()

print('pml dofs', sum(fes.FreeDofs()))


fes.SetDefinedOn(m.Materials('inner|transmission'))
fes.Update()

print('inner dofs', sum(fes.FreeDofs()))



fes.SetDefinedOn(~m.Materials('.*'))
fes.SetDefinedOn(m.Boundaries([2]))
fes.Update()

print('boundary dofs', sum(fes.FreeDofs()))
