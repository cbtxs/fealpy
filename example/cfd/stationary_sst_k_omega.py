from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation import StationaryIncompressibleRANS, StationaryTurbulentKineticEnergy, StationarySpecificDissipationRate
from fealpy.cfd.simulation.fem.stationary_sst_k_omega.stationary_incompressible_rans import Ossen
import matplotlib.pyplot as plt
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace, FunctionSpace
from fealpy.typing import Index, _S
from fealpy.backend import backend_manager as bm


geom = PipeGeometry()
geom.build()

mesher = PipeMesh(geom, mesh_size=0.3)
mesh = mesher.generate_mesh()

# 网格可视化
# fig = plt.figure()
# axes = fig.add_subplot(111, projection='3d')
# mesh.add_plot(axes)
# plt.savefig("1.png")

pde = PipeBendTurbulentFlow()
eq = StationaryIncompressibleRANS(pde = pde)

_uspace = LagrangeFESpace(mesh, p=2)
uspace = TensorFunctionSpace(_uspace, (mesh.GD, -1))
pspace = LagrangeFESpace(mesh, p=1)
kspace = LagrangeFESpace(mesh, p=2)
omegaspace = LagrangeFESpace(mesh, p=2)

u0 = uspace.function()
p0 = pspace.function()
k0 = kspace.function()
omega0 = omegaspace.function()



points = mesh.interpolation_points(p=2)
bcs = bm.array([[1, 0, 0, 0],
                [0.5, 0.5, 0, 0],
                [0.5, 0, 0.5, 0],
                [0.5, 0, 0, 0.5],
                [0, 1, 0, 0],
                [0, 0.5, 0.5, 0],
                [0, 0.5, 0, 0.5],
                [0, 0, 1, 0],
                [0, 0, 0.5, 0.5],
                [0, 0, 0, 1]])

print("points", points.shape)
print("ugdof", uspace.number_of_global_dofs())
print("u0", u0.shape)
print("k0", k0.shape)
print("omega0", omega0.shape)
print("edge", mesh.number_of_edges())
print("nodes", mesh.number_of_nodes())
print("c2d", uspace.cell_to_dof())
mu_t = pde.tur_mu(u0=u0, k0=k0, omega0=omega0, bcs=bcs)




eq.set_constitutive("viscosity")




