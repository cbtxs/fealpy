from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation import StationaryIncompressibleRANS, StationaryTurbulentKineticEnergy, StationarySpecificDissipationRate
from fealpy.cfd.simulation.fem.stationary_sst_k_omega.stationary_incompressible_rans import Ossen
from fealpy.cfd.simulation.fem.stationary_sst_k_omega.stationary_turbulent_kinetic_energy import StationaryTurbulentKineticEnergyPicard
from fealpy.cfd.simulation.fem.stationary_sst_k_omega.stationary_specific_dissipation_rate import StationarySpecificDissipationRatePicard
import matplotlib.pyplot as plt
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
from fealpy.solver import spsolve, cg
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
eq_k = StationaryTurbulentKineticEnergy(pde = pde)
eq_omega = StationarySpecificDissipationRate(pde = pde)
fem = Ossen(equation = eq, mesh = mesh)
fem_k = StationaryTurbulentKineticEnergyPicard(equation = eq_k, mesh = mesh)
fem_omega = StationarySpecificDissipationRatePicard(equation = eq_omega, mesh = mesh)

uspace = fem.uspace
pspace = fem.pspace
kspace = LagrangeFESpace(mesh, p=2)
omegaspace = LagrangeFESpace(mesh, p=2)

u0 = uspace.function()
u1 = uspace.function()
p0 = pspace.function()
p1 = pspace.function()
k0 = kspace.function()
k1 = kspace.function()
omega0 = omegaspace.function()
omega1 = omegaspace.function()

k0.array = 0.00292 * bm.ones(k0.array.shape)
omega0.array = 1.41 * bm.ones(omega0.array.shape)
ugdof = uspace.number_of_global_dofs()


for i in range(1):

    BForm = fem.BForm()
    LForm = fem.LForm()
    fem.update(u0=u0, k0=k0, omega0=omega0)
    A = BForm.assembly()
    b = LForm.assembly()
    A, b = fem.apply_bc(A, b, pde=pde)
    x = cg(A, b)
    print(x)

    u1[:] = x[:ugdof]
    p1[:] = x[ugdof:]

    res_u = mesh.error(u0, u1)
    res_p = mesh.error(p0, p1)
    print("res_u", res_u)
    print("res_p", res_p)
    if res_u + res_p < 1e-8:
        break   

    u0[:] = u1
    p0[:] = p1

    mesh.nodedata["uh"] = u1.reshape(3, -1).T
    mesh.nodedata["ph"] = p1
    mesh.to_vtk(f"stationary_sst_k_omega_{i+1}.vtu")

    BForm_k = fem_k.BForm()
    LForm_k = fem_k.LForm()
    fem_k.update(u1=u1, k0 = k0, omega0=omega0, mu_t=fem.mu_t)
    A_k = BForm_k.assembly()
    print("A_k", A_k.shape)
    b_k = LForm_k.assembly()
    print("b_k", b_k.shape)
    # A_k, b_k = fem_k.apply_bc(A_k, b_k, pde=pde)

    k1[:] = cg(A_k, b_k)
    k0[:] = k1

    BForm_omega = fem_omega.BForm()
    LForm_omega = fem_omega.LForm()
    fem_omega.update(u1, k1, omega0, mu_t = fem.mu_t)
    A_omega = BForm_omega.assembly()
    b_omega = LForm_omega.assembly()

    omega1[:] = cg(A_omega, b_omega)
    omega0[:] = omega1








