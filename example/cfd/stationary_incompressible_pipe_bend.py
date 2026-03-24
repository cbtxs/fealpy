from fealpy.cfd.model.stationary_incompressible_navier_stokes.pipe_bend_3d import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.solver import spsolve, cg
from fealpy.backend import backend_manager as bm

options = {
    'backend': 'numpy',
    'pde': 1,
    'init_mesh': 'tri',
    'box': [0.0, 2.2, 0.0, 0.41],
    'center': (0.2, 0.2),
    'radius': 0.05,
    'n_circle': 1000,
    'lc': 0.004,
    'rho': 1.0,
    'mu': 1e-3,
    'method': 'Newton',
    'solve': 'direct',
    'apply_bc': 'cylinder',
    'postprocess': 'res',
    'run': 'main_cylinder',
    'maxit': 1,
    'maxstep': 1000,
    'tol': 1e-10
}

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

equation = StationaryIncompressibleNS(pde=pde)
fem = Ossen(equation=equation, mesh=mesh)

u0 = fem.uspace.function()
u1 = fem.uspace.function()
p0 = fem.pspace.function()
p1 = fem.pspace.function()

for i in range(1000):
    BForm = fem.BForm()
    LForm = fem.LForm()
    fem.update(u0=u0)
    A = BForm.assembly() 
    b = LForm.assembly()
    A, b = fem.apply_bc(A, b, pde)
    if equation.pressure_neumann == True:
        A, b = fem.lagrange_multiplier(A, b)
    x = cg(A, b)

    ugdof = fem.uspace.number_of_global_dofs()
    
    u1[:] = x[:ugdof]
    if equation.pressure_neumann == True:
        p1[:] = x[ugdof:-1]
    else:
        p1[:] = x[ugdof:]

    mesh.nodedata["uh"] = u1.reshape(3, -1).T
    mesh.nodedata["ph"] = p1
    mesh.to_vtk(f"stationary_sst_k_omega_{i+1}.vtu")

    res_u = mesh.error(u0, u1)
    res_p = mesh.error(p0, p1)
    print("res_u", res_u)
    print("res_p", res_p)

    if res_u + res_p < 1e-8:
        break

    u0[:] = u1[:]
    p0[:] = p1[:]

