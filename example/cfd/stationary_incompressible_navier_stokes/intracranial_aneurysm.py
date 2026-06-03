from fealpy.backend import backend_manager as bm
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.cfd.model import CFDPDEModelManager

options = {
    'backend': 'numpy',
    'method': 'Ossen',
    'solve': 'direct',
    'apply_bc': 'cylinder',
    'postprocess': 'res',
    'run': 'main',
    'maxit': 1,
    'maxstep': 1000,
    'tol': 1e-10
}

bm.set_backend(options['backend'])
from fealpy.cfd.model.stationary_incompressible_navier_stokes.intracranial_aneurysm_3d import IntracranialAneurysm3d
pde = IntracranialAneurysm3d()
model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh = pde.mesh, options = options)




def to_vtk(uh1, ph1, i):
    mesh.nodedata['ph'] = ph1
    mesh.nodedata['uh'] = uh1.reshape(3,-1).T
    mesh.to_vtk(f'stationary_2d{i+1}.vtu')


mesh = pde.mesh
maxit = options['maxit']
maxstep = options['maxstep']
tol = options['tol']

for i in range(maxit):
    print(f"number of cells: {mesh.number_of_cells()}")
    uh0 = model.fem.uspace.function()
    ph0 = model.fem.pspace.function()
    for j in range(maxstep):

        BForm, LForm = model.linear_system()  
        model.fem.update(uh0)
        A = BForm.assembly() 
        b = LForm.assembly()
        A, b = model.fem.apply_bc(A, b, pde)
        x = model.solve(A, b)

        ugdof = model.fem.uspace.number_of_global_dofs()
        uh1= model.fem.uspace.function()
        ph1 = model.fem.pspace.function()
        uh1[:] = x[:ugdof]
        ph1[:] = x[ugdof:]

        to_vtk(uh1, ph1, j)

        res_u = mesh.error(uh0, uh1)
        res_p = mesh.error(ph0, ph1)
        print(f"res_u: {res_u}, res_p: {res_p}")
        if res_u + res_p < tol:
            print(f"Converged at iteration {j+1}")
            break 
        uh0[:] = uh1
        ph0[:] = ph1











exit()


pde = IntracranialAneurysm3d()

mesh = pde.mesh
print("num_face", mesh.inlet_face_index.shape)
mesh.to_vtk("fealpy_mesh.vtu")
print("流入法向量", mesh.face_unit_normal(mesh.inlet_face_index).shape)

pde.is_inlet_boundary(mesh.node)