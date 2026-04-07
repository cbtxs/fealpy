from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen, Newton
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.solver import spsolve, cg, gmres
from fealpy.backend import backend_manager as bm
from fealpy.mesher import ElbowPipeMesher
from fealpy.mesh import TetrahedronMesh

params = {
    "D": 1.0,                     # 管道内径 1.0 m (对应半径 0.5 m)
    "bend_angle": 90.0,           # 90度弯曲
    "R_bend_inner": 2.3,          # 使得中心曲率半径 Rc = (2.3 + 0.5) * D = 2.8D
    "L_in_ratio": 10.0,           # 上游直管段 10m / 1m = 10.0
    "L_out_ratio": 15.0,          # 下游直管段 15m / 1m = 15.0
    "wall_thickness": 0.05,       # 报告未给定，基于1m管径假定一个合理值 (如 50mm)
    "mesh_size_global": 0.3,     # 使用默认网格大小策略
    "mesh_size_bend": 0.3,
    "mesh_size_interface": 0.3,
}
mesher = ElbowPipeMesher(params)
tetra_mesh = mesher.init_mesh()
tetra_mesh.to_vtk("pipe_bend_mesh.vtu")
region_tags = tetra_mesh.celldata["region"]
fluid_cell_indices = bm.where(region_tags == 1)[0]

def fluid_mesh(full_mesh):
    """
    从完整的 FSI 网格中安全地提取纯流体网格，并清理冗余节点。
    """
    old_nodes = full_mesh.entity('node')
    old_cells = full_mesh.entity('cell')
    cell_tags = full_mesh.celldata['region'] 
    
    is_fluid_cell = (cell_tags == 1)
    fluid_cells_old_idx = old_cells[is_fluid_cell]
    unique_nodes, new_cell_nodes = bm.unique(fluid_cells_old_idx, return_inverse=True)
    fluid_nodes = old_nodes[unique_nodes]
    fluid_cells = new_cell_nodes.reshape(fluid_cells_old_idx.shape)
    fluid_mesh = TetrahedronMesh(fluid_nodes, fluid_cells)
    
    return fluid_mesh

mesh = fluid_mesh(tetra_mesh)
pde = PipeBendTurbulentFlow()
equation = StationaryIncompressibleNS(pde=pde)
fem = Ossen(equation=equation, mesh=mesh)

u0 = fem.uspace.function()
u1 = fem.uspace.function()
p0 = fem.pspace.function()
p1 = fem.pspace.function()

for i in range(1):
    BForm = fem.BForm()
    LForm = fem.LForm()
    fem.update(u0=u0)
    A = BForm.assembly() 
    b = LForm.assembly()
    A, b = fem.apply_bc(A, b, pde)
    if equation.pressure_neumann == True:
        A, b = fem.lagrange_multiplier(A, b)
    x = spsolve(A, b)

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
    # u0[:] = 0.5 * u1[:] + 0.5 * u0[:]
    # p0[:] = 0.5 * p1[:] + 0.5 * p0[:]


from fealpy.mesh import TriangleMesh
from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
mesh_dict = mesher.mesh_data()
node_id, cell_flat = bm.unique(mesh_dict["interface_tri"], return_inverse=True)
node = mesh_dict["node"][node_id]
cell = cell_flat.reshape(-1, 3)
tri_interface = TriangleMesh(node, cell)

is_wall = pde.is_wall_boundary(mesh.entity('node'))
p = p1[is_wall]
pressurespace = LagrangeFESpace(mesh=tri_interface, p=1)
pressure = pressurespace.function()
pressure[:] = p

pressspace = TensorFunctionSpace(pressurespace, (3, -1))
press = pressspace.function()

v0 = node[cell[:, 1], :] - node[cell[:, 0], :]
v1 = node[cell[:, 2], :] - node[cell[:, 0], :]
nv = bm.cross(v0, v1)
S = bm.sqrt(bm.sum(nv**2, axis=1))/2
nv = nv / bm.sqrt(bm.sum(nv**2, axis=1))[:, None]

n2c = tri_interface.node_to_cell()
ws = bm.ones(n2c.shape)
ws *= S
ws = n2c.mul(ws)
ws = ws.toarray()
ws_sum = bm.sum(ws, axis=1)
ws = ws / ws_sum[:, None]
nv = ws @ nv
press[:] = (pressure[:, None] * nv).T.reshape(-1)

tri_interface.nodedata["press"] = press.reshape(3, -1).T
tri_interface.to_vtk("pressure.vtu")

from fealpy.csm.model.linear_elasticity.elbow_pipe_model import ElbowPipeModel
from fealpy.decorator import cartesian, barycentric

solid_pde = ElbowPipeModel(params=params)
solid_mesh = solid_pde.init_mesh()

@cartesian
def distance_t0_wallline(p):
    R_pipe = 0.5  
    R_bend = 2.8

    x = p[..., 0]
    y = p[..., 1]
    z = p[..., 2]

    dist_to_axis_up = bm.sqrt(y**2 + z**2)
    d_up = dist_to_axis_up - R_pipe

    dist_to_center_xy = bm.sqrt(x**2 + (y - R_bend)**2)
    dist_to_axis_bend = bm.sqrt((dist_to_center_xy - R_bend)**2 + z**2)
    d_bend = dist_to_axis_bend - R_pipe
    dist_to_axis_down = bm.sqrt((x - R_bend)**2 + z**2)
    d_down = dist_to_axis_down - R_pipe
    d = bm.where(x <= 0, d_up, 
                    bm.where(y >= R_bend, d_down, d_bend))

    return bm.maximum(d, 1e-15)

@cartesian
def is_inwall_boundary(p):
    d = distance_t0_wallline(p)
    atol = 1e-12
    on_boundary = (bm.abs(d)<atol)
    return on_boundary

is_inwall = is_inwall_boundary(solid_mesh.node)
space = LagrangeFESpace(mesh=solid_mesh, p=1)
solid_pspace = TensorFunctionSpace(space, (3, -1))
solid_p = solid_pspace.function()
solid_p.reshape(3, -1)[:, is_inwall] = press.reshape(3, -1)
solid_mesh.to_vtk("solidpressure.vtu")

@barycentric
def SI_source(bcs, index):
    result = solid_p(bcs, index)
    return result

import argparse

# Argument parsing
parser = argparse.ArgumentParser(description=
        """
        Finite element analysis for fluid-structure interaction (FSI) in hydraulic valve systems,
        with linear elasticity for structural deformation in steady-state conditions.
        """)

parser.add_argument('--backend',
        default='numpy', type=str,
        help="Default backend is numpy")

parser.add_argument('--pde',
                    default=solid_pde, type=int,
                    help="index of the linear elasticity  model, default is 4")

parser.add_argument('--mesh_type',
                    default='uniform_tet', type=str,
                    help="Type of mesh, default is uniform_tet")

parser.add_argument('--space_degree',
        default=1, type=int,
        help="Degree of Lagrange finite element space, default is 1")

parser.add_argument('--E', 
                    default=2.1e11, type=float, 
                    help="Young's modulus (E) in GPa for the elastic material")

parser.add_argument('--nu',
                    default=0.3, type=float,
                    help="Poisson's ratio (nu) for the elastic material, default is 0.3")

parser.add_argument('--rho',
                    default=7800, type=float,
                    help="density for the elastic material, default is 7800")

parser.add_argument('--pbar_log',
                    default=True, type=bool,
                    help='Whether to show progress bar, default is True')

parser.add_argument('--log_level',
                    default='INFO', type=str,
                    help='Log level, default is INFO, options are DEBUG, INFO, WARNING, ERROR, CRITICAL')

options = vars(parser.parse_args())


from fealpy.backend import bm
bm.set_backend(options['backend'])

from fealpy.csm.fem.hydraulic_pipe_lfem_model import  HydraulicPipeLFEMModel
model = HydraulicPipeLFEMModel(options)


A, F = model.linear_system()
model.SI.source = SI_source
A = A.assembly()
F = F.assembly()
A1, F1 = model.apply_bc(A, F)
uh = model.solve(A1, F1)
print("max displacement:", float(bm.max(bm.abs(uh))))
print(float(bm.linalg.norm(uh)))
model.show(uh)
print("-----------------------------")
