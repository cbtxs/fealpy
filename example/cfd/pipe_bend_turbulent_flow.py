from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_geo_mesh import PipeGeometry, PipeMesh
from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen, Newton
from fealpy.cfd.stationary_incompressible_navier_stokes_lfem_model import StationaryIncompressibleNSLFEMModel
from fealpy.solver import spsolve, cg, gmres
from fealpy.backend import backend_manager as bm
from fealpy.mesher import ElbowPipeMesher
from fealpy.mesh import TetrahedronMesh

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

params = {
    "D": 1.0,                     # 管道内径 1.0 m (对应半径 0.5 m)
    "bend_angle": 90.0,           # 90度弯曲
    "R_bend_inner": 2.3,          # 使得中心曲率半径 Rc = (2.3 + 0.5) * D = 2.8D
    "L_in_ratio": 10.0,           # 上游直管段 10m / 1m = 10.0
    "L_out_ratio": 15.0,          # 下游直管段 15m / 1m = 15.0
    "wall_thickness": 0.05,       # 报告未给定，基于1m管径假定一个合理值 (如 50mm)
    "mesh_size_global": 0.25,     # 使用默认网格大小策略
    "mesh_size_bend": 0.25,
    "mesh_size_interface": 0.25,
}
mesher = ElbowPipeMesher(params)
tetra_mesh = mesher.init_mesh()
tetra_mesh.to_vtk("pipe_bend_mesh.vtu")
region_tags = tetra_mesh.celldata["region"]
fluid_cell_indices = bm.where(region_tags == 1)[0]

def extract_fluid_mesh(full_mesh):
    """
    从完整的 FSI 网格中安全地提取纯流体网格，并清理冗余节点。
    """
    # 1. 获取全局节点和单元
    old_nodes = full_mesh.entity('node')
    old_cells = full_mesh.entity('cell')
    
    # 2. 获取单元的物理组标签 (假设存在 celldata 中，FEALPy 通常将其存为 'physical' 或类似键名)
    cell_tags = full_mesh.celldata['region'] 
    
    # 3. 找到所有属于流体的单元的布尔索引
    is_fluid_cell = (cell_tags == 1)
    
    # 4. 提取流体单元（此时单元内部的节点编号仍然是基于旧的全局 old_nodes 的索引）
    fluid_cells_old_idx = old_cells[is_fluid_cell]
    
    # 5. 剔除悬空节点，并重新映射节点编号
    unique_nodes, new_cell_nodes = bm.unique(fluid_cells_old_idx, return_inverse=True)
    
    # 6. 生成崭新且干净的流体节点坐标矩阵
    fluid_nodes = old_nodes[unique_nodes]
    
    # 7. 将扁平化的新节点索引重新 reshape 为 (N_cells, 4) 的四面体连接矩阵
    fluid_cells = new_cell_nodes.reshape(fluid_cells_old_idx.shape)
    
    # 8. 构建并返回全新的干净流体网格
    fluid_mesh = TetrahedronMesh(fluid_nodes, fluid_cells)
    
    return fluid_mesh

mesh = extract_fluid_mesh(tetra_mesh)

pde = PipeBendTurbulentFlow()

equation = StationaryIncompressibleNS(pde=pde)
fem = Ossen(equation=equation, mesh=mesh)
# fem = Newton(equation=equation, mesh=mesh)

u0 = fem.uspace.function()
u1 = fem.uspace.function()
p0 = fem.pspace.function()
p1 = fem.pspace.function()

for i in range(100):
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

