from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike

from .coupling_interface import CouplingInetrface


class FSIFEMModel:
    """
    """
    def __init__(self, mesher,
                 fsi_interface: CouplingInetrface, 
                 max_iter: int = 100, 
                 tolerance: float = 1e-6):
        self.fsi_interface = fsi_interface
        self.max_iter = max_iter  # 最大迭代次数
        self.tolerance = tolerance  # 收敛容忍度
        self.mesher = mesher
        self.mesh = self.mesher.init_mesh()
        self.fluid_mesh = self.extract_fluid_mesh()
        # self.solid_mesh = self.extract_solid_mesh(self.mesh)
    
    def extract_fluid_mesh(self):
        """
        从完整的 FSI 网格中安全地提取纯流体网格，并清理冗余节点。
        """
        from fealpy.mesh import TetrahedronMesh
        mesh = self.mesh
        old_nodes = mesh.entity('node')
        old_cells = mesh.entity('cell')
        cell_tags = mesh.celldata['region'] 
        is_fluid_cell = (cell_tags == 1)
        fluid_cells_old_idx = old_cells[is_fluid_cell]
        unique_nodes, new_cell_nodes = bm.unique(fluid_cells_old_idx, return_inverse=True)
        fluid_nodes = old_nodes[unique_nodes]
        fluid_cells = new_cell_nodes.reshape(fluid_cells_old_idx.shape)
        fluid_mesh = TetrahedronMesh(fluid_nodes, fluid_cells)
        
        return fluid_mesh

    def run(self):
        """
        执行流固耦合算法
        """
        from fealpy.cfd.model.stationary_incompressible_sst_k_omega.pipe_bend_turbulent_flow import PipeBendTurbulentFlow
        from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
        from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen, Newton
        from fealpy.solver import spsolve
        from fealpy.mesh import TriangleMesh

        for iteration in range(self.max_iter):
            # 1. 流体 → 固体：将流体压力传递给固体模型
            fluid_mesh = self.fluid_mesh
            pde = PipeBendTurbulentFlow()
            equation = StationaryIncompressibleNS(pde=pde)
            fem = Ossen(equation=equation, mesh=fluid_mesh)

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

                fluid_mesh.nodedata["uh"] = u1.reshape(3, -1).T
                fluid_mesh.nodedata["ph"] = p1
                fluid_mesh.to_vtk(f"stationary_sst_k_omega_{i+1}.vtu")

                res_u = fluid_mesh.error(u0, u1)
                res_p = fluid_mesh.error(p0, p1)
                print("res_u", res_u)
                print("res_p", res_p)

                if res_u + res_p < 1e-8:
                    break

                u0[:] = u1[:]
                p0[:] = p1[:]
    
            mesh_dict = self.mesher.mesh_data()
            tri_interface = TriangleMesh(mesh_dict["node"], mesh_dict["interface_tri"])
            index_wall = bm.unique(bm.concat(tri_interface))

            p = p1[index_wall].array


            # 2. 固体 → 流体：将固体模型的位移传回流体模型
            displacement = self.solid_model.calculate_displacement()  # 固体模型计算位移
            self.fsi_interface.transfer_solid_to_fluid(displacement)

            # 3. 检查收敛性（可以使用位移变化、压力变化等作为标准）
            if self.check_convergence(pressure, displacement):
                print(f"Converged at iteration {iteration + 1}")
                break
            else:
                print(f"Iteration {iteration + 1} not converged.")

    def check_convergence(self, last_pressure, last_displacement):
        """
        检查流固耦合的收敛性
        """
        pressure_change = self.fluid_model.calculate_pressure_change(last_pressure)
        displacement_change = self.solid_model.calculate_displacement_change(last_displacement)

        return pressure_change < self.tolerance and displacement_change < self.tolerance