from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike, Index, _S, CoefLike


class HydraulicPipeFSIFEMModel:
    """
    """
    def __init__(self, options: dict, pde):
        self.options = options
        self.options["pde"] = pde
        self.max_iter = options["max_iter"]  # 最大迭代次数
        self.tolerance = options["tolerance"]  # 收敛容忍度
        self.pde = pde

    def run(self):
        """
        执行流固耦合算法
        """
        from fealpy.cfd.equation.stationary_incompressible_ns import StationaryIncompressibleNS
        from fealpy.cfd.simulation.fem.stationary_incompressible_ns import Ossen, Newton
        from fealpy.cfd import StationaryIncompressibleNSLFEMModel
        from fealpy.solver import spsolve
        from fealpy.mesh import TriangleMesh
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

        for iteration in range(self.max_iter):
            # 1. 求解流体方程，计算压力
            pde = self.pde
            fluid_model = StationaryIncompressibleNSLFEMModel(pde=pde, mesh=pde.fluid_mesh, options=self.options)
            u1, p1 = fluid_model.run()
            pde.fluid_mesh.nodedata["u"] = u1.reshape(3, -1).T
            pde.fluid_mesh.nodedata["p"] = p1
            pde.fluid_mesh.to_vtk("fluid.vtu")

            # 2. 压力传递
            from .coupling_interface import CouplingInterface
            interface = CouplingInterface(pde=pde)
            p_interface = interface.pressure_to_interface(p1 = p1)
            p_solid = interface.pressure_to_solid(p_interface)
            # mesh_dict = pde.mesher.mesh_data()
            # node_id, cell_flat = bm.unique(mesh_dict["interface_tri"], return_inverse=True)
            # node = mesh_dict["node"][node_id]
            # cell = cell_flat.reshape(-1, 3)
            # tri_interface = TriangleMesh(node, cell)

            # is_wall = pde.is_wall_boundary(pde.fluid_mesh.entity('node'))
            # p = p1[is_wall]
            # pressurespace = LagrangeFESpace(mesh=tri_interface, p=1)
            # pressure = pressurespace.function()
            # pressure[:] = p

            # pressspace = TensorFunctionSpace(pressurespace, (3, -1))
            # press = pressspace.function()

            # v0 = node[cell[:, 1], :] - node[cell[:, 0], :]
            # v1 = node[cell[:, 2], :] - node[cell[:, 0], :]
            # nv = bm.cross(v0, v1)
            # S = bm.sqrt(bm.sum(nv**2, axis=1))/2
            # nv = nv / bm.sqrt(bm.sum(nv**2, axis=1))[:, None]

            # n2c = tri_interface.node_to_cell()
            # ws = bm.ones(n2c.shape)
            # ws *= S
            # ws = n2c.mul(ws)
            # ws = ws.toarray()
            # ws_sum = bm.sum(ws, axis=1)
            # ws = ws / ws_sum[:, None]
            # nv = ws @ nv
            # press[:] = (pressure[:, None] * nv).T.reshape(-1)

            # from fealpy.csm.fem.hydraulic_pipe_lfem_model import  HydraulicPipeLFEMModel
            # from fealpy.decorator import cartesian, barycentric

            # @cartesian
            # def distance_t0_wallline(p):
            #     R_pipe = 0.5  
            #     R_bend = 2.8

            #     x = p[..., 0]
            #     y = p[..., 1]
            #     z = p[..., 2]

            #     dist_to_axis_up = bm.sqrt(y**2 + z**2)
            #     d_up = dist_to_axis_up - R_pipe

            #     dist_to_center_xy = bm.sqrt(x**2 + (y - R_bend)**2)
            #     dist_to_axis_bend = bm.sqrt((dist_to_center_xy - R_bend)**2 + z**2)
            #     d_bend = dist_to_axis_bend - R_pipe
            #     dist_to_axis_down = bm.sqrt((x - R_bend)**2 + z**2)
            #     d_down = dist_to_axis_down - R_pipe
            #     d = bm.where(x <= 0, d_up, 
            #                     bm.where(y >= R_bend, d_down, d_bend))

            #     return bm.maximum(d, 1e-15)

            # @cartesian
            # def is_inwall_boundary(p):
            #     d = distance_t0_wallline(p)
            #     atol = 1e-12
            #     on_boundary = (bm.abs(d)<atol)
            #     return on_boundary

            # is_inwall = is_inwall_boundary(pde.solid_mesh.node)
            # space = LagrangeFESpace(mesh=pde.solid_mesh, p=1)
            # solid_pspace = TensorFunctionSpace(space, (3, -1))
            # solid_p = solid_pspace.function()
            # solid_p.reshape(3, -1)[:, is_inwall] = press.reshape(3, -1)

            # 3. 求解固体方程，计算位移
            from fealpy.csm.fem.hydraulic_pipe_lfem_model import  HydraulicPipeLFEMModel
            from fealpy.decorator import cartesian, barycentric
            model = HydraulicPipeLFEMModel(self.options)
            model.set_pde(pde)
            A, F = model.linear_system()
            @barycentric
            def SI_source(bcs, index):
                result = p_solid(bcs, index)
                return result
            model.SI.source = SI_source
            A = A.assembly()
            F = F.assembly()
            A1, F1 = model.apply_bc(A, F)
            uh = model.solve(A1, F1)
            print("max displacement:", float(bm.max(bm.abs(uh))))
            print(float(bm.linalg.norm(uh)))
            model.show(uh)
            print("-----------------------------")


            exit()
            # 4. 检查收敛性（可以使用位移变化、压力变化等作为标准）
            if self.check_convergence(pressure, uh):
                print(f"Converged at iteration {iteration + 1}")
                break
            else:
                print(f"Iteration {iteration + 1} not converged.")

            # 5. 网格更新
            self.fsi_interface.transfer_solid_to_fluid(uh)

            

    def check_convergence(self, last_pressure, last_displacement):
        """
        检查流固耦合的收敛性
        """
        pressure_change = self.fluid_model.calculate_pressure_change(last_pressure)
        displacement_change = self.solid_model.calculate_displacement_change(last_displacement)

        return pressure_change < self.tolerance and displacement_change < self.tolerance