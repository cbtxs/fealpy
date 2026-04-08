from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm

class CouplingInterface:
    """FSI coupling operator driven by interface triangles.

    - Pressure on fluid interface -> nodal force on solid (traction -p n, lumped to vertices).
    - Solid displacement (NN_solid, GD) -> fluid interface displacement / velocity.

    Use :meth:`from_volume_mesh` with ``node`` and ``interface_tri`` from
    :meth:`fealpy.mesher.gmsh_fsi_pipe_mesher.BaseGmshFSIPipeMesher.extract_mesh_data`,
    or pass two :class:`TriangleMesh` instances (``__init__``).
    """

    def __init__(self, pde):
        self.pde = pde

    def interface_normal(self):
        '''
            计算外法向量
        '''
        pde = self.pde
        tri_interface = pde.interface_mesh
        node = tri_interface.node
        cell = tri_interface.cell
        v0 = node[cell[:, 1], :] - node[cell[:, 0], :]
        v1 = node[cell[:, 2], :] - node[cell[:, 0], :]
        nv = bm.cross(v0, v1)
        S = bm.sqrt(bm.sum(nv**2, axis=1))/2
        # 单元中心处法向量
        nv = nv / bm.sqrt(bm.sum(nv**2, axis=1))[:, None]

        n2c = tri_interface.node_to_cell()
        # 根据点附近单元面积计算单元对点的权重
        ws = bm.ones(n2c.shape)
        ws *= S
        ws = n2c.mul(ws)
        ws = ws.toarray()
        ws_sum = bm.sum(ws, axis=1)
        ws = ws / ws_sum[:, None]
        # 节点处法向量（加权平均）
        nv = ws @ nv
        return nv

    def pressure_on_interface(self, p1: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        pde = self.pde
        is_wall = pde.is_wall_boundary(pde.fluid_mesh.entity('node'))
        p = p1[is_wall]
        pressurespace = LagrangeFESpace(mesh=pde.interface_mesh, p=1)
        pressure = pressurespace.function()
        pressure[:] = p

        pressspace = TensorFunctionSpace(pressurespace, (3, -1))
        press = pressspace.function()

        nv = self.interface_normal()
        press[:] = (pressure[:, None] * nv).T.reshape(-1)

        return press
    
    def pressure_on_solid(self, press: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

        pde = self.pde
        is_inwall = pde.is_wall_boundary(pde.solid_mesh.node)
        space = LagrangeFESpace(mesh=pde.solid_mesh, p=1)
        solid_pspace = TensorFunctionSpace(space, (3, -1))
        solid_p = solid_pspace.function()
        solid_p.reshape(3, -1)[:, is_inwall] = press.reshape(3, -1)

        return solid_p
    
    def shear_stress_on_interface(self, u1: TensorLike) -> TensorLike:
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace
        pde = self.pde
        is_wall = pde.is_wall_boundary(pde.fluid_mesh.interpolation_points(p=2))
        space = LagrangeFESpace(mesh=pde.interface_mesh, p=2)
        print("space", space.number_of_global_dofs())
        uspace = TensorFunctionSpace(space, (3, -1))
        print("uspace", uspace.number_of_global_dofs())
        velocity = uspace.function()
        velocity.reshape(3, -1)[:] = u1.reshape(3, -1)[:, is_wall]

        return velocity
        

    def shear_stress_on_solid():
        pass

    def solid_disp_to_fluid_interface(
        self,
        solid_disp: TensorLike,
        *,
        dt: Optional[float] = None,
        compute_velocity: bool = True,
    ) -> Tuple[TensorLike, Optional[TensorLike]]:
        s_ids = self.solid.iface_node_ids
        disp_s_iface = solid_disp[s_ids]

        ni_f = int(bm.to_numpy(self.fluid.iface_node_ids).shape[0])
        gd = int(bm.to_numpy(solid_disp).shape[1])
        disp_f_iface = bm.zeros((ni_f, gd), dtype=solid_disp.dtype)

        idx = self._s2f
        disp_f_iface[idx] = disp_s_iface

        vel_f_iface = None
        if compute_velocity:
            if dt is None:
                raise ValueError("dt is required when compute_velocity=True")
            if self._solid_disp_prev is None:
                vel_f_iface = disp_f_iface / dt
            else:
                prev = self._solid_disp_prev
                disp_s_prev_iface = prev[s_ids]
                disp_f_prev = bm.zeros_like(disp_f_iface)
                disp_f_prev[idx] = disp_s_prev_iface
                vel_f_iface = (disp_f_iface - disp_f_prev) / dt

        self._solid_disp_prev = solid_disp
        return disp_f_iface, vel_f_iface
