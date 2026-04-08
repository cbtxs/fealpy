from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm


# @dataclass
# class InterfaceMesh:
#     """Interface triangle geometry for FSI coupling.

#     Primary data from Gmsh-style output is ``node`` + ``face`` (``interface_tri``).
#     ``normals`` and ``areas`` are computed in :func:`build_interface_mesh`.
#     Use :attr:`iface_node_ids` for unique interface vertex indices — it is **not** a
#     separate mesher field; it is always derived from ``face``.
#     """

#     tri: TriangleMesh
#     node: TensorLike  # (NN, GD) same coordinate array as the parent volume mesh when from Gmsh
#     face: TensorLike  # (NF, 3) triangle connectivity (global node ids into ``node``)
#     normals: TensorLike  # (NF, GD) unit face normals (from cross product of edges)
#     areas: TensorLike  # (NF,) triangle areas

#     @property
#     def iface_node_ids(self) -> TensorLike:
#         """Sorted unique global node indices appearing in ``face`` (derived, not from Gmsh)."""
#         return bm.unique(self.face.reshape(-1))


# def _triangle_normals_areas(node: TensorLike, face: TensorLike) -> Tuple[TensorLike, TensorLike]:
#     x0 = node[face[:, 0]]
#     x1 = node[face[:, 1]]
#     x2 = node[face[:, 2]]
#     e1 = x1 - x0
#     e2 = x2 - x0
#     cr = bm.cross(e1, e2)
#     norm_cr = bm.sqrt(bm.sum(cr * cr, axis=1))
#     areas = 0.5 * norm_cr
#     normals = cr / (norm_cr[:, None] + 1e-30)
#     return normals, areas


# def build_interface_mesh(tri: TriangleMesh) -> InterfaceMesh:
#     """Compute normals and areas from a surface ``TriangleMesh`` (Gmsh: node + interface_tri)."""
#     node = tri.entity("node")
#     face = tri.entity("cell")
#     normals, areas = _triangle_normals_areas(node, face)
#     return InterfaceMesh(
#         tri=tri,
#         node=node,
#         face=face,
#         normals=normals,
#         areas=areas,
#     )


# def _numpy_nearest_indices(src_np: TensorLike, dst_np: TensorLike) -> TensorLike:
#     """src (NS, GD), dst (ND, GD) -> (NS,) index of nearest dst row per src row."""
#     diff = src_np[:, bm.newaxis, :] - dst_np[bm.newaxis, :, :]
#     d2 = bm.sum(diff * diff, axis=2)
#     return bm.argmin(d2, axis=1).astype(bm.int64)


class CouplingInterface:
    """FSI coupling operator driven by interface triangles.

    - Pressure on fluid interface -> nodal force on solid (traction -p n, lumped to vertices).
    - Solid displacement (NN_solid, GD) -> fluid interface displacement / velocity.

    Use :meth:`from_volume_mesh` with ``node`` and ``interface_tri`` from
    :meth:`fealpy.mesher.gmsh_fsi_pipe_mesher.BaseGmshFSIPipeMesher.extract_mesh_data`,
    or pass two :class:`TriangleMesh` instances (``__init__``).
    """

    def __init__(
        self,
        pde
    ):
        # if mapping not in ("nearest", "direct_by_coord"):
        #     raise ValueError('mapping must be "nearest" or "direct_by_coord"')

        # self.fluid = build_interface_mesh(fluid_interface)
        # self.solid = build_interface_mesh(solid_interface)
        # self._s2f = self._build_node_mapping(
        #     src_coords=self.solid.node[self.solid.iface_node_ids],
        #     dst_coords=self.fluid.node[self.fluid.iface_node_ids],
        #     mode=mapping,
        #     tol=coord_tol,
        # )
        # self._solid_disp_prev: Optional[TensorLike] = None
        self.pde = pde

    # @classmethod
    # def from_volume_mesh(
    #     cls,
    #     node: TensorLike,
    #     interface_tri: TensorLike,
    #     *,
    #     solid_interface_tri: Optional[TensorLike] = None,
    #     mapping: str = "nearest",
    #     coord_tol: float = 1e-10,
    # ) -> CouplingInterface:
    #     """Build from volume-node array and interface triangles (Gmsh FSI convention).

    #     ``node`` and ``interface_tri`` are the ``"node"`` and ``"interface_tri"`` entries
    #     returned by ``ElbowPipeMesher(...).run()`` / ``mesh_data()``.

    #     For one conformal tet mesh, fluid and solid share the same FSI surface
    #     connectivity; omit ``solid_interface_tri``. If the solid side uses a different
    #     cell array (same global ``node`` ordering), pass it explicitly.
    #     """
    #     fluid_iface = TriangleMesh(node, interface_tri)
    #     if solid_interface_tri is None:
    #         return cls(fluid_iface, fluid_iface, mapping=mapping, coord_tol=coord_tol)
    #     solid_iface = TriangleMesh(node, solid_interface_tri)
    #     return cls(fluid_iface, solid_iface, mapping=mapping, coord_tol=coord_tol)

    # def _build_node_mapping(
    #     self,
    #     src_coords: TensorLike,
    #     dst_coords: TensorLike,
    #     mode: str,
    #     tol: float,
    # ) -> TensorLike:
    #     src_np = bm.to_numpy(src_coords)
    #     dst_np = bm.to_numpy(dst_coords)

    #     if mode == "direct_by_coord":
    #         out = bm.empty(src_np.shape[0], dtype=bm.int64)
    #         for i, p in enumerate(src_np):
    #             d = bm.linalg.norm(dst_np - p[None, :], axis=1)
    #             j = int(bm.argmin(d))
    #             if d[j] > tol:
    #                 raise ValueError(
    #                     f"direct_by_coord failed at {i}: min_dist={d[j]}"
    #                 )
    #             out[i] = j
    #         return out

    #     if mode == "nearest":
    #         return _numpy_nearest_indices(src_np, dst_np)

    #     raise ValueError(f"unknown mapping mode: {mode}")

    # def _global_to_local_lookup(self, global_ids: TensorLike) -> TensorLike:
    #     gids = bm.to_numpy(global_ids).astype(bm.int64)
    #     m = int(gids.max())
    #     table = -bm.ones(m + 1, dtype=bm.int64)
    #     table[gids] = bm.arange(gids.shape[0], dtype=bm.int64)
    #     return table

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
        # 单元法向量
        nv = nv / bm.sqrt(bm.sum(nv**2, axis=1))[:, None]

        n2c = tri_interface.node_to_cell()
        # 根据点附近单元面积计算单元对点的权重
        ws = bm.ones(n2c.shape)
        ws *= S
        ws = n2c.mul(ws)
        ws = ws.toarray()
        ws_sum = bm.sum(ws, axis=1)
        ws = ws / ws_sum[:, None]
        # 点的法向量（加权平均）
        nv = ws @ nv
        return nv

    def pressure_to_interface(self, p1: TensorLike) -> TensorLike:
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
    
    def pressure_to_solid(self, press: TensorLike) -> TensorLike:
        from fealpy.decorator import cartesian
        from fealpy.functionspace import LagrangeFESpace, TensorFunctionSpace

        @cartesian
        def distance_to_wallline(p):
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
            d = distance_to_wallline(p)
            atol = 1e-12
            on_boundary = (bm.abs(d)<atol)
            return on_boundary

        pde = self.pde
        is_inwall = is_inwall_boundary(pde.solid_mesh.node)
        space = LagrangeFESpace(mesh=pde.solid_mesh, p=1)
        solid_pspace = TensorFunctionSpace(space, (3, -1))
        solid_p = solid_pspace.function()
        solid_p.reshape(3, -1)[:, is_inwall] = press.reshape(3, -1)

        return solid_p

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
