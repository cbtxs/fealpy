from dataclasses import dataclass
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.mesh import TriangleMesh

@dataclass
class InterfaceMesh:
    """Interface triangle geometry derived from TriangleMesh(node, interface_tri)."""

    tri: TriangleMesh
    node: TensorLike  # (NN, GD)
    face: TensorLike  # (NF, 3) triangle connectivity (global or local node ids)
    iface_node_ids: TensorLike  # (NI,) unique node ids on interface
    normals: TensorLike  # (NF, GD) unit normals
    areas: TensorLike  # (NF,) triangle areas


def _numpy_nearest_indices(src_np: TensorLike, dst_np: TensorLike) -> TensorLike:
    """src (NS, GD), dst (ND, GD) -> (NS,) index of nearest dst row per src row."""
    diff = src_np[:, bm.newaxis, :] - dst_np[bm.newaxis, :, :]
    d2 = bm.sum(diff * diff, axis=2)
    return bm.argmin(d2, axis=1).astype(bm.int64)


class CouplingInterface:
    """FSI coupling operator driven by interface triangles (e.g. elbow pipe interface_tri).

    - Pressure on fluid interface -> nodal force on solid (traction -p n, lumped to vertices).
    - Solid displacement (NN_solid, GD) -> fluid interface displacement / velocity.

    Pass TriangleMesh for each side; no fluid_model / solid_model required.
    """

    def __init__(
        self,
        fluid_interface: TriangleMesh,
        solid_interface: TriangleMesh,
        *,
        mapping: str = "nearest",
        coord_tol: float = 1e-10,
    ):
        if mapping not in ("nearest", "direct_by_coord"):
            raise ValueError('mapping must be "nearest" or "direct_by_coord"')

        self.fluid = self._build_interface_mesh(fluid_interface)
        self.solid = self._build_interface_mesh(solid_interface)
        self._s2f = self._build_node_mapping(
            src_coords=self.solid.node[self.solid.iface_node_ids],
            dst_coords=self.fluid.node[self.fluid.iface_node_ids],
            mode=mapping,
            tol=coord_tol,
        )
        self._solid_disp_prev: Optional[TensorLike] = None

    def _build_interface_mesh(self, tri: TriangleMesh) -> InterfaceMesh:
        node = tri.entity("node")
        face = tri.entity("cell")
        normals, areas = self._triangle_normals_areas(node, face)
        iface_node_ids = bm.unique(face.reshape(-1))
        return InterfaceMesh(
            tri=tri,
            node=node,
            face=face,
            iface_node_ids=iface_node_ids,
            normals=normals,
            areas=areas,
        )

    def _triangle_normals_areas(self, node: TensorLike, face: TensorLike):
        x0 = node[face[:, 0]]
        x1 = node[face[:, 1]]
        x2 = node[face[:, 2]]
        e1 = x1 - x0
        e2 = x2 - x0
        cr = bm.cross(e1, e2)
        norm_cr = bm.sqrt(bm.sum(cr * cr, axis=1))
        areas = 0.5 * norm_cr
        normals = cr / (norm_cr[:, None] + 1e-30)
        return normals, areas

    def _build_node_mapping(
        self,
        src_coords: TensorLike,
        dst_coords: TensorLike,
        mode: str,
        tol: float,
    ) -> TensorLike:
        src_np = bm.to_numpy(src_coords)
        dst_np = bm.to_numpy(dst_coords)

        if mode == "direct_by_coord":
            out = bm.empty(src_np.shape[0], dtype=bm.int64)
            for i, p in enumerate(src_np):
                d = bm.linalg.norm(dst_np - p[None, :], axis=1)
                j = int(bm.argmin(d))
                if d[j] > tol:
                    raise ValueError(
                        f"direct_by_coord failed at {i}: min_dist={d[j]}"
                    )
                out[i] = j
            return out

        if mode == "nearest":
            return _numpy_nearest_indices(src_np, dst_np)

        raise ValueError(f"unknown mapping mode: {mode}")

    def _global_to_local_lookup(self, global_ids: TensorLike) -> TensorLike:
        gids = bm.to_numpy(global_ids).astype(bm.int64)
        m = int(gids.max())
        table = -bm.ones(m + 1, dtype=bm.int64)
        table[gids] = bm.arange(gids.shape[0], dtype=bm.int64)
        return table

    def pressure_to_solid_nodal_force(
        self,
        pressure: TensorLike,
        *,
        pressure_type: str = "face",
        face_mapping: str = "assume_same_topology",
    ) -> TensorLike:
        if pressure_type not in ("face", "node"):
            raise ValueError('pressure_type must be "face" or "node"')
        if face_mapping != "assume_same_topology":
            raise ValueError('only face_mapping="assume_same_topology" is implemented')

        if pressure_type == "face":
            p_face = pressure
        else:
            p_node = pressure
            g2l = self._global_to_local_lookup(self.fluid.iface_node_ids)
            f = self.fluid.face
            l0 = g2l[bm.to_numpy(f[:, 0]).astype(bm.int64)]
            l1 = g2l[bm.to_numpy(f[:, 1]).astype(bm.int64)]
            l2 = g2l[bm.to_numpy(f[:, 2]).astype(bm.int64)]
            p_face = (p_node[l0] + p_node[l1] + p_node[l2]) / 3.0

        n = self.fluid.normals
        A = self.fluid.areas
        F_face = (-p_face)[:, None] * n * A[:, None]

        if int(bm.to_numpy(self.solid.face).shape[0]) != int(
            bm.to_numpy(self.fluid.face).shape[0]
        ):
            raise ValueError(
                "fluid/solid interface triangle count mismatch; use the same interface "
                "connectivity (e.g. remapped interface_tri) on both sides."
            )
        face_s = self.solid.face

        F_solid = bm.zeros_like(self.solid.node)
        for lv in range(3):
            vid = face_s[:, lv]
            F_solid[vid] += F_face / 3.0
        return F_solid

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
