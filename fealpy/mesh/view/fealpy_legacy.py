
from typing import Literal

from ...backend import bm
from ...backend import Tensor, Index
from ..storage import MeshBlock
from ..topology.boundary import BoundaryInfo
from .entity_view import EntityView
from .mesh import Mesh

__all__ = ["FEALPyMesh"]


class FEALPyMesh(Mesh):
    """Provides a view of the mesh compatible with FEALPy's API."""
    def __init__(self, block: MeshBlock, /):
        super().__init__(block)
        self.ftype = block.positions.dtype

        node_block = block.sectors.get("node")
        if node_block is not None:
            self.itype = node_block.indices.dtype
        else:
            self.itype = bm.int32

        self.device = getattr(block.positions, "device", None)

    @property
    def localEdge(self) -> Tensor:
        top_name = self._get_block_by_top_dim(self.top_dimension()).block.schema_name
        if top_name != "tet":
            raise NotImplementedError(f"localEdge is only implemented for tetrahedra, got {top_name!r}")
        return bm.asarray(
            [[2, 3], [3, 1], [1, 2], [2, 0], [0, 3], [0, 1]],
            dtype=self.itype,
        )

    @property
    def localFace(self) -> Tensor:
        top_name = self._get_block_by_top_dim(self.top_dimension()).block.schema_name
        if top_name != "tet":
            raise NotImplementedError(f"localFace is only implemented for tetrahedra, got {top_name!r}")
        return bm.asarray(
            [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]],
            dtype=self.itype,
        )

    def _etype_to_dim(self, etype: str | int) -> int:
        if isinstance(etype, int):
            return etype
        if etype == "cell":
            return self.top_dimension()
        if etype == "face":
            return self.top_dimension() - 1
        if etype == "edge":
            return 1
        if etype == "node":
            return 0
        raise KeyError(f"{etype!r} is not a valid entity name")

    def _boundary_info_by_top_dim(self, top_dim: int) -> BoundaryInfo:
        block = self._get_block_by_top_dim(top_dim)
        return block.boundary()

    def _pair_to_edge_index(self, pairs: Tensor, edge: Tensor) -> Tensor:
        sorted_edge = bm.sort(edge, axis=1)
        sorted_pairs = bm.sort(pairs, axis=-1)

        nn = self.number_of_nodes()
        edge_key = sorted_edge[:, 0] * nn + sorted_edge[:, 1]
        pair_key = sorted_pairs[..., 0] * nn + sorted_pairs[..., 1]

        order = bm.argsort(edge_key)
        sorted_key = edge_key[order]
        hit = bm.searchsorted(sorted_key, bm.reshape(pair_key, (-1,)))
        return bm.reshape(order[hit], pair_key.shape)

    def _get_block_by_top_dim(self, top_dim: int) -> EntityView:
        if top_dim < 0:
            top_dim += self.top_dimension() + 1

        for block in self.block.sectors.values():
            if block.schema.top_dim == top_dim:
                return EntityView(self.block, block)
        raise ValueError(f"No block found with top dimension: {top_dim}")

    # Meta

    def entity(self, name: str | int | Literal["cell", "face", "edge", "node"], /) -> Tensor:
        """Provides a view of the entity with the given name."""
        if isinstance(name, int):
            top_dim = name
        elif name == "cell":
            top_dim = self.top_dimension()
        elif name == "face":
            top_dim = self.top_dimension() - 1
        elif name == "edge":
            top_dim = 1
        elif name == "node":
            return self.block.positions
        else:
            raise ValueError(f"Unknown entity name: {name}")

        return self._get_block_by_top_dim(top_dim).indices

    @property
    def cell(self) -> Tensor:
        return self.entity("cell")

    @property
    def face(self) -> Tensor:
        return self.entity("face")

    @property
    def edge(self) -> Tensor:
        return self.entity("edge")

    @property
    def node(self) -> Tensor:
        return self.entity("node")

    def entity_barycenter(self, etype: str | int, index: Index | None = None) -> Tensor:
        top_dim = self._etype_to_dim(etype)
        block = self._get_block_by_top_dim(top_dim)
        return block.barycenter(index=index)

    def number_of_cells(self) -> int:
        return self._get_block_by_top_dim(self.top_dimension()).size()

    def number_of_faces(self) -> int:
        return self._get_block_by_top_dim(self.top_dimension() - 1).size()

    def number_of_edges(self) -> int:
        return self._get_block_by_top_dim(1).size()

    def number_of_nodes(self) -> int:
        return self.block.positions.shape[0]

    def multi_index_matrix(self, p: int, etype: int):
        """Returns the multi-index matrix for polynomial degree p and number of nodes n."""
        from ..topology.ipoints import InterpolationPoints
        return InterpolationPoints.multi_index_matrix(p, etype + 1)

    def quadrature_formula(self, q: int, etype: str | int = "cell", qtype: str = "legendre"):
        """Returns FEALPy quadrature object for simplex entities."""
        etype = self._etype_to_dim(etype)
        td = self.top_dimension()

        if etype == 1:
            from fealpy.quadrature import GaussLegendreQuadrature
            return GaussLegendreQuadrature(q)

        if etype == 2:
            from fealpy.quadrature import TriangleQuadrature
            if q > 9:
                from fealpy.quadrature.stroud_quadrature import StroudQuadrature
                return StroudQuadrature(2, q)
            return TriangleQuadrature(q)

        if etype == 3:
            if td != 3:
                raise ValueError(f"Entity dim 3 requires a 3D mesh, got top dimension {td}")
            if q > 7:
                from fealpy.quadrature.stroud_quadrature import StroudQuadrature
                return StroudQuadrature(3, q)
            from fealpy.quadrature import TetrahedronQuadrature
            return TetrahedronQuadrature(q)

        raise ValueError(f"Unsupported entity dimension for quadrature: {etype}")

    # Topology

    def cell_to_face(self) -> Tensor:
        """Returns the mapping from cells to faces."""
        cell_block = self._get_block_by_top_dim(self.top_dimension())
        face_block = self._get_block_by_top_dim(self.top_dimension() - 1)
        return cell_block.to(face_block).tgt_indices

    def cell_to_edge(self) -> Tensor:
        """Returns the mapping from cells to edges."""
        cell = self.entity("cell")
        edge = self.entity("edge")
        top_name = self._get_block_by_top_dim(self.top_dimension()).block.schema_name

        if top_name != "tet":
            cell_block = self._get_block_by_top_dim(self.top_dimension())
            edge_block = self._get_block_by_top_dim(1)
            return cell_block.to(edge_block).tgt_indices

        local_edge = self.localEdge
        local_pairs = cell[:, local_edge]
        return self._pair_to_edge_index(local_pairs, edge)

    def face_to_edge(self) -> Tensor:
        """Returns the mapping from faces to edges."""
        face = self.entity("face")
        edge = self.entity("edge")
        local_edge = bm.asarray([[1, 2], [2, 0], [0, 1]], dtype=self.itype)
        local_pairs = face[:, local_edge]
        return self._pair_to_edge_index(local_pairs, edge)

    def boundary_cell_flag(self) -> Tensor:
        return self._boundary_info_by_top_dim(self.top_dimension()).mask

    def boundary_face_flag(self) -> Tensor:
        return self._boundary_info_by_top_dim(self.top_dimension() - 1).mask

    def boundary_edge_flag(self) -> Tensor:
        return self._boundary_info_by_top_dim(1).mask

    def boundary_node_flag(self) -> Tensor:
        return self._boundary_info_by_top_dim(0).mask

    def boundary_cell_index(self) -> Tensor:
        return self._boundary_info_by_top_dim(self.top_dimension()).index

    def boundary_face_index(self) -> Tensor:
        return self._boundary_info_by_top_dim(self.top_dimension() - 1).index

    def boundary_edge_index(self) -> Tensor:
        return self._boundary_info_by_top_dim(1).index

    def boundary_node_index(self) -> Tensor:
        return self._boundary_info_by_top_dim(0).index

    def cell_to_edge_sign(self) -> Tensor:
        """Returns the sign of edges for each cell."""
        cell = self.entity("cell")
        edge = self.entity("edge")
        c2e = self.cell_to_edge()
        local_edge = self.localEdge

        local_pair = cell[:, local_edge]
        global_pair = edge[c2e]
        return bm.all(local_pair == global_pair, axis=-1)

    def face_to_edge_sign(self) -> Tensor:
        """Returns the sign of edges for each face."""
        face = self.entity("face")
        edge = self.entity("edge")
        f2e = self.face_to_edge()
        sign = bm.zeros((face.shape[0], 3), dtype=bm.bool)
        n = [1, 2, 0]
        for i in range(3):
            sign[:, i] = face[:, n[i]] == edge[f2e[:, i], 0]
        return sign

    def cell_to_face_permutation(self, *, locFace: Tensor | None = None) -> Tensor:
        """Returns the permutation of faces for each cell."""
        cell = self.entity("cell")
        face = self.entity("face")
        c2f = self.cell_to_face()

        if locFace is None:
            locFace = self.localFace
        else:
            locFace = bm.asarray(locFace, dtype=self.itype)

        nc = cell.shape[0]
        c2f_glo = face[bm.reshape(c2f, (-1,))]
        c2f_loc = bm.reshape(cell[:, locFace], (-1, 3))

        c2f_glo = bm.argsort(c2f_glo, axis=1)
        c2f_loc = bm.argsort(c2f_loc, axis=1)

        order = c2f_loc[bm.arange(nc * locFace.shape[0])[:, None], c2f_glo]
        return bm.reshape(order, (nc, locFace.shape[0], 3))

    # Geometry

    def bc_to_point(self, bc: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        if isinstance(bc, tuple):
            raise NotImplementedError("Tensor-product barycentric coordinates are not implemented")

        nvert = bc.shape[-1]
        if nvert == self.top_dimension() + 1:
            entity = self.entity("cell")
        elif nvert == self.top_dimension():
            entity = self.entity("face")
        elif nvert == 2:
            entity = self.entity("edge")
        else:
            raise ValueError(f"Unsupported barycentric dimension: {nvert}")

        if index is not None:
            entity = entity[index]

        points = self.block.positions[entity]
        return bm.einsum("...j,cjd->c...d", bc, points)

    def entity_measure(
        self,
        name: str | Literal["cell", "face", "edge", "node"],
        /,
        *,
        index: Index | None = None
    ) -> Tensor:
        """Returns the measure (length, area, volume) of the specified entity."""
        top_dim = self._etype_to_dim(name)
        if top_dim == 0:
            return bm.zeros((1,), dtype=self.ftype)

        block = self._get_block_by_top_dim(top_dim)
        return block.measure(index=index)

    def edge_tangent(self, *, index: Index | None = None) -> Tensor:
        """Returns the tangent vector of edges."""
        block = self._get_block_by_top_dim(1)
        return block.tangent(index=index)

    def edge_unit_tangent(self, *, index: Index | None = None) -> Tensor:
        """Returns the unit tangent vector of edges."""
        block = self._get_block_by_top_dim(1)
        tangent = block.tangent(index=index)
        norm = bm.linalg.vector_norm(tangent, axis=1, keepdims=True)
        return tangent / norm

    def face_normal(self, *, index: Index | None = None) -> Tensor:
        """Returns the normal vector of faces."""
        block = self._get_block_by_top_dim(self.top_dimension() - 1)
        return block.normal(index=index)

    def face_unit_normal(self, *, index: Index | None = None) -> Tensor:
        """Returns the unit normal vector of faces."""
        block = self._get_block_by_top_dim(self.top_dimension() - 1)
        normal = block.normal(index=index)
        norm = bm.linalg.vector_norm(normal, axis=1, keepdims=True)
        return normal / norm

    def grad_lambda(
        self,
        index: Index | None = None,
        TD: int | None = None,
    ) -> Tensor:
        if TD is None:
            TD = self.top_dimension()

        if TD == self.top_dimension():
            block = self._get_block_by_top_dim(self.top_dimension())
        elif TD == self.top_dimension() - 1:
            block = self._get_block_by_top_dim(self.top_dimension() - 1)
        elif TD == 1:
            block = self._get_block_by_top_dim(1)
        else:
            raise ValueError(f"Unsupported top dimension: {TD}")

        return block.grad_lambda(index=index)

    def grad_face_lambda(self, index: Index | None = None):
        return self.grad_lambda(index=index, TD=self.top_dimension() - 1)
