
from collections.abc import Iterable, Callable
from dataclasses import dataclass
from typing import Literal

from ...backend import bm
from ...backend import Tensor, Index
from ..topology.boundary import BoundaryInfo
from .entity_view import EntityView
from .mesh import Mesh

__all__ = ["FEALPyMesh"]


@dataclass
class FEALPyMesh(Mesh):
    """Provides a view of the mesh compatible with FEALPy's API."""
    def __post_init__(self):
        self.ftype = self.block.positions.dtype

        node_block = self.block.sectors.get("node")
        if node_block is not None:
            self.itype = node_block.indices.dtype
        else:
            self.itype = bm.int32

        self.device = getattr(self.block.positions, "device", None)

    def entity_view(self, etype: str | int, /) -> EntityView:
        """Provides a view for the entity with the given dimension.

        Raises ValueError if multiple types of entities with the given dimension are found.
        Use ``entity_views`` to get views of all entities with the given dimension."""
        top_dim = self._ensure_top_dim(etype)
        sectors = self.entity_view_by_topdim(top_dim)

        if len(sectors) > 1:
            raise ValueError(f"Multiple types of entities with top dimension {top_dim} found: {sectors}")

        return sectors[0]

    @property
    def localEdge(self) -> Tensor:
        cell_sec = self.entity_view(-1)
        edge_sec = self.entity_view(1)
        data = cell_sec.schema.local_entity(edge_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

    @property
    def localFace(self) -> Tensor:
        cell_sec = self.entity_view(-1)
        face_sec = self.entity_view(-2)
        data = cell_sec.schema.local_entity(face_sec.schema.name)
        return bm.asarray(data, dtype=self.itype, device=self.device)

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
        block = self.entity_view(top_dim)
        return block.boundary()

    # Meta

    def entity(self, etype: str | int, /) -> Tensor:
        """Provides the indices/positions of the entity with the given dimension.

        Raises ValueError if multiple types of entities with the given dimension are found.
        Use ``entities`` or ``entity_view`` to get all entities with the given dimension."""
        top_dim = self._ensure_top_dim(etype)

        if top_dim > 0:
            return self.entity_view(etype).indices
        return self.block.positions

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
        block = self.entity_view(top_dim)
        return block.barycenter(index=index)

    # [Shape functions]

    def shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None
    ) -> Tensor:
        return self.entity_view(-1).shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    cell_shape_function = shape_function

    def face_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None
    ) -> Tensor:
        return self.entity_view(-2).shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def edge_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None
    ) -> Tensor:
        return self.entity_view(1).shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def grad_shape_function(
        self,
        bcs: Tensor | tuple[Tensor, ...],
        p: int | tuple[int, ...] = 1,
        *,
        index: Index | None = None,
        variables: Literal['b', 'u', 'x'] = "u",
        mi=None
    ) -> Tensor:
        return self.entity_view(-1).grad_shape_function(
            bcs, p=p, index=index, variables=variables, mi=mi
        )

    def number_of_cells(self) -> int:
        return self.entity_view(self.top_dimension()).size()

    def number_of_faces(self) -> int:
        return self.entity_view(self.top_dimension() - 1).size()

    def number_of_edges(self) -> int:
        return self.entity_view(1).size()

    def number_of_nodes(self) -> int:
        return self.block.positions.shape[0]

    def number_of_global_ipoints(self, p: int | tuple[int, ...]) -> int:
        total = 0
        for name in self.block.sectors:
            sector_view = self.entity_view_by_name(name)
            total += sector_view.num_multi_index(p, internal=True) * sector_view.size()
        return total

    def number_of_local_ipoints(self, p: int | tuple[int, ...], iptype: str | int = "cell") -> int:
        return self.entity_view(iptype).num_multi_index(p)

    def multi_index_matrix(self, p: int | tuple[int, ...], etype: int | str = "cell") -> Tensor:
        """Returns the multi-index matrix for polynomial degree p and number of nodes n."""
        sec = self.entity_view(etype)
        return sec.multi_index_matrix(p)

    def quadrature_formula(self, q: int, etype: str | int = "cell", qtype: str = "legendre"):
        """Returns FEALPy quadrature object for simplex entities."""
        return self.entity_view(etype).quadrature_formula(q, qtype)

    # Topology

    def cell_to_face(self) -> Tensor:
        """Returns the mapping from cells to faces."""
        cell_sec = self.entity_view(-1)
        face_sec = self.entity_view(-2)
        return cell_sec.to(face_sec).tgt_indices

    def cell_to_edge(self) -> Tensor:
        """Returns the mapping from cells to edges."""
        cell_sec = self.entity_view(-1)
        edge_sec = self.entity_view(1)
        return cell_sec.to(edge_sec).tgt_indices

    def face_to_edge(self) -> Tensor:
        """Returns the mapping from faces to edges."""
        face_sec = self.entity_view(-2)
        edge_sec = self.entity_view(1)
        return face_sec.to(edge_sec).tgt_indices

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

    def cell_to_edge_sign(self) -> Tensor: # TODO: remove implementation here
        """Returns the sign of edges for each cell."""
        cell_sec = self.entity_view(-1)
        edge_sec = self.entity_view(1)
        c2e = self.cell_to_edge()

        local_pair = cell_sec.indices[:, self.localEdge]
        global_pair = edge_sec.indices[c2e]

        return bm.all(local_pair == global_pair, axis=-1)

    def face_to_edge_sign(self) -> Tensor: # TODO: remove implementation here
        """Returns the sign of edges for each face."""
        face_sec = self.entity_view(-2)
        edge_sec = self.entity_view(1)
        f2e = self.face_to_edge()
        sign = bm.zeros((face_sec.indices.shape[0], 3), dtype=bm.bool)
        local_f2e = face_sec.schema.local_entity("edge")
        n = [item[0] for item in local_f2e]

        for i in range(len(n)):
            sign[:, i] = face_sec.indices[:, n[i]] == edge_sec.indices[f2e[:, i], 0]

        return sign

    def cell_to_ipoint(self, p: int, index: Index | None = None) -> Tensor:
        from ..ipoints import to_ipoint
        view = self.entity_view(-1)
        result = to_ipoint(self, view.schema.name, p)
        return result if index is None else result[index]

    def face_to_ipoint(self, p: int, index: Index | None = None) -> Tensor:
        from ..ipoints import to_ipoint
        view = self.entity_view(-2)
        result = to_ipoint(self, view.schema.name, p)
        return result if index is None else result[index]

    def interpolation_points(
        self,
        p: int,
        entity: str | int | Iterable[str] | Iterable[int] | None = None,
        index: Index | None = None
    ) -> Tensor:
        """Get the interpolation points for the given entity and order.

        Parameters:
            p (int): The degree of interpolation.
            entity (str | int | Iterable[str] | Iterable[int] | None, optional):
                The entity or entities for which to compute interpolation points.

        Returns:
            Tensor: A tensor of shape (num_ip, GD) containing the interpolation points.
        """
        from ..ipoints import ipoints
        if entity is None:
            entity_iter = range(self.top_dimension()+1)
        elif not isinstance(entity, Iterable) or isinstance(entity, str):
            entity_iter = (entity,)
        else:
            entity_iter = entity
        views: list[EntityView] = []
        for et in entity_iter:
            views.extend(self.entity_view_by_topdim(et))
        names = [sec.schema.name for sec in views]
        ips = ipoints(self, p, names)

        if index is None:
            return ips
        return ips[index, :]

    # def cell_to_face_permutation(self, *, locFace: Tensor | None = None) -> Tensor:
    #     """Returns the permutation of faces for each cell."""
    #     cell = self.entity("cell")
    #     face = self.entity("face")
    #     c2f = self.cell_to_face()

    #     if locFace is None:
    #         locFace = self.localFace
    #     else:
    #         locFace = bm.asarray(locFace, dtype=self.itype)

    #     nc = cell.shape[0]
    #     c2f_glo = face[bm.reshape(c2f, (-1,))]
    #     c2f_loc = bm.reshape(cell[:, locFace], (-1, 3))

    #     c2f_glo = bm.argsort(c2f_glo, axis=1)
    #     c2f_loc = bm.argsort(c2f_loc, axis=1)

    #     order = c2f_loc[bm.arange(nc * locFace.shape[0])[:, None], c2f_glo]
    #     return bm.reshape(order, (nc, locFace.shape[0], 3))

    # Geometry

    def bc_to_point(self, bc: Tensor | tuple[Tensor, ...], *, index: Index | None = None) -> Tensor:
        if not isinstance(bc, tuple):
            bc = (bc,)

        top = sum(b.shape[1] - 1 for b in bc)
        sec = self.entity_view(top)
        return sec.bc_to_point(bc, index=index)

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

        block = self.entity_view(top_dim)
        return block.measure(index=index)

    def edge_tangent(self, *, index: Index | None = None) -> Tensor:
        """Returns the tangent vector of edges."""
        block = self.entity_view(1)
        return block.tangent(index=index)

    def edge_unit_tangent(self, *, index: Index | None = None) -> Tensor:
        """Returns the unit tangent vector of edges."""
        block = self.entity_view(1)
        tangent = block.tangent(index=index)
        norm = bm.linalg.vector_norm(tangent, axis=1, keepdims=True)
        return tangent / norm

    def error(
        self,
        f1: Callable[..., Tensor],
        f2: Callable[..., Tensor],
        /,
        power: float = 2.0,
        q: int = 3,
        *,
        cell_axis: bool = False,
        index: Index | None = None
    ) -> Tensor:
        """Returns the error between two functions defined on the mesh."""
        cell_sec = self.entity_view(-1)
        return cell_sec.error(f1, f2, power=power, q=q, cell_axis=cell_axis, index=index)

    def face_normal(self, *, index: Index | None = None) -> Tensor:
        """Returns the normal vector of faces."""
        block = self.entity_view(self.top_dimension() - 1)
        return block.normal(index=index)

    def face_unit_normal(self, *, index: Index | None = None) -> Tensor:
        """Returns the unit normal vector of faces."""
        block = self.entity_view(self.top_dimension() - 1)
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
            block = self.entity_view(self.top_dimension())
        elif TD == self.top_dimension() - 1:
            block = self.entity_view(self.top_dimension() - 1)
        elif TD == 1:
            block = self.entity_view(1)
        else:
            raise ValueError(f"Unsupported top dimension: {TD}")

        return block.grad_lambda(index=index)

    def grad_face_lambda(self, index: Index | None = None):
        return self.grad_lambda(index=index, TD=self.top_dimension() - 1)
