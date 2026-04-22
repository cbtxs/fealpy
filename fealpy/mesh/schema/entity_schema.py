
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, ParamSpec

from ...backend import bm
from ...backend import Tensor, Index
from ..storage import EntitySector, MeshBlock, Relation
from ..topology.boundary import BoundaryInfo

__all__ = [
    "EntitySchema",
    "ShapedEntitySchema"
]

P = ParamSpec("P")
EntityShape = Literal["node", "edge", "tri", "quad", "prism", "pyramid", "tet", "hex"]


@dataclass(slots=True, frozen=True)
class EntityContext:
    block: MeshBlock
    sector: EntitySector


class EntitySchema:
    name: ClassVar[str]
    top_dim: ClassVar[int]
    local_faces: ClassVar[dict[str, list[list[int]]]] = {}
    orientation: ClassVar[list[tuple[int, ...]]] = []

    ### [Entity Topology] ###

    @classmethod
    def boundary(cls, ctx: EntityContext) -> BoundaryInfo:
        """Boundary information of the entity."""
        raise NotImplementedError()

    @classmethod
    def local_entity(cls, tgt_name: str, /) -> list[list[int]]:
        """Local entity indices of the target entity."""
        raise NotImplementedError()

    @classmethod
    def relation(cls, ctx: EntityContext, tgt_name: EntityShape) -> Relation:
        """Compute the relation between two entities."""
        raise NotImplementedError()

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Number of entities in the sector."""
        raise NotImplementedError()

    ### [Multi-Indices] ###

    @classmethod
    def multi_index(cls, order: int | tuple[int, ...], *, internal: bool = False) -> Tensor:
        """Multi-index of the entity."""
        raise NotImplementedError()

    @classmethod
    def multi_index_sort(cls, multi_index: Tensor, /) -> Tensor:
        """Return a stable sort indices of the multi-index."""
        raise NotImplementedError()

    @classmethod
    def num_multi_index(cls, order: int | tuple[int, ...], *, internal: bool = False) -> int:
        """Number of multi-indices."""
        raise NotImplementedError()

    ### [Geometric Computations] ###

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the barycenter of the entity."""
        raise NotImplementedError()

    @classmethod
    def barycentric(cls, ctx: EntityContext, index: Index | None, func: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
        """Transform functions from cartesian to barycentric coordinates."""
        raise NotImplementedError()

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """Geometric dimension of the cell."""
        raise NotImplementedError()

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Gradient of barycentric coordinates."""
        raise NotImplementedError()

    @classmethod
    def integral(cls, ctx: EntityContext, index: Index | None, func: Callable[[Tensor], Tensor], q: int) -> Tensor:
        """Integral of a barycentric function."""
        raise NotImplementedError()

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the measure of the entity."""
        raise NotImplementedError()

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the normal vector of the entity."""
        raise NotImplementedError()

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the tangent vector of the entity."""
        raise NotImplementedError()

    @classmethod
    def transform(cls, ctx: EntityContext, func: Callable[P, Tensor], kind: str = "value") -> Callable[P, Tensor]:
        """Transform functions from reference to physical element."""
        raise NotImplementedError()


class ShapedEntitySchema(EntitySchema):
    """Entity schema with a fixed shape (e.g., a triangle or quadrilateral)."""

    @classmethod
    def boundary(cls, ctx: EntityContext) -> BoundaryInfo:
        """Boundary information of the entity."""
        from ..topology.boundary import BoundaryInferencer

        if ctx.block._cache_boundary_info is None:
            ctx.block._cache_boundary_info = BoundaryInferencer.infer_all(ctx.block)

        return ctx.block._cache_boundary_info[ctx.sector.schema_name]

    @classmethod
    def local_entity(cls, tgt_name: str, /) -> list[list[int]]:
        """Local entity indices of the target entity."""
        from ..topology.builder import LocalIndicesInferer
        local_indices = LocalIndicesInferer.infer(cls, tgt_name)
        return local_indices

    @classmethod
    def relation(cls, ctx: EntityContext, tgt_name: EntityShape) -> Relation:
        """Compute the relation between two entities."""
        src_name = ctx.sector.schema_name
        relation = ctx.block.relations.get((src_name, tgt_name))

        if relation is None:
            from ..topology.builder import TopologyInferer
            TopologyInferer.infer(ctx.block, src_name, tgt_name)
            relation = ctx.block.relations.get((src_name, tgt_name))

        if relation is None:
            raise ValueError(f"relation from {src_name!r} to {tgt_name!r} not found")

        return relation

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Number of entities in the sector."""
        return ctx.sector.indices.shape[0]
