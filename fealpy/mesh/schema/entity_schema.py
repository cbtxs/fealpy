
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, overload, ParamSpec, TYPE_CHECKING

from ...backend import bm, Tensor, Index
from ..storage import EntitySector, MeshBlock, Relation
from ..topology.boundary import BoundaryInfo

if TYPE_CHECKING:
    from ...quadrature import Quadrature

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
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> Tensor:
        """Multi-index of the entity, with one column per vertex."""
        raise NotImplementedError()

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        """Number of multi-indices."""
        return int(cls.multi_index(order, internal=internal).shape[0])

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
    def bc_to_point(cls, ctx: EntityContext, index: Index | None, bcs: tuple[Tensor, ...]) -> Tensor:
        """Convert barycentric coordinates to physical points."""
        raise NotImplementedError()

    @classmethod
    def first_fundamental_form(cls, J: Tensor) -> Tensor:
        """First fundamental form of the entity."""
        return bm.einsum("...km, ...kn -> ...mn", J, J)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """Geometric dimension of the cell."""
        raise NotImplementedError()

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        """Gradient of lowest-order shape functions.

        If ref is False, return gradients with respect to physical Cartesian
        coordinates. With bcs=None the shape is (NC, ldof, GD); with bcs set
        the shape is (NC, NQ, ldof, GD).

        If ref is True, return gradients with respect to the reference
        barycentric coordinates. With bcs=None the shape is (NC, ldof, num_bc);
        with bcs set the shape is (NC, NQ, ldof, num_bc).
        """
        raise NotImplementedError()

    @classmethod
    def grad_shape_function(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None,
        variables: str = "u",
        mi: Tensor | None = None,
    ) -> Tensor:
        """Gradient of shape functions."""
        raise NotImplementedError()

    @classmethod
    def integral(cls, ctx: EntityContext, index: Index | None, func: Callable[[Tensor], Tensor], q: int) -> Tensor:
        """Integral of a barycentric function."""
        raise NotImplementedError()

    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Jacobi matrix of the transformation from reference to physical element."""
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
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device = None) -> "Quadrature":
        """Quadrature formula for the entity."""
        raise NotImplementedError()

    @classmethod
    def shape_function(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None,
        variables: str = "u",
        mi: Tensor | None = None,
    ) -> Tensor:
        """Shape functions."""
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

    ### [Geometric Computations] ###

    @classmethod
    def barycentric(cls, ctx: EntityContext, index: Index | None, func: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]:
        """Compute the barycentric coordinates of the entity."""
        from functools import wraps
        @wraps(func)
        def wrapper(bcs: tuple[Tensor, ...]) -> Tensor:
            points = cls.bc_to_point(ctx, index, bcs) # [NC, NQ, GD]
            return func(points)
        return wrapper

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def integral(cls, ctx: EntityContext, index: Index | None, func: Callable[[Tensor], Tensor], q: int) -> Tensor:
        """Integral of a barycentric function."""
        quadrature = cls.quadrature_formula(q)
        bcs, ws = quadrature.get_quadrature_points_and_weights()
        points = cls.bc_to_point(ctx, index, bcs) # [NC, NQ, GD]
        values = func(points)
        return bm.einsum("cq..., q -> c...", values, ws)


def _make_sure_interval_bcs(bcs: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
    return tuple(
        bc if bc.ndim >= 2 else bm.stack((bc, 1. - bc), dim=-1)
        for bc in bcs
    )


@overload
def _require_bcs_tuple(bcs: Any, name: str) -> tuple[Tensor, ...]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[1]) -> tuple[Tensor]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[2]) -> tuple[Tensor, Tensor]: ...
@overload
def _require_bcs_tuple(bcs: Any, name: str, n: Literal[3]) -> tuple[Tensor, Tensor, Tensor]: ...
def _require_bcs_tuple(bcs: Any, name: str, n: int | None = None) -> tuple[Tensor, ...]:
    if not isinstance(bcs, tuple):
        raise TypeError(f"{name} expects barycentric coordinates as a tuple of tensors, got {type(bcs).__name__}")

    if n is None:
        return bcs

    if len(bcs) == n:
        pass
    elif len(bcs) == 1:
        bcs = (bcs[0],) * n
    else:
        raise ValueError(f"{name} expects {n} barycentric tensors, got {len(bcs)}")

    return bcs


@overload
def _require_order_tuple(p: Any, name: str) -> tuple[int, ...]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[1]) -> tuple[int]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[2]) -> tuple[int, int]: ...
@overload
def _require_order_tuple(p: Any, name: str, n: Literal[3]) -> tuple[int, int, int]: ...
def _require_order_tuple(p: Any, name: str, n: int | None = None) -> tuple[int, ...]:
    """Ensure that the polynomial degree is a tuple of integers.

    Parameters:
        p (tuple[int, ...]): The input polynomial degree(s).
        name (str): The name of the function for error messages.
        n (int | None): The expected number of polynomial degrees.
            If None, no check is performed. If an integer, the length of the
            tuple must be either 1 or n.
            1-length means that the same degree is used for all dimensions,
            returning a tuple of length n with the repeated degree.

    Returns:
        tuple[int, ...]: A tuple of polynomial degrees with length n (if n is not None).
    """
    if not isinstance(p, tuple):
        raise TypeError(f"{name} expects polynomial degrees as a tuple of integers, got {type(p).__name__}")

    if not all(isinstance(pi, int) for pi in p):
        raise TypeError(f"{name} expects polynomial degrees as integers, got {p}")

    if not all(pi >= 0 for pi in p):
        raise ValueError(f"{name} expects non-negative polynomial degrees, got {p}")

    if n is None:
        return p

    if len(p) == n:
        pass
    elif len(p) == 1:
        p = (p[0],) * n
    else:
        raise ValueError(f"{name} expects {n} polynomial degrees, got {len(p)}")

    return p
