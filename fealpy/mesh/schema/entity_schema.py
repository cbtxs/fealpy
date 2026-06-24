
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


@dataclass(slots=True, frozen=True)
class EntityContext:
    block: MeshBlock
    sector: EntitySector


class EntitySchema:
    name: ClassVar[str]
    top_dim: ClassVar[int]
    OFace: ClassVar[dict[str, list[list[int]]]] = {}
    SFace: ClassVar[dict[str, list[list[int]]]] = {}
    orientation: ClassVar[list[tuple[int, ...]]] = []

    ### [Entity Topology] ###

    @classmethod
    def boundary(cls, ctx: EntityContext) -> BoundaryInfo:
        """Boundary information of the entity.

        Returns:
            NamedTuple:
            - index: Tensor of shape (num_boundary,) containing the indices
                of boundary entities.
            - mask: Tensor of shape (num_entity,) containing a boolean mask
                indicating whether each entity is a boundary entity.
            - count: Tensor of shape (num_entity,) containing the count of
                adjacent top-dimensional entities for each entity, or None if
                not applicable.
        """
        raise NotImplementedError()

    @classmethod
    def local_entity(cls, tgt_name: str, /, indexing: Literal["o", "s"] = "o") -> list[list[int]]:
        """Local entity indices of the target entity.

        Parameters:
            tgt_name (str): The name of the target entity.
            indexing (Literal["o", "s"], optional): The indexing method. Defaults to "o".

        Returns:
            list[list[int]]: The local entity indices.
        """
        raise NotImplementedError()

    @classmethod
    def relation(cls, ctx: EntityContext, tgt_name: str) -> Relation:
        """Compute the relation between two entities."""
        raise NotImplementedError()

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Number of entities in the sector."""
        raise NotImplementedError()

    ### [Multi-Indices] ###

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        """Multi-index of the entity, with one column per vertex."""
        raise NotImplementedError()

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        """Number of multi-indices."""
        return int(cls.multi_index(order, internal=internal).shape[0])

    @classmethod
    def global_permutations(cls, ctx: EntityContext, tgt_name: str) -> Tensor:
        """Permutation indices from local to global."""
        raise NotImplementedError()

    @classmethod
    def vo_to_do(cls, order: tuple[int, ...]) -> dict[tuple[int, ...], Tensor]:
        """Return the mapping from vertex orientation to DoF ordering."""
        from ..topology.ipoints import multi_index_sort
        result: dict[tuple[int, ...], Tensor] = {}

        for v_o in cls.orientation:
            mi = cls.multi_index(order, internal=True, tensorprod=False)
            d_o = multi_index_sort(mi[:, v_o])
            result[tuple(v_o)] = d_o

        return result

    ### [Geometric Computations] ###

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the barycenter of the entity."""
        raise NotImplementedError()

    @classmethod
    def barycentric(cls, ctx: EntityContext, func: Callable[[Tensor], Tensor], index: Index | None) -> Callable[[Tensor | tuple[Tensor, ...]], Tensor]:
        """Transform functions from cartesian to barycentric coordinates."""
        raise NotImplementedError()

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Convert barycentric coordinates to physical points."""
        raise NotImplementedError()

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        """Geometric dimension of the cell."""
        raise NotImplementedError()

    @classmethod
    def grad_shape_function_barycentric(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        """Gradient of shape functions to barycentric coordinates.

        Parameters:
            bcs (tuple[Tensor, ...]): Barycentric coordinates of evaluation points, with shape (NQ, num_bc).
            p (tuple[int, ...]): Polynomial degree(s) of the shape functions.

        Returns:
            Tensor: The gradient of shape functions with shape (NQ, num_shape, num_bc), where
                NQ is the number of points, num_shape is the number of shape functions,
                and num_bc is the number of barycentric coordinates.
        """
        raise NotImplementedError()

    @classmethod
    def grad_shape_function_cartesian(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None
    ) -> Tensor:
        """Gradient of shape functions to cartesian coordinates."""
        from ..transform import piola_transform_covariant
        grad_ref = cls.grad_shape_function_reference(bcs, p)[None, ...]
        J = cls.jacobi_matrix(ctx, bcs, index=index)[..., None, :, :]
        # [NC, NQ, num_shape, ref_dim], [NC, NQ, num_shape, GD, ref_dim]
        return piola_transform_covariant(grad_ref, J)

    @classmethod
    def grad_shape_function_reference(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        """Gradient of shape functions to reference coordinates.

        Parameters:
            bcs (tuple[Tensor, ...]): Barycentric coordinates of evaluation points, with shape (NQ, num_bc).
            p (tuple[int, ...]): Polynomial degree(s) of the shape functions.

        Returns:
            Tensor: The gradient of shape functions with shape (NQ, num_shape, ref_dim), where
                NQ is the number of points, num_shape is the number of shape functions,
                and ref_dim is the dimension of the reference element.
        """
        raise NotImplementedError()

    @classmethod
    def integral(
        cls,
        ctx: EntityContext,
        func: Callable[[Tensor | tuple[Tensor, ...]], Tensor],
        q: int, index: Index | None
    ) -> Tensor:
        """Integral of a barycentric function."""
        raise NotImplementedError()

    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        """Jacobi matrix of the transformation from reference to physical element.

        Parameters:
            ctx (EntityContext): The entity context containing the mesh block and sector information.
            bcs (tuple[Tensor, ...]): Barycentric coordinates of evaluation points, with shape (NQ, num_bc).
            index (Index | None): The index of the entity in the sector, or None for all entities.

        Returns:
            Tensor: The Jacobi matrix with shape (NC, NQ, GD, ref_dim), where
                NC is the number of cells, NQ is the number of points,
                GD is the geometric dimension, and ref_dim is the number of
                reference coordinates (typically equal to the topological
                dimension of the entity).
        """
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
    def shape_function(cls, bcs: tuple[Tensor, ...], p: tuple[int, ...]) -> Tensor:
        """Shape functions."""
        raise NotImplementedError()

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        """Compute the tangent vector of the entity."""
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
    def local_entity(cls, tgt_name: str, /, indexing: Literal["o", "s"] = "o") -> list[list[int]]:
        if indexing == "o":
            if tgt_name in cls.OFace:
                return cls.OFace[tgt_name]
        elif indexing == "s":
            if tgt_name in cls.SFace:
                return cls.SFace[tgt_name]
        else:
            raise ValueError(f"indexing must be 'o' or 's', got {indexing!r}")
        raise ValueError(f"local entity {tgt_name!r} is not defined for {cls.name!r}")

    @classmethod
    def relation(cls, ctx: EntityContext, tgt_name: str) -> Relation:
        src_name = ctx.sector.schema_name
        relation = ctx.block.relations.get((src_name, tgt_name))

        if relation is None:
            from ..topology.builder import TopologyInferer
            try:
                TopologyInferer.infer(ctx.block, src_name, tgt_name)
                relation = ctx.block.relations.get((src_name, tgt_name))
            except ValueError:
                pass

        if relation is None:
            from ..topology.builder import TopologyBuilder
            from .registry import SCHEMA_REGISTRY
            tgt_schema = SCHEMA_REGISTRY[tgt_name]

            if cls.top_dim > tgt_schema.top_dim:
                TopologyBuilder.construct(ctx.block, src_name)
            elif cls.top_dim < tgt_schema.top_dim:
                TopologyBuilder.construct(ctx.block, tgt_name)
            else:
                raise ValueError(f"Cannot construct relation from {src_name!r} "
                                 f"to {tgt_name!r} with the same topological dimension")

            relation = ctx.block.relations.get((src_name, tgt_name))

        if relation is None:
            raise ValueError(f"relation from {src_name!r} to {tgt_name!r} not found")

        return relation

    @classmethod
    def size(cls, ctx: EntityContext) -> int:
        """Number of entities in the sector."""
        return ctx.sector.indices.shape[0]

    @classmethod
    def global_permutations(cls, ctx: EntityContext, tgt_name: str) -> Tensor:
        from .utils import argpermute
        cell_indices = ctx.sector.indices
        local_face = cls.local_entity(tgt_name, indexing="s")
        face_indices = ctx.block.get_sector(tgt_name).indices
        cell_to_face = cls.relation(ctx, tgt_name).tgt_indices
        return argpermute(
            cell_indices[:, local_face],
            face_indices[cell_to_face],
            dtype=bm.uint8
        )

    ### [Geometric Computations] ###

    @classmethod
    def barycentric(cls, ctx: EntityContext, func: Callable[[Tensor], Tensor], index: Index | None) -> Callable[[Tensor], Tensor]:
        """Compute the barycentric coordinates of the entity."""
        from functools import wraps
        from ...decorator import barycentric
        @wraps(func)
        @barycentric
        def wrapper(bcs: Tensor | tuple[Tensor, ...]) -> Tensor:
            if not isinstance(bcs, tuple):
                bcs = (bcs,)
            points = cls.bc_to_point(ctx, bcs, index) # [NC, NQ, GD]
            return func(points)
        return wrapper

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def integral(cls, ctx: EntityContext, func: Callable[[Tensor], Tensor], q: int, index: Index | None) -> Tensor:
        """Integral of a barycentric function."""
        quadrature = cls.quadrature_formula(q)
        bcs, ws = quadrature.get_quadrature_points_and_weights()

        if not getattr(func, "coordtype", None) == "barycentric":
            func = cls.barycentric(ctx, func, index)
        values = func(bcs)
        measure = cls.measure(ctx, index)

        return bm.einsum("c, q, cq... -> c...", measure, ws, values)


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
