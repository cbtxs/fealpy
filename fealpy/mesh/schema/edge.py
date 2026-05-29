from ...backend import bm
from ...backend import Index, Tensor
from ..topology.ipoints import InterpolationPoints
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["EdgeSchema"]


class EdgeSchema(ShapedEntitySchema):
    name = "edge"
    top_dim = 1
    local_faces = {
        "node": [[0], [1]]
    }
    ccw = {
        "node": [[0], [1]]
    }

    @classmethod
    def _entity(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        edge = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(edge.shape) == 1:
            edge = bm.reshape(edge, (1, -1))
        return edge

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        return ctx.block.positions[cls._entity(ctx, index)]

    @classmethod
    def _parse_order(cls, p: tuple[int, ...]) -> int:
        if not isinstance(p, tuple):
            raise TypeError(f"edge multi_index expects a tuple of integers, got {type(p).__name__}")
        if len(p) != 1:
            raise ValueError(f"edge multi_index expects one order value, got {len(p)}")

        order = p[0]
        if not isinstance(order, int):
            raise TypeError(f"edge multi_index order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"edge multi_index order must be non-negative, got {order}")
        return order

    @classmethod
    def multi_index(cls, p: tuple[int, ...]) -> Tensor:
        order = cls._parse_order(p)
        return InterpolationPoints.multi_index_matrix(order, 2)

    @classmethod
    def multi_index_sort(cls, multi_index: Tensor, /) -> Tensor:
        if len(multi_index.shape) != 2 or multi_index.shape[1] != 2:
            raise ValueError("edge multi_index_sort expects a tensor of shape (N, 2)")
        if multi_index.shape[0] == 0:
            return bm.asarray([], dtype=bm.int32)
        return bm.lexsort(tuple(reversed(multi_index.T)), axis=0)

    @classmethod
    def num_multi_index(cls, p: tuple[int, ...]) -> int:
        order = cls._parse_order(p)
        return order + 1

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)

    @classmethod
    def barycentric(cls, ctx: EntityContext, index: Index | None, func):
        points = cls._points(ctx, index)

        def wrapper(bc: Tensor) -> Tensor:
            x = bm.einsum("...j,cjd->c...d", bc, points)
            return func(x)

        return wrapper

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        if not isinstance(bcs, tuple):
            raise TypeError(f"edge barycentric coordinates expect a tuple, got {type(bcs).__name__}")
        if len(bcs) != 1:
            raise ValueError(f"edge barycentric coordinates expect one tensor, got {len(bcs)}")
        if bcs[0].shape[-1] != 2:
            raise ValueError(f"edge barycentric coordinates expect last dimension 2, got {bcs[0].shape[-1]}")

        points = cls._points(ctx, index)
        return bm.einsum("...j,cjd->c...d", bcs[0], points)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        sqnorm = bm.sum(tangent * tangent, axis=1, keepdims=True)
        g1 = tangent / sqnorm
        g0 = -g1
        return bm.stack([g0, g1], axis=1)

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre"):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported edge quadrature type: {qtype!r}")
        from fealpy.quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        return bm.linalg.vector_norm(tangent, axis=1)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        points = cls._points(ctx, index)
        tangent = points[:, 1, :] - points[:, 0, :]
        gd = points.shape[-1]
        sqnorm = bm.sum(tangent * tangent, axis=1)

        if bm.any(sqnorm == 0):
            raise ValueError("degenerate edge has no well-defined normal directions")

        if gd == 1:
            return bm.zeros((points.shape[0], 0, gd), dtype=ctx.block.positions.dtype)

        if gd == 2:
            normal = bm.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
            return normal[:, None, :]

        if gd == 3:
            axis = bm.argmin(bm.abs(tangent), axis=1)
            ref = bm.zeros(tangent.shape, dtype=ctx.block.positions.dtype)
            ref = bm.set_at(ref, (axis == 0, 0), 1.0)
            ref = bm.set_at(ref, (axis == 1, 1), 1.0)
            ref = bm.set_at(ref, (axis == 2, 2), 1.0)
            n1 = bm.linalg.cross(tangent, ref)
            n2 = bm.linalg.cross(tangent, n1)
            return bm.stack([n1, n2], axis=1)

        raise NotImplementedError(f"edge normal is only implemented for GD <= 3, got {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        points = cls._points(ctx, index)
        return (points[:, 1, :] - points[:, 0, :])[:, None, :]

    @classmethod
    def transform(cls, ctx: EntityContext, func, kind: str = "value"):
        points = ctx.block.positions[ctx.sector.indices]

        def wrapper(bc: Tensor) -> Tensor:
            x = bm.einsum("...j,cjd->c...d", bc, points)
            value = func(x)
            if kind == "value":
                return value
            raise NotImplementedError(f"Unsupported edge transform kind: {kind!r}")

        return wrapper
