from ...backend import bm
from ...backend import Index, Tensor
from ..topology.ipoints import MultiIndex as _MI
from .entity_schema import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

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
    orientation = [(0, 1), (1, 0)]

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
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        p = _require_order_tuple(order, "edge multi_index", 1)[0]
        if internal:
            mi = _MI.multi_index_inner(p, 2)
        else:
            mi = _MI.multi_index_matrix(p, 2)
        return mi

    @classmethod
    def num_multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> int:
        p = _require_order_tuple(order, "edge num_multi_index", 1)[0]
        if internal:
            return p - 1 if p > 1 else 0
        return p + 1

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "edge bc_to_point", 1)
        if bcs[0].shape[-1] != 2:
            raise ValueError(f"edge barycentric coordinates expect last dimension 2, got {bcs[0].shape[-1]}")

        points = cls._points(ctx, index)
        return bm.einsum("...j,cjd->c...d", bcs[0], points)

    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "edge shape_function", 1)
        p = _require_order_tuple(p, "edge shape_function", 1)
        if bcs[0].shape[-1] != 2:
            raise ValueError(f"edge shape_function expects last dimension 2, got {bcs[0].shape[-1]}")
        phi = bm.simplex_shape_function(bcs[0], p[0], mi)
        if variables == "u":
            return phi
        if variables == "x":
            return phi[None, ...]
        raise ValueError(f"Unsupported variables: {variables!r}")

    @classmethod
    def grad_shape_function(
        cls,
        ctx: EntityContext,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "edge grad_shape_function", 1)
        p = _require_order_tuple(p, "edge grad_shape_function", 1)
        if bcs[0].shape[-1] != 2:
            raise ValueError(f"edge grad_shape_function expects last dimension 2, got {bcs[0].shape[-1]}")
        ref = bm.simplex_grad_shape_function(bcs[0], p[0], mi)

        if variables == "u":
            return ref
        if variables == "x":
            Dlambda = cls.grad_lambda(ctx, index, ref=False)
            grad = bm.einsum("...ij, kjm -> k...im", ref, Dlambda)
            return grad
        raise ValueError(f"Unsupported variables: {variables!r}")

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        points = cls._points(ctx, index)
        nc = int(points.shape[0])
        if ref:
            grad = bm.broadcast_to(
                bm.eye(2, dtype=ctx.block.positions.dtype)[None, :, :],
                (nc, 2, 2),
            )
        else:
            tangent = points[:, 1, :] - points[:, 0, :]
            sqnorm = bm.sum(tangent * tangent, axis=1, keepdims=True)
            g1 = tangent / sqnorm
            g0 = -g1
            grad = bm.stack([g0, g1], axis=1)
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "edge grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (nc, nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported edge quadrature type: {qtype!r}")
        from fealpy.quadrature import GaussLegendreQuadrature
        return GaussLegendreQuadrature(q, device=device)

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
