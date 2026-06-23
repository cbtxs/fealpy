from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

from ...quadrature import Quadrature

__all__ = ["NodeSchema", "PointQuadrature"]


class PointQuadrature(Quadrature):
    def __init__(self, *, dtype=None):
        dtype = bm.float64 if dtype is None else dtype
        self.quadpts = (bm.asarray([[1.0]], dtype=dtype),)
        self.weights = bm.asarray([1.0], dtype=dtype)

    def number_of_quadrature_points(self) -> int:
        return int(self.weights.shape[0])

    def get_quadrature_points_and_weights(self) -> tuple[tuple[Tensor, ...], Tensor]:
        return self.quadpts, self.weights

    def get_quadrature_point_and_weight(self, i: int) -> tuple[tuple[Tensor, ...], Tensor]:
        return tuple(qp[i:i + 1] for qp in self.quadpts), self.weights[i]

    def __len__(self) -> int:
        return self.number_of_quadrature_points()

    def __getitem__(self, i: int) -> tuple[tuple[Tensor, ...], Tensor]:
        return self.get_quadrature_point_and_weight(i)


class NodeSchema(ShapedEntitySchema):
    name = "node"
    top_dim = 0
    OFace = {}
    SFace = {}
    orientation = [(0,)]

    @classmethod
    def _indices(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        node = ctx.sector.indices if index is None else ctx.sector.indices[index]
        return bm.reshape(node, (-1,))

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        return ctx.block.positions[cls._indices(ctx, index)]

    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...]
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "node shape_function", 1)
        p = _require_order_tuple(p, "node shape_function", 1)
        if bcs[0].shape[-1] != 1:
            raise ValueError(f"node shape_function expects last dimension 1, got {bcs[0].shape[-1]}")
        return bm.ones((bcs[0].shape[0], 1), dtype=bcs[0].dtype)

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
        node = cls._indices(ctx, index)
        dim = 1 if variables == "u" else cls.geo_dimension(ctx)
        grad = bm.zeros((1, 1, dim), dtype=ctx.block.positions.dtype)
        bcs = _require_bcs_tuple(bcs, "node grad_shape_function", 1)
        _ = _require_order_tuple(p, "node grad_shape_function", 1)
        nq = int(bcs[0].shape[0])
        if variables == "x":
            return bm.broadcast_to(grad, (node.shape[0], nq, 1, dim))
        return bm.broadcast_to(grad, (nq, 1, dim))

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        points = cls.barycenter(ctx, index)
        if not isinstance(bcs, tuple):
            raise TypeError(f"node barycentric coordinates expect a tuple, got {type(bcs).__name__}")
        if len(bcs) != 1:
            raise ValueError(f"node barycentric coordinates expect one tensor, got {len(bcs)}")
        if bcs[0].shape[-1] != 1:
            raise ValueError(f"node barycentric coordinates expect last dimension 1, got {bcs[0].shape[-1]}")
        expected = bm.ones(bcs[0].shape, dtype=bcs[0].dtype)
        if not bm.allclose(bcs[0], expected):
            raise ValueError("node barycentric coordinates must be identically 1")
        return bm.einsum("...j,cjd->c...d", bcs[0], points[:, None, :])

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        node = cls._indices(ctx, index)
        dim = 1 if ref else cls.geo_dimension(ctx)
        grad = bm.zeros((node.shape[0], 1, dim), dtype=ctx.block.positions.dtype)
        if bcs is None:
            return grad
        bcs = _require_bcs_tuple(bcs, "node grad_lambda", 1)
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (node.shape[0], nq, 1, dim))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None) -> "Quadrature":
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported node quadrature type: {qtype!r}")
        if q < 1:
            raise ValueError(f"node quadrature order must be positive, got {q}")
        return PointQuadrature()

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        node = cls._indices(ctx, index)
        return bm.ones((node.shape[0],), dtype=ctx.block.positions.dtype)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        node = cls._indices(ctx, index)
        gd = cls.geo_dimension(ctx)
        basis = bm.eye(gd, dtype=ctx.block.positions.dtype)
        return bm.broadcast_to(basis, (node.shape[0], gd, gd))

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        node = cls._indices(ctx, index)
        gd = cls.geo_dimension(ctx)
        return bm.zeros((node.shape[0], 0, gd), dtype=ctx.block.positions.dtype)

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False, tensorprod: bool = True) -> Tensor:
        p = _require_order_tuple(order, "node multi_index", 1)[0]

        mi = bm.asarray([[p]], dtype=bm.int32)
        if tensorprod:
            from ..topology.ipoints import multi_index_tensorprod
            return multi_index_tensorprod(mi)
        return mi
