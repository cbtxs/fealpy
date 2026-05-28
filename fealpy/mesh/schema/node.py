from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["NodeSchema"]


class NodeSchema(ShapedEntitySchema):
    name = "node"
    top_dim = 0
    local_faces = {}
    ccw = {}

    @classmethod
    def _indices(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        node = ctx.sector.indices if index is None else ctx.sector.indices[index]
        return bm.reshape(node, (-1,))

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        return ctx.block.positions[cls._indices(ctx, index)]

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        points = cls.barycenter(ctx, index)
        if not isinstance(bcs, tuple):
            raise TypeError(f"node barycentric coordinates expect a tuple, got {type(bcs).__name__}")
        if len(bcs) != 1:
            raise ValueError(f"node barycentric coordinates expect one tensor, got {len(bcs)}")
        if bcs[0].shape[-1] != 1:
            raise ValueError(f"node barycentric coordinates expect last dimension 1, got {bcs[0].shape[-1]}")
        return bm.einsum("...j,cjd->c...d", bcs[0], points[:, None, :])

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        node = cls._indices(ctx, index)
        gd = cls.geo_dimension(ctx)
        return bm.zeros((node.shape[0], 1, gd), dtype=ctx.block.positions.dtype)

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
    def multi_index(cls, p: tuple[int, ...]) -> Tensor:
        if not isinstance(p, tuple):
            raise TypeError(f"node multi_index expects a tuple of integers, got {type(p).__name__}")
        if len(p) != 1:
            raise ValueError(f"node multi_index expects one order value, got {len(p)}")

        order = p[0]
        if not isinstance(order, int):
            raise TypeError(f"node multi_index order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"node multi_index order must be non-negative, got {order}")
        return bm.asarray([[order]], dtype=bm.int32)
