from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["EdgeSchema"]


class EdgeSchema(ShapedEntitySchema):
    name = "edge"
    top_dim = 1
    local_faces = {
        "node": [[0], [1]]
    }

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        edge = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[edge]
        return bm.mean(points, axis=1)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        edge = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[edge]
        tangent = points[:, 1, :] - points[:, 0, :]
        sqnorm = bm.sum(tangent * tangent, axis=1, keepdims=True)
        g1 = tangent / sqnorm
        g0 = -g1
        return bm.stack([g0, g1], axis=1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        edge = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[edge]
        tangent = points[:, 1, :] - points[:, 0, :]
        return bm.linalg.vector_norm(tangent, axis=1)

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        edge = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[edge]
        return points[:, 1, :] - points[:, 0, :]