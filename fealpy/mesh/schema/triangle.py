from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["TriangleSchema"]


class TriangleSchema(ShapedEntitySchema):
    name = "tri"
    top_dim = 2
    local_faces = {
        'edge': [[0, 1], [0, 2], [1, 2]]
    }

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[tri]
        return bm.mean(points, axis=1)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[tri]

        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        jac = bm.stack([v1, v2], axis=-1)
        jac_t = bm.einsum("nij->nji", jac)
        metric = bm.linalg.matmul(jac_t, jac)
        metric_inv = bm.linalg.inv(metric)
        grads = bm.linalg.matmul(metric_inv, jac_t)

        g1 = grads[:, 0, :]
        g2 = grads[:, 1, :]
        g0 = -g1 - g2
        return bm.stack([g0, g1, g2], axis=1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[tri]

        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        normal = bm.linalg.cross(v1, v2)
        return bm.linalg.vector_norm(normal, axis=1) * 0.5

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[tri]
        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        return bm.linalg.cross(v1, v2)