from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["TetrahedronSchema"]


class TetrahedronSchema(ShapedEntitySchema):
    name = "tet"
    top_dim = 3
    local_faces = {
        'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    }

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[tet]
        return bm.mean(points, axis=1)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        node = ctx.block.positions
        NC = tet.shape[0]
        Dlambda = bm.zeros((NC, 4, 3), dtype=node.dtype)
        volume = cls.measure(ctx, index)

        for i in range(4):
            j, k, m = ctx.sector.schema.local_faces["tri"][i]
            vjk = node[tet[:, k], :] - node[tet[:, j], :]
            vjm = node[tet[:, m], :] - node[tet[:, j], :]
            Dlambda[:, i, :] = bm.linalg.cross(vjm, vjk) / (6*volume[:, None])

        return Dlambda

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[tet]

        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        v3 = points[:, 3, :] - points[:, 0, :]
        jac = bm.stack([v1, v2, v3], axis=-1)
        return bm.abs(bm.linalg.det(jac)) / 6.0