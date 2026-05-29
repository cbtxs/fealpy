from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["QuadrilateralSchema"]


class QuadrilateralSchema(ShapedEntitySchema):
    name = "quad"
    top_dim = 2
    local_faces = {'edge': [[0, 1], [1, 2], [2, 3], [3, 0]]}
    ccw = {'edge': [[0, 1], [1, 2], [2, 3], [3, 0]]}

    @classmethod
    def multi_index(cls, order: int | tuple[int, ...], *, internal: bool = False) -> Tensor:
        if isinstance(order, int):
            px = py = order
        else:
            if len(order) == 1:
                px = py = order[0]
            elif len(order) == 2:
                px, py = order
            else:
                raise ValueError("quadrilateral multi-index expects one or two integers")

        i = bm.repeat(bm.arange(px + 1), py + 1).reshape(-1, py + 1)
        multi_index0 = i.flatten().reshape(-1, 1)
        multi_index1 = i.T.flatten().reshape(-1, 1)
        mi = bm.concatenate([multi_index0, multi_index1], axis=1)

        if internal:
            flag = ((mi[:, 0] > 0) & (mi[:, 0] < px) &
                    (mi[:, 1] > 0) & (mi[:, 1] < py))
            return mi[flag]
        return mi

    @classmethod
    def multi_index_sort(cls, multi_index: Tensor, /) -> Tensor:
        return bm.lexsort((multi_index[:, 1], multi_index[:, 0]))

    @classmethod
    def num_multi_index(cls, order: int | tuple[int, ...], *, internal: bool = False) -> int:
        mi = cls.multi_index(order, internal=internal)
        return int(mi.shape[0])

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        if not isinstance(bcs, tuple) or len(bcs) != 2:
            raise TypeError("quadrilateral bc_to_point expects a tuple of two tensors")

        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad[:, [0, 3, 1, 2]]]

        bc0 = bcs[0].reshape(-1, 2)
        bc1 = bcs[1].reshape(-1, 2)
        bc = bm.einsum("im,jn->ijmn", bc0, bc1).reshape(-1, 4)
        return bm.einsum("qj,cjd->cqd", bc, points)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = None):
        from ...quadrature import QuadrangleQuadrature
        return QuadrangleQuadrature(q)

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]

        vr = 0.5 * ((points[:, 1, :] - points[:, 0, :]) +
                    (points[:, 2, :] - points[:, 3, :]))
        vs = 0.5 * ((points[:, 3, :] - points[:, 0, :]) +
                    (points[:, 2, :] - points[:, 1, :]))

        jac = bm.stack([vr, vs], axis=-1)
        jac_t = bm.einsum("nij->nji", jac)
        metric = bm.einsum("nik,nkj->nij", jac_t, jac)
        metric_inv = bm.linalg.inv(metric)
        grads = bm.einsum("nik,nkj->nij", metric_inv, jac_t)

        ref_grads = bm.asarray(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=points.dtype,
        )
        return bm.einsum("ld, ndg->nlg", ref_grads, grads)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]

        v0 = points[:, 1, :] - points[:, 0, :]
        v1 = points[:, 2, :] - points[:, 0, :]
        v2 = points[:, 3, :] - points[:, 0, :]

        if points.shape[-1] == 2:
            cross01 = v0[:, 0] * v1[:, 1] - v0[:, 1] * v1[:, 0]
            cross12 = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            return 0.5 * (bm.abs(cross01) + bm.abs(cross12))

        cross0 = bm.cross(v0, v1)
        cross1 = bm.cross(v1, v2)
        area0 = bm.sqrt(bm.sum(cross0 * cross0, axis=1))
        area1 = bm.sqrt(bm.sum(cross1 * cross1, axis=1))
        return 0.5 * (area0 + area1)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]

        v0 = points[:, 1, :] - points[:, 0, :]
        v1 = points[:, 2, :] - points[:, 0, :]
        v2 = points[:, 3, :] - points[:, 0, :]
        return bm.cross(v0, v1) + bm.cross(v1, v2)

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        quad = ctx.sector.indices if index is None else ctx.sector.indices[index]
        points = ctx.block.positions[quad]

        vr = 0.5 * ((points[:, 1, :] - points[:, 0, :]) +
                    (points[:, 2, :] - points[:, 3, :]))
        vs = 0.5 * ((points[:, 3, :] - points[:, 0, :]) +
                    (points[:, 2, :] - points[:, 1, :]))
        return bm.stack([vr, vs], axis=1)
