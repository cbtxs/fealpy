from ...backend import bm
from ...backend import Index, Tensor
from ..topology.ipoints import InterpolationPoints
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["TriangleSchema"]


class TriangleSchema(ShapedEntitySchema):
    name = "tri"
    top_dim = 2
    local_faces = {
        'edge': [[0, 1], [0, 2], [1, 2]]
    }
    ccw = {
        "edge": [[1, 2], [2, 0], [0, 1]]
    }

    @classmethod
    def _selected_triangles(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tri.shape) == 1:
            tri = bm.reshape(tri, (1, -1))
        return tri

    @classmethod
    def multi_index(cls, order: tuple[int, ...]) -> Tensor:
        if not isinstance(order, tuple):
            raise TypeError(
                f"triangle multi_index expects a tuple of integers, got {type(order).__name__}"
            )
        if len(order) != 1:
            raise ValueError(f"triangle multi_index expects one order value, got {len(order)}")

        p = order[0]
        if not isinstance(p, int):
            raise TypeError(f"triangle multi_index order must be an integer, got {type(p).__name__}")
        if p < 0:
            raise ValueError(f"triangle multi_index order must be non-negative, got {p}")
        return InterpolationPoints.multi_index_matrix(p, cls.top_dim + 1)

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        return bm.mean(points, axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        if not isinstance(bcs, tuple):
            bcs = (bcs,)
        if len(bcs) != 1:
            raise ValueError("triangle schema expects a single barycentric tensor")
        bc = bcs[0]
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        return bm.einsum("...j,cjd->c...d", bc, points)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        node = ctx.block.positions
        if ref:
            grad = bm.broadcast_to(
                bm.eye(3, dtype=node.dtype)[None, :, :],
                (tri.shape[0], 3, 3),
            )
        else:
            gd = int(node.shape[1])
            if gd == 2:
                grad = bm.triangle_grad_lambda_2d(tri, node)
            elif gd == 3:
                grad = bm.triangle_grad_lambda_3d(tri, node)
            else:
                raise ValueError(f"unsupported geometric dimension: {gd}")
        if bcs is None:
            return grad
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (tri.shape[0], nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str = "legendre"):
        if qtype != "legendre":
            raise ValueError(f"unsupported quadrature type: {qtype}")
        if q > 9:
            from ...quadrature.stroud_quadrature import StroudQuadrature
            return StroudQuadrature(2, q)
        from ...quadrature import TriangleQuadrature
        return TriangleQuadrature(q)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        node = ctx.block.positions
        gd = int(node.shape[1])
        if gd == 2:
            return bm.simplex_measure(tri, node)
        if gd == 3:
            points = node[tri]
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            normal = bm.cross(v1, v2)
            return bm.linalg.vector_norm(normal, axis=1) * 0.5
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        gd = int(points.shape[2])
        if gd == 2:
            return bm.zeros((points.shape[0], 0, gd), **bm.context(points))
        if gd == 3:
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            normal = bm.cross(v1, v2)
            return bm.expand_dims(normal, axis=1)
        raise ValueError(f"unsupported geometric dimension: {gd}")

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tri = cls._selected_triangles(ctx, index)
        points = ctx.block.positions[tri]
        t0 = points[:, 1, :] - points[:, 0, :]
        t1 = points[:, 2, :] - points[:, 0, :]
        return bm.stack([t0, t1], axis=1)
