from ...backend import bm
from ...backend import Index, Tensor
from ..topology.ipoints import InterpolationPoints
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["TetrahedronSchema"]


class TetrahedronSchema(ShapedEntitySchema):
    name = "tet"
    top_dim = 3
    local_faces = {
        'tri': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]
    }
    ccw = {
        'tri': [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
    }

    @classmethod
    def _entity(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        return tet

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        return ctx.block.positions[cls._entity(ctx, index)]

    @classmethod
    def _require_geo_dimension_3(cls, ctx: EntityContext) -> None:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")

    @classmethod
    def _parse_order(cls, p: tuple[int, ...]) -> int:
        if not isinstance(p, tuple):
            raise TypeError(
                f"tetrahedron multi_index expects a tuple of integers, got {type(p).__name__}"
            )
        if len(p) != 1:
            raise ValueError(f"tetrahedron multi_index expects one order value, got {len(p)}")

        order = p[0]
        if not isinstance(order, int):
            raise TypeError(f"tetrahedron multi_index order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"tetrahedron multi_index order must be non-negative, got {order}")
        return order

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        return bm.mean(cls._points(ctx, index), axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        if not isinstance(bcs, tuple):
            raise TypeError(f"tetrahedron barycentric coordinates expect a tuple, got {type(bcs).__name__}")
        if len(bcs) != 1:
            raise ValueError(f"tetrahedron barycentric coordinates expect one tensor, got {len(bcs)}")
        if bcs[0].shape[-1] != 4:
            raise ValueError(f"tetrahedron barycentric coordinates expect last dimension 4, got {bcs[0].shape[-1]}")

        points = cls._points(ctx, index)
        return bm.einsum("...j,cjd->c...d", bcs[0], points)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def grad_lambda(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        cls._require_geo_dimension_3(ctx)
        inv_jac = bm.linalg.inv(cls.jacobi_matrix(ctx, index))
        g1 = inv_jac[:, 0, :]
        g2 = inv_jac[:, 1, :]
        g3 = inv_jac[:, 2, :]
        g0 = -g1 - g2 - g3
        return bm.stack([g0, g1, g2, g3], axis=1)

    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        cls._require_geo_dimension_3(ctx)
        points = cls._points(ctx, index)
        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        v3 = points[:, 3, :] - points[:, 0, :]
        return bm.stack([v1, v2, v3], axis=-1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        cls._require_geo_dimension_3(ctx)
        jac = cls.jacobi_matrix(ctx, index)
        return bm.abs(bm.linalg.det(jac)) / 6.0

    @classmethod
    def multi_index(cls, p: tuple[int, ...]) -> Tensor:
        order = cls._parse_order(p)
        return InterpolationPoints.multi_index_matrix(order, 4)

    @classmethod
    def multi_index_sort(cls, multi_index: Tensor, /) -> Tensor:
        if len(multi_index.shape) != 2 or multi_index.shape[1] != 4:
            raise ValueError("tetrahedron multi_index_sort expects a tensor of shape (N, 4)")
        if multi_index.shape[0] == 0:
            return bm.asarray([], dtype=bm.int32)
        return bm.lexsort(tuple(reversed(multi_index.T)), axis=0)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        cls._require_geo_dimension_3(ctx)
        tet = cls._entity(ctx, index)
        return bm.zeros((tet.shape[0], 0, 3), dtype=ctx.block.positions.dtype)

    @classmethod
    def num_multi_index(cls, p: tuple[int, ...]) -> int:
        order = cls._parse_order(p)
        return (order + 1) * (order + 2) * (order + 3) // 6

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre"):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported tetrahedron quadrature type: {qtype!r}")
        if q > 7:
            from fealpy.quadrature.stroud_quadrature import StroudQuadrature
            return StroudQuadrature(3, q)
        from fealpy.quadrature import TetrahedronQuadrature
        return TetrahedronQuadrature(q)

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        cls._require_geo_dimension_3(ctx)
        points = cls._points(ctx, index)
        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        v3 = points[:, 3, :] - points[:, 0, :]
        return bm.stack([v1, v2, v3], axis=1)
