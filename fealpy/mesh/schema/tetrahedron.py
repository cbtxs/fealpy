from ...backend import bm
from ...backend import Index, Tensor
from ..topology.ipoints import InterpolationPoints
from .entity_schema import EntityContext, ShapedEntitySchema, _require_bcs_tuple

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
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        return bm.mean(points, axis=1)

    @classmethod
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: int = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "tetrahedron shape_function", 1)
        if bcs[0].shape[-1] != 4:
            raise ValueError(f"tetrahedron shape_function expects last dimension 4, got {bcs[0].shape[-1]}")
        phi = bm.simplex_shape_function(bcs[0], p, mi)
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
        p: int = 1,
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "tetrahedron grad_shape_function", 1)
        if bcs[0].shape[-1] != 4:
            raise ValueError(f"tetrahedron grad_shape_function expects last dimension 4, got {bcs[0].shape[-1]}")
        ref = bm.simplex_grad_shape_function(bcs[0], p, mi)
        if variables == "u":
            return ref
        if variables == "x":
            Dlambda = cls.grad_lambda(ctx, index, ref=False)
            grad = bm.einsum("...ij, kjm -> k...im", ref, Dlambda)
            return grad
        raise ValueError(f"Unsupported variables: {variables!r}")

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "tetrahedron bc_to_point", 1)
        if bcs[0].shape[-1] != 4:
            raise ValueError(f"tetrahedron barycentric coordinates expect last dimension 4, got {bcs[0].shape[-1]}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        return bm.einsum("...j,cjd->c...d", bcs[0], points)

    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        if ref:
            grad = bm.broadcast_to(
                bm.eye(4, dtype=ctx.block.positions.dtype)[None, :, :],
                (tet.shape[0], 4, 4),
            )
        else:
            gd = cls.geo_dimension(ctx)
            if gd != 3:
                raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")
            points = ctx.block.positions[tet]
            v1 = points[:, 1, :] - points[:, 0, :]
            v2 = points[:, 2, :] - points[:, 0, :]
            v3 = points[:, 3, :] - points[:, 0, :]
            jac = bm.stack([v1, v2, v3], axis=-1)
            inv_jac = bm.linalg.inv(jac)
            g1 = inv_jac[:, 0, :]
            g2 = inv_jac[:, 1, :]
            g3 = inv_jac[:, 2, :]
            g0 = -g1 - g2 - g3
            grad = bm.stack([g0, g1, g2, g3], axis=1)
        if bcs is None:
            return grad
        nq = int(bcs[0].shape[0])
        return bm.broadcast_to(grad[:, None, :, :], (tet.shape[0], nq, grad.shape[1], grad.shape[2]))

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        v3 = points[:, 3, :] - points[:, 0, :]
        jac = bm.stack([v1, v2, v3], axis=-1)
        return bm.abs(bm.linalg.det(jac)) / 6.0

    @classmethod
    def multi_index(cls, order: tuple[int, ...]) -> Tensor:
        if not isinstance(order, tuple):
            raise TypeError(
                f"tetrahedron multi_index expects a tuple of integers, got {type(order).__name__}"
            )
        if len(order) != 1:
            raise ValueError(f"tetrahedron multi_index expects one order value, got {len(order)}")

        order = order[0]
        if not isinstance(order, int):
            raise TypeError(f"tetrahedron multi_index order must be an integer, got {type(order).__name__}")
        if order < 0:
            raise ValueError(f"tetrahedron multi_index order must be non-negative, got {order}")
        return InterpolationPoints.multi_index_matrix(order, 4)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        return bm.zeros((tet.shape[0], 0, 3), dtype=ctx.block.positions.dtype)

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
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"tetrahedron geometry requires GD == 3, got {gd}")

        tet = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(tet.shape) == 1:
            tet = bm.reshape(tet, (1, -1))
        points = ctx.block.positions[tet]
        v1 = points[:, 1, :] - points[:, 0, :]
        v2 = points[:, 2, :] - points[:, 0, :]
        v3 = points[:, 3, :] - points[:, 0, :]
        return bm.stack([v1, v2, v3], axis=1)
