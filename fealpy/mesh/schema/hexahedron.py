from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema

__all__ = ["HexahedronSchema"]


class HexahedronSchema(ShapedEntitySchema):
    name = "hex"
    top_dim = 3
    local_faces = {
        "quad": [
            [0, 1, 2, 3], [4, 5, 6, 7],
            [0, 1, 4, 5], [2, 3, 6, 7],
            [0, 2, 4, 6], [1, 3, 5, 7],
        ]
    }
    ccw = {
        "quad": [
            [0, 2, 3, 1], [4, 5, 7, 6],
            [0, 1, 5, 4], [2, 6, 7, 3],
            [0, 4, 6, 2], [1, 3, 7, 5],
        ]
    }

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        return bm.mean(ctx.block.positions[cell], axis=1)

    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, ...], index: Index | None) -> Tensor:
        if not isinstance(bcs, tuple):
            raise TypeError(f"hexahedron barycentric coordinates expect a tuple, got {type(bcs).__name__}")
        if len(bcs) != 3:
            raise ValueError(f"hexahedron barycentric coordinates expect three tensors, got {len(bcs)}")
        for bc in bcs:
            if bc.shape[-1] != 2:
                raise ValueError(
                    f"hexahedron barycentric coordinate tensors expect last dimension 2, got {bc.shape[-1]}"
                )

        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        points = ctx.block.positions[cell][:, [0, 4, 2, 6, 1, 5, 3, 7], :]
        points = bm.reshape(points, (-1, 2, 2, 2, cls.geo_dimension(ctx)))
        u, v, w = bcs
        return bm.einsum("ia,jb,kc,nabce->nijke", u, v, w, points)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])

    @classmethod
    def multi_index(cls, p: tuple[int, ...]) -> Tensor:
        if not isinstance(p, tuple):
            raise TypeError(
                f"hexahedron multi_index expects a tuple of integers, got {type(p).__name__}"
            )
        if len(p) == 1:
            px, py, pz = p[0], p[0], p[0]
        elif len(p) == 3:
            px, py, pz = p
        else:
            raise ValueError(f"hexahedron multi_index expects one or three order values, got {len(p)}")

        for value in (px, py, pz):
            if not isinstance(value, int):
                raise TypeError(f"hexahedron multi_index order must be an integer, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"hexahedron multi_index order must be non-negative, got {value}")

        ix = bm.arange(px + 1, dtype=bm.int32)
        iy = bm.arange(py + 1, dtype=bm.int32)
        iz = bm.arange(pz + 1, dtype=bm.int32)
        shape = (px + 1, py + 1, pz + 1)
        multi_index0 = bm.broadcast_to(ix[:, None, None], shape).reshape(-1, 1)
        multi_index1 = bm.broadcast_to(iy[None, :, None], shape).reshape(-1, 1)
        multi_index2 = bm.broadcast_to(iz[None, None, :], shape).reshape(-1, 1)
        return bm.concatenate([multi_index0, multi_index1, multi_index2], axis=-1)

    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")

        qf = cls.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        u, v, w = bcs
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        points = ctx.block.positions[cell][:, [0, 4, 2, 6, 1, 5, 3, 7], :]
        points = bm.reshape(points, (-1, 2, 2, 2, 3))
        du = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], u.shape)
        dv = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], v.shape)
        dw = bm.broadcast_to(bm.asarray([-1.0, 1.0], dtype=ctx.block.positions.dtype)[None, :], w.shape)

        ju = bm.einsum("ia,jb,kc,nabce->nijke", du, v, w, points)
        jv = bm.einsum("ia,jb,kc,nabce->nijke", u, dv, w, points)
        jw = bm.einsum("ia,jb,kc,nabce->nijke", u, v, dw, points)
        jac = bm.stack([ju, jv, jw], axis=-1)
        det = bm.abs(bm.linalg.det(jac))
        n = bcs[0].shape[0]
        det = bm.reshape(det, (-1, n, n, n))
        weight = bm.reshape(ws, (n, n, n))
        return bm.einsum("ijk,nijk->n", weight, det)

    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        return bm.zeros((cell.shape[0], 0, 3), dtype=ctx.block.positions.dtype)

    @classmethod
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre"):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported hexahedron quadrature type: {qtype!r}")
        from fealpy.quadrature import GaussLegendreQuadrature, TensorProductQuadrature

        qf = GaussLegendreQuadrature(q)
        return TensorProductQuadrature((qf, qf, qf))

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None) -> Tensor:
        gd = cls.geo_dimension(ctx)
        if gd != 3:
            raise ValueError(f"hexahedron geometry requires GD == 3, got {gd}")
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        points = ctx.block.positions[cell]
        t0 = points[:, 1, :] - points[:, 0, :]
        t1 = points[:, 2, :] - points[:, 0, :]
        t2 = points[:, 4, :] - points[:, 0, :]
        return bm.stack([t0, t1, t2], axis=1)
