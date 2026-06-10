from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import (
    EntityContext,
    ShapedEntitySchema,
    _require_bcs_tuple,
    _require_order_tuple,
)

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
        bcs = _require_bcs_tuple(bcs, "hexahedron bc_to_point", 3)
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
    def shape_function(
        cls,
        bcs: tuple[Tensor, ...],
        p: tuple[int, ...],
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "hexahedron shape_function", 3)
        p = _require_order_tuple(p, "hexahedron shape_function", 3)
        for bc in bcs:
            if bc.shape[-1] != 2:
                raise ValueError("hexahedron shape_function expects three interval barycentric tensors")
        phi = bm.tensorprod(*(bm.simplex_shape_function(bc, p, mi) for bc, p in zip(bcs, p)))
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
        p: tuple[int, ...],
        *,
        index: Index | None = None,
        variables: str = "u",
        mi=None,
    ) -> Tensor:
        bcs = _require_bcs_tuple(bcs, "hexahedron grad_shape_function", 3)
        p = _require_order_tuple(p, "hexahedron grad_shape_function", 3)
        for bc in bcs:
            if bc.shape[-1] != 2:
                raise ValueError("hexahedron grad_shape_function expects three interval barycentric tensors")

        phi0, phi1, phi2 = (bm.simplex_shape_function(bc, p, mi) for bc, p in zip(bcs, p))
        R0, R1, R2 = (bm.simplex_grad_shape_function(bc, p, mi) for bc, p in zip(bcs, p))
        Dlambda = bm.asarray([[-1.0], [1.0]], dtype=phi0.dtype)
        R0 = bm.einsum("...ij,jn->...in", R0, Dlambda)
        R1 = bm.einsum("...ij,jn->...in", R1, Dlambda)
        R2 = bm.einsum("...ij,jn->...in", R2, Dlambda)
        ref = bm.concatenate(
            [
                R0[:, None, :, None, None, :] * phi1[None, :, None, :, None, None] * phi2[None, None, None, :, :, None],
                phi0[:, None, :, None, None, None] * R1[None, :, None, :, None, :] * phi2[None, None, None, :, :, None],
                phi0[:, None, :, None, None, None] * phi1[None, :, None, :, None, None] * R2[None, None, None, :, :, :],
            ],
            axis=-1,
        ).reshape(-1, phi0.shape[1] * phi1.shape[1] * phi2.shape[1], 3)

        if variables == "u":
            return ref
        if variables == "x":
            if p != 1:
                raise NotImplementedError("hexahedron grad_shape_function currently only supports p=1 in physical space")
            return cls.grad_lambda(ctx, index, bcs=bcs, ref=False)
        raise ValueError(f"Unsupported variables: {variables!r}")

    @classmethod
    def multi_index(cls, order: tuple[int, ...], *, internal: bool = False) -> Tensor:
        order = _require_order_tuple(order, "hexahedron multi_index", 3)
        px, py, pz = order

        if internal:
            ix = bm.arange(1, px, dtype=bm.int32)
            iy = bm.arange(1, py, dtype=bm.int32)
            iz = bm.arange(1, pz, dtype=bm.int32)
            shape = (max(px - 1, 0), max(py - 1, 0), max(pz - 1, 0))
        else:
            ix = bm.arange(px + 1, dtype=bm.int32)
            iy = bm.arange(py + 1, dtype=bm.int32)
            iz = bm.arange(pz + 1, dtype=bm.int32)
            shape = (px + 1, py + 1, pz + 1)
        multi_index0 = bm.broadcast_to(ix[:, None, None], shape).reshape(-1, 1)
        multi_index1 = bm.broadcast_to(iy[None, :, None], shape).reshape(-1, 1)
        multi_index2 = bm.broadcast_to(iz[None, None, :], shape).reshape(-1, 1)
        return bm.concatenate([multi_index0, multi_index1, multi_index2], axis=-1)


    @classmethod
    def grad_lambda(
        cls,
        ctx: EntityContext,
        index: Index | None,
        bcs: tuple[Tensor, ...] | None = None,
        *,
        ref: bool = False,
    ) -> Tensor:
        cell = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(cell.shape) == 1:
            cell = bm.reshape(cell, (1, -1))
        if bcs is None:
            bcs = (
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
                bm.asarray([[0.5, 0.5]], dtype=ctx.block.positions.dtype),
            )
            squeeze_q = True
        else:
            bcs = _require_bcs_tuple(bcs, "hexahedron grad_lambda", 3)
            squeeze_q = False
        u, v, w = bcs
        u0, u1 = u[:, 0], u[:, 1]
        v0, v1 = v[:, 0], v[:, 1]
        w0, w1 = w[:, 0], w[:, 1]
        z = bm.zeros_like(u0)

        def ref_row(ua, ub, va, vb, wa, wb):
            return bm.stack([ua, ub, va, vb, wa, wb], axis=-1)

        ref_grad = bm.stack([
            ref_row(v0*w0, z, u0*w0, z, u0*v0, z),
            ref_row(z, v0*w0, u1*w0, z, u1*v0, z),
            ref_row(v1*w0, z, z, u0*w0, u0*v1, z),
            ref_row(z, v1*w0, z, u1*w0, u1*v1, z),
            ref_row(v0*w1, z, u0*w1, z, z, u0*v0),
            ref_row(z, v0*w1, u1*w1, z, z, u1*v0),
            ref_row(v1*w1, z, z, u0*w1, z, u0*v1),
            ref_row(z, v1*w1, z, u1*w1, z, u1*v1),
        ], axis=1)
        if ref:
            grad = bm.broadcast_to(ref_grad[None, :, :, :], (cell.shape[0], ref_grad.shape[0], 8, 6))
            return grad[:, 0, :, :] if squeeze_q else grad

        dphi = bm.stack([
            bm.stack([-v0*w0, -u0*w0, -u0*v0], axis=-1),
            bm.stack([ v0*w0, -u1*w0, -u1*v0], axis=-1),
            bm.stack([-v1*w0,  u0*w0, -u0*v1], axis=-1),
            bm.stack([ v1*w0,  u1*w0, -u1*v1], axis=-1),
            bm.stack([-v0*w1, -u0*w1,  u0*v0], axis=-1),
            bm.stack([ v0*w1, -u1*w1,  u1*v0], axis=-1),
            bm.stack([-v1*w1,  u0*w1,  u0*v1], axis=-1),
            bm.stack([ v1*w1,  u1*w1,  u1*v1], axis=-1),
        ], axis=1)
        points = ctx.block.positions[cell]
        J = bm.einsum("qit,cid->cqtd", dphi, points)
        Jt = bm.einsum("cqtd->cqdt", J)
        metric = bm.einsum("cqtd,cqsd->cqts", J, J)
        metric_inv = bm.linalg.inv(metric)
        grad = bm.einsum("cqdt,cqts,qis->cqid", Jt, metric_inv, dphi)
        return grad[:, 0, :, :] if squeeze_q else grad

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
    def quadrature_formula(cls, q: int, qtype: str | None = "legendre", device=None):
        if qtype not in (None, "legendre"):
            raise ValueError(f"unsupported hexahedron quadrature type: {qtype!r}")
        from fealpy.quadrature import GaussLegendreQuadrature, TensorProductQuadrature

        qf = GaussLegendreQuadrature(q, device=device)
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
