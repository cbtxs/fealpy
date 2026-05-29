from ...backend import bm
from ...backend import Index, Tensor
from .entity_schema import EntityContext, ShapedEntitySchema
from ..topology.ipoints import InterpolationPoints

__all__ = ["PrismSchema"]


class PrismSchema(ShapedEntitySchema):
    name = "prism"
    top_dim = 3
    local_faces = {
        'tri': [[0, 1, 2], [3, 4, 5]],
        'quad': [[0, 1, 3, 4], [0, 2, 3, 5], [1, 2, 4, 5]]
    }
    ccw = {
        'tri': [[0, 2, 1], [3, 4, 5]],
        'quad': [[0, 1, 4, 3], [0, 3, 5, 2], [1, 2, 5, 4]]
    }

    @classmethod
    def _entity(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        entity = ctx.sector.indices if index is None else ctx.sector.indices[index]
        if len(entity.shape) == 1:
            entity = entity[None, :]
        return entity

    @classmethod
    def _points(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        prism = cls._entity(ctx, index)
        return ctx.block.positions[prism]

    @classmethod
    def _tp_points(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        prism = cls._entity(ctx, index)
        return ctx.block.positions[prism[:, [0, 3, 1, 4, 2, 5]]]

    @classmethod
    def barycenter(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        """Compute barycenters of prism entities.

        Return shape: (NC, GD).
        """
        points = cls._points(ctx, index)
        return bm.mean(points, axis=1)
    
    @classmethod
    def measure(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        """Compute the volume of a prism.

        ∫_K dx = ∫_{\hat K} sqrt(det(G)) dξ, where G = J^T J.
        """
        qf = cls.quadrature_formula(2)
        bcs, ws = qf.get_quadrature_points_and_weights()
        G = cls.first_fundamental_form(ctx, bcs, index=index)
        l = bm.sqrt(bm.linalg.det(G))
        return 0.5 * bm.einsum("q,cq->c", ws, l)

    @classmethod
    def geo_dimension(cls, ctx: EntityContext) -> int:
        return int(ctx.block.positions.shape[1])
    
    @classmethod
    def normal(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        """Prism volume entities have no normal directions in 3D."""
        prism = cls._entity(ctx, index)
        GD = cls.geo_dimension(ctx)
        return bm.zeros((prism.shape[0], 0, GD), dtype=ctx.block.positions.dtype)

    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        """Tangent directions are not defined for prism volume entities."""
        prism = cls._entity(ctx, index)
        GD = cls.geo_dimension(ctx)
        return bm.zeros((prism.shape[0], 0, GD), dtype=ctx.block.positions.dtype)
    
    @classmethod
    def tangent(cls, ctx: EntityContext, index: Index | None = None) -> Tensor:
        """Compute tangent directions of prism entities.

        Return shape: (NC, 3, GD).
        """
        points = cls._points(ctx, index)
        return points[:, [1, 2, 3], :] - points[:, [0], :]

    # quadrature
    @classmethod
    def quadrature_formula(cls, q: int, qtype: str = "legendre"):
        from ...quadrature import (
            GaussLegendreQuadrature,
            TensorProductQuadrature,
            TriangleQuadrature,
        )

        qf0 = TriangleQuadrature(q)
        qf1 = GaussLegendreQuadrature(q)
        return TensorProductQuadrature((qf0, qf1))
    
    # shape function
    @classmethod
    def shape_function(cls, bcs: tuple[Tensor, Tensor], p: int = 1, *, index: Index | None = None,
                    variables: str = "u", mi: Tensor | None = None) -> Tensor:
        """Compute the shape function values on the reference prism.

        Parameters
            bcs : tuple[Tensor, Tensor]
                Tuple of barycentric coordinates on triangle and interval.
            p : int, default=1
                Polynomial degree.
            index : Index | None, optional
                Reserved for interface compatibility.
            variables : str, default='u'
                Variable space, either 'u' or 'x'.
            mi : Tensor | None, optional
                Multi-index matrix.

        Returns
            Tensor
                Shape function values.
                'u': (NQ, ldof).
                'x': (1, NQ, ldof).
        """
        raw_phi = [bm.simplex_shape_function(bc, p) for bc in bcs]
        phi = bm.tensorprod(*raw_phi)

        if variables == "u":
            return phi
        if variables == "x":
            return phi[None, ...]

        raise ValueError(f"Unsupported variables: {variables!r}")
    
    @classmethod
    def grad_shape_function(cls, ctx: EntityContext, bcs: tuple[Tensor, Tensor], p: int = 1, *,
                            index: Index | None = None, variables: str = "u",
                            mi: Tensor | None = None) -> Tensor:
        """Compute the gradient of shape functions with respect to reference variables.

        Parameters
            bcs : tuple[Tensor, Tensor]
                Tuple of barycentric coordinates on triangle and interval.
            p : int, default=1
                Polynomial degree.
            index : Index | None, optional
                Reserved for interface compatibility.
            variables : str, default='u'
                Variable space. Currently only 'u' is supported.
            mi : Tensor | None, optional
                Multi-index matrix.

        Returns
            Tensor
                'u': (NQ, ldof, 3).
        """
        Dlambda0 = bm.array([[-1, -1], [1, 0], [0, 1]], dtype=bcs[0].dtype)
        Dlambda1 = bm.array([[-1], [1]], dtype=bcs[1].dtype)

        phi0 = bm.simplex_shape_function(bcs[0], p)
        phi1 = bm.simplex_shape_function(bcs[1], p)

        R0 = bm.simplex_grad_shape_function(bcs[0], p)
        R1 = bm.simplex_grad_shape_function(bcs[1], p)

        gphi0 = bm.einsum("...ij,jn->...in", R0, Dlambda0)
        gphi1 = bm.einsum("...ij,jn->...in", R1, Dlambda1)

        n = len(bcs[0]) * len(bcs[1])
        gxy = gphi0[:, None, :, None, :] * phi1[None, :, None, :, None]
        gz = phi0[:, None, :, None, None] * gphi1[None, :, None, :, :]

        gphi = bm.concatenate([gxy, gz], axis=-1)
        gphi = gphi.reshape(n, (p + 1) * (p + 1) * (p + 2) // 2, 3)

        if variables == "u":
            return gphi

        if variables == "x":
            G, J = cls.first_fundamental_form(ctx, bcs, index=index, return_jacobi=True)
            G = bm.linalg.inv(G)
            return bm.einsum("cqkm,cqmn,qln->cqlk", J, G, gphi)

        raise ValueError(f"Unsupported variables: {variables!r}")
    
    # ipoint
    @classmethod
    def multi_index(cls, p: tuple[int, int]) -> Tensor:
        """Compute the multi-index matrix on reference prism.

        Return tensor-product multi-index of triangle and interval.
        """
        p0, p1 = p
        mi0 = InterpolationPoints.multi_index_matrix(p0, 3)
        mi1 = InterpolationPoints.multi_index_matrix(p1, 2)

        mi0 = bm.repeat(mi0[:, None, :], mi1.shape[0], axis=1)
        mi1 = bm.repeat(mi1[None, :, :], mi0.shape[0], axis=0)

        return bm.concatenate([mi0, mi1], axis=-1).reshape(-1, 5)
    
    @classmethod
    def bc_to_point(cls, ctx: EntityContext, bcs: tuple[Tensor, Tensor],
                    index: Index | None = None) -> Tensor:
        """Convert barycentric coordinates to Cartesian coordinates.

        x = sum_i phi_i x_i on the physical prism.
        """
        phi = cls.shape_function(bcs)
        points = cls._tp_points(ctx, index)
        return bm.einsum("cim,qi->cqm", points, phi)

    # jacobi
    @classmethod
    def jacobi_matrix(cls, ctx: EntityContext, bcs: tuple[Tensor, Tensor], index: Index | None = None,
                    etype: str = "cell", ftype=None, return_grad: bool = False) -> Tensor:
        """Compute the Jacobian matrix of the reference-to-physical prism map.

        For p = 1, x(eta, zeta, xi) = sum_i phi_i x_i, where
        phi = [(1-eta-zeta)(1-xi), (1-eta-zeta)xi, eta(1-xi), eta xi, zeta(1-xi), zeta xi].

        Parameters
            bcs : tuple[Tensor, Tensor]
                Tuple[(NQ0, 3), (NQ1, 2)], the integration points.
            index : Index | None, optional
                Cell index.
            etype : str, default='cell'
                Reserved for compatibility.
            ftype : optional
                Reserved for compatibility.
            return_grad : bool, default=False
                Whether to return reference gradients.

        Returns
            Tensor
                J: (NC, NQ, 3, GD).
                gphi: (NQ, 6, 3), if return_grad is True.
        """
        prism = ctx.sector.indices if index is None else ctx.sector.indices[index]
        node = ctx.block.positions

        gphi = cls.grad_shape_function(ctx, bcs, p=1, variables="u")
        points = node[prism[:, [0, 3, 1, 4, 2, 5]]]
        J = bm.einsum("cim,qin->cqmn", points, gphi)

        if return_grad:
            return J, gphi
        return J

    @classmethod
    def first_fundamental_form(cls, ctx: EntityContext, bcs: tuple[Tensor, Tensor], index: Index | None = None,
                            etype: str = "cell", ftype=None, return_jacobi: bool = False,
                            return_grad: bool = False):
        """Compute the first fundamental form of the Lagrange prism.

        G = J^T J, where J is the Jacobian matrix of the reference-to-physical map.
        """
        J, gphi = cls.jacobi_matrix(ctx, bcs, index=index, return_grad=True)
        TD = J.shape[-1]
        shape = J.shape[0:-2] + (TD, TD)
        data = [[0 for _ in range(TD)] for _ in range(TD)]

        for i in range(TD):
            data[i][i] = bm.einsum("...d,...d->...", J[..., i], J[..., i])
            for j in range(i + 1, TD):
                data[i][j] = bm.einsum("...d,...d->...", J[..., i], J[..., j])
                data[j][i] = data[i][j]

        data = [val.reshape(val.shape + (1,)) for row in data for val in row]
        G = bm.concatenate(data, axis=-1).reshape(shape)

        if not return_jacobi and not return_grad:
            return G
        if return_jacobi and not return_grad:
            return G, J
        if not return_jacobi and return_grad:
            return G, gphi
        return G, J, gphi
