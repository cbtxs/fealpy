"""Finite-volume Dirichlet boundary-condition application helpers."""

from fealpy.backend import backend_manager as bm
from fealpy.sparse import spdiags

from .fvm_geometry import FVMGeometry, boundary_face_flag


class DirichletBC:
    """Apply prescribed boundary values to FVM matrices and RHS vectors.

    The class contains term-specific helpers because a Dirichlet value enters a
    finite-volume diffusion operator, convection boundary flux, and divergence
    block in different algebraic forms.  It is a boundary-condition application
    layer, not a general PDE solver and not a place for SIMPLE/PISO iteration
    rules.

    Future cleanup should keep the main ``DiffusionApply`` and
    ``ConvectionApply`` contracts intact.
    """

    def __init__(self, mesh, gd, threshold=None, geometry=None):
        """Store boundary value data for later algebraic application.

        Args:
            mesh: Computational mesh for the control-volume domain.
            gd: Callable returning prescribed values at physical points.
            threshold: Optional selector for applying values on part of the boundary.
        """
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold
        self.geometry = geometry

    def DiffusionApply(self, A, b, coef=1.0, threshold=None):
        """Add boundary-face Dirichlet contribution for diffusion operators.

        For a boundary face, the prescribed value contributes an implicit
        owner-cell diagonal term and a matching RHS term.  This is the standard
        FVM face-flux form for Dirichlet data, and it supports scalar and
        component-wise vector fields.

        Args:
            A (sparse matrix): System matrix to be modified.
            b (ndarray): Right-hand side vector to be modified.

        Returns:
            tuple: (A, b)
                - A (sparse matrix): Modified system matrix with boundary conditions applied.
                - b (ndarray): Modified right-hand side vector with boundary contributions.
        """
        NC = self.mesh.number_of_cells()
        geometry = self.geometry if self.geometry is not None else FVMGeometry(self.mesh)
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        _, Ef_abs, _ = geometry.over_relaxed_decomposition()
        boundary_integrator = Ef_abs[boundary_faces] / geometry.mag_d_f[boundary_faces]
        boundary_points = geometry.face_center[boundary_faces]
        if not isinstance(coef, (int, float)):
            coef = bm.array(coef)
            if coef.shape == ():
                pass
            elif coef.ndim == 1 and coef.shape[0] == self.mesh.number_of_faces():
                coef = coef[boundary_faces]
            else:
                raise ValueError("coef must be scalar or face-wise.")
        boundary_integrator = coef * boundary_integrator
        if threshold is not None:
            bd_flag = boundary_face_flag(boundary_points, threshold)
            boundary_faces = boundary_faces[bd_flag]
            boundary_integrator = boundary_integrator[bd_flag]
            boundary_points = boundary_points[bd_flag]
        boundary_integrator = bm.array(boundary_integrator, dtype=b.dtype)
        boundary_owner = geometry.owner[boundary_faces]
        # Scalar field: boundary_value shape (NE,), vector field: (NE, D).
        boundary_value = self.gd(boundary_points)[..., None]
        bdIdx = bm.zeros(NC, dtype=boundary_integrator.dtype)
        bdIdx = bm.index_add(bdIdx, boundary_owner, boundary_integrator, axis=0)
        D = boundary_value.shape[1]
        if D > 1:
            bdIdx = bm.tile(bdIdx, (D,))
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0
        if D == 1:
            bd_correct = (boundary_integrator[:, None] * boundary_value).reshape(-1)
            bd_correct = bm.array(bd_correct, dtype=b.dtype)
            b = bm.index_add(b, boundary_owner, bd_correct, axis=0)
        else:
            boundary_value = bm.squeeze(boundary_value, axis=-1)
            bd_correct = boundary_integrator[:, None] * boundary_value
            bd_correct = bm.swapaxes(bd_correct, 0, 1).flatten()
            bd_correct = bm.array(bd_correct, dtype=b.dtype)
            indices = bm.concat(
                [boundary_owner + component * NC for component in range(D)]
            )
            b = bm.index_add(b, indices, bd_correct, axis=0)
        return A, b

    def ConvectionApply(self, b, coef, threshold=None):
        """
        Apply Dirichlet boundary values to a finite-volume convection RHS.

        The interior convection operator only assembles owner-neighbour face
        contributions. On boundary faces the prescribed value contributes the
        known flux ``-(coef_f · S_f) g_D`` to the owner cell RHS.
        """
        if coef is None:
            return b

        geometry = self.geometry if self.geometry is not None else FVMGeometry(self.mesh)
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        NC = self.mesh.number_of_cells()
        Sf = geometry.S_f[boundary_faces]
        boundary_points = geometry.face_center[boundary_faces]
        coef = bm.array(coef)
        if coef.ndim == 1 and coef.shape[0] == self.mesh.number_of_faces():
            flux = coef[boundary_faces]
        elif coef.ndim == 2 and coef.shape[0] == self.mesh.number_of_faces():
            flux = bm.einsum("ij,ij->i", coef[boundary_faces], Sf)
        else:
            raise ValueError("coef must be a face-wise scalar flux or vector face field.")

        if threshold is not None:
            bd_flag = boundary_face_flag(boundary_points, threshold)
            boundary_faces = boundary_faces[bd_flag]
            boundary_points = boundary_points[bd_flag]
            flux = flux[bd_flag]

        boundary_owner = geometry.owner[boundary_faces]
        boundary_value = self.gd(boundary_points)

        if len(boundary_value.shape) == 1:
            bd_correct = bm.array(flux * boundary_value, dtype=b.dtype)
            b = bm.index_add(b, boundary_owner, bd_correct, axis=0, alpha=-1)
            return b

        bd_correct = -flux[:, None] * boundary_value
        bd_correct = bm.array(bd_correct, dtype=b.dtype)
        indices = bm.concat(
            [boundary_owner + component * NC for component in range(boundary_value.shape[1])]
        )
        b = bm.index_add(b, indices, bm.swapaxes(bd_correct, 0, 1).flatten(), axis=0)
        return b
