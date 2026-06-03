"""Finite-volume Dirichlet boundary-condition application helpers."""

from inspect import signature

from fealpy.backend import backend_manager as bm
from fealpy.sparse import spdiags

from .backend_utils import as_backend_array, cast_like
from .fvm_geometry import FVMGeometry


class DirichletBC:
    """Apply prescribed boundary values to FVM matrices and RHS vectors.

    The class contains term-specific helpers because a Dirichlet value enters a
    finite-volume diffusion operator, convection boundary flux, and divergence
    block in different algebraic forms.  It is a boundary-condition application
    layer, not a general PDE solver and not a place for SIMPLE/PISO iteration
    rules.

    Future cleanup should keep the main ``DiffusionApply`` and
    ``ConvectionApply`` contracts intact, while moving shared boundary selector
    logic out of this class and retiring legacy value-pinning paths after the
    old staggered solvers are refactored.
    """

    def __init__(self, mesh, gd, threshold=None):
        """Store boundary value data for later algebraic application.

        Args:
            mesh: Computational mesh for the control-volume domain.
            gd: Callable returning prescribed values at physical points.
            threshold: Optional selector for applying values on part of the boundary.
        """
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold

    def ThresholdApply(self, A, f, uh=None):
        """Pin selected boundary-cell unknowns to prescribed values.

        This helper is useful for cell-centred unknowns stored directly on
        boundary cells.  It is not the face-flux Dirichlet treatment used by
        diffusion operators; that contract is handled by ``DiffusionApply``.
        This path is kept for legacy staggered solvers and can be revisited
        when those models are cleaned up.

        Args:
            A (sparse matrix): System matrix to be modified.
            f (ndarray): Right-hand side vector to be modified.
            uh (ndarray, optional): Solution vector to store boundary values. If None, initialized as zeros.

        Returns:
            tuple: (A, f)
                - A (sparse matrix): Modified system matrix with boundary conditions applied.
                - f (ndarray): Modified right-hand side vector with boundary contributions.

        Raises:
            ValueError: If threshold is not a callable function.
        """
        total_bd_idx = self.mesh.boundary_cell_index()
        points = self.mesh.entity_barycenter('cell')
        NC = self.mesh.number_of_cells()
        bd_node = points[total_bd_idx]
        if callable(self.threshold):
            try:
                # Try applying condition to x-coordinate only
                x = bd_node[:, 0]
                bd_idx = self.threshold(x)
                bd_idx = as_backend_array(bd_idx, dtype=bm.bool)
                if not bm.any(bd_idx):  # Check if bd_idx is all False
                    y = bd_node[:, 1]
                    bd_idx = self.threshold(y)
                    bd_idx = as_backend_array(bd_idx, dtype=bm.bool)
            except Exception:
                # Fall back to applying condition to full node coordinates
                bd_idx = self.threshold(bd_node)
                bd_idx = as_backend_array(bd_idx, dtype=bm.bool)
        else:
            raise ValueError("self.threshold must be a callable (e.g., lambda x: (x==0.5)|(x==2.5) or a function).")
        index = total_bd_idx[bd_idx]
        bdFlag_u = bm.zeros(NC, dtype=getattr(f, "dtype", None))
        bdFlag_u = bm.set_at(bdFlag_u, index, 1)
        D0 = spdiags(1 - bdFlag_u, 0, A.shape[0], A.shape[0])  # Keeps interior equations
        D1 = spdiags(bdFlag_u, 0, A.shape[0], A.shape[0])      # Identity on boundary nodes
        # Apply boundary conditions to the matrix
        if uh is None:
            # Initialize uh as a zero vector if not provided
            if hasattr(A, 'values_context'):
                uh = bm.zeros(A.shape[0], **A.values_context())
            else:
                uh = bm.zeros(A.shape[0], dtype=A.dtype)
        uh = bm.set_at(uh, index, self.gd(points[index]))
        f = f - A @ uh
        f = bm.set_at(f, index, uh[index])
        A = D0.matmul(A.matmul(D0)) + D1
        return A, f

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
        geometry = FVMGeometry(self.mesh)
        bd_edge = bm.nonzero(geometry.is_boundary)[0]
        _, Ef_abs, _ = geometry.over_relaxed_decomposition()
        bd_integrator = Ef_abs[bd_edge] / geometry.mag_d_f[bd_edge]
        bdedgepoint = geometry.face_center[bd_edge]
        if threshold is not None:
            bd_flag = self._boundary_face_flag(bdedgepoint, threshold)
            bd_edge = bd_edge[bd_flag]
            bd_integrator = bd_integrator[bd_flag]
            bdedgepoint = bdedgepoint[bd_flag]
            coef = self._select_boundary_coef(coef, bd_flag)
        coef = self._normalize_boundary_coef(coef, bd_integrator.shape[0])
        bd_integrator = coef * bd_integrator
        bd_integrator = cast_like(bd_integrator, b)
        bde2c = geometry.owner[bd_edge]
        # Scalar field: bd_u shape (NE,), vector field: (NE, D).
        bd_u = self.gd(bdedgepoint)[..., None]
        bdIdx = bm.zeros(NC, dtype=bd_integrator.dtype)
        bdIdx = bm.index_add(bdIdx, bde2c, bd_integrator, axis=0)
        D = bd_u.shape[1]
        if D > 1:
            bdIdx = bm.tile(bdIdx, (D,))
        A_0 = spdiags(bdIdx, 0, A.shape[0], A.shape[1])
        A = A + A_0
        if D == 1:
            bd_correct = (bd_integrator[:, None] * bd_u).reshape(-1)
            bd_correct = cast_like(bd_correct, b)
            b = bm.index_add(b, bde2c, bd_correct, axis=0)
        else:
            bd_u = bm.squeeze(bd_u, axis=-1)
            bd_correct = bd_integrator[:, None] * bd_u
            bd_correct = bm.swapaxes(bd_correct, 0, 1).flatten()
            bd_correct = cast_like(bd_correct, b)
            new_arr = bde2c + NC
            bde2c = bm.concat([bde2c, new_arr])
            b = bm.index_add(b, bde2c, bd_correct, axis=0)
        return A, b

    def _boundary_face_flag(self, points, threshold):
        """
        Evaluate a face-based threshold on boundary face centers.

        The threshold convention follows FEALPy's coordinate-selector style:
        ``lambda x: ...`` receives the x-coordinate of boundary face centers,
        ``lambda y: ...`` receives the y-coordinate, and a non-axis name such
        as ``lambda p: ...`` receives the full point array.

        This selector logic is not Dirichlet-specific.  If Neumann and gradient
        reconstruction boundary handling are refactored together, this should
        become a small shared boundary-selection utility.
        """
        if not callable(threshold):
            raise ValueError("threshold must be a callable boundary face selector.")

        axis = self._threshold_coordinate_axis(threshold)
        if axis is not None:
            if axis >= points.shape[1]:
                raise ValueError(
                    f"threshold requests coordinate axis {axis}, "
                    f"but boundary face centers have dimension {points.shape[1]}."
                )
            return self._validate_boundary_face_flag(
                threshold(points[:, axis]), points.shape[0]
            )

        return self._validate_boundary_face_flag(threshold(points), points.shape[0])

    def _threshold_coordinate_axis(self, threshold):
        """
        Infer whether a threshold wants a single coordinate component.

        This keeps ``DiffusionApply`` compatible with calls such as
        ``threshold=lambda x: ...`` without adding a separate axis argument.
        Unknown parameter names are treated as full-point thresholds.
        """
        try:
            params = list(signature(threshold).parameters.values())
        except (TypeError, ValueError):
            return None

        positional = [
            p for p in params
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) != 1:
            return None

        return {"x": 0, "y": 1, "z": 2}.get(positional[0].name)

    def _validate_boundary_face_flag(self, flag, n_boundary_face):
        flag = as_backend_array(flag, dtype=bm.bool)
        if flag.shape == (n_boundary_face,):
            return flag
        raise ValueError(
            "threshold must return a boolean array with one entry per boundary face."
        )

    def _select_boundary_coef(self, coef, bd_flag):
        # This and ``_normalize_boundary_coef`` are only a two-stage adapter for
        # thresholded ``DiffusionApply`` calls.  Once the boundary coefficient
        # contract is stable, they can be folded into one clearer normalization
        # path or moved to a shared boundary-coefficient helper.
        if isinstance(coef, (int, float)):
            return coef

        coef = as_backend_array(coef)
        if coef.shape == ():
            return coef
        n_boundary_face = bd_flag.shape[0]
        n_selected_face = int(bm.to_numpy(bm.sum(bd_flag)))
        if coef.shape[0] == n_boundary_face:
            return coef[bd_flag]
        if coef.shape[0] == n_selected_face:
            return coef

        raise ValueError(
            "coef must be scalar, selected-boundary-face-wise, or boundary-face-wise."
        )

    def _normalize_boundary_coef(self, coef, n_boundary_face):
        if isinstance(coef, (int, float)):
            return coef

        coef = as_backend_array(coef)
        if coef.shape == ():
            return coef
        if coef.shape[0] == n_boundary_face:
            return coef

        raise ValueError(
            "coef must be scalar or have one entry per selected boundary face."
        )

    def DivApply(self, b):
        """
        Apply Dirichlet boundary values to a 2D divergence RHS.

        This method modifies the right-hand side vector `b` to account for Dirichlet boundary 
        conditions in the divergence term, incorporating boundary face contributions and 
        vector field normals.

        This helper is not used by the current collocated SIMPLE/PISO paths.
        Keep it until the divergence-boundary contract is reviewed, then either
        connect it to a real solver path or move it out of the main boundary
        operator.

        Args:
            b (ndarray): Right-hand side vector to be modified.

        Returns:
            ndarray: Modified right-hand side vector with boundary contributions.
        """
        NC = self.mesh.number_of_cells()
        geometry = FVMGeometry(self.mesh)
        bd_edge = bm.nonzero(geometry.is_boundary)[0]
        bdedgepoint = geometry.face_center[bd_edge]
        bdSf = geometry.S_f[bd_edge]
        bde2c = geometry.owner[bd_edge]
        # Current FVM Navier-Stokes paths store 2D vector fields as [u, v].
        bd_u = self.gd(bdedgepoint)
        bd_correct = bd_u * bdSf
        bd_correct = bm.swapaxes(bd_correct, 0, 1).flatten()
        bd_correct = cast_like(bd_correct, b)
        new_arr = bde2c + NC
        bde2c = bm.concat([bde2c, new_arr])
        b = bm.index_add(b, bde2c, bd_correct, axis=0, alpha=-1)
        return b

    def ConvectionApply(self, b, coef, threshold=None):
        """
        Apply Dirichlet boundary values to a finite-volume convection RHS.

        The interior convection operator only assembles owner-neighbour face
        contributions. On boundary faces the prescribed value contributes the
        known flux ``-(coef_f · S_f) g_D`` to the owner cell RHS.
        """
        if coef is None:
            return b

        geometry = FVMGeometry(self.mesh)
        bd_edge = bm.nonzero(geometry.is_boundary)[0]
        NC = self.mesh.number_of_cells()
        Sf = geometry.S_f[bd_edge]
        bdedgepoint = geometry.face_center[bd_edge]
        flux = self._boundary_convection_flux(coef, bd_edge, Sf)

        if threshold is not None:
            bd_flag = self._boundary_face_flag(bdedgepoint, threshold)
            bd_edge = bd_edge[bd_flag]
            bdedgepoint = bdedgepoint[bd_flag]
            flux = flux[bd_flag]

        bde2c = geometry.owner[bd_edge]
        bd_value = self.gd(bdedgepoint)

        if len(bd_value.shape) == 1:
            bd_correct = cast_like(flux * bd_value, b)
            b = bm.index_add(b, bde2c, bd_correct, axis=0, alpha=-1)
            return b

        bd_correct = -flux[:, None] * bd_value
        bd_correct = cast_like(bd_correct, b)
        indices = bm.concat(
            [bde2c + component * NC for component in range(bd_value.shape[1])]
        )
        b = bm.index_add(b, indices, bm.swapaxes(bd_correct, 0, 1).flatten(), axis=0)
        return b

    def _boundary_convection_flux(self, coef, bd_edge, Sf):
        coef = as_backend_array(coef)
        NF = self.mesh.number_of_faces()
        NBD = bd_edge.shape[0]

        if len(coef.shape) == 1:
            if coef.shape[0] == NF:
                return coef[bd_edge]
            if coef.shape[0] == NBD:
                return coef
            raise ValueError(
                "coef must be face-wise or boundary-face-wise when it is scalar."
            )

        if len(coef.shape) == 2:
            if coef.shape[0] == NF:
                boundary_coef = coef[bd_edge]
            elif coef.shape[0] == NBD:
                boundary_coef = coef
            else:
                raise ValueError(
                    "coef must be face-wise or boundary-face-wise when it is vector-valued."
                )
            return bm.einsum("ij,ij->i", boundary_coef, Sf)

        raise ValueError("coef must be a scalar face flux or a vector face field.")
