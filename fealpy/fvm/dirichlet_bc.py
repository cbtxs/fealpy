"""Finite-volume Dirichlet boundary-condition algebra."""

from fealpy.backend import backend_manager as bm
from fealpy.decorator import variantmethod
from fealpy.sparse import spdiags

from .fvm_geometry import FVMGeometry, boundary_face_flag


class DirichletBC:
    """Apply prescribed boundary values to FVM matrices and RHS vectors.

    The class contains term-specific helpers because a Dirichlet value enters a
    finite-volume diffusion operator, convection boundary flux, and divergence
    block in different algebraic forms.  It only applies already-defined PDE
    boundary data to assembled algebraic systems; engineering boundary mapping
    and SIMPLE/PISO iteration rules live outside this class.
    """

    def __init__(
        self,
        mesh,
        gd,
        threshold=None,
        geometry=None,
        component=None,
        diffusion_method="over_relaxed",
        nonorthogonal_eps=0.05,
    ):
        """Store boundary value data for later algebraic application.

        Args:
            mesh: Computational mesh for the control-volume domain.
            gd: Callable returning prescribed values at physical points.
            threshold: Optional selector for applying values on part of the boundary.
            geometry: Optional reusable FVM geometry.  When omitted, the class
                constructs one for standalone Poisson/tests/debug calls.
            component: Optional component index for applying a vector boundary
                function to a scalar component system.
        """
        self.mesh = mesh
        self.gd = gd
        self.threshold = threshold
        self.geometry = geometry if geometry is not None else FVMGeometry(mesh)
        self.component = component
        self._static_boundary_patch = None
        if nonorthogonal_eps <= 0.0:
            raise ValueError("nonorthogonal_eps must be positive.")
        self.nonorthogonal_eps = float(nonorthogonal_eps)
        if diffusion_method not in self.diffusion_boundary_data:
            raise ValueError(
                f"unknown diffusion method: {diffusion_method!r}"
            )
        self.diffusion_method = diffusion_method
        self.diffusion_boundary_data.set(diffusion_method)

    def boundary_patch(self, geometry, threshold):
        """Return selected boundary faces and centers for this fixed mesh."""
        cacheable = geometry is self.geometry and threshold is self.threshold
        if cacheable and self._static_boundary_patch is not None:
            return self._static_boundary_patch

        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        boundary_points = geometry.face_center[boundary_faces]
        if threshold is not None:
            boundary_flag = boundary_face_flag(boundary_points, threshold)
            boundary_faces = boundary_faces[boundary_flag]
            boundary_points = boundary_points[boundary_flag]

        result = (boundary_faces, boundary_points)
        if cacheable:
            self._static_boundary_patch = result
        return result

    def system_components(self, b, A=None):
        """Return the number of component-major fields represented by ``b``."""
        NC = self.geometry.NC
        if A is not None and (A.shape[0] != A.shape[1] or A.shape[0] != b.shape[0]):
            raise ValueError("DirichletBC expects a square matrix matching the RHS size.")
        if b.shape[0] == NC:
            return 1
        if b.shape[0] % NC == 0:
            return b.shape[0] // NC
        raise ValueError("RHS size must be NC or an integer multiple of NC.")

    def matrix_components(self, A):
        """Return the number of component-major fields represented by ``A``."""
        NC = self.geometry.NC
        if A.shape[0] != A.shape[1]:
            raise ValueError("DirichletBC expects a square matrix.")
        if A.shape[0] == NC:
            return 1
        if A.shape[0] % NC == 0:
            return A.shape[0] // NC
        raise ValueError("matrix size must be NC or an integer multiple of NC.")

    def boundary_values(self, points, components: int):
        """Return boundary values compatible with the algebraic system shape."""
        value = bm.array(self.gd(points))
        if self.component is not None:
            if components != 1:
                raise ValueError("component can only be used with scalar component systems.")
            if value.ndim == 1:
                if self.component != 0:
                    raise ValueError("scalar boundary data only has component 0.")
                return value
            if value.ndim == 2:
                if self.component < 0 or self.component >= value.shape[1]:
                    raise ValueError(
                        f"component {self.component} is out of bounds for boundary data "
                        f"with {value.shape[1]} components."
                    )
                return value[:, self.component]
            raise ValueError(f"Unsupported boundary value shape {value.shape}.")

        if components == 1:
            if value.ndim == 1:
                return value
            if value.ndim == 2 and value.shape[1] == 1:
                return value[:, 0]
            raise ValueError(
                "scalar Dirichlet system received vector boundary data; "
                "pass component=<index> or a scalar boundary function."
            )

        if value.ndim != 2 or value.shape[1] != components:
            raise ValueError(
                "vector Dirichlet system expects boundary data with shape "
                f"(N, {components}), got {value.shape}."
            )
        return value

    @variantmethod("over_relaxed")
    def diffusion_boundary_data(self, coef=1.0, threshold=None, dtype=None):
        """Return owner cells, implicit coefficients, and points for Dirichlet faces."""
        geometry = self.geometry
        decomposition = geometry.diffusion_face_decomposition("over_relaxed")
        return self._diffusion_boundary_data_from_factor(
            geometry,
            decomposition.orthogonal_factor,
            coef=coef,
            threshold=threshold,
            dtype=dtype,
        )

    @diffusion_boundary_data.register("bounded_over_relaxed")
    def diffusion_boundary_data(self, coef=1.0, threshold=None, dtype=None):
        geometry = self.geometry
        decomposition = geometry.diffusion_face_decomposition(
            "bounded_over_relaxed", eps=self.nonorthogonal_eps
        )
        return self._diffusion_boundary_data_from_factor(
            geometry,
            decomposition.orthogonal_factor,
            coef=coef,
            threshold=threshold,
            dtype=dtype,
        )

    @diffusion_boundary_data.register("uncorrected")
    def diffusion_boundary_data(self, coef=1.0, threshold=None, dtype=None):
        geometry = self.geometry
        decomposition = geometry.diffusion_face_decomposition("uncorrected")
        return self._diffusion_boundary_data_from_factor(
            geometry,
            decomposition.orthogonal_factor,
            coef=coef,
            threshold=threshold,
            dtype=dtype,
        )

    def _diffusion_boundary_data_from_factor(
        self,
        geometry,
        orthogonal_factor,
        *,
        coef,
        threshold,
        dtype,
    ):
        threshold = self.threshold if threshold is None else threshold
        boundary_faces, boundary_points = self.boundary_patch(geometry, threshold)
        boundary_integrator = orthogonal_factor[boundary_faces]
        if not isinstance(coef, (int, float)):
            coef = bm.array(coef)
            if coef.shape == ():
                pass
            elif coef.ndim == 1 and coef.shape[0] == geometry.NF:
                coef = coef[boundary_faces]
            else:
                raise ValueError("coef must be scalar or face-wise.")
        boundary_integrator = coef * boundary_integrator
        if dtype is not None:
            boundary_integrator = bm.array(boundary_integrator, dtype=dtype)
        return geometry.owner[boundary_faces], boundary_integrator, boundary_points

    def apply_diffusion_matrix(self, A, coef=1.0, threshold=None):
        """Add the implicit Dirichlet diffusion diagonal to ``A``."""
        NC = self.geometry.NC
        components = self.matrix_components(A)
        dtype = getattr(getattr(A, "values", None), "dtype", None)
        boundary_owner, boundary_integrator, _ = self.diffusion_boundary_data(
            coef=coef,
            threshold=threshold,
            dtype=dtype,
        )
        boundary_diagonal = bm.zeros(NC, dtype=boundary_integrator.dtype)
        boundary_diagonal = bm.index_add(
            boundary_diagonal,
            boundary_owner,
            boundary_integrator,
            axis=0,
        )
        if components > 1:
            boundary_diagonal = bm.tile(boundary_diagonal, (components,))
        return A + spdiags(
            boundary_diagonal,
            0,
            A.shape[0],
            A.shape[1],
            index_dtype=A.itype,
        )

    def apply_diffusion_rhs(self, b, coef=1.0, threshold=None):
        """Add the explicit Dirichlet diffusion RHS contribution to ``b``."""
        NC = self.geometry.NC
        components = self.system_components(b)
        boundary_owner, boundary_integrator, boundary_points = self.diffusion_boundary_data(
            coef=coef,
            threshold=threshold,
            dtype=b.dtype,
        )
        boundary_value = self.boundary_values(boundary_points, components)
        if components == 1:
            boundary_rhs = boundary_integrator * boundary_value
            boundary_rhs = bm.array(boundary_rhs, dtype=b.dtype)
            return bm.index_add(b, boundary_owner, boundary_rhs, axis=0)

        boundary_rhs = boundary_integrator[:, None] * boundary_value
        boundary_rhs = bm.swapaxes(boundary_rhs, 0, 1).flatten()
        boundary_rhs = bm.array(boundary_rhs, dtype=b.dtype)
        indices = bm.concat(
            [boundary_owner + component * NC for component in range(components)]
        )
        return bm.index_add(b, indices, boundary_rhs, axis=0)

    def apply_diffusion(self, A, b, coef=1.0, threshold=None):
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
        self.system_components(b, A)
        return (
            self.apply_diffusion_matrix(A, coef=coef, threshold=threshold),
            self.apply_diffusion_rhs(b, coef=coef, threshold=threshold),
        )

    def apply_convection(self, b, coef, threshold=None):
        """
        Apply Dirichlet boundary values to a finite-volume convection RHS.

        The interior convection operator only assembles owner-neighbour face
        contributions. On boundary faces the prescribed value contributes the
        known flux ``-(coef_f · S_f) g_D`` to the owner cell RHS.
        """
        if coef is None:
            return b

        threshold = self.threshold if threshold is None else threshold
        geometry = self.geometry
        boundary_faces, boundary_points = self.boundary_patch(geometry, threshold)
        NC = geometry.NC
        components = self.system_components(b)
        Sf = geometry.S_f[boundary_faces]
        coef = bm.array(coef)
        if coef.ndim == 1 and coef.shape[0] == geometry.NF:
            flux = coef[boundary_faces]
        elif coef.ndim == 2 and coef.shape[0] == geometry.NF:
            flux = bm.einsum("ij,ij->i", coef[boundary_faces], Sf)
        else:
            raise ValueError("coef must be a face-wise scalar flux or vector face field.")

        boundary_owner = geometry.owner[boundary_faces]
        boundary_value = self.boundary_values(boundary_points, components)

        if components == 1:
            boundary_rhs = bm.array(flux * boundary_value, dtype=b.dtype)
            b = bm.index_add(b, boundary_owner, boundary_rhs, axis=0, alpha=-1)
            return b

        boundary_rhs = -flux[:, None] * boundary_value
        boundary_rhs = bm.array(boundary_rhs, dtype=b.dtype)
        indices = bm.concat(
            [boundary_owner + component * NC for component in range(components)]
        )
        b = bm.index_add(b, indices, bm.swapaxes(boundary_rhs, 0, 1).flatten(), axis=0)
        return b
