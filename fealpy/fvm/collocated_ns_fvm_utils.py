"""Shared internal operators for collocated Navier-Stokes FVM solvers."""

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm
from fealpy.sparse import COOTensor, CSRTensor, spdiags

from .convection_integrator import ConvectionIntegrator
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .scalar_cross_diffusion_integrator import ScalarCrossDiffusionIntegrator
from .scalar_source_integrator import ScalarSourceIntegrator
from .gradient_reconstruct import GradientReconstruct
from .face_gradient import reconstruct_face_gradient
from .div_reconstruct import DivergenceReconstruct
from .dirichlet_bc import DirichletBC
from .fvm_geometry import FVMGeometry
from .fvm_linear_solver import FVMLinearSolver, FVMLinearSolverConfig


class CollocatedNSFVMOperators:
    """Internal mixin for common collocated Navier-Stokes FVM algebra.

    Velocity fields use two shapes deliberately: physical operators take
    cell-vector values with shape ``(NC, GD)``, while assembled linear systems
    use component-major dof vectors with shape ``(GD*NC,)``.
    """

    @staticmethod
    def _as_positive_scalar(value, name: str) -> float:
        if callable(value):
            value = value()
        try:
            scalar = float(value)
        except TypeError:
            scalar = float(bm.to_numpy(value))
        if scalar <= 0.0:
            raise ValueError(f"{name} must be positive.")
        return scalar

    @staticmethod
    def _as_nonnegative_scalar(value, name: str) -> float:
        if callable(value):
            value = value()
        try:
            scalar = float(value)
        except TypeError:
            scalar = float(bm.to_numpy(value))
        if scalar < 0.0:
            raise ValueError(f"{name} must be non-negative.")
        return scalar

    def _init_collocated_discretization(
        self,
        degree: int,
        velocity_dirichlet,
        *,
        pressure_gradient_method: str = "extended_lsq",
        velocity_gradient_method: str = "extended_lsq",
        velocity_dirichlet_threshold=None,
        pressure_dirichlet=None,
        pressure_dirichlet_threshold=None,
        with_divergence: bool = False,
        with_velocity_dirichlet_bc: bool = False,
    ) -> None:
        self.p = degree
        self.GD = self.mesh.geo_dimension()
        self.space = ScaledMonomialSpace(self.mesh, degree)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(self.GD, -1))
        self.points = self.mesh.entity_barycenter("cell")
        self.epoints = self.mesh.entity_barycenter("face")

        self.pressure_gradient = GradientReconstruct(
            self.mesh,
            method=pressure_gradient_method,
            gd=pressure_dirichlet,
            bc_type="dirichlet" if pressure_dirichlet is not None else None,
            threshold=pressure_dirichlet_threshold,
        )
        self.velocity_gradient = GradientReconstruct(
            self.mesh,
            method=velocity_gradient_method,
            gd=velocity_dirichlet,
            bc_type="dirichlet",
            threshold=velocity_dirichlet_threshold,
        )
        self.fvm_geometry = FVMGeometry(self.mesh)
        self.velocity_dirichlet = velocity_dirichlet
        self.velocity_dirichlet_threshold = velocity_dirichlet_threshold
        if with_divergence:
            self.divergence = DivergenceReconstruct(self.mesh)
        if with_velocity_dirichlet_bc:
            self.velocity_dirichlet_bc = DirichletBC(
                self.mesh,
                velocity_dirichlet,
                threshold=velocity_dirichlet_threshold,
            )

        self.e2c = self.fvm_geometry.face_to_cell
        self.last_nonorthogonal_iterations = 0
        self.last_momentum_nonorthogonal_iterations = 0
        self.last_pressure_nonorthogonal_iterations = 0

    def cell_vector_to_dofs(self, cell_vector):
        """Return component-major algebraic dofs from ``(NC, GD)`` cell vectors."""
        return bm.swapaxes(cell_vector, 0, 1).flatten()

    def dofs_to_cell_vector(self, dofs):
        """Return ``(NC, GD)`` cell vectors from component-major algebraic dofs."""
        return bm.stack(
            [
                dofs[component * self.NC : (component + 1) * self.NC]
                for component in range(self.GD)
            ],
            axis=-1,
        )

    def component_cell_diagonal(self, diagonal):
        """Repeat a cell diagonal once per velocity component."""
        return bm.concatenate([diagonal for _ in range(self.GD)], axis=0)

    def component_response(self, a_p):
        """Return the cell response ``V/a_P`` as an ``(NC, GD)`` vector field."""
        return bm.stack(
            [
                self.cm
                / a_p[component * self.NC : (component + 1) * self.NC]
                for component in range(self.GD)
            ],
            axis=-1,
        )

    def _init_linear_solver(self, options):
        linear_solver = options.get("linear_solver")
        if linear_solver is not None and hasattr(linear_solver, "solve"):
            return linear_solver

        config = options.get("linear_solver_config")
        if isinstance(config, dict):
            config = FVMLinearSolverConfig(**config)
        if config is not None:
            return FVMLinearSolver(config)

        try:
            device = str(bm.get_device(self.cm))
        except Exception:
            device = "cpu"
        config = FVMLinearSolverConfig(
            backend=bm.backend_name,
            device=device,
            solver=linear_solver or "auto",
        )
        return FVMLinearSolver(config)

    def momentum_diffusion_matrix(self, diffusion_coef):
        """Assemble and cache the shared momentum diffusion matrix."""
        cache = getattr(self, "_momentum_diffusion_matrix_cache", None)
        if cache is None:
            cache = {}
            self._momentum_diffusion_matrix_cache = cache

        try:
            key = ("scalar", float(diffusion_coef))
        except TypeError:
            key = ("object", id(diffusion_coef))

        if key not in cache:
            cache[key] = BilinearForm(self.velocity_space).add_integrator(
                ScalarDiffusionIntegrator(q=self.p + 2, coef=diffusion_coef)
            ).assembly()
        return cache[key]

    def momentum_convection_matrix(self, convection_face_velocity, interpolation: str):
        """Assemble the momentum convection matrix for the active face velocity."""
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(
            ConvectionIntegrator(
                q=self.p + 2,
                coef=convection_face_velocity,
                interpolation=interpolation,
            )
        )
        return bform.assembly()

    def momentum_time_matrix(self, density, time_step):
        """Return the implicit backward-Euler momentum time matrix.

        For P0 finite volumes this is the diagonal contribution
        ``rho * V_C / dt`` repeated for each velocity component.
        """
        time_step = float(time_step)
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")

        density = bm.array(density, dtype=self.cm.dtype)
        if density.shape == ():
            cell_diagonal = density * self.cm / time_step
        else:
            if density.shape != self.cm.shape:
                raise ValueError("density must be scalar or cell-wise.")
            cell_diagonal = density * self.cm / time_step

        total_dofs = self.GD * self.NC
        values = self.component_cell_diagonal(cell_diagonal)
        return CSRTensor(
            crow=bm.arange(total_dofs + 1),
            col=bm.arange(total_dofs),
            values=values,
            spshape=(total_dofs, total_dofs),
        )

    def momentum_time_source(self, previous_velocity, density, time_step):
        """Return the old-time RHS contribution ``rho * V_C / dt * U^n``."""
        time_step = float(time_step)
        if time_step <= 0.0:
            raise ValueError("time_step must be positive.")

        density = bm.array(density, dtype=self.cm.dtype)
        if density.shape == ():
            cell_diagonal = density * self.cm / time_step
        else:
            if density.shape != self.cm.shape:
                raise ValueError("density must be scalar or cell-wise.")
            cell_diagonal = density * self.cm / time_step

        return self.cell_vector_to_dofs(previous_velocity * cell_diagonal[:, None])

    def add_velocity_natural_convection_diagonal(
        self,
        matrix,
        convection_face_velocity,
        threshold,
    ):
        """Add natural-boundary convection fluxes to the owner diagonal."""
        if threshold is None:
            return matrix

        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[boundary_faces]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return matrix

        selected = boundary_faces[flag]
        flux = bm.einsum(
            "ij,ij->i",
            convection_face_velocity[selected],
            self.fvm_geometry.S_f[selected],
        )
        owner = self.fvm_geometry.owner[selected]
        diagonal = bm.zeros(self.NC, dtype=flux.dtype)
        diagonal = bm.index_add(diagonal, owner, flux, axis=0)
        components = matrix.shape[0] // self.NC
        diagonal = bm.concatenate([diagonal for _ in range(components)], axis=0)
        return matrix + spdiags(diagonal, 0, matrix.shape[0], matrix.shape[1])

    def momentum_source_vector(self, source):
        """Assemble the shared cell-integrated momentum source vector."""
        return LinearForm(self.velocity_space).add_integrator(
            ScalarSourceIntegrator(source, q=self.p + 2)
        ).assembly()

    def pressure_gradient_source(self, pressure):
        """Return the cell-integrated pressure-gradient source vector."""
        grad_p = self.pressure_gradient.cell_gradient(pressure)
        return bm.concatenate(
            [
                bm.einsum("i,i->i", grad_p[:, component], self.cm)
                for component in range(grad_p.shape[1])
            ]
        )

    def pressure_response_face_coefficient(self, a_p, interpolation_method=None):
        """Interpolate cell pressure response ``V/a_P`` to faces."""
        if interpolation_method is None:
            interpolation_method = getattr(self, "face_interpolation_method", None)
        if interpolation_method is None:
            controls = getattr(self, "controls", None)
            if controls is not None and hasattr(controls, "face_interpolation"):
                interpolation_method = controls.face_interpolation(
                    "pressure_response_interpolation"
                )
            elif controls is not None and hasattr(controls, "face_interpolation_method"):
                interpolation_method = controls.face_interpolation_method
            else:
                interpolation_method = "linear"

        response = self.cm / a_p[: self.NC]
        return self.face_interpolate_cell_scalar(
            response,
            method=interpolation_method,
        )

    def pressure_orthogonal_flux(self, pressure, response_coef):
        """Return the implicit orthogonal pressure-Laplacian flux."""
        _, mag_E_f, _ = self.fvm_geometry.over_relaxed_decomposition()
        coefficient = response_coef * mag_E_f / self.fvm_geometry.mag_d_f
        jump = pressure[self.e2c[:, 0]] - pressure[self.e2c[:, 1]]
        return coefficient * jump

    def add_pressure_dirichlet_flux(
        self,
        flux,
        pressure,
        response_coef,
        boundary_value,
        threshold,
    ):
        """Set pressure-induced fluxes on pressure Dirichlet boundary faces."""
        if threshold is None or boundary_value is None:
            return flux

        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[boundary_faces]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return flux

        selected = boundary_faces[flag]
        _, mag_E_f, _ = self.fvm_geometry.over_relaxed_decomposition()
        coefficient = (
            response_coef[selected]
            * mag_E_f[selected]
            / self.fvm_geometry.mag_d_f[selected]
        )
        owner = self.fvm_geometry.owner[selected]
        bd_value = boundary_value(face_centers[flag])
        bd_flux = coefficient * (pressure[owner] - bd_value)
        return bm.set_at(flux, selected, bd_flux)

    def compute_cross_diffusion(self, velocity: TensorLike) -> TensorLike:
        """Assemble the explicit non-orthogonal momentum diffusion correction."""
        grad_u = self.velocity_gradient.cell_gradient(velocity)
        grad_f = reconstruct_face_gradient(self.mesh, grad_u)
        return LinearForm(self.velocity_space).add_integrator(
            ScalarCrossDiffusionIntegrator(
                grad_f=grad_f,
                coef=getattr(self, "diffusion_coef", getattr(self, "mu", 1.0)),
                geometry=self.fvm_geometry,
                boundary_policy="all",
            )
        ).assembly()

    def face_interpolate_cell_scalar(self, cell_values, method: str = "linear"):
        """Linearly interpolate a cell scalar to faces using face geometry."""
        e2c = self.e2c[:, :2]
        if method == "linear":
            owner_weight = self.fvm_geometry.linear_owner_weight()
        elif method == "average":
            owner_weight = 0.5 * bm.ones_like(self.fvm_geometry.mag_S_f)
            owner_weight = bm.where(self.fvm_geometry.is_internal, owner_weight, 1.0)
        else:
            raise ValueError("method must be 'average' or 'linear'.")
        return owner_weight * cell_values[e2c[:, 0]] + (1.0 - owner_weight) * cell_values[e2c[:, 1]]

    def face_interpolate_cell_vector(self, cell_vectors, method: str = "linear"):
        """Linearly interpolate a cell vector to faces using face geometry."""
        e2c = self.e2c[:, :2]
        if method == "linear":
            owner_weight = self.fvm_geometry.linear_owner_weight()
        elif method == "average":
            owner_weight = 0.5 * bm.ones_like(self.fvm_geometry.mag_S_f)
            owner_weight = bm.where(self.fvm_geometry.is_internal, owner_weight, 1.0)
        else:
            raise ValueError("method must be 'average' or 'linear'.")
        return (
            owner_weight[:, None] * cell_vectors[e2c[:, 0]]
            + (1.0 - owner_weight)[:, None] * cell_vectors[e2c[:, 1]]
        )

    def face_flux(self, face_velocity):
        """Return the signed surface flux ``phi_f = u_f dot S_f``."""
        return bm.einsum("ij,ij->i", face_velocity, self.fvm_geometry.S_f)

    def divergence_from_flux(self, phi):
        """Scatter signed face fluxes to the cell flux imbalance."""
        return self.fvm_geometry.scatter_face_flux_to_cells(phi)

    def _pressure_nonorthogonal_cross_flux(
        self,
        pressure: TensorLike,
        response_coef: TensorLike,
    ) -> TensorLike:
        """Return the explicit non-orthogonal flux induced by a pressure field.

        The face-gradient interpolation follows the active face-interpolation
        setting used by the collocated pressure-correction route.  Pressure
        boundary cross flux remains zero on non-coupled boundary faces,
        matching the pressure Laplacian correction route.
        """
        grad_p = self.pressure_gradient.cell_gradient(pressure)
        controls = getattr(self, "controls", None)
        face_method = getattr(
            self,
            "face_interpolation_method",
            getattr(controls, "face_interpolation_method", "average"),
        )
        grad_f = reconstruct_face_gradient(self.mesh, grad_p, interpolation_method=face_method)
        T_f = self.fvm_geometry.bounded_over_relaxed_decomposition()[2]
        cross_flux = response_coef * bm.einsum("ij,ij->i", T_f, grad_f)
        return bm.where(self.fvm_geometry.is_boundary, 0.0, cross_flux)

    def _assemble_pressure_gauge_matrix(self, coef, q: int):
        A = BilinearForm(self.space).add_integrator(
            ScalarDiffusionIntegrator(q=q, coef=coef)
        ).assembly()
        gauge_index = bm.stack(
            [
                bm.zeros(self.NC, dtype=bm.int32),
                bm.arange(self.NC, dtype=bm.int32),
            ],
            axis=0,
        )
        A1 = COOTensor(gauge_index, self.cm, spshape=(1, self.NC))
        A = BlockForm([[A, A1.T], [A1, None]])
        return A.assembly_sparse_matrix(format="csr")

    def _boundary_face_coefficient(self, coef):
        """Return boundary-face coefficients from scalar or face-wise data."""
        if isinstance(coef, (int, float)):
            return coef

        coef = bm.array(coef)
        if coef.shape == ():
            return coef

        boundary_faces = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        if coef.shape[0] == self.mesh.number_of_faces():
            return coef[boundary_faces]
        if coef.shape[0] == boundary_faces.shape[0]:
            return coef
        return coef

    def enforce_face_flux(self, face_velocity, target_flux):
        """Adjust only the normal component of a vector face velocity."""
        Sf = self.fvm_geometry.S_f
        current_flux = self.face_flux(face_velocity)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        return face_velocity + ((target_flux - current_flux) / Sf_dot_Sf)[:, None] * Sf

    def velocity_pressure_correction(self, cell_velocity, pressure_field, a_p):
        """Apply ``U <- U - rAU grad(p)`` for the supplied pressure argument.

        In SIMPLE callers this argument is a pressure correction ``p'``.  In
        the current PISO route it is the corrected pressure state, not an
        increment.
        """
        grad_p = self.pressure_gradient.cell_gradient(pressure_field)
        return cell_velocity - self.component_response(a_p) * grad_p

    def pressure_free_velocity(self, cell_velocity, pressure, a_p):
        """Remove the current pressure-gradient contribution from velocity."""
        grad_p = self.pressure_gradient.cell_gradient(pressure)
        return cell_velocity + self.component_response(a_p) * grad_p
