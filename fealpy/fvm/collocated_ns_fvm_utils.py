"""Shared internal operators for collocated Navier-Stokes FVM solvers."""

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm
from fealpy.sparse import COOTensor

from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .scalar_cross_diffusion_integrator import ScalarCrossDiffusionIntegrator
from .gradient_reconstruct import GradientReconstruct
from .face_gradient import reconstruct_face_gradient
from .div_reconstruct import DivergenceReconstruct
from .dirichlet_bc import DirichletBC
from .fvm_geometry import FVMGeometry, face_interpolation_owner_weight
from .fvm_linear_solver import FVMLinearSolver, FVMLinearSolverConfig


class CollocatedNSFVMOperators:
    """Internal mixin for common collocated Navier-Stokes FVM algebra."""

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
        self.space = ScaledMonomialSpace2d(self.mesh, degree)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))
        self.points = self.mesh.entity_barycenter("cell")
        self.epoints = self.mesh.entity_barycenter("edge")

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
        self.edge_measure = self.mesh.entity_measure("edge")
        self.last_nonorthogonal_iterations = 0
        self.last_momentum_nonorthogonal_iterations = 0
        self.last_pressure_nonorthogonal_iterations = 0

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

    def _cell_velocity(self, velocity: TensorLike) -> TensorLike:
        if velocity.ndim == 1:
            return bm.stack([velocity[:self.NC], velocity[self.NC:]], axis=-1)
        return velocity

    @staticmethod
    def _flatten_velocity(cell_velocity: TensorLike) -> TensorLike:
        return cell_velocity.flatten(order="F")

    def compute_cross_diffusion(self, velocity: TensorLike) -> TensorLike:
        """Assemble the explicit non-orthogonal momentum diffusion correction."""
        cell_velocity = self._cell_velocity(velocity)
        flat_velocity = velocity if velocity.ndim == 1 else self._flatten_velocity(velocity)
        grad_u = self.velocity_gradient.cell_gradient(cell_velocity)
        grad_f = reconstruct_face_gradient(self.mesh, grad_u)
        return LinearForm(self.velocity_space).add_integrator(
            ScalarCrossDiffusionIntegrator(
                flat_velocity,
                grad_f,
                coef=getattr(self, "diffusion_coef", getattr(self, "mu", 1.0)),
                geometry=self.fvm_geometry,
                boundary_policy="all",
            )
        ).assembly()

    def _correct_momentum_nonorthogonal_diffusion(
        self,
        matrix,
        rhs,
        velocity,
        previous_velocity,
        *,
        max_iter: int,
        tol: float,
        iteration_attr: str,
    ):
        if max_iter == 0:
            setattr(self, iteration_attr, 0)
            return velocity

        previous_velocity = (
            self._flatten_velocity(previous_velocity)
            if previous_velocity.ndim == 2
            else previous_velocity
        )
        cross = self.compute_cross_diffusion(previous_velocity)
        corrected_velocity = velocity
        setattr(self, iteration_attr, 0)
        for iteration in range(1, max_iter + 1):
            next_velocity = self.linear_solver.solve(matrix, rhs + cross)
            setattr(self, iteration_attr, iteration)
            if bm.max(bm.abs(next_velocity - corrected_velocity)) < tol:
                return next_velocity
            corrected_velocity = next_velocity
            cross = self.compute_cross_diffusion(corrected_velocity)
        return corrected_velocity

    def _boundary_face_velocity(self):
        bd_edge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        return bd_edge, self.velocity_dirichlet(self.fvm_geometry.face_center[bd_edge])

    def face_interpolation_owner_weight(self, method: str = "linear"):
        """Return owner-side interpolation weights for faces."""
        return face_interpolation_owner_weight(self.mesh, method=method)

    def face_interpolate_cell_scalar(self, cell_values, method: str = "linear"):
        """Linearly interpolate a cell scalar to faces using face geometry."""
        e2c = self.e2c[:, :2]
        owner_weight = self.face_interpolation_owner_weight(method=method)
        return owner_weight * cell_values[e2c[:, 0]] + (1.0 - owner_weight) * cell_values[e2c[:, 1]]

    def face_interpolate_cell_vector(self, cell_vectors, method: str = "linear"):
        """Linearly interpolate a cell vector to faces using face geometry."""
        cell_vectors = self._cell_velocity(cell_vectors)
        e2c = self.e2c[:, :2]
        owner_weight = self.face_interpolation_owner_weight(method=method)
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

    def _pressure_correction_cross_flux(
        self,
        pressure: TensorLike,
        response_coef: TensorLike,
    ) -> TensorLike:
        """Return the explicit non-orthogonal flux induced by a pressure field.

        For OpenFOAM-aligned PISO diagnostics this uses the same internal
        face-gradient interpolation semantic as the momentum explicit source:
        ``face_interpolation_method="linear"`` selects OpenFOAM-style linear
        interpolation.  Pressure boundary cross flux remains zero on non-coupled
        boundary faces, matching the pressure Laplacian correction route.
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

    def _solve_pressure_correction_with_cross_rhs(
        self,
        rhs,
        coef,
        *,
        q: int,
        nonorthogonal_max_iter: int,
        nonorthogonal_tol: float,
        cross_flux,
        dirichlet_value=None,
        dirichlet_threshold=None,
    ):
        if nonorthogonal_max_iter < 0:
            raise ValueError("nonorthogonal_max_iter must be non-negative.")
        if nonorthogonal_tol <= 0.0:
            raise ValueError("nonorthogonal_tol must be positive.")

        has_dirichlet = dirichlet_value is not None and dirichlet_threshold is not None
        if has_dirichlet:
            A = BilinearForm(self.space).add_integrator(ScalarDiffusionIntegrator(q=q, coef=coef)).assembly()
            boundary_coef = self._boundary_face_coefficient(coef)
            pressure_bc = DirichletBC(self.mesh, dirichlet_value)
        else:
            A = self._assemble_pressure_gauge_matrix(coef, q=q)
            b0 = bm.array([0])

        def solve_with_cross_rhs(cross):
            b = rhs + cross
            if has_dirichlet:
                A_bc, b_bc = pressure_bc.DiffusionApply(A, b, coef=boundary_coef, threshold=dirichlet_threshold)
                return self.linear_solver.solve(A_bc, b_bc)

            b = bm.concatenate([b, b0], axis=0)
            return self.linear_solver.solve(A, b)[:-1]

        cross_rhs = bm.zeros_like(rhs)
        if nonorthogonal_max_iter == 0:
            self.last_pressure_nonorthogonal_iterations = 0
            return solve_with_cross_rhs(cross_rhs)

        self.last_pressure_nonorthogonal_iterations = 0
        for iteration in range(1, nonorthogonal_max_iter + 1):
            next_pressure = solve_with_cross_rhs(cross_rhs)
            next_cross_rhs = self.divergence_from_flux(cross_flux(next_pressure, coef))
            self.last_pressure_nonorthogonal_iterations = iteration
            if bm.max(bm.abs(next_cross_rhs - cross_rhs)) < nonorthogonal_tol:
                return next_pressure
            cross_rhs = next_cross_rhs
        return next_pressure

    def enforce_face_flux(self, face_velocity, target_flux):
        """Adjust only the normal component of a vector face velocity."""
        Sf = self.fvm_geometry.S_f
        current_flux = self.face_flux(face_velocity)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        return face_velocity + ((target_flux - current_flux) / Sf_dot_Sf)[:, None] * Sf

    def apply_face_velocity_dirichlet(self, face_velocity, boundary_velocity=None):
        """Apply externally supplied velocity Dirichlet data on boundary faces."""
        if boundary_velocity is None:
            return face_velocity

        face_velocity = bm.array(face_velocity)
        bd_edge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        boundary_velocity = bm.array(boundary_velocity)
        if boundary_velocity.shape[0] == self.mesh.number_of_faces():
            boundary_velocity = boundary_velocity[bd_edge]
        face_velocity[bd_edge] = boundary_velocity
        return face_velocity

    def apply_boundary_flux_constraint(self, flux, boundary_velocity=None):
        """Set boundary scalar fluxes to the prescribed boundary velocity flux."""
        if boundary_velocity is None:
            return flux

        constrained = bm.array(flux)
        bd_edge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        boundary_velocity = bm.array(boundary_velocity)
        if boundary_velocity.shape[0] == self.mesh.number_of_faces():
            boundary_velocity = boundary_velocity[bd_edge]
        target_flux = bm.einsum("ij,ij->i", boundary_velocity, self.fvm_geometry.S_f[bd_edge])
        return bm.set_at(constrained, bd_edge, target_flux)

    def velocity_pressure_correction(self, u_flat, pressure_field, a_p):
        """Apply ``U <- U - rAU grad(p)`` for the supplied pressure argument.

        In SIMPLE callers this argument is a pressure correction ``p'``.  In
        the current PISO route it is the corrected pressure state, not an
        increment.
        """
        grad_p = self.pressure_gradient.cell_gradient(pressure_field)
        u_cell = self._cell_velocity(u_flat)
        u_cell = u_cell - (self.cm / a_p[:self.NC])[:, None] * grad_p
        return self._flatten_velocity(u_cell)

    def pressure_free_velocity(self, u_flat, pressure, a_p):
        """Remove the current pressure-gradient contribution from velocity."""
        grad_p = self.pressure_gradient.cell_gradient(pressure)
        u_cell = self._cell_velocity(u_flat)
        u_cell = u_cell + (self.cm / a_p[:self.NC])[:, None] * grad_p
        return self._flatten_velocity(u_cell)
