"""Collocated SIMPLE solver core for steady incompressible Navier-Stokes."""

import logging
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm
from .collocated_ns_fvm_utils import CollocatedNSFVMOperators
from .dirichlet_bc import DirichletBC
from .fvm_linear_solver import FVMLinearSolverConfig
from .rhie_chow import RhieChowInterpolation
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .simple_iteration_control import SimpleIterationControl
from .solver_controls import SimpleSolverControls
from .pressure_correction_control import (
    PressureRelaxationConfig,
    pressure_correction_converged,
)


class CollocatedSimpleSolver(CollocatedNSFVMOperators):
    """Algorithm core for 2D steady collocated SIMPLE solves."""

    def __init__(
        self,
        *,
        mesh,
        diffusion_coef,
        convection_coef,
        source,
        boundary_conditions,
        controls=None,
        linear_solver=None,
        linear_solver_config=None,
        logger=None,
        log_level="WARNING",
        pbar_log=False,
    ):
        """Initialize the reusable SIMPLE algorithm state."""
        self.controls = controls or SimpleSolverControls()
        if logger is None:
            logger = logging.getLogger(self.__class__.__name__)
            logger.propagate = False
            logger.setLevel(log_level)
            if not logger.hasHandlers():
                if pbar_log:
                    from fealpy.logs import TqdmLoggingHandler

                    logger.addHandler(TqdmLoggingHandler())
                else:
                    from fealpy.logs import handler

                    logger.addHandler(handler)
        self.logger = logger
        self.diffusion_coef = self._as_positive_scalar(diffusion_coef, "diffusion_coef")
        self.convection_coef = self._as_nonnegative_scalar(
            convection_coef, "convection_coef"
        )
        self.source = source
        self.mesh = mesh
        self.cm = self.mesh.entity_measure("cell")
        self.NC = self.mesh.number_of_cells()
        self.iteration_control = SimpleIterationControl(self.mesh, self.logger)
        required = (
            "conditions_for",
            "dirichlet_threshold",
            "dirichlet_value",
            "natural_threshold",
            "boundary_face_velocity",
            "has_pressure_dirichlet",
            "pressure_dirichlet_threshold",
            "pressure_dirichlet_value",
        )
        missing = [name for name in required if not hasattr(boundary_conditions, name)]
        if missing:
            raise TypeError("boundary_conditions must provide " + ", ".join(required))
        self.boundary_conditions = boundary_conditions
        self.velocity_dirichlet_data = self.boundary_conditions.dirichlet_value("velocity")
        self.velocity_dirichlet_threshold = self.boundary_conditions.dirichlet_threshold("velocity")
        self.velocity_natural_threshold = (
            self.boundary_conditions.natural_threshold("velocity")
            if self.boundary_conditions.conditions_for("velocity", "natural")
            else None
        )
        self.has_pressure_dirichlet = self.boundary_conditions.has_pressure_dirichlet()
        self.pressure_dirichlet_threshold = (
            self.boundary_conditions.pressure_dirichlet_threshold()
            if self.has_pressure_dirichlet
            else None
        )
        self.pressure_dirichlet_data = (
            self.boundary_conditions.pressure_dirichlet_value()
            if self.has_pressure_dirichlet
            else None
        )
        self._init_collocated_discretization(
            self.controls.space_degree,
            self.velocity_dirichlet_data,
            pressure_gradient_method=self.controls.pressure_gradient_method,
            velocity_gradient_method=self.controls.velocity_gradient_method,
            velocity_dirichlet_threshold=self.velocity_dirichlet_threshold,
            pressure_dirichlet=self.pressure_dirichlet_data,
            pressure_dirichlet_threshold=self.pressure_dirichlet_threshold,
            with_divergence=True,
            with_velocity_dirichlet_bc=True,
        )
        solver_config = linear_solver_config
        if solver_config is None and linear_solver is None:
            solver_config = FVMLinearSolverConfig()
        self.linear_solver = self._init_linear_solver(
            {"linear_solver": linear_solver, "linear_solver_config": solver_config}
        )

    @staticmethod
    def _zero_pressure_correction(points):
        """Pressure-fixed boundaries impose zero pressure correction."""
        return bm.zeros(points.shape[0], dtype=points.dtype)

    def temporary_velocity(self, p, uf, u0) -> Tuple[TensorLike, TensorLike]:
        """Solve momentum equation for the intermediate velocity."""
        convection_face_velocity = self.convection_coef * uf
        B = self.momentum_diffusion_matrix(self.diffusion_coef)
        if self.convection_coef != 0.0:
            B = B + self.momentum_convection_matrix(
                convection_face_velocity,
                self.controls.face_interpolation("momentum_face_interpolation"),
            )
            B = self._apply_velocity_natural_convection(B, uf)
        f = self.momentum_source_vector(self.source)
        threshold = self.velocity_dirichlet_threshold
        B, f = self.velocity_dirichlet_bc.DiffusionApply(
            B, f, coef=self.diffusion_coef, threshold=threshold
        )
        if self.convection_coef != 0.0:
            f = self.velocity_dirichlet_bc.ConvectionApply(
                f, convection_face_velocity, threshold=threshold
            )
        B = B.tocoo().coalesce().tocsr()
        ap = B.diags().values
        f = f - self.pressure_gradient_source(p)
        u = self.linear_solver.solve(B, f)

        u = self.correct_momentum_nonorthogonal_diffusion(
            B,
            f,
            u,
            u0,
            max_iter=self.controls.momentum_nonorthogonal_max_iter,
            tol=self.controls.momentum_nonorthogonal_tol,
            iteration_attr="last_nonorthogonal_iterations",
        )

        return ap, u

    def correct_momentum_nonorthogonal_diffusion(
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
        """Picard-correct the SIMPLE momentum equation for non-orthogonal diffusion."""
        if max_iter == 0:
            setattr(self, iteration_attr, 0)
            return velocity

        correction_velocity = (
            bm.stack([previous_velocity[: self.NC], previous_velocity[self.NC :]], axis=-1)
            if previous_velocity.ndim == 1
            else previous_velocity
        )
        cross = self.compute_cross_diffusion(correction_velocity)
        corrected_velocity = velocity
        setattr(self, iteration_attr, 0)
        for iteration in range(1, max_iter + 1):
            next_velocity = self.linear_solver.solve(matrix, rhs + cross)
            setattr(self, iteration_attr, iteration)
            if bm.max(bm.abs(next_velocity - corrected_velocity)) < tol:
                return next_velocity
            corrected_velocity = next_velocity
            correction_velocity = bm.stack(
                [corrected_velocity[: self.NC], corrected_velocity[self.NC :]],
                axis=-1,
            )
            cross = self.compute_cross_diffusion(correction_velocity)
        return corrected_velocity

    def _apply_velocity_natural_convection(self, matrix, uf):
        """Add zero-gradient velocity outlet convection as owner diagonal."""
        return self.add_velocity_natural_convection_diagonal(
            matrix,
            self.convection_coef * uf,
            self.velocity_natural_threshold,
        )

    def pressure_correction_flux(self, p_corr: TensorLike, response_coef: TensorLike) -> TensorLike:
        """Return the full pressure-correction flux used to correct mass flux."""
        orthogonal_flux = self.pressure_orthogonal_flux(p_corr, response_coef)
        cross_flux = self._pressure_nonorthogonal_cross_flux(p_corr, response_coef)
        flux = orthogonal_flux - cross_flux
        return self._add_pressure_dirichlet_boundary_flux(flux, p_corr, response_coef)

    def _add_pressure_dirichlet_boundary_flux(
        self,
        flux: TensorLike,
        p_corr: TensorLike,
        response_coef: TensorLike,
    ) -> TensorLike:
        """Add pressure-correction flux on pressure Dirichlet boundary faces."""
        return self.add_pressure_dirichlet_flux(
            flux,
            p_corr,
            response_coef,
            self._zero_pressure_correction,
            self.pressure_dirichlet_threshold,
        )

    def correct_face_velocity_with_pressure_correction(
        self,
        uf: TensorLike,
        p_corr: TensorLike,
        response_coef: TensorLike,
        bd_edge: TensorLike,
        bdedgeu: TensorLike,
    ) -> TensorLike:
        """Correct only the normal face velocity component from ``p_corr``."""
        Sf = self.fvm_geometry.S_f
        delta_phi = self.pressure_correction_flux(p_corr, response_coef)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        uf = uf + (delta_phi / Sf_dot_Sf)[:, None] * Sf
        return bm.set_at(uf, bd_edge, bdedgeu)

    def pressure_correct(
        self,
        ap: TensorLike,
        uf: TensorLike,
        *,
        response_coef: Optional[TensorLike] = None,
        nonorthogonal_max_iter: Optional[int] = None,
        nonorthogonal_tol: Optional[float] = None,
    ) -> TensorLike:
        """Solve the SIMPLE pressure-correction equation."""
        if nonorthogonal_max_iter is None:
            nonorthogonal_max_iter = self.controls.pressure_nonorthogonal_max_iter
        if nonorthogonal_tol is None:
            nonorthogonal_tol = self.controls.pressure_nonorthogonal_tol
        if nonorthogonal_max_iter < 0:
            raise ValueError("nonorthogonal_max_iter must be non-negative.")
        if nonorthogonal_tol <= 0.0:
            raise ValueError("nonorthogonal_tol must be positive.")

        dp_edge = (
            self.pressure_response_face_coefficient(
                ap,
                self.controls.face_interpolation("pressure_response_interpolation"),
            )
            if response_coef is None
            else response_coef
        )
        div_u = self.divergence.Reconstruct(uf)
        rhs = -div_u
        has_dirichlet = self.has_pressure_dirichlet
        if has_dirichlet:
            matrix = BilinearForm(self.space).add_integrator(
                ScalarDiffusionIntegrator(q=2, coef=dp_edge)
            ).assembly()
            boundary_coef = self._boundary_face_coefficient(dp_edge)
            pressure_bc = DirichletBC(self.mesh, self._zero_pressure_correction)
        else:
            matrix = self._assemble_pressure_gauge_matrix(dp_edge, q=2)
            gauge_rhs = bm.zeros(1, dtype=rhs.dtype)

        def solve_with_cross_rhs(cross_rhs):
            if has_dirichlet:
                matrix_bc, rhs_bc = pressure_bc.DiffusionApply(
                    matrix,
                    rhs + cross_rhs,
                    coef=boundary_coef,
                    threshold=self.pressure_dirichlet_threshold,
                )
                return self.linear_solver.solve(matrix_bc, rhs_bc)

            matrix_rhs = bm.concatenate([rhs + cross_rhs, gauge_rhs], axis=0)
            return self.linear_solver.solve(matrix, matrix_rhs)[:-1]

        cross_rhs = bm.zeros_like(rhs)
        if nonorthogonal_max_iter == 0:
            self.last_pressure_nonorthogonal_iterations = 0
            return solve_with_cross_rhs(cross_rhs)

        self.last_pressure_nonorthogonal_iterations = 0
        p_corr = cross_rhs
        for iteration in range(1, nonorthogonal_max_iter + 1):
            p_corr = solve_with_cross_rhs(cross_rhs)
            next_cross_rhs = self.divergence_from_flux(
                self._pressure_nonorthogonal_cross_flux(p_corr, dp_edge)
            )
            self.last_pressure_nonorthogonal_iterations = iteration
            if bm.max(bm.abs(next_cross_rhs - cross_rhs)) < nonorthogonal_tol:
                return p_corr
            cross_rhs = next_cross_rhs
        return p_corr

    def rhie_chow_face_velocity(self, u, ap, p, response_coef, bd_edge, bdedgeu):
        """Construct Rhie-Chow face velocity and enforce velocity Dirichlet data."""
        try:
            uf = self.rhie_chow.Interpolation(
                u, ap, p, face_response_coefficient=response_coef
            )
        except TypeError as exc:
            if "face_response_coefficient" not in str(exc):
                raise
            uf = self.rhie_chow.Interpolation(u, ap, p)
        return bm.set_at(uf, bd_edge, bdedgeu)

    def _build_rhie_chow_interpolation(self):
        """Build Rhie-Chow interpolation with active pressure boundary data."""
        try:
            return RhieChowInterpolation(
                self.mesh,
                pressure_gradient_method=self.controls.rhie_chow_pressure_gradient_method,
                velocity_interpolation=self.controls.face_interpolation(
                    "rhie_chow_velocity_interpolation"
                ),
                pressure_dirichlet=self.pressure_dirichlet_data,
                pressure_dirichlet_threshold=self.pressure_dirichlet_threshold,
            )
        except TypeError as exc:
            if (
                "pressure_dirichlet" not in str(exc)
                and "pressure_gradient_method" not in str(exc)
                and "velocity_interpolation" not in str(exc)
            ):
                raise
            return RhieChowInterpolation(self.mesh)

    def _boundary_face_velocity(self):
        """Return only faces with prescribed velocity data when using patches."""
        try:
            return self.boundary_conditions.boundary_face_velocity("velocity", mesh=self.mesh)
        except TypeError:
            return self.boundary_conditions.boundary_face_velocity("velocity")

    def solve(
        self,
        max_iter: int = 100,
        tol: float = 1e-5,
        relax: float = 0.03,
        tol_mass=None,
        tol_pressure_update=None,
        adaptive_pressure_relax: bool = True,
        relaxation_config: Optional[PressureRelaxationConfig] = None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Run the SIMPLE outer iteration."""
        tol_mass, tol_pressure_update = self.iteration_control.tolerances(
            tol, tol_mass, tol_pressure_update
        )
        relaxation = self.iteration_control.pressure_relaxation_controller(
            relax,
            adaptive_pressure_relax,
            relaxation_config,
        )
        pressure_relax = relax
        field_dtype = self.cm.dtype
        p = bm.zeros(self.NC, dtype=field_dtype)
        uf = bm.zeros((self.mesh.number_of_faces(), 2), dtype=field_dtype)
        u = bm.zeros(2 * self.NC, dtype=field_dtype)
        ap, u = self.temporary_velocity(p, uf, u)
        self.residuals = []
        bd_edge, bdedgeu = self._boundary_face_velocity()
        self.rhie_chow = self._build_rhie_chow_interpolation()

        for iteration in range(1, max_iter + 1):
            response_coef = self.pressure_response_face_coefficient(
                ap,
                self.controls.face_interpolation("pressure_response_interpolation"),
            )
            uf = self.rhie_chow_face_velocity(
                u, ap, p, response_coef, bd_edge, bdedgeu
            )
            p_corr = self.pressure_correct(ap, uf, response_coef=response_coef)
            pressure_relax, p_update, residual = (
                self.iteration_control.pressure_update_step(
                    self.residuals,
                    uf,
                    p_corr,
                    p,
                    pressure_relax,
                    relaxation,
                    nonorthogonal_iterations=(
                        self.last_pressure_nonorthogonal_iterations
                    ),
                    momentum_nonorthogonal_iterations=(
                        self.last_nonorthogonal_iterations
                    ),
                )
            )
            self.iteration_control.log_iteration(iteration, residual)

            if pressure_correction_converged(residual, tol_mass, tol_pressure_update):
                self.logger.info("Converged.")
                break

            p += p_update
            uf = self.correct_face_velocity_with_pressure_correction(
                uf, p_corr, response_coef, bd_edge, bdedgeu
            )
            ap, u = self.temporary_velocity(p, uf, u)

        self.uh = u[:self.NC]
        self.vh = u[self.NC:]
        self.ph = p
        return self.uh, self.vh, self.ph
