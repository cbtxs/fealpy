"""Collocated SIMPLE solver core for steady incompressible Navier-Stokes."""

import logging
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from .collocated_ns_components import (
    CollocatedNSFVMComponents,
    MomentumPredictorResult,
)
from .dirichlet_bc import DirichletBC
from .engineering_boundary_conditions import PDEBoundaryConditions
from .gradient_reconstruct import GradientReconstruct
from .fvm_linear_solver import init_fvm_linear_solver
from .collocated_face_velocity_reconstruct import RhieChowInterpolation
from .face_flux_reconstruct import FaceFluxReconstruct
from .solver_controls import SimpleSolverControls, nonnegative_scalar, positive_scalar
from .simple_residual import (
    normalized_cell_integral_residual,
    simple_fixed_point_converged,
    simple_iteration_residual,
    simple_tolerances,
)
from .solver_diagnostics import (
    equation_residual_converged,
    inexact_inner_tolerance,
    log_simple_iteration,
    normalized_equation_residual,
)


class CollocatedSimpleSolver(CollocatedNSFVMComponents):
    """Algorithm core for steady collocated SIMPLE solves."""

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
        self.momentum_linear_solver = self.controls.momentum_linear_solver
        self.pressure_linear_solver = self.controls.pressure_linear_solver
        self.pressure_gauge_linear_solver = self.controls.pressure_gauge_linear_solver
        self.pressure_nullspace_linear_solver = (
            self.controls.pressure_nullspace_linear_solver
        )
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
        self.diffusion_coef = positive_scalar(diffusion_coef, "diffusion_coef")
        self.convection_coef = nonnegative_scalar(convection_coef, "convection_coef")
        self.source = source
        self.mesh = mesh
        if not isinstance(boundary_conditions, PDEBoundaryConditions):
            raise TypeError(
                "boundary_conditions must be PDEBoundaryConditions; convert "
                "engineering boundary data before constructing the solver."
            )
        self.boundary_conditions = boundary_conditions
        self.dirichlet_velocity_data = self.boundary_conditions.dirichlet_value("velocity")
        self.dirichlet_velocity_threshold = self.boundary_conditions.dirichlet_threshold("velocity")
        self.natural_velocity_threshold = (
            self.boundary_conditions.natural_threshold("velocity")
            if self.boundary_conditions.has_natural("velocity")
            else None
        )
        self.neumann_velocity_data = (
            self.boundary_conditions.neumann_value("velocity")
            if self.boundary_conditions.has_neumann("velocity")
            else None
        )
        self.neumann_velocity_threshold = (
            self.boundary_conditions.neumann_threshold("velocity")
            if self.boundary_conditions.has_neumann("velocity")
            else None
        )
        self.has_dirichlet_pressure = self.boundary_conditions.has_dirichlet("pressure")
        self.dirichlet_pressure_threshold = (
            self.boundary_conditions.dirichlet_threshold("pressure")
            if self.has_dirichlet_pressure
            else None
        )
        self.dirichlet_pressure_data = (
            self.boundary_conditions.dirichlet_value("pressure")
            if self.has_dirichlet_pressure
            else None
        )
        self.init_collocated_discretization(
            self.controls.space_degree,
            self.dirichlet_velocity_data,
            pressure_gradient_method=self.controls.pressure_gradient_method,
            velocity_gradient_method=self.controls.velocity_gradient_method,
            gradient_layer_weights=self.controls.gradient_layer_weights,
            gradient_boundary_weight=self.controls.gradient_boundary_weight,
            diffusion_method=self.controls.diffusion_method,
            diffusion_nonorthogonal_eps=self.controls.diffusion_nonorthogonal_eps,
            dirichlet_velocity_threshold=self.dirichlet_velocity_threshold,
            dirichlet_pressure=self.dirichlet_pressure_data,
            dirichlet_pressure_threshold=self.dirichlet_pressure_threshold,
            with_dirichlet_velocity_bc=True,
            geometry=self.boundary_conditions.geometry,
        )
        self.pressure_correction_bc = None
        self.pressure_correction_gradient = self.pressure_gradient
        if self.has_dirichlet_pressure:
            self.pressure_correction_bc = DirichletBC(
                self.mesh,
                self.zero_pressure_correction,
                threshold=self.dirichlet_pressure_threshold,
                geometry=self.fvm_geometry,
                diffusion_method=self.controls.diffusion_method,
                nonorthogonal_eps=self.controls.diffusion_nonorthogonal_eps,
            )
            self.pressure_correction_gradient = GradientReconstruct(
                self.mesh,
                method=self.controls.pressure_gradient_method,
                boundary_value=self.zero_pressure_correction,
                boundary_type="dirichlet",
                boundary_threshold=self.dirichlet_pressure_threshold,
                layer_weights=self.controls.gradient_layer_weights,
                boundary_weight=self.controls.gradient_boundary_weight,
                geometry=self.fvm_geometry,
            )
        self.rhie_chow = RhieChowInterpolation(
            self.mesh,
            pressure_gradient_method=self.controls.rhie_chow_pressure_gradient_method,
            gradient_layer_weights=self.controls.gradient_layer_weights,
            gradient_boundary_weight=self.controls.gradient_boundary_weight,
            velocity_interpolation=self.controls.face_interpolation_method,
            dirichlet_pressure=self.dirichlet_pressure_data,
            dirichlet_pressure_threshold=self.dirichlet_pressure_threshold,
            geometry=self.fvm_geometry,
        )
        self.face_flux_reconstruct = FaceFluxReconstruct(
            self.mesh,
            method=self.controls.face_flux_correction_scheme,
            geometry=self.fvm_geometry,
            quadrature_order=self.controls.face_flux_quadrature_order,
            max_stencil_layers=self.controls.face_flux_max_stencil_layers,
            max_condition=self.controls.face_flux_max_condition,
        )
        self.linear_solver = init_fvm_linear_solver(
            linear_solver,
            linear_solver_config,
            reference=self.cm,
        )

    def zero_pressure_correction(self, points):
        """Return the homogeneous Dirichlet value for SIMPLE pressure correction.

        If the physical pressure is fixed on a boundary, SIMPLE must solve the
        pressure-correction equation with ``p' = 0`` there.  Otherwise the
        pressure update ``p <- p + alpha_p p'`` would change a prescribed
        pressure boundary value.
        """
        return bm.zeros(points.shape[0], dtype=points.dtype)

    def steady_momentum_source_vector(self):
        """Return the cached steady cell-integrated momentum source RHS."""
        source_vector = getattr(self, "_steady_momentum_source_vector", None)
        if source_vector is None:
            source_vector = self.momentum_source_vector(self.source)
            self._steady_momentum_source_vector = source_vector
        return bm.copy(source_vector)

    def temporary_velocity(
        self,
        p,
        uf,
        u0,
        *,
        pressure_gradient=None,
        nonorthogonal_tol=None,
    ) -> MomentumPredictorResult:
        """Return the named steady momentum predictor result."""
        return self.solve_steady_momentum_predictor(
            p,
            uf,
            u0,
            pressure_gradient=pressure_gradient,
            nonorthogonal_tol=nonorthogonal_tol,
        )

    def pressure_correction_flux(
        self,
        p_corr: TensorLike,
        response_coef: TensorLike,
        *,
        pressure_gradient=None,
    ) -> TensorLike:
        """Return the full pressure-correction flux used to correct mass flux."""
        if pressure_gradient is None:
            pressure_gradient = self.pressure_correction_gradient.cell_gradient(
                p_corr
            )
        orthogonal_flux = self.pressure_orthogonal_flux(p_corr, response_coef)
        cross_flux = self.pressure_nonorthogonal_cross_flux(
            p_corr,
            response_coef,
            interpolation_method=self.controls.face_interpolation(
                "pressure_response_interpolation"
            ),
            pressure_gradient=pressure_gradient,
            boundary_threshold=self.dirichlet_pressure_threshold,
        )
        flux = orthogonal_flux - cross_flux
        return self.add_dirichlet_pressure_flux(
            flux,
            p_corr,
            response_coef,
            self.zero_pressure_correction,
            self.dirichlet_pressure_threshold,
        )

    def correct_face_velocity_with_pressure_correction(
        self,
        uf: TensorLike,
        p_corr: TensorLike,
        response_coef: TensorLike,
        boundary_faces: TensorLike,
        boundary_velocity: TensorLike,
        pressure_gradient=None,
    ) -> TensorLike:
        """Correct only the normal face velocity component from ``p_corr``."""
        Sf = self.fvm_geometry.S_f
        delta_phi = self.pressure_correction_flux(
            p_corr,
            response_coef,
            pressure_gradient=pressure_gradient,
        )
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        uf = uf + (delta_phi / Sf_dot_Sf)[:, None] * Sf
        return bm.set_at(uf, boundary_faces, boundary_velocity)

    def pressure_correct(
        self,
        ap: TensorLike,
        uf: TensorLike,
        *,
        response_coef: Optional[TensorLike] = None,
        nonorthogonal_max_iter: Optional[int] = None,
        nonorthogonal_tol: Optional[float] = None,
        nonorthogonal_atol: Optional[float] = None,
    ) -> TensorLike:
        """Solve the SIMPLE pressure-correction equation."""
        if nonorthogonal_max_iter is None:
            nonorthogonal_max_iter = self.controls.pressure_nonorthogonal_max_iter
        if nonorthogonal_tol is None:
            nonorthogonal_tol = self.controls.pressure_nonorthogonal_tol
        if nonorthogonal_atol is None:
            nonorthogonal_atol = self.controls.pressure_nonorthogonal_atol
        if nonorthogonal_max_iter < 0:
            raise ValueError("nonorthogonal_max_iter must be non-negative.")
        if nonorthogonal_tol <= 0.0:
            raise ValueError("nonorthogonal_tol must be positive.")
        if nonorthogonal_atol < 0.0:
            raise ValueError("nonorthogonal_atol must be non-negative.")

        face_response_coef = (
            self.pressure_response_face_coefficient(
                ap,
                self.controls.face_interpolation("pressure_response_interpolation"),
            )
            if response_coef is None
            else response_coef
        )
        div_u = self.divergence_from_flux(self.compute_face_flux(uf))
        rhs = -div_u
        has_dirichlet = self.has_dirichlet_pressure
        pressure_constraint = self.controls.pressure_constraint
        if has_dirichlet:
            base_matrix = self.pressure_diffusion_matrix(face_response_coef)
            pressure_bc = self.pressure_correction_bc
            matrix = pressure_bc.apply_diffusion_matrix(
                base_matrix,
                coef=face_response_coef,
                threshold=self.dirichlet_pressure_threshold,
            )
        elif pressure_constraint == "nullspace":
            matrix = self.pressure_diffusion_matrix(face_response_coef)
        else:
            matrix = self.pressure_gauge_matrix(face_response_coef)
            gauge_rhs = bm.zeros(1, dtype=rhs.dtype)

        def system_rhs(cross_rhs):
            if has_dirichlet:
                return pressure_bc.apply_diffusion_rhs(
                    rhs + cross_rhs,
                    coef=face_response_coef,
                    threshold=self.dirichlet_pressure_threshold,
                )
            if pressure_constraint == "nullspace":
                return self.project_pressure_rhs_to_range(rhs + cross_rhs)
            return bm.concatenate([rhs + cross_rhs, gauge_rhs], axis=0)

        def algebraic_solution(pressure):
            if has_dirichlet or pressure_constraint == "nullspace":
                return pressure
            return bm.concatenate(
                [pressure, bm.zeros(1, dtype=pressure.dtype)], axis=0
            )

        def solve_with_cross_rhs(cross_rhs):
            matrix_rhs = system_rhs(cross_rhs)
            if has_dirichlet:
                return self.solve_linear_system(
                    matrix,
                    matrix_rhs,
                    solver=self.pressure_linear_solver,
                )
            if pressure_constraint == "nullspace":
                return self.zero_mean_pressure(
                    self.linear_solver.solve_constant_nullspace(
                        matrix,
                        matrix_rhs,
                        self.cm,
                        solver=self.pressure_nullspace_linear_solver,
                    )
                )
            return self.solve_linear_system(
                matrix,
                matrix_rhs,
                solver=self.pressure_gauge_linear_solver,
            )[:-1]

        cross_rhs = bm.zeros_like(rhs)
        p_corr = solve_with_cross_rhs(cross_rhs)
        self.last_pressure_nonorthogonal_iterations = 0
        self.last_pressure_nonorthogonal_residual = None
        self.last_pressure_nonorthogonal_relative_update = 0.0
        if nonorthogonal_max_iter == 0:
            return p_corr

        for correction in range(nonorthogonal_max_iter + 1):
            next_cross_rhs = self.divergence_from_flux(
                self.pressure_nonorthogonal_cross_flux(
                    p_corr,
                    face_response_coef,
                    interpolation_method=self.controls.face_interpolation(
                        "pressure_response_interpolation"
                    ),
                    pressure_gradient=(
                        self.pressure_correction_gradient.cell_gradient(p_corr)
                    ),
                    boundary_threshold=self.dirichlet_pressure_threshold,
                )
            )
            matrix_rhs = system_rhs(next_cross_rhs)
            metrics = normalized_equation_residual(
                matrix @ algebraic_solution(p_corr),
                matrix_rhs,
            )
            self.last_pressure_nonorthogonal_residual = metrics
            if equation_residual_converged(
                metrics,
                rtol=nonorthogonal_tol,
                atol=nonorthogonal_atol,
            ):
                return p_corr
            if correction == nonorthogonal_max_iter:
                break

            previous = p_corr
            cross_rhs = next_cross_rhs
            p_corr = solve_with_cross_rhs(cross_rhs)
            update = float(bm.to_numpy(bm.linalg.norm(p_corr - previous)))
            scale = float(bm.to_numpy(bm.linalg.norm(p_corr)))
            self.last_pressure_nonorthogonal_relative_update = (
                update / max(scale, 1.0e-30)
            )
            self.last_pressure_nonorthogonal_iterations = correction + 1

        raise RuntimeError(
            "pressure non-orthogonal correction did not converge before "
            "pressure_nonorthogonal_max_iter"
        )

    def rhie_chow_face_velocity(
        self,
        u,
        p,
        response_coef,
        boundary_faces,
        boundary_velocity,
        pressure_gradient=None,
    ):
        """Construct Rhie-Chow face velocity and enforce velocity Dirichlet data."""
        base_face_velocity = self.spatial_face_velocity(
            u,
            boundary_faces=boundary_faces,
            boundary_face_average=(
                boundary_velocity
                if self.controls.face_flux_correction_scheme != "none"
                else None
            ),
        )
        gradient_difference = self.rhie_chow.pressure_gradient_difference(
            p,
            pressure_gradient=pressure_gradient,
        )
        uf = base_face_velocity - response_coef[:, None] * gradient_difference
        return bm.set_at(uf, boundary_faces, boundary_velocity)

    def solve(
        self,
        max_iter: int = 100,
        tol: float = 1e-5,
        relax: float = 0.3,
        tol_momentum=None,
        tol_mass=None,
    ) -> Tuple[TensorLike, TensorLike]:
        """Run the SIMPLE outer iteration."""
        tol_momentum, tol_mass = simple_tolerances(
            tol, tol_momentum, tol_mass
        )
        if tol_momentum <= 0.0:
            raise ValueError("tol_momentum must be positive.")
        if tol_mass <= 0.0:
            raise ValueError("tol_mass must be positive.")
        if relax <= 0.0:
            raise ValueError("relax must be positive.")
        momentum_predictor_tol = inexact_inner_tolerance(
            self.controls.momentum_nonorthogonal_tol,
            tol_momentum,
        )
        field_dtype = self.cm.dtype
        p = bm.zeros(self.NC, dtype=field_dtype)
        uf = bm.zeros((self.NF, self.GD), dtype=field_dtype)
        u = bm.zeros((self.NC, self.GD), dtype=field_dtype)
        pressure_gradient = self.pressure_gradient.cell_gradient(p)
        predictor = self.temporary_velocity(
            p,
            uf,
            u,
            pressure_gradient=pressure_gradient,
            nonorthogonal_tol=momentum_predictor_tol,
        )
        correction_ap = predictor.correction_denominator
        spatial_ap = predictor.spatial_diagonal
        u = predictor.velocity
        self.residuals = []
        self.converged = False
        self.termination_reason = "max_iter"
        self.outer_iterations = 0
        if self.controls.face_flux_correction_scheme == "none":
            boundary_faces, boundary_velocity = (
                self.boundary_conditions.boundary_face_velocity(
                    "velocity",
                    mesh=self.mesh,
                )
            )
        else:
            boundary_faces, boundary_velocity = (
                self.boundary_conditions.boundary_face_velocity_average(
                    "velocity",
                    mesh=self.mesh,
                    quadrature_order=self.controls.face_flux_quadrature_order,
                )
            )

        for iteration in range(1, max_iter + 1):
            correction_response_coef = self.pressure_response_face_coefficient(
                correction_ap,
                self.controls.face_interpolation("pressure_response_interpolation"),
            )
            spatial_response_coef = self.pressure_response_face_coefficient(
                spatial_ap,
                self.controls.face_interpolation("pressure_response_interpolation"),
            )
            uf = self.rhie_chow_face_velocity(
                u,
                p,
                spatial_response_coef,
                boundary_faces,
                boundary_velocity,
                pressure_gradient=pressure_gradient,
            )
            p_corr = self.pressure_correct(
                correction_ap,
                uf,
                response_coef=correction_response_coef,
            )
            relaxed_p_corr = relax * p_corr
            relaxed_p_corr_gradient = (
                self.pressure_correction_gradient.cell_gradient(relaxed_p_corr)
            )
            previous_pressure = bm.copy(p)
            p_update = relaxed_p_corr
            p += p_update
            u = self.velocity_pressure_correction(
                u,
                p_update,
                correction_ap,
                pressure_gradient=relaxed_p_corr_gradient,
            )
            pressure_gradient = self.pressure_gradient.cell_gradient(p)
            uf = self.rhie_chow_face_velocity(
                u,
                p,
                spatial_response_coef,
                boundary_faces,
                boundary_velocity,
                pressure_gradient=pressure_gradient,
            )
            residual = simple_iteration_residual(
                self.mesh,
                uf,
                p_corr,
                previous_pressure,
                nonorthogonal_iterations=(
                    self.last_pressure_nonorthogonal_iterations
                ),
                momentum_nonorthogonal_iterations=(
                    self.last_momentum_nonorthogonal_iterations
                ),
                stopping_face_velocity=uf,
                geometry=self.fvm_geometry,
            )
            self.residuals.append(residual)
            momentum = self.steady_momentum_balance(
                p,
                u,
                uf,
            )
            momentum_norm = normalized_cell_integral_residual(
                momentum["lhs"],
                momentum["rhs"],
                self.cm,
            )
            residual["momentum_residual_absolute"] = momentum_norm[
                "absolute_l2"
            ]
            residual["momentum_residual_relative"] = momentum_norm[
                "relative_l2"
            ]
            residual["momentum_nonorthogonal_tolerance"] = (
                momentum_predictor_tol
            )
            self.outer_iterations = iteration
            log_simple_iteration(self.logger, iteration, residual)

            if simple_fixed_point_converged(
                residual,
                tol_momentum,
                tol_mass,
            ):
                self.converged = True
                self.termination_reason = "fixed_point_residuals"
                self.logger.info("Converged.")
                break

            momentum_predictor_tol = inexact_inner_tolerance(
                self.controls.momentum_nonorthogonal_tol,
                tol_momentum,
                residual["momentum_residual_relative"],
            )
            predictor = self.temporary_velocity(
                p,
                uf,
                u,
                pressure_gradient=pressure_gradient,
                nonorthogonal_tol=momentum_predictor_tol,
            )
            correction_ap = predictor.correction_denominator
            spatial_ap = predictor.spatial_diagonal
            u = predictor.velocity

        self.velocity = u
        self.pressure = p
        self.face_velocity = uf
        self.face_flux = self.compute_face_flux(uf)
        return self.velocity, self.pressure
