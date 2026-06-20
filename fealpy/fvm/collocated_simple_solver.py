"""Collocated SIMPLE solver core for steady incompressible Navier-Stokes."""

import logging
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from .collocated_ns_components import CollocatedNSFVMComponents
from .dirichlet_bc import DirichletBC
from .fvm_linear_solver import init_fvm_linear_solver
from .rhie_chow import RhieChowInterpolation
from .solver_controls import SimpleSolverControls, nonnegative_scalar, positive_scalar
from .simple_residual import (
    log_simple_residual,
    pressure_correction_converged,
    simple_pressure_update_step,
    simple_tolerances,
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
        self.cm = self.mesh.entity_measure("cell")
        self.NC = self.mesh.number_of_cells()
        required = (
            "conditions_for",
            "dirichlet_threshold",
            "dirichlet_value",
            "natural_threshold",
            "neumann_threshold",
            "neumann_value",
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
        self.velocity_neumann_data = (
            self.boundary_conditions.neumann_value("velocity")
            if self.boundary_conditions.conditions_for("velocity", "neumann")
            else None
        )
        self.velocity_neumann_threshold = (
            self.boundary_conditions.neumann_threshold("velocity")
            if self.boundary_conditions.conditions_for("velocity", "neumann")
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
        self.init_collocated_discretization(
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
        self.rhie_chow = RhieChowInterpolation(
            self.mesh,
            pressure_gradient_method=self.controls.rhie_chow_pressure_gradient_method,
            velocity_interpolation=self.controls.face_interpolation(
                "rhie_chow_velocity_interpolation"
            ),
            pressure_dirichlet=self.pressure_dirichlet_data,
            pressure_dirichlet_threshold=self.pressure_dirichlet_threshold,
            geometry=self.fvm_geometry,
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

    def temporary_velocity(self, p, uf, u0) -> Tuple[TensorLike, TensorLike]:
        """Solve momentum equation for the intermediate velocity."""
        return self.solve_steady_momentum_predictor(p, uf, u0)

    def pressure_correction_flux(self, p_corr: TensorLike, response_coef: TensorLike) -> TensorLike:
        """Return the full pressure-correction flux used to correct mass flux."""
        orthogonal_flux = self.pressure_orthogonal_flux(p_corr, response_coef)
        cross_flux = self.pressure_nonorthogonal_cross_flux(
            p_corr,
            response_coef,
            interpolation_method=self.controls.face_interpolation(
                "pressure_response_interpolation"
            ),
        )
        flux = orthogonal_flux - cross_flux
        return self.add_pressure_dirichlet_flux(
            flux,
            p_corr,
            response_coef,
            self.zero_pressure_correction,
            self.pressure_dirichlet_threshold,
        )

    def correct_face_velocity_with_pressure_correction(
        self,
        uf: TensorLike,
        p_corr: TensorLike,
        response_coef: TensorLike,
        boundary_faces: TensorLike,
        boundary_velocity: TensorLike,
    ) -> TensorLike:
        """Correct only the normal face velocity component from ``p_corr``."""
        Sf = self.fvm_geometry.S_f
        delta_phi = self.pressure_correction_flux(p_corr, response_coef)
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

        face_response_coef = (
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
        pressure_constraint = self.controls.pressure_constraint
        if has_dirichlet:
            matrix = self.pressure_diffusion_matrix(face_response_coef)
            pressure_bc = DirichletBC(
                self.mesh,
                self.zero_pressure_correction,
                geometry=self.fvm_geometry,
            )
        elif pressure_constraint == "nullspace":
            matrix = self.pressure_diffusion_matrix(face_response_coef)
        else:
            matrix = self.pressure_gauge_matrix(face_response_coef)
            gauge_rhs = bm.zeros(1, dtype=rhs.dtype)

        def solve_with_cross_rhs(cross_rhs):
            if has_dirichlet:
                matrix_bc, rhs_bc = pressure_bc.apply_diffusion(
                    matrix,
                    rhs + cross_rhs,
                    coef=face_response_coef,
                    threshold=self.pressure_dirichlet_threshold,
                )
                return self.solve_linear_system(
                    matrix_bc,
                    rhs_bc,
                    solver=self.pressure_linear_solver,
                )

            if pressure_constraint == "nullspace":
                projected_rhs = self.project_pressure_rhs_to_range(rhs + cross_rhs)
                return self.zero_mean_pressure(
                    self.linear_solver.solve_constant_nullspace(
                        matrix,
                        projected_rhs,
                        self.cm,
                        solver=self.pressure_nullspace_linear_solver,
                    )
                )

            matrix_rhs = bm.concatenate([rhs + cross_rhs, gauge_rhs], axis=0)
            return self.solve_linear_system(
                matrix,
                matrix_rhs,
                solver=self.pressure_gauge_linear_solver,
            )[:-1]

        cross_rhs = bm.zeros_like(rhs)
        if nonorthogonal_max_iter == 0:
            self.last_pressure_nonorthogonal_iterations = 0
            return solve_with_cross_rhs(cross_rhs)

        self.last_pressure_nonorthogonal_iterations = 0
        p_corr = cross_rhs
        for iteration in range(1, nonorthogonal_max_iter + 1):
            p_corr = solve_with_cross_rhs(cross_rhs)
            next_cross_rhs = self.divergence_from_flux(
                self.pressure_nonorthogonal_cross_flux(
                    p_corr,
                    face_response_coef,
                    interpolation_method=self.controls.face_interpolation(
                        "pressure_response_interpolation"
                    ),
                )
            )
            self.last_pressure_nonorthogonal_iterations = iteration
            if bm.max(bm.abs(next_cross_rhs - cross_rhs)) < nonorthogonal_tol:
                return p_corr
            cross_rhs = next_cross_rhs
        return p_corr

    def rhie_chow_face_velocity(
        self, u, ap, p, response_coef, boundary_faces, boundary_velocity
    ):
        """Construct Rhie-Chow face velocity and enforce velocity Dirichlet data."""
        uf = self.rhie_chow.Interpolation(
            u, ap, p, face_response_coefficient=response_coef
        )
        return bm.set_at(uf, boundary_faces, boundary_velocity)

    def solve(
        self,
        max_iter: int = 100,
        tol: float = 1e-5,
        relax: float = 0.3,
        tol_mass=None,
        tol_pressure_correction=None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Run the SIMPLE outer iteration."""
        tol_mass, tol_pressure_correction = simple_tolerances(
            tol, tol_mass, tol_pressure_correction
        )
        if tol_pressure_correction <= 0.0:
            raise ValueError("tol_pressure_correction must be positive.")
        if relax <= 0.0:
            raise ValueError("relax must be positive.")
        field_dtype = self.cm.dtype
        p = bm.zeros(self.NC, dtype=field_dtype)
        uf = bm.zeros((self.mesh.number_of_faces(), self.GD), dtype=field_dtype)
        u = bm.zeros(self.GD * self.NC, dtype=field_dtype)
        ap, u = self.temporary_velocity(p, uf, u)
        self.residuals = []
        boundary_faces, boundary_velocity = self.boundary_conditions.boundary_face_velocity(
            "velocity",
            mesh=self.mesh,
        )

        for iteration in range(1, max_iter + 1):
            response_coef = self.pressure_response_face_coefficient(
                ap,
                self.controls.face_interpolation("pressure_response_interpolation"),
            )
            uf = self.rhie_chow_face_velocity(
                u, ap, p, response_coef, boundary_faces, boundary_velocity
            )
            p_corr = self.pressure_correct(ap, uf, response_coef=response_coef)
            relaxed_p_corr = relax * p_corr
            uf_corrected = self.correct_face_velocity_with_pressure_correction(
                uf, relaxed_p_corr, response_coef, boundary_faces, boundary_velocity
            )
            p_update, residual = simple_pressure_update_step(
                self.residuals,
                self.mesh,
                uf,
                p_corr,
                p,
                pressure_relax=relax,
                nonorthogonal_iterations=(
                    self.last_pressure_nonorthogonal_iterations
                ),
                momentum_nonorthogonal_iterations=self.last_nonorthogonal_iterations,
                stopping_face_velocity=uf_corrected,
                geometry=self.fvm_geometry,
            )
            log_simple_residual(self.logger, iteration, residual)

            p += p_update
            uf = uf_corrected
            u = self.cell_vector_to_dofs(
                self.velocity_pressure_correction(
                    self.dofs_to_cell_vector(u),
                    p_update,
                    ap,
                )
            )
            if pressure_correction_converged(
                residual,
                tol_mass,
                tol_pressure_correction=tol_pressure_correction,
            ):
                self.logger.info("Converged.")
                break

            ap, u = self.temporary_velocity(p, uf, u)

        self.velocity = self.dofs_to_cell_vector(u)
        self.velocity_components = [
            self.velocity[:, component] for component in range(self.GD)
        ]
        self.uh = self.velocity_components[0]
        self.vh = (
            self.velocity_components[1]
            if self.GD > 1
            else bm.zeros_like(self.uh)
        )
        self.ph = p
        return self.uh, self.vh, self.ph
