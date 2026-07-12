"""Collocated PISO solver core for transient incompressible Navier-Stokes."""

import logging
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.fem import LinearForm

from .collocated_ns_components import CollocatedNSFVMComponents
from .deviatoric_stress_source import DeviatoricStressSourceIntegrator
from .dirichlet_bc import DirichletBC
from .engineering_boundary_conditions import (
    apply_boundary_flux_constraint,
    apply_face_velocity_constraint,
)
from .solver_controls import PisoSolverControls, positive_scalar
from .collocated_face_velocity_reconstruct import RhieChowInterpolation
from .scalar_cross_diffusion_integrator import ScalarCrossDiffusionIntegrator
from .fvm_linear_solver import init_fvm_linear_solver
from .solver_diagnostics import (
    pressure_correction_diagnostics,
    record_piso_corrector_diagnostics,
)


class CollocatedPisoSolver(CollocatedNSFVMComponents):
    """Algorithm core for transient collocated PISO solves."""

    def __init__(
        self,
        *,
        mesh,
        diffusion_coef,
        convection_coef,
        source,
        boundary_conditions,
        controls=None,
        initial_solution=None,
        linear_solver=None,
        linear_solver_config=None,
        logger=None,
        log_level="WARNING",
        pbar_log=False,
    ):
        self.controls = controls or PisoSolverControls()
        self.momentum_linear_solver = self.controls.momentum_linear_solver
        self.pressure_linear_solver = self.controls.pressure_linear_solver
        self.pressure_gauge_linear_solver = self.controls.pressure_gauge_linear_solver
        self.pressure_nullspace_linear_solver = (
            self.controls.pressure_nullspace_linear_solver
        )
        self.logger = logger or self._build_logger(
            self.__class__.__name__,
            pbar_log=pbar_log,
            log_level=log_level,
        )
        self.corrector_diagnostics = []
        self.mu = positive_scalar(diffusion_coef, "diffusion_coef")
        self.rho = positive_scalar(convection_coef, "convection_coef")
        self.diffusion_coef = self.mu
        self.convection_coef = self.rho
        self.source = source
        self.mesh = mesh
        self.cm = self.mesh.entity_measure("cell")
        self.cell_center = self.mesh.entity_barycenter("cell")
        self.face_center = self.mesh.entity_barycenter("face")
        self.NC = self.mesh.number_of_cells()
        required = (
            "conditions_for",
            "dirichlet_threshold",
            "dirichlet_value",
            "natural_threshold",
            "neumann_threshold",
            "neumann_value",
            "has_pressure_dirichlet",
            "boundary_face_velocity",
        )
        missing = [name for name in required if not hasattr(boundary_conditions, name)]
        if missing:
            raise TypeError("boundary_conditions must provide " + ", ".join(required))
        self.boundary_conditions = boundary_conditions
        self.velocity_dirichlet_value = self.boundary_conditions.dirichlet_value("velocity")
        self.velocity_dirichlet_threshold = self.boundary_conditions.dirichlet_threshold(
            "velocity"
        )
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
        self.pressure_dirichlet_value = None
        self.pressure_dirichlet_threshold = None
        if self.boundary_conditions.has_pressure_dirichlet():
            self.pressure_dirichlet_value = (
                self.boundary_conditions.pressure_dirichlet_value()
            )
            self.pressure_dirichlet_threshold = (
                self.boundary_conditions.pressure_dirichlet_threshold()
            )
        self.initial_solution_callback = initial_solution
        self._init_discretization(self.controls.space_degree)
        self.linear_solver = init_fvm_linear_solver(
            linear_solver,
            linear_solver_config,
            reference=self.cm,
        )

    @staticmethod
    def _build_logger(name, *, pbar_log=False, log_level="WARNING"):
        """Create a logger compatible with ``ComputationalModel`` defaults."""
        logger = logging.getLogger(name)
        logger.propagate = False
        logger.setLevel(log_level)
        if not logger.hasHandlers():
            if pbar_log:
                from fealpy.logs import TqdmLoggingHandler

                logger.addHandler(TqdmLoggingHandler())
            else:
                from fealpy.logs import handler

                logger.addHandler(handler)
        return logger

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  Time steps: {self.controls.nt}\n"
            f"  PISO correctors: {self.controls.n_correctors}\n"
            f"  Momentum nonorthogonal corrections: "
            f"{self.controls.momentum_nonorthogonal_max_iter}\n"
            f"  Pressure nonorthogonal corrections: "
            f"{self.controls.pressure_nonorthogonal_max_iter}\n"
        )

    def set_space(self, degree: int) -> None:
        """Rebuild the collocated discretization with a new polynomial degree."""
        self._init_discretization(degree)

    def _init_discretization(self, degree: int) -> None:
        self.init_collocated_discretization(
            degree,
            self.velocity_dirichlet_value,
            pressure_gradient_method=self.controls.pressure_gradient_method,
            velocity_gradient_method=self.controls.velocity_gradient_method,
            velocity_dirichlet_threshold=self.velocity_dirichlet_threshold,
            pressure_dirichlet=self.pressure_dirichlet_value,
            pressure_dirichlet_threshold=self.pressure_dirichlet_threshold,
            with_velocity_dirichlet_bc=True,
        )
        self.rhie_chow = RhieChowInterpolation(
            self.mesh,
            pressure_gradient_method=self.controls.rhie_chow_pressure_gradient_method,
            velocity_interpolation=self.controls.face_interpolation_method,
            pressure_dirichlet=self.pressure_dirichlet_value,
            pressure_dirichlet_threshold=self.pressure_dirichlet_threshold,
            geometry=self.fvm_geometry,
        )

    def initial_solution(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        if self.initial_solution_callback is None:
            raise NotImplementedError("initial_solution must be supplied.")
        return self.initial_solution_callback()

    def temporary_velocity(self, U0, Uf0, p0, t):
        return self.solve_transient_momentum_predictor(U0, Uf0, p0, t)

    def boundary_corrected_momentum_explicit_source(self, velocity):
        """Return boundary-corrected explicit viscous RHS for momentum prediction.

        This route is the momentum-side counterpart of the pressure non-orthogonal
        cross-flux route: internal face gradients are built with linear face
        interpolation, and velocity boundary gradients are corrected with patch
        normal derivatives before the explicit cross and deviatoric-stress
        sources are assembled.
        """
        face_gradient = self.boundary_corrected_momentum_face_gradient(velocity)
        cross_source = LinearForm(self.velocity_space).add_integrator(
            ScalarCrossDiffusionIntegrator(
                grad_f=face_gradient,
                coef=self.mu,
                geometry=self.fvm_geometry,
                boundary_policy="zero",
            )
        ).assembly()
        stress_source = LinearForm(self.velocity_space).add_integrator(
            DeviatoricStressSourceIntegrator(
                face_gradient,
                coef=self.mu,
                geometry=self.fvm_geometry,
            )
        ).assembly()
        return cross_source + stress_source

    def boundary_corrected_momentum_face_gradient(self, velocity):
        """Return boundary-corrected face gradient for explicit momentum sources.

        Keep this semantic in sync with the pressure non-orthogonal correction
        when comparing solver routes.  Pressure cross flux follows
        ``controls.face_interpolation_method``; this momentum route deliberately uses
        linear interpolation and patch normal-gradient boundary correction.
        """
        return self.boundary_corrected_velocity_face_gradient(
            velocity,
            interpolation_method="linear",
        )

    def assemble_pressure_state_system(self, rhs, coef, cross_rhs):
        """Assemble the pressure-state linear system for one nonOrth solve."""
        has_dirichlet = (
            self.pressure_dirichlet_value is not None
            and self.pressure_dirichlet_threshold is not None
        )
        if has_dirichlet:
            matrix = self.pressure_diffusion_matrix(coef)
            pressure_bc = DirichletBC(
                self.mesh,
                self.pressure_dirichlet_value,
                geometry=self.fvm_geometry,
            )
            return pressure_bc.apply_diffusion(
                matrix,
                rhs + cross_rhs,
                coef=coef,
                threshold=self.pressure_dirichlet_threshold,
            )

        if self.controls.pressure_constraint == "nullspace":
            return self.pressure_diffusion_matrix(coef), rhs + cross_rhs

        matrix = self.pressure_gauge_matrix(coef)
        rhs = bm.concatenate([rhs + cross_rhs, bm.zeros(1, dtype=rhs.dtype)], axis=0)
        return matrix, rhs

    def solve_pressure_state_equation(self, rhs, a_p, initial_pressure_state: Optional[TensorLike] = None):
        coef = self.pressure_response_face_coefficient(
            a_p,
            self.controls.face_interpolation_method,
        )
        max_iter = int(self.controls.pressure_nonorthogonal_max_iter)
        if max_iter < 0:
            raise ValueError("pressure_nonorthogonal_max_iter must be non-negative.")

        explicit_cross_flux = bm.zeros(self.mesh.number_of_faces(), dtype=rhs.dtype)
        cross_rhs = bm.zeros_like(rhs)
        has_pressure_dirichlet = (
            self.pressure_dirichlet_value is not None
            and self.pressure_dirichlet_threshold is not None
        )
        use_nullspace = (
            self.controls.pressure_constraint == "nullspace"
            and not has_pressure_dirichlet
        )
        if max_iter == 0:
            matrix, matrix_rhs = self.assemble_pressure_state_system(rhs, coef, cross_rhs)
            if use_nullspace:
                matrix_rhs = self.project_pressure_rhs_to_range(matrix_rhs)
                pressure = self.zero_mean_pressure(
                    self.linear_solver.solve_constant_nullspace(
                        matrix,
                        matrix_rhs,
                        self.cm,
                        solver=self.pressure_nullspace_linear_solver,
                    )
                )
            else:
                solver = (
                    self.pressure_linear_solver
                    if has_pressure_dirichlet
                    else self.pressure_gauge_linear_solver
                )
                solution = self.solve_linear_system(
                    matrix,
                    matrix_rhs,
                    solver=solver,
                )
                pressure = solution[: self.NC]
            self.last_pressure_nonorthogonal_iterations = 0
        else:
            correction_pressure = initial_pressure_state
            pressure = initial_pressure_state
            for iteration in range(1, max_iter + 1):
                if correction_pressure is not None:
                    explicit_cross_flux = self.pressure_nonorthogonal_cross_flux(
                        correction_pressure,
                        coef,
                        interpolation_method=self.controls.face_interpolation_method,
                    )
                    cross_rhs = self.divergence_from_flux(explicit_cross_flux)

                matrix, matrix_rhs = self.assemble_pressure_state_system(rhs, coef, cross_rhs)
                if use_nullspace:
                    matrix_rhs = self.project_pressure_rhs_to_range(matrix_rhs)
                    pressure = self.zero_mean_pressure(
                        self.linear_solver.solve_constant_nullspace(
                            matrix,
                            matrix_rhs,
                            self.cm,
                            solver=self.pressure_nullspace_linear_solver,
                        )
                    )
                else:
                    solver = (
                        self.pressure_linear_solver
                        if has_pressure_dirichlet
                        else self.pressure_gauge_linear_solver
                    )
                    solution = self.solve_linear_system(
                        matrix,
                        matrix_rhs,
                        solver=solver,
                    )
                    pressure = solution[: self.NC]
                self.last_pressure_nonorthogonal_iterations = iteration
                if iteration == max_iter:
                    break

                correction_pressure = pressure

        orthogonal_flux = self.pressure_orthogonal_flux(pressure, coef)
        flux_without_boundary = orthogonal_flux - explicit_cross_flux
        boundary_pressure_flux = self.add_pressure_dirichlet_flux(
            bm.zeros_like(orthogonal_flux),
            pressure,
            coef,
            self.pressure_dirichlet_value,
            self.pressure_dirichlet_threshold,
        )
        pressure_flux = self.add_pressure_dirichlet_flux(
            flux_without_boundary,
            pressure,
            coef,
            self.pressure_dirichlet_value,
            self.pressure_dirichlet_threshold,
        )
        flux_parts = {
            "orthogonal_flux": orthogonal_flux,
            "cross_flux": explicit_cross_flux,
            "boundary_pressure_flux": boundary_pressure_flux,
        }
        return pressure, pressure_flux, flux_parts

    # Local algebra for the PISO pressure-corrector step below.
    #
    # The generic collocated FVM pieces live in ``collocated_ns_components``.
    # The current default PISO step solves a pressure state from a pressure-free
    # velocity estimate and then applies the matching velocity and face-flux
    # correction:
    #
    #     U_free <- U + rAU grad(p_old),  rAU = V / a_p
    #     solve L(p_new) = -div(phi_free)
    #     U_new <- U_free - rAU grad(p_new)
    #     phi_new <- phi_free + phi_{p_new}
    #
    def rhie_chow_face_velocity(
        self,
        cell_velocity,
        a_p,
        pressure,
        target_flux=None,
        boundary_faces=None,
        boundary_velocity=None,
        face_response_coefficient=None,
    ):
        """Build a collocated face velocity for the current PISO substep."""
        if face_response_coefficient is None:
            face_response_coefficient = self.pressure_response_face_coefficient(
                a_p,
                self.controls.face_interpolation_method,
            )
        face_velocity = self.rhie_chow.Interpolation(
            self.cell_vector_to_dofs(cell_velocity),
            a_p,
            pressure,
            face_response_coefficient=face_response_coefficient,
        )
        if target_flux is not None:
            face_velocity = self.enforce_face_flux(face_velocity, target_flux)
        return apply_face_velocity_constraint(
            face_velocity,
            boundary_faces,
            boundary_velocity,
        )

    def pressure_free_flux(self, intermediate_velocity, pressure, a_p):
        """Return pressure-free velocity and matching interpolated face flux."""
        pressure_free_velocity = self.pressure_free_velocity(
            intermediate_velocity,
            pressure,
            a_p,
        )
        face_velocity = self.face_interpolate_cell_vector(
            pressure_free_velocity,
            method=self.controls.face_interpolation_method,
        )
        return pressure_free_velocity, self.face_flux(face_velocity)

    def transient_face_flux_correction(self, previous_cell_velocity, previous_face_velocity, a_p):
        """Return the Euler ``rAU_f * ddtCorr(U, face_flux)`` correction."""
        if (
            not self.controls.use_transient_flux_correction
            or previous_cell_velocity is None
            or previous_face_velocity is None
        ):
            return bm.zeros(self.mesh.number_of_faces(), dtype=self.cm.dtype)

        previous_face_flux = self.face_flux(previous_face_velocity)
        previous_cell_flux = self.face_flux(
            self.face_interpolate_cell_vector(
                previous_cell_velocity,
                method=self.controls.face_interpolation_method,
            )
        )
        flux_correction = previous_face_flux - previous_cell_flux
        denominator = bm.abs(previous_face_flux) + 1.0e-300
        limiter = 1.0 - bm.minimum(bm.abs(flux_correction) / denominator, 1.0)
        limiter = bm.where(self.fvm_geometry.is_boundary, 0.0, limiter)
        flux_correction = limiter * flux_correction
        response_coef = self.pressure_response_face_coefficient(
            a_p,
            self.controls.face_interpolation_method,
        )
        return response_coef * flux_correction / self.controls.tau

    def piso_neighbour_velocity_correction(
        self, corrected_velocity, predicted_velocity, momentum_matrix, a_p
    ):
        """Add the PISO neighbour-velocity compensation before correction two.

        The momentum matrix is stored on the left-hand side, so its off-diagonal
        action is the negative of ``sum_N a_PN delta_U_N`` in the control-volume
        derivation.  Therefore the explicit PISO compensation is

            -(A delta_U - diag(A) delta_U) / diag(A).
        """
        delta_u = corrected_velocity - predicted_velocity
        delta_dofs = self.cell_vector_to_dofs(delta_u)
        offdiag_delta = self.momentum_matrix_action(momentum_matrix, delta_dofs)
        offdiag_delta = offdiag_delta - a_p * delta_dofs
        offdiag_cell = self.dofs_to_cell_vector(offdiag_delta)
        a_p_cell = bm.stack(
            [
                a_p[component * self.NC : (component + 1) * self.NC]
                for component in range(self.GD)
            ],
            axis=-1,
        )
        return corrected_velocity - offdiag_cell / a_p_cell

    def pressure_correction_step(
        self,
        intermediate_velocity,
        pressure,
        a_p,
        boundary_faces,
        boundary_velocity,
        previous_cell_velocity=None,
        previous_face_velocity=None,
        return_diagnostics=False,
    ):
        """Perform one PISO pressure-correction step.

        The pressure equation is solved for the pressure state associated with
        the current pressure-free velocity estimate, using the pressure-state
        PISO corrector form without introducing a separate flux abstraction.
        """
        pressure_free_velocity, flux = self.pressure_free_flux(
            intermediate_velocity,
            pressure,
            a_p,
        )
        transient_flux = self.transient_face_flux_correction(
            previous_cell_velocity,
            previous_face_velocity,
            a_p,
        )
        flux = flux + transient_flux
        flux = apply_boundary_flux_constraint(
            flux,
            boundary_faces,
            boundary_velocity,
            self.fvm_geometry.S_f,
        )
        free_divergence = self.divergence_from_flux(flux)
        pressure_state, pressure_flux, pressure_flux_parts = (
            self.solve_pressure_state_equation(
                -free_divergence,
                a_p,
                initial_pressure_state=pressure,
            )
        )
        corrected_velocity = self.velocity_pressure_correction(
            pressure_free_velocity,
            pressure_state,
            a_p,
        )
        corrected_flux = flux + pressure_flux
        corrected_flux = apply_boundary_flux_constraint(
            corrected_flux,
            boundary_faces,
            boundary_velocity,
            self.fvm_geometry.S_f,
        )
        if not return_diagnostics:
            return corrected_velocity, pressure_state, corrected_flux

        diagnostics = pressure_correction_diagnostics(
            free_divergence=free_divergence,
            corrected_divergence=self.divergence_from_flux(corrected_flux),
            free_flux=flux,
            corrected_flux=corrected_flux,
            pressure_flux=pressure_flux,
            pressure_flux_parts=pressure_flux_parts,
            transient_flux=transient_flux,
            pressure_nonorthogonal_iterations=self.last_pressure_nonorthogonal_iterations,
        )
        return corrected_velocity, pressure_state, corrected_flux, diagnostics

    def piso_pressure_corrector_loop(
        self,
        predicted_velocity,
        initial_pressure,
        a_p,
        momentum_matrix,
        boundary_faces,
        boundary_velocity,
        previous_cell_velocity,
        previous_face_velocity,
        *,
        n_correctors,
        step,
        time,
        corrector_callback=None,
    ):
        """Run the repeated PISO pressure-corrector loop inside one time step.

        The first corrector uses the momentum-predicted velocity.  Later
        correctors first apply the neighbour-velocity compensation, then solve
        the same pressure-state correction step.
        """
        previous_corrector_velocity = predicted_velocity
        velocity = predicted_velocity
        pressure = initial_pressure
        face_flux = None
        diagnostics_enabled = (
            self.controls.diagnostics_enabled or corrector_callback is not None
        )
        for correction in range(1, n_correctors + 1):
            if correction == 1:
                intermediate_velocity = velocity
                splitting_linf = 0.0
            else:
                intermediate_velocity = self.piso_neighbour_velocity_correction(
                    velocity, previous_corrector_velocity, momentum_matrix, a_p
                )
                splitting_linf = float(
                    bm.to_numpy(bm.max(bm.abs(intermediate_velocity - velocity)))
                )

            correction_result = self.pressure_correction_step(
                intermediate_velocity,
                pressure,
                a_p,
                boundary_faces,
                boundary_velocity,
                previous_cell_velocity,
                previous_face_velocity,
                return_diagnostics=diagnostics_enabled,
            )
            if not diagnostics_enabled:
                next_velocity, next_pressure, face_flux = correction_result
            else:
                next_velocity, next_pressure, face_flux, diagnostics = correction_result
                record_piso_corrector_diagnostics(
                    self.corrector_diagnostics,
                    corrector_callback,
                    diagnostics,
                    step=step,
                    time=time,
                    correction=correction,
                    n_correctors=n_correctors,
                    splitting_linf=splitting_linf,
                    current_velocity=velocity,
                    next_velocity=next_velocity,
                    current_pressure=pressure,
                    next_pressure=next_pressure,
                    a_p=a_p,
                    current_face_flux=face_flux,
                    boundary_faces=boundary_faces,
                    boundary_velocity=boundary_velocity,
                    face_response_coefficient=self.pressure_response_face_coefficient(
                        a_p,
                        self.controls.face_interpolation_method,
                    ),
                    rhie_chow_face_velocity=self.rhie_chow_face_velocity,
                    face_flux_operator=self.face_flux,
                )

            previous_corrector_velocity = velocity
            velocity = next_velocity
            pressure = next_pressure

        return velocity, pressure, face_flux

    def advance_time_step(
        self,
        U0,
        Uf0,
        p0,
        *,
        t,
        step,
        n_correctors,
        corrector_callback=None,
    ):
        """Advance one time step through predictor, correctors, and face flux."""
        tau = self.controls.tau
        previous_cell_velocity = U0
        previous_face_velocity = Uf0
        predicted_velocity, a_p, momentum_matrix = self.temporary_velocity(
            U0, Uf0, p0, t + tau
        )
        boundary_faces, boundary_velocity = self.boundary_conditions.boundary_face_velocity(
            "velocity",
            mesh=self.mesh,
        )
        current_velocity, current_pressure, face_flux = self.piso_pressure_corrector_loop(
            predicted_velocity,
            p0,
            a_p,
            momentum_matrix,
            boundary_faces,
            boundary_velocity,
            previous_cell_velocity,
            previous_face_velocity,
            n_correctors=n_correctors,
            step=step,
            time=t + tau,
            corrector_callback=corrector_callback,
        )
        next_face_velocity = self.rhie_chow_face_velocity(
            current_velocity,
            a_p,
            current_pressure,
            face_flux,
            boundary_faces=boundary_faces,
            boundary_velocity=boundary_velocity,
            face_response_coefficient=self.pressure_response_face_coefficient(
                a_p,
                self.controls.face_interpolation_method,
            ),
        )
        return current_velocity, next_face_velocity, current_pressure, face_flux

    def solve(
        self,
        U0=None,
        Uf0=None,
        p0=None,
        n_correctors=None,
        snapshot_callback=None,
        snapshot_interval=None,
        snapshot_start_step=None,
        corrector_callback=None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        if U0 is None or Uf0 is None or p0 is None:
            U0, Uf0, p0 = self.initial_solution()
        self.corrector_diagnostics = []

        controls = self.controls
        n_correctors = (
            controls.n_correctors
            if n_correctors is None
            else int(n_correctors)
        )
        PisoSolverControls.validate_piso_controls(n_correctors)
        snapshot_interval = (
            controls.snapshot_interval
            if snapshot_interval is None
            else int(snapshot_interval)
        )
        snapshot_start_step = (
            controls.snapshot_start_step
            if snapshot_start_step is None
            else int(snapshot_start_step)
        )
        PisoSolverControls.validate_snapshot_controls(snapshot_interval, snapshot_start_step)

        current_velocity = None
        current_pressure = p0
        for n in range(controls.nt):
            t = controls.duration[0] + n * controls.tau
            step = n + 1
            U0, Uf0, p0, face_flux = self.advance_time_step(
                U0,
                Uf0,
                p0,
                t=t,
                step=step,
                n_correctors=n_correctors,
                corrector_callback=corrector_callback,
            )
            current_velocity = U0
            current_pressure = p0
            if (
                snapshot_callback is not None
                and step >= snapshot_start_step
                and (step - snapshot_start_step) % snapshot_interval == 0
            ):
                snapshot_callback(
                    step=step,
                    time=t + controls.tau,
                    model=self,
                    cell_velocity=U0,
                    face_velocity=Uf0,
                    pressure=p0,
                    flux=face_flux,
                )

        self.velocity = current_velocity
        self.velocity_components = [
            current_velocity[:, component] for component in range(self.GD)
        ]
        self.uh = self.velocity_components[0]
        self.vh = (
            self.velocity_components[1]
            if self.GD > 1
            else bm.zeros_like(self.uh)
        )
        self.ph = current_pressure
        return self.uh, self.vh, self.ph
