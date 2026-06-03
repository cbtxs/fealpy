"""Collocated PISO solver core for transient incompressible Navier-Stokes."""

import logging
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm, LinearForm
from fealpy.sparse import CSRTensor, spdiags

from .collocated_ns_fvm_utils import CollocatedNSFVMOperators
from .convection_integrator import ConvectionIntegrator
from .deviatoric_stress_source import DeviatoricStressSourceIntegrator
from .dirichlet_bc import DirichletBC
from .face_gradient import reconstruct_face_gradient
from .piso_solver_data import PisoSolverControls
from .rhie_chow import RhieChowInterpolation
from .scalar_cross_diffusion_integrator import ScalarCrossDiffusionIntegrator
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .scalar_source_integrator import ScalarSourceIntegrator
from fealpy.decorator import cartesian


class CollocatedPisoSolver(CollocatedNSFVMOperators):
    """Algorithm core for 2D transient collocated PISO solves."""

    validate_piso_controls = staticmethod(PisoSolverControls.validate_piso_controls)
    validate_snapshot_controls = staticmethod(
        PisoSolverControls.validate_snapshot_controls
    )
    validate_face_interpolation_method = staticmethod(
        PisoSolverControls.validate_face_interpolation_method
    )
    validate_transient_flux_correction_limiter = staticmethod(
        PisoSolverControls.validate_transient_flux_correction_limiter
    )
    validate_momentum_explicit_correction = staticmethod(
        PisoSolverControls.validate_momentum_explicit_correction
    )
    validate_nonorthogonal_controls = staticmethod(
        PisoSolverControls.validate_nonorthogonal_controls
    )

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
        self._init_piso_solver(
            mesh=mesh,
            diffusion_coef=diffusion_coef,
            convection_coef=convection_coef,
            source=source,
            boundary_conditions=boundary_conditions,
            controls=controls,
            initial_solution=initial_solution,
            linear_solver=linear_solver,
            linear_solver_config=linear_solver_config,
            logger=logger,
            log_level=log_level,
            pbar_log=pbar_log,
        )

    def _init_piso_solver(
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
        """Initialize the reusable PISO algorithm state."""
        self.controls = controls or PisoSolverControls()
        self.logger = logger or self._build_logger(
            self.__class__.__name__,
            pbar_log=pbar_log,
            log_level=log_level,
        )
        self.duration = tuple(self.controls.duration)
        self.nt = int(self.controls.nt)
        self.tau = self.controls.tau
        self.n_correctors = int(self.controls.n_correctors)
        self.snapshot_interval = int(self.controls.snapshot_interval)
        self.snapshot_start_step = int(self.controls.snapshot_start_step)
        self.use_transient_flux_correction = bool(
            self.controls.use_transient_flux_correction
        )
        self.transient_flux_correction_limiter = (
            self.controls.transient_flux_correction_limiter
        )
        self.momentum_explicit_correction = self.controls.momentum_explicit_correction
        self.face_interpolation_method = self.controls.face_interpolation_method
        self.momentum_nonorthogonal_max_iter = int(
            self.controls.momentum_nonorthogonal_max_iter
        )
        self.momentum_nonorthogonal_tol = float(
            self.controls.momentum_nonorthogonal_tol
        )
        self.pressure_nonorthogonal_max_iter = int(
            self.controls.pressure_nonorthogonal_max_iter
        )
        self.pressure_nonorthogonal_tol = float(
            self.controls.pressure_nonorthogonal_tol
        )
        self.mu = self._as_positive_scalar(diffusion_coef, "diffusion_coef")
        self.rho = self._as_positive_scalar(convection_coef, "convection_coef")
        self.diffusion_coef = self.mu
        self.convection_coef = self.rho
        self.source = source
        self.mesh = mesh
        self.cm = self.mesh.entity_measure("cell")
        self.points = self.mesh.entity_barycenter("cell")
        self.epoints = self.mesh.entity_barycenter("edge")
        self.NC = self.mesh.number_of_cells()
        self.boundary_conditions = self._validate_boundary_conditions(
            boundary_conditions
        )
        self.initial_solution_callback = initial_solution
        self._init_discretization(self.controls.space_degree)
        self.linear_solver = self._init_solver_backend(
            linear_solver,
            linear_solver_config,
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

    def _validate_boundary_conditions(self, boundary_conditions):
        """Validate boundary-condition data required by the PISO solver."""
        required = (
            "conditions_for",
            "dirichlet_threshold",
            "dirichlet_value",
            "has_pressure_dirichlet",
            "boundary_face_velocity",
        )
        missing = [
            name for name in required if not hasattr(boundary_conditions, name)
        ]
        if missing:
            raise TypeError(
                "boundary_conditions must provide " + ", ".join(required)
            )
        return boundary_conditions

    def _init_solver_backend(self, linear_solver, linear_solver_config):
        """Initialize the linear solver from explicit solver inputs."""
        if linear_solver is not None and hasattr(linear_solver, "solve"):
            return linear_solver

        options = {}
        if linear_solver is not None:
            options["linear_solver"] = linear_solver
        if linear_solver_config is not None:
            options["linear_solver_config"] = linear_solver_config
        return self._init_linear_solver(options)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  Time steps: {self.nt}\n"
            f"  PISO correctors: {self.n_correctors}\n"
            f"  Momentum nonorthogonal corrections: "
            f"{self.momentum_nonorthogonal_max_iter}\n"
            f"  Pressure nonorthogonal corrections: "
            f"{self.pressure_nonorthogonal_max_iter}\n"
            f"  Transient flux correction limiter: "
            f"{self.transient_flux_correction_limiter}\n"
            f"  Momentum explicit correction: "
            f"{self.momentum_explicit_correction}\n"
        )

    def _velocity_dirichlet_data(self):
        return self.boundary_conditions.dirichlet_value("velocity")

    def _velocity_dirichlet_threshold(self):
        return self.boundary_conditions.dirichlet_threshold("velocity")

    def _has_pressure_dirichlet(self):
        return self.boundary_conditions.has_pressure_dirichlet()

    def _pressure_dirichlet_threshold(self):
        if not self._has_pressure_dirichlet():
            return None
        return self.boundary_conditions.pressure_dirichlet_threshold()

    def _pressure_dirichlet_data(self):
        if not self._has_pressure_dirichlet():
            return None
        return self.boundary_conditions.pressure_dirichlet_value()

    def set_space(self, degree: int) -> None:
        """Rebuild the collocated discretization with a new polynomial degree."""
        self._init_discretization(degree)

    def _init_discretization(self, degree: int) -> None:
        velocity_dirichlet = self._velocity_dirichlet_data()
        velocity_dirichlet_threshold = self._velocity_dirichlet_threshold()
        pressure_dirichlet = self._pressure_dirichlet_data()
        pressure_dirichlet_threshold = self._pressure_dirichlet_threshold()
        self._init_collocated_discretization(
            degree,
            velocity_dirichlet,
            pressure_gradient_method=self.controls.pressure_gradient_method,
            velocity_gradient_method=self.controls.velocity_gradient_method,
            velocity_dirichlet_threshold=velocity_dirichlet_threshold,
            pressure_dirichlet=pressure_dirichlet,
            pressure_dirichlet_threshold=pressure_dirichlet_threshold,
            with_velocity_dirichlet_bc=True,
        )
        self.rhie_chow = RhieChowInterpolation(
            self.mesh,
            pressure_gradient_method=(
                self.controls.rhie_chow_pressure_gradient_method
            ),
            velocity_interpolation=self.face_interpolation_method,
            pressure_dirichlet=pressure_dirichlet,
            pressure_dirichlet_threshold=pressure_dirichlet_threshold,
        )

    def initial_solution(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        if self.initial_solution_callback is None:
            raise NotImplementedError("initial_solution must be supplied.")
        return self.initial_solution_callback()

    def _pressure_gradient_integrator(self, pressure):
        grad_p = self.pressure_gradient.cell_gradient(pressure)
        p1 = bm.einsum("i,i->i", grad_p[:, 0], self.cm)
        p2 = bm.einsum("i,i->i", grad_p[:, 1], self.cm)
        return bm.concatenate((p1, p2))

    def temporary_velocity(
        self,
        U0,
        Uf0,
        p0,
        t,
        return_matrix=False,
    ):
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2, coef=self.mu))
        bform.add_integrator(
            ConvectionIntegrator(
                q=self.p + 2,
                coef=self.rho * Uf0,
                interpolation=self.face_interpolation_method,
            )
        )
        A = bform.assembly()
        A = self._apply_velocity_natural_convection(A, Uf0)

        M = CSRTensor(
            crow=bm.arange(2*self.NC + 1),
            col=bm.arange(2*self.NC),
            values=bm.concatenate([
                self.rho * self.cm / self.tau,
                self.rho * self.cm / self.tau,
            ]),
            spshape=(2*self.NC, 2*self.NC),
        )

        @cartesian
        def src(p):
            return self.source(p, t)

        f = LinearForm(self.velocity_space).add_integrator(
            ScalarSourceIntegrator(src, q=self.p + 2)
        ).assembly()

        p_grad_integrator = self._pressure_gradient_integrator(p0)
        A = A + M
        b = (
            f
            + (U0 * (self.rho * self.cm / self.tau)[:, None]).flatten(order="F")
        )
        b = b - p_grad_integrator
        velocity_dirichlet_threshold = self._velocity_dirichlet_threshold()
        A, b = self.velocity_dirichlet_bc.DiffusionApply(
            A,
            b,
            coef=self.mu,
            threshold=velocity_dirichlet_threshold,
        )
        b = self.velocity_dirichlet_bc.ConvectionApply(
            b,
            self.rho * Uf0,
            threshold=velocity_dirichlet_threshold,
        )
        if self.momentum_explicit_correction == "openfoam":
            b = b + self.boundary_corrected_momentum_explicit_source(U0)
        # FEALPy sparse assembly can leave duplicate entries here.
        A = A.tocoo().coalesce().tocsr()
        a_p = A.diags().values
        U = self.linear_solver.solve(A, b)
        if self.momentum_explicit_correction == "current":
            U = self.correct_momentum_nonorthogonal_diffusion(A, b, U, U0)
        else:
            self.last_momentum_nonorthogonal_iterations = 1
        if return_matrix:
            return U, a_p, A
        return U, a_p

    def boundary_corrected_momentum_explicit_source(self, velocity):
        """Return boundary-corrected explicit viscous RHS for momentum prediction.

        This route is the momentum-side counterpart of the pressure non-orthogonal
        cross-flux route: internal face gradients are built with linear face
        interpolation, and velocity boundary gradients are corrected with patch
        normal derivatives before the explicit cross and deviatoric-stress
        sources are assembled.
        """
        face_gradient = self.boundary_corrected_momentum_face_gradient(velocity)
        cell_velocity = self._cell_velocity(velocity)
        flat_velocity = self._flatten_velocity(cell_velocity)
        cross_source = LinearForm(self.velocity_space).add_integrator(
            ScalarCrossDiffusionIntegrator(
                flat_velocity,
                face_gradient,
                coef=self.mu,
                geometry=self.fvm_geometry,
                boundary_policy="zero",
            )
        ).assembly()
        stress_source = LinearForm(self.velocity_space).add_integrator(
            DeviatoricStressSourceIntegrator(
                face_gradient,
                coef=self.mu,
            )
        ).assembly()
        return cross_source + stress_source

    def boundary_corrected_momentum_face_gradient(self, velocity):
        """Return boundary-corrected face gradient for explicit momentum sources.

        Keep this semantic in sync with the pressure non-orthogonal correction
        when comparing solver routes.  Pressure cross flux follows
        ``face_interpolation_method``; this route deliberately uses linear
        interpolation and patch normal-gradient boundary correction.  The legacy
        ``momentum_explicit_correction="current"`` path does not use this
        boundary-corrected face-gradient construction.
        """
        cell_velocity = self._cell_velocity(velocity)
        cell_gradient = self.velocity_gradient.cell_gradient(cell_velocity)
        kwargs = {}

        dirichlet_faces, dirichlet_values = self._boundary_face_velocity()
        if dirichlet_faces is not None and dirichlet_faces.shape[0] > 0:
            kwargs["dirichlet_faces"] = dirichlet_faces
            kwargs["dirichlet_values"] = dirichlet_values

        natural_faces = self._velocity_natural_faces()
        if natural_faces is not None and natural_faces.shape[0] > 0:
            kwargs["neumann_faces"] = natural_faces
            kwargs["neumann_sn_grad"] = bm.zeros(
                (natural_faces.shape[0], cell_velocity.shape[1]),
                dtype=cell_velocity.dtype,
            )

        return reconstruct_face_gradient(
            self.mesh,
            cell_gradient,
            cell_values=cell_velocity,
            interpolation_method="linear",
            **kwargs,
        )

    def _velocity_natural_threshold(self):
        if self.boundary_conditions is None:
            return None
        if not self.boundary_conditions.conditions_for("velocity", "natural"):
            return None
        return self.boundary_conditions.natural_threshold("velocity")

    def _velocity_natural_faces(self):
        threshold = self._velocity_natural_threshold()
        if threshold is None:
            return None

        bd_edge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[bd_edge]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return bd_edge[:0]
        return bd_edge[flag]

    def _apply_velocity_natural_convection(self, matrix, face_velocity):
        """Add zero-gradient velocity outlet convection as owner diagonal."""
        selected = self._velocity_natural_faces()
        if selected is None or selected.shape[0] == 0:
            return matrix

        flux = bm.einsum(
            "ij,ij->i",
            self.rho * face_velocity[selected],
            self.fvm_geometry.S_f[selected],
        )
        owner = self.fvm_geometry.owner[selected]
        diagonal = bm.zeros(self.NC, dtype=flux.dtype)
        diagonal = bm.index_add(diagonal, owner, flux, axis=0)
        diagonal = bm.concatenate([diagonal, diagonal], axis=0)
        return matrix + spdiags(diagonal, 0, matrix.shape[0], matrix.shape[1])

    def correct_momentum_nonorthogonal_diffusion(
        self,
        matrix,
        rhs,
        velocity,
        previous_velocity,
    ):
        """Picard-correct the explicit non-orthogonal momentum diffusion RHS."""
        max_iter = self.momentum_nonorthogonal_max_iter
        if max_iter == 0:
            self.last_momentum_nonorthogonal_iterations = 0
            return velocity

        return self._correct_momentum_nonorthogonal_diffusion(
            matrix,
            rhs,
            velocity,
            previous_velocity,
            max_iter=max_iter,
            tol=self.momentum_nonorthogonal_tol,
            iteration_attr="last_momentum_nonorthogonal_iterations",
        )

    def pressure_response_face_coefficient(self, a_p):
        """Return face response ``D_f`` used by the PISO pressure-state solve."""
        dp = self.cm/a_p[:self.NC]
        return self.face_interpolate_cell_scalar(
            dp,
            method=self.face_interpolation_method,
        )

    def solve_pressure_correction(
        self,
        rhs,
        a_p,
        nonorthogonal_max_iter: Optional[int] = None,
        nonorthogonal_tol: Optional[float] = None,
        initial_pressure_state: Optional[TensorLike] = None,
    ):
        """Solve the PISO pressure-state equation.

        The historical method name refers to the pressure-corrector step.  The
        returned array is the corrected pressure state, not a pressure
        increment ``p'``.
        """
        pressure, _, _ = self._solve_pressure_correction_state_and_flux(
            rhs,
            a_p,
            nonorthogonal_max_iter=nonorthogonal_max_iter,
            nonorthogonal_tol=nonorthogonal_tol,
            initial_pressure_state=initial_pressure_state,
        )
        return pressure

    def _assemble_pressure_state_system(self, rhs, coef, cross_rhs):
        """Assemble the pressure-state linear system for one nonOrth solve."""
        has_dirichlet = (
            self._pressure_dirichlet_data() is not None
            and self._pressure_dirichlet_threshold() is not None
        )
        if has_dirichlet:
            matrix = BilinearForm(self.space).add_integrator(
                ScalarDiffusionIntegrator(q=self.p + 2, coef=coef)
            ).assembly()
            pressure_bc = DirichletBC(self.mesh, self._pressure_dirichlet_data())
            return pressure_bc.DiffusionApply(
                matrix,
                rhs + cross_rhs,
                coef=self._boundary_face_coefficient(coef),
                threshold=self._pressure_dirichlet_threshold(),
            )

        matrix = self._assemble_pressure_gauge_matrix(coef, q=self.p + 2)
        rhs = bm.concatenate(
            [rhs + cross_rhs, bm.zeros(1, dtype=rhs.dtype)],
            axis=0,
        )
        return matrix, rhs

    def _solve_pressure_correction_state_and_flux(
        self,
        rhs,
        a_p,
        nonorthogonal_max_iter: Optional[int] = None,
        nonorthogonal_tol: Optional[float] = None,
        initial_pressure_state: Optional[TensorLike] = None,
    ):
        coef = self.pressure_response_face_coefficient(a_p)
        nonorthogonal_max_iter = (
            self.pressure_nonorthogonal_max_iter
            if nonorthogonal_max_iter is None
            else int(nonorthogonal_max_iter)
        )
        nonorthogonal_tol = (
            self.pressure_nonorthogonal_tol
            if nonorthogonal_tol is None
            else float(nonorthogonal_tol)
        )
        if nonorthogonal_max_iter < 0:
            raise ValueError("nonorthogonal_max_iter must be non-negative.")
        if nonorthogonal_tol <= 0.0:
            raise ValueError("nonorthogonal_tol must be positive.")

        # OpenFOAM's nNonOrthogonalCorrectors is a fixed loop count.  The
        # tolerance is kept as a validated option for API compatibility, but
        # this PISO route does not stop the final pressure-system sequence early.
        if nonorthogonal_max_iter > 0 and initial_pressure_state is not None:
            explicit_cross_flux = self.pressure_correction_cross_flux(
                initial_pressure_state,
                coef,
            )
            cross_rhs = self.divergence_from_flux(explicit_cross_flux)
        else:
            cross_rhs = bm.zeros_like(rhs)
            explicit_cross_flux = bm.zeros(self.mesh.number_of_faces(), dtype=rhs.dtype)
        pressure = None
        n_solves = nonorthogonal_max_iter + 1
        for iteration in range(1, n_solves + 1):
            matrix, matrix_rhs = self._assemble_pressure_state_system(
                rhs,
                coef,
                cross_rhs,
            )
            solution = self.linear_solver.solve(matrix, matrix_rhs)
            pressure = solution[: self.NC]
            self.last_pressure_nonorthogonal_iterations = iteration
            if iteration == n_solves:
                break

            explicit_cross_flux = self.pressure_correction_cross_flux(pressure, coef)
            cross_rhs = self.divergence_from_flux(explicit_cross_flux)

        orthogonal_flux = self.pressure_correction_orthogonal_flux(pressure, coef)
        flux_without_boundary = orthogonal_flux - explicit_cross_flux
        boundary_pressure_flux = self._add_pressure_dirichlet_boundary_flux(
            bm.zeros_like(orthogonal_flux),
            pressure,
            coef,
        )
        pressure_flux = self._add_pressure_dirichlet_boundary_flux(
            flux_without_boundary,
            pressure,
            coef,
        )
        flux_parts = {
            "orthogonal_flux": orthogonal_flux,
            "cross_flux": explicit_cross_flux,
            "boundary_pressure_flux": boundary_pressure_flux,
        }
        return pressure, pressure_flux, flux_parts

    def _boundary_face_velocity(self):
        if self.boundary_conditions is not None:
            try:
                return self.boundary_conditions.boundary_face_velocity(
                    "velocity", mesh=self.mesh
                )
            except TypeError:
                return self.boundary_conditions.boundary_face_velocity("velocity")
        return super()._boundary_face_velocity()

    def _apply_selected_face_velocity_dirichlet(
        self,
        face_velocity,
        boundary_faces,
        boundary_velocity,
    ):
        if boundary_velocity is None:
            return face_velocity
        if boundary_faces is None:
            return self.apply_face_velocity_dirichlet(
                face_velocity,
                boundary_velocity,
            )
        return bm.set_at(
            bm.array(face_velocity),
            boundary_faces,
            bm.array(boundary_velocity),
        )

    def _apply_selected_boundary_flux_constraint(
        self,
        flux,
        boundary_faces,
        boundary_velocity,
    ):
        if boundary_velocity is None:
            return flux
        if boundary_faces is None:
            return self.apply_boundary_flux_constraint(flux, boundary_velocity)
        target_flux = bm.einsum(
            "ij,ij->i",
            bm.array(boundary_velocity),
            self.fvm_geometry.S_f[boundary_faces],
        )
        return bm.set_at(bm.array(flux), boundary_faces, target_flux)

    # Local algebra for the PISO pressure-corrector step below.
    #
    # The generic collocated FVM pieces live in ``collocated_ns_fvm_utils``.
    # The current default PISO step solves a pressure state from a pressure-free
    # velocity estimate and then applies the matching velocity and face-flux
    # correction:
    #
    #     U_free <- U + rAU grad(p_old),  rAU = V / a_p
    #     solve L(p_new) = -div(phi_free)
    #     U_new <- U_free - rAU grad(p_new)
    #     phi_new <- phi_free + phi_{p_new}
    #
    def pressure_correction_flux(self, pressure_state, a_p):
        """Return the PISO pressure-state flux contribution.

        The method name is retained for compatibility with pressure-corrector
        diagnostics.  The argument is a pressure state, not an increment.
        """
        coef = self.pressure_response_face_coefficient(a_p)
        return self.pressure_correction_flux_from_coefficient(
            pressure_state, coef
        )

    def pressure_correction_flux_from_coefficient(
        self,
        pressure_state,
        coef,
        explicit_cross_flux=None,
    ):
        """Return pressure-state flux from a face response coefficient."""
        if explicit_cross_flux is None:
            explicit_cross_flux = self.pressure_correction_cross_flux(
                pressure_state,
                coef,
            )
        flux = (
            self.pressure_correction_orthogonal_flux(pressure_state, coef)
            - explicit_cross_flux
        )
        return self._add_pressure_dirichlet_boundary_flux(
            flux,
            pressure_state,
            coef,
        )

    def _add_pressure_dirichlet_boundary_flux(self, flux, pressure, coef):
        """Add pressure-state flux on pressure Dirichlet boundary faces."""
        threshold = self._pressure_dirichlet_threshold()
        pressure_dirichlet = self._pressure_dirichlet_data()
        if threshold is None or pressure_dirichlet is None:
            return flux

        bd_edge = bm.nonzero(self.fvm_geometry.is_boundary)[0]
        face_centers = self.fvm_geometry.face_center[bd_edge]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return flux

        selected = bd_edge[flag]
        _, ef_abs, _ = self.fvm_geometry.over_relaxed_decomposition()
        coefficient = (
            ef_abs[selected]
            / self.fvm_geometry.mag_d_f[selected]
            * bm.array(coef)[selected]
        )
        owner = self.fvm_geometry.owner[selected]
        bd_value = pressure_dirichlet(face_centers[flag])
        bd_flux = coefficient * (pressure[owner] - bd_value)
        return bm.set_at(flux, selected, bd_flux)

    def pressure_correction_orthogonal_flux(self, pressure_state, coef):
        """Return the implicit orthogonal flux induced by a pressure state."""
        e2c = self.e2c
        _, ef_abs, _ = self.fvm_geometry.over_relaxed_decomposition()
        kf = ef_abs / self.fvm_geometry.mag_d_f * coef
        return kf*(pressure_state[e2c[:, 0]] - pressure_state[e2c[:, 1]])

    def pressure_correction_cross_flux(self, pressure_state, coef):
        """Return the explicit non-orthogonal flux induced by a pressure state."""
        return self._pressure_correction_cross_flux(pressure_state, coef)

    def rhie_chow_face_velocity(
        self,
        u_flat,
        a_p,
        pressure,
        target_flux=None,
        boundary_faces=None,
        boundary_velocity=None,
        face_response_coefficient=None,
    ):
        """Build a collocated face velocity for the current PISO substep."""
        if face_response_coefficient is None:
            face_response_coefficient = self.pressure_response_face_coefficient(a_p)
        face_velocity = self.rhie_chow.Interpolation(
            u_flat,
            a_p,
            pressure,
            face_response_coefficient=face_response_coefficient,
        )
        if target_flux is not None:
            face_velocity = self.enforce_face_flux(face_velocity, target_flux)
        return self._apply_selected_face_velocity_dirichlet(
            face_velocity,
            boundary_faces,
            boundary_velocity,
        )

    def cell_velocity_face_flux(self, cell_velocity):
        """Return the face flux from interpolated cell-centred velocity."""
        return self.face_flux(
            self.face_interpolate_cell_vector(
                cell_velocity,
                method=self.face_interpolation_method,
            )
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
            method=self.face_interpolation_method,
        )
        return pressure_free_velocity, self.face_flux(face_velocity)

    def transient_face_flux_correction(
        self,
        previous_cell_velocity,
        previous_face_velocity,
        a_p,
    ):
        """Return the Euler ``rAU_f * ddtCorr(U, phi)`` face-flux correction."""
        if (
            not self.use_transient_flux_correction
            or previous_cell_velocity is None
            or previous_face_velocity is None
        ):
            return bm.zeros(self.mesh.number_of_faces(), dtype=self.cm.dtype)

        previous_face_flux = self.face_flux(previous_face_velocity)
        previous_cell_flux = self.cell_velocity_face_flux(previous_cell_velocity)
        flux_correction = previous_face_flux - previous_cell_flux
        if self.transient_flux_correction_limiter == "openfoam":
            denominator = bm.abs(previous_face_flux) + 1.0e-300
            limiter = 1.0 - bm.minimum(bm.abs(flux_correction) / denominator, 1.0)
            limiter = bm.where(self.fvm_geometry.is_boundary, 0.0, limiter)
            flux_correction = limiter * flux_correction
        response_coef = self.pressure_response_face_coefficient(a_p)
        return response_coef * flux_correction / self.tau

    def operator_splitting_velocity_correction(
        self, corrected_velocity, predicted_velocity, momentum_matrix, a_p
    ):
        """Add the PISO neighbour-velocity compensation before correction two.

        The momentum matrix is stored on the left-hand side, so its off-diagonal
        action is the negative of ``sum_N a_PN delta_U_N`` in the control-volume
        derivation.  Therefore the explicit PISO compensation is

            -(A delta_U - diag(A) delta_U) / diag(A).
        """
        delta_u = corrected_velocity - predicted_velocity
        offdiag_delta = momentum_matrix @ delta_u - a_p * delta_u
        return corrected_velocity - offdiag_delta / a_p

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
        the current pressure-free velocity estimate, matching the OpenFOAM PISO
        pressure-corrector semantics without exposing an HbyA-style abstraction.
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
        flux = self._apply_selected_boundary_flux_constraint(
            flux,
            boundary_faces,
            boundary_velocity,
        )
        free_divergence = self.divergence_from_flux(flux)
        pressure_state, pressure_flux, pressure_flux_parts = (
            self._solve_pressure_correction_state_and_flux(
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
        corrected_flux = self._apply_selected_boundary_flux_constraint(
            corrected_flux,
            boundary_faces,
            boundary_velocity,
        )
        if not return_diagnostics:
            return corrected_velocity, pressure_state, corrected_flux

        def linf(value):
            return float(bm.to_numpy(bm.max(bm.abs(value))))

        coef = self.pressure_response_face_coefficient(a_p)
        orthogonal_flux = pressure_flux_parts["orthogonal_flux"]
        cross_flux = pressure_flux_parts["cross_flux"]
        boundary_pressure_flux = pressure_flux_parts["boundary_pressure_flux"]
        diagnostics = {
            "pressure_free_divergence_linf": linf(free_divergence),
            "pressure_corrected_divergence_linf": linf(
                self.divergence_from_flux(corrected_flux)
            ),
            "pressure_free_flux_linf": linf(flux),
            "pressure_corrected_flux_linf": linf(corrected_flux),
            "pressure_flux_linf": linf(pressure_flux),
            "pressure_orthogonal_flux_linf": linf(orthogonal_flux),
            "pressure_cross_flux_linf": linf(cross_flux),
            "pressure_boundary_flux_linf": linf(boundary_pressure_flux),
            "transient_flux_correction_linf": linf(transient_flux),
            "pressure_nonorthogonal_iterations": int(
                self.last_pressure_nonorthogonal_iterations
            ),
        }
        return corrected_velocity, pressure_state, corrected_flux, diagnostics

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

        n_correctors = (
            self.n_correctors if n_correctors is None else int(n_correctors)
        )
        self.validate_piso_controls(n_correctors)
        snapshot_interval = (
            self.snapshot_interval
            if snapshot_interval is None
            else int(snapshot_interval)
        )
        snapshot_start_step = (
            self.snapshot_start_step
            if snapshot_start_step is None
            else int(snapshot_start_step)
        )
        self.validate_snapshot_controls(snapshot_interval, snapshot_start_step)

        current_velocity = None
        current_pressure = p0
        for n in range(self.nt):
            t = self.duration[0] + n * self.tau
            step = n + 1
            previous_cell_velocity = U0
            previous_face_velocity = Uf0
            u_tem, a_p, momentum_matrix = self.temporary_velocity(
                U0, Uf0, p0, t + self.tau, return_matrix=True
            )
            boundary_faces, boundary_velocity = self._boundary_face_velocity()

            previous_velocity = u_tem
            current_velocity = u_tem
            current_pressure = p0
            phi = None
            correction = 0
            while True:
                correction += 1
                if correction == 1:
                    intermediate_velocity = current_velocity
                    splitting_linf = 0.0
                else:
                    intermediate_velocity = self.operator_splitting_velocity_correction(
                        current_velocity, previous_velocity, momentum_matrix, a_p
                    )
                    splitting_linf = float(
                        bm.to_numpy(
                            bm.max(bm.abs(intermediate_velocity - current_velocity))
                        )
                    )

                correction_result = self.pressure_correction_step(
                    intermediate_velocity,
                    current_pressure,
                    a_p,
                    boundary_faces,
                    boundary_velocity,
                    previous_cell_velocity,
                    previous_face_velocity,
                    return_diagnostics=corrector_callback is not None,
                )
                if corrector_callback is None:
                    next_velocity, next_pressure, phi = correction_result
                    diagnostics = None
                else:
                    next_velocity, next_pressure, phi, diagnostics = correction_result
                    target_face_velocity = self.rhie_chow_face_velocity(
                        next_velocity,
                        a_p,
                        next_pressure,
                        phi,
                        boundary_faces=boundary_faces,
                        boundary_velocity=boundary_velocity,
                        face_response_coefficient=(
                            self.pressure_response_face_coefficient(a_p)
                        ),
                    )
                    target_flux_error = self.face_flux(target_face_velocity) - phi
                    diagnostics.update(
                        {
                            "step": int(step),
                            "time": float(t + self.tau),
                            "corrector": int(correction),
                            "n_correctors": int(n_correctors),
                            "operator_splitting_compensation_linf": splitting_linf,
                            "velocity_update_linf": float(
                                bm.to_numpy(
                                    bm.max(bm.abs(next_velocity - current_velocity))
                                )
                            ),
                            "pressure_update_linf": float(
                                bm.to_numpy(
                                    bm.max(bm.abs(next_pressure - current_pressure))
                                )
                            ),
                            "rhie_chow_flux_error_linf": float(
                                bm.to_numpy(bm.max(bm.abs(target_flux_error)))
                            ),
                        }
                    )
                    corrector_callback(**diagnostics)
                should_continue = correction < n_correctors
                previous_velocity = current_velocity
                current_velocity = next_velocity
                current_pressure = next_pressure
                if not should_continue:
                    break

            Uf0 = self.rhie_chow_face_velocity(
                current_velocity,
                a_p,
                current_pressure,
                phi,
                boundary_faces=boundary_faces,
                boundary_velocity=boundary_velocity,
                face_response_coefficient=self.pressure_response_face_coefficient(a_p),
            )

            U0 = bm.stack(
                [current_velocity[:self.NC], current_velocity[self.NC:]], axis=-1
            )
            p0 = current_pressure
            if (
                snapshot_callback is not None
                and step >= snapshot_start_step
                and (step - snapshot_start_step) % snapshot_interval == 0
            ):
                snapshot_callback(
                    step=step,
                    time=t + self.tau,
                    model=self,
                    cell_velocity=U0,
                    face_velocity=Uf0,
                    pressure=p0,
                    flux=phi,
                )

        self.uh = current_velocity[:self.NC]
        self.vh = current_velocity[self.NC:]
        self.ph = current_pressure
        return self.uh, self.vh, self.ph
