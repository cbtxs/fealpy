"""Collocated SIMPLE solver core for steady incompressible Navier-Stokes."""

import logging
from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.fem import BilinearForm, LinearForm
from fealpy.sparse import spdiags

from .collocated_ns_fvm_utils import CollocatedNSFVMOperators
from .convection_integrator import ConvectionIntegrator
from .fvm_linear_solver import FVMLinearSolverConfig
from .rhie_chow import RhieChowInterpolation
from .scalar_diffusion_integrator import ScalarDiffusionIntegrator
from .scalar_source_integrator import ScalarSourceIntegrator
from .simple_iteration_control import SimpleIterationControl
from .simple_solver_data import SimpleSolverControls
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
        self._init_simple_solver(
            mesh=mesh,
            diffusion_coef=diffusion_coef,
            convection_coef=convection_coef,
            source=source,
            boundary_conditions=boundary_conditions,
            controls=controls,
            linear_solver=linear_solver,
            linear_solver_config=linear_solver_config,
            logger=logger,
            log_level=log_level,
            pbar_log=pbar_log,
        )

    def _init_simple_solver(
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
        self.logger = logger or self._build_logger(
            self.__class__.__name__,
            pbar_log=pbar_log,
            log_level=log_level,
        )
        self.diffusion_coef = self._as_positive_scalar(
            diffusion_coef, "diffusion_coef"
        )
        self.convection_coef = self._as_positive_scalar(
            convection_coef, "convection_coef"
        )
        self.source = source
        self.mesh = mesh
        self.cm = self.mesh.entity_measure("cell")
        self.NC = self.mesh.number_of_cells()
        self.iteration_control = SimpleIterationControl(self.mesh, self.logger)
        self.boundary_conditions = self._validate_boundary_conditions(
            boundary_conditions
        )
        self._init_discretization(degree=self.controls.space_degree)
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
        """Validate boundary-condition data required by the SIMPLE solver."""
        required = ("dirichlet_threshold", "dirichlet_value", "boundary_face_velocity")
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
        options["linear_solver_config"] = (
            FVMLinearSolverConfig()
            if linear_solver_config is None
            else linear_solver_config
        )
        return self._init_linear_solver(options)

    def _velocity_dirichlet_data(self):
        """Return velocity Dirichlet value callable for the active BC path."""
        return self.boundary_conditions.dirichlet_value("velocity")

    def _velocity_dirichlet_threshold(self):
        """Return active velocity Dirichlet patch threshold."""
        return self.boundary_conditions.dirichlet_threshold("velocity")

    def _velocity_natural_threshold(self):
        """Return active velocity natural patch threshold."""
        if not self.boundary_conditions.conditions_for("velocity", "natural"):
            return None
        return self.boundary_conditions.natural_threshold("velocity")

    def _has_pressure_dirichlet(self):
        """Return whether the active engineering BC has pressure Dirichlet data."""
        return self.boundary_conditions.has_pressure_dirichlet()

    def _pressure_dirichlet_threshold(self):
        """Return pressure Dirichlet threshold for engineering outlets."""
        if not self._has_pressure_dirichlet():
            return None
        return self.boundary_conditions.pressure_dirichlet_threshold()

    def _pressure_dirichlet_data(self):
        """Return pressure Dirichlet values for momentum pressure gradients."""
        if not self._has_pressure_dirichlet():
            return None
        return self.boundary_conditions.pressure_dirichlet_value()

    @staticmethod
    def _zero_pressure_correction(points):
        """Pressure-fixed boundaries impose zero pressure correction."""
        return bm.zeros(points.shape[0], dtype=points.dtype)

    def _init_discretization(self, degree: int = 0) -> None:
        """Initialize spaces and reusable FVM operators."""
        self._init_collocated_discretization(
            degree,
            self._velocity_dirichlet_data(),
            pressure_gradient_method=self.controls.pressure_gradient_method,
            velocity_gradient_method=self.controls.velocity_gradient_method,
            velocity_dirichlet_threshold=self._velocity_dirichlet_threshold(),
            pressure_dirichlet=self._pressure_dirichlet_data(),
            pressure_dirichlet_threshold=self._pressure_dirichlet_threshold(),
            with_divergence=True,
            with_velocity_dirichlet_bc=True,
        )

    def temporary_velocity(self, p, uf, u0) -> Tuple[TensorLike, TensorLike]:
        """Solve momentum equation for the intermediate velocity."""
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(
            ScalarDiffusionIntegrator(q=self.p + 2, coef=self.diffusion_coef)
        )
        convection_face_velocity = self._convection_face_velocity(uf)
        bform.add_integrator(
            ConvectionIntegrator(
                q=self.p + 2,
                coef=convection_face_velocity,
                interpolation=self.controls.face_interpolation(
                    "momentum_face_interpolation"
                ),
            )
        )
        B = bform.assembly()
        B = self._apply_velocity_natural_convection(B, uf)
        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.source, q=self.p + 2))
        f = lform.assembly()
        threshold = self._velocity_dirichlet_threshold()
        B, f = self.velocity_dirichlet_bc.DiffusionApply(
            B, f, coef=self.diffusion_coef, threshold=threshold
        )
        f = self.velocity_dirichlet_bc.ConvectionApply(
            f, convection_face_velocity, threshold=threshold
        )
        B = B.tocoo().coalesce().tocsr()
        ap = B.diags().values
        grad_p = self.pressure_gradient.cell_gradient(p)
        p1 = bm.einsum("i,i->i", grad_p[:, 0], self.cm)
        p2 = bm.einsum("i,i->i", grad_p[:, 1], self.cm)
        p_grad_integrator = bm.concatenate((p1, p2))
        f = f - p_grad_integrator
        u = self.linear_solver.solve(B, f)

        u = self._correct_momentum_nonorthogonal_diffusion(
            B,
            f,
            u,
            u0,
            max_iter=self.controls.momentum_nonorthogonal_max_iter,
            tol=self.controls.momentum_nonorthogonal_tol,
            iteration_attr="last_nonorthogonal_iterations",
        )

        return ap, u

    def _convection_face_velocity(self, face_velocity):
        """Return the face velocity used by the momentum convection operator."""
        return self.convection_coef * face_velocity

    def _apply_velocity_natural_convection(self, matrix, uf):
        """Add zero-gradient velocity outlet convection as owner diagonal."""
        threshold = self._velocity_natural_threshold()
        if threshold is None:
            return matrix

        bd_edge = self.mesh.boundary_face_index()
        face_centers = self.mesh.entity_barycenter("face")[bd_edge]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return matrix

        selected = bd_edge[flag]
        flux = bm.einsum(
            "ij,ij->i",
            self._convection_face_velocity(uf)[selected],
            self.mesh.edge_normal()[selected],
        )
        owner = self.e2c[selected, 0]
        diagonal = bm.zeros(self.NC, dtype=flux.dtype)
        diagonal = bm.index_add(diagonal, owner, flux, axis=0)
        diagonal = bm.concatenate([diagonal, diagonal], axis=0)
        return matrix + spdiags(diagonal, 0, matrix.shape[0], matrix.shape[1])

    def _pressure_response_face_coefficient(self, ap: TensorLike) -> TensorLike:
        """Return the SIMPLE pressure-correction face response."""
        dp = self.cm / ap[:self.NC]
        return self.face_interpolate_cell_scalar(
            dp,
            method=self.controls.face_interpolation(
                "pressure_response_interpolation"
            ),
        )

    def _pressure_correction_orthogonal_flux(
        self,
        p_corr: TensorLike,
        response_coef: TensorLike,
    ) -> TensorLike:
        """Return the implicit orthogonal pressure-correction face flux."""
        Sf = self.nonorthogonal_geometry.face_area_vector()
        d = self.nonorthogonal_geometry.cell_center_vector()
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        d_dot_Sf = bm.einsum("ij,ij->i", d, Sf)
        coefficient = response_coef * Sf_dot_Sf / d_dot_Sf
        jump = p_corr[self.e2c[:, 0]] - p_corr[self.e2c[:, 1]]
        return coefficient * jump

    def pressure_correction_flux(
        self,
        p_corr: TensorLike,
        response_coef: TensorLike,
    ) -> TensorLike:
        """Return the full pressure-correction flux used to correct mass flux."""
        flux = (
            self._pressure_correction_orthogonal_flux(p_corr, response_coef)
            - self._pressure_correction_cross_flux(p_corr, response_coef)
        )
        return self._add_pressure_dirichlet_boundary_flux(
            flux,
            p_corr,
            response_coef,
        )

    def _add_pressure_dirichlet_boundary_flux(
        self,
        flux: TensorLike,
        p_corr: TensorLike,
        response_coef: TensorLike,
    ) -> TensorLike:
        """Add pressure-correction flux on pressure Dirichlet boundary faces."""
        threshold = self._pressure_dirichlet_threshold()
        if threshold is None:
            return flux

        bd_edge = self.mesh.boundary_face_index()
        face_centers = self.mesh.entity_barycenter("face")[bd_edge]
        flag = threshold(face_centers)
        if not bool(bm.to_numpy(bm.any(flag))):
            return flux

        selected = bd_edge[flag]
        Sf = self.nonorthogonal_geometry.face_area_vector()[selected]
        d = self.nonorthogonal_geometry.cell_center_vector()[selected]
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        d_dot_Sf = bm.einsum("ij,ij->i", d, Sf)
        response_coef = bm.array(response_coef)
        coefficient = response_coef[selected] * Sf_dot_Sf / d_dot_Sf
        owner = self.e2c[selected, 0]
        bd_value = self._zero_pressure_correction(face_centers[flag])
        bd_flux = coefficient * (p_corr[owner] - bd_value)
        return bm.set_at(flux, selected, bd_flux)

    def correct_face_velocity_with_pressure_correction(
        self,
        uf: TensorLike,
        p_corr: TensorLike,
        response_coef: TensorLike,
        bd_edge: TensorLike,
        bdedgeu: TensorLike,
    ) -> TensorLike:
        """Correct only the normal face velocity component from ``p_corr``."""
        Sf = self.nonorthogonal_geometry.face_area_vector()
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
            self._pressure_response_face_coefficient(ap)
            if response_coef is None
            else response_coef
        )
        div_u = self.divergence.Reconstruct(uf)
        return self._solve_pressure_correction_with_cross_rhs(
            -div_u,
            dp_edge,
            q=2,
            nonorthogonal_max_iter=nonorthogonal_max_iter,
            nonorthogonal_tol=nonorthogonal_tol,
            cross_flux=self._pressure_correction_cross_flux,
            dirichlet_value=(
                self._zero_pressure_correction
                if self._has_pressure_dirichlet()
                else None
            ),
            dirichlet_threshold=self._pressure_dirichlet_threshold(),
        )

    @staticmethod
    def _face_velocity(rhie_chow, u, ap, p, response_coef, bd_edge, bdedgeu):
        """Construct Rhie-Chow face velocity and enforce velocity Dirichlet data."""
        try:
            uf = rhie_chow.Interpolation(
                u, ap, p, face_response_coefficient=response_coef
            )
        except TypeError:
            uf = rhie_chow.Interpolation(u, ap, p)
        return bm.set_at(uf, bd_edge, bdedgeu)

    def _build_rhie_chow_interpolation(self):
        """Build Rhie-Chow interpolation with active pressure boundary data."""
        try:
            return RhieChowInterpolation(
                self.mesh,
                pressure_gradient_method=(
                    self.controls.rhie_chow_pressure_gradient_method
                ),
                velocity_interpolation=self.controls.face_interpolation(
                    "rhie_chow_velocity_interpolation"
                ),
                pressure_dirichlet=self._pressure_dirichlet_data(),
                pressure_dirichlet_threshold=self._pressure_dirichlet_threshold(),
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
            return self.boundary_conditions.boundary_face_velocity(
                "velocity", mesh=self.mesh
            )
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
        rhie_chow = self._build_rhie_chow_interpolation()

        for iteration in range(1, max_iter + 1):
            response_coef = self._pressure_response_face_coefficient(ap)
            uf = self._face_velocity(
                rhie_chow, u, ap, p, response_coef, bd_edge, bdedgeu
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

            if pressure_correction_converged(
                residual, tol_mass, tol_pressure_update
            ):
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
