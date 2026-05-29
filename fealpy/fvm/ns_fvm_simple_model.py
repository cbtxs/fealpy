from typing import Optional, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel

from fealpy.fem import BilinearForm, LinearForm

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    ConvectionIntegrator,
    RhieChowInterpolation,
)
from .collocated_ns_fvm_utils import CollocatedNSFVMOperators
from .simple_residual import (
    cell_l2_norm,
    collocated_mass_residual,
    relative_l2_update,
)
from .pressure_correction_control import (
    PressureRelaxationConfig,
    PressureRelaxationController,
    format_pressure_correction_log,
    pressure_correction_converged,
)


class NSFVMSimpleModel(ComputationalModel, CollocatedNSFVMOperators):
    """
    Finite Volume SIMPLE solver for 2D steady incompressible Navier–Stokes equations.
    """

    def __init__(self, options):
        super().__init__(
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        self.options = options
        self.pde = self._resolve_navier_stokes_pde(options["pde"])
        self.mesh = self._init_mesh(options)
        self.cm = self.mesh.entity_measure("cell")
        self.NC = self.mesh.number_of_cells()
        self._init_discretization(degree=options.get("space_degree", 0))
        self.linear_solver = self._init_linear_solver(options)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def _init_mesh(self, options):
        """Build the PDE default mesh and apply optional uniform refinement."""
        return self._init_navier_stokes_mesh(
            options,
            default_mesh_type="uniform_tri",
        )

    def _init_discretization(self, degree: int = 0) -> None:
        """Initialize spaces and reusable FVM operators."""
        self._init_collocated_discretization(
            degree,
            self.pde.dirichlet_velocity,
            with_divergence=True,
            with_velocity_dirichlet_bc=True,
        )

    def temporary_velocity(self, p, uf, u0) -> Tuple[TensorLike, TensorLike]:
        """Solve momentum eqn for intermediate velocity u*."""
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        bform.add_integrator(ConvectionIntegrator(q=self.p + 2, coef=uf))
        B = bform.assembly()
        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()
        B, f = self.velocity_dirichlet_bc.DiffusionApply(B, f)
        f = self.velocity_dirichlet_bc.ConvectionApply(f, uf)
        # FEALPy sparse assembly can leave duplicate entries here.
        B = B.tocoo().coalesce().tocsr()
        ap = B.diags().values
        grad_p = self.pressure_gradient.cell_gradient(p)  # (NC, 2)
        p1 = bm.einsum('i,i->i', grad_p[:,0], self.cm)
        p2 = bm.einsum('i,i->i', grad_p[:,1], self.cm)
        p_grad_integrator = bm.concatenate((p1,p2))
        f = f - p_grad_integrator
        u = self.linear_solver.solve(B, f)

        u = self._correct_momentum_nonorthogonal_diffusion(
            B,
            f,
            u,
            u0,
            max_iter=10,
            tol=10e-5,
            iteration_attr="last_nonorthogonal_iterations",
        )
        
        return ap, u
    
    def _pressure_response_face_coefficient(self, ap: TensorLike) -> TensorLike:
        """Return the historical SIMPLE pressure-correction face response.

        This coefficient is not the standard momentum response ``V/a_p``.
        It is kept here as an experimental route because it is much less
        sensitive to pressure relaxation in the current SIMPLE loop.  The
        same face coefficient must also be passed to Rhie-Chow interpolation
        so that the pressure-correction equation and face-velocity response
        remain algebraically consistent.
        """
        dp = 1.0 / ap[:self.NC]
        return 0.5 * (dp[self.e2c[:, 0]] + dp[self.e2c[:, 1]]) * self.edge_measure

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
        """Return the full pressure-correction flux used to correct mass flux.

        The orthogonal part follows ``ScalarDiffusionIntegrator``.  The
        non-orthogonal part follows the explicit cross-diffusion convention:
        the pressure-induced velocity flux is ``orthogonal - cross`` because
        velocity correction contains ``-D grad(p')``.
        """
        return (
            self._pressure_correction_orthogonal_flux(p_corr, response_coef)
            - self._pressure_correction_cross_flux(p_corr, response_coef)
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
        nonorthogonal_max_iter: int = 10,
        nonorthogonal_tol: float = 1.0e-5,
    ) -> TensorLike:
        """Solve the SIMPLE pressure-correction equation."""
        if nonorthogonal_max_iter < 0:
            raise ValueError("nonorthogonal_max_iter must be non-negative.")
        if nonorthogonal_tol <= 0.0:
            raise ValueError("nonorthogonal_tol must be positive.")

        dp_edge = (
            self._pressure_response_face_coefficient(ap)
            if response_coef is None
            else response_coef
        )
        div_u = self.divergence.Reconstruct(uf)  # (NC,)
        return self._solve_pressure_correction_with_cross_rhs(
            -div_u,
            dp_edge,
            q=2,
            nonorthogonal_max_iter=nonorthogonal_max_iter,
            nonorthogonal_tol=nonorthogonal_tol,
            cross_flux=self._pressure_correction_cross_flux,
        )

    def _simple_residual(self, uf, p_corr, p_update, p, pressure_relax, action):
        """Build one pressure-correction residual record."""
        return {
            "mass": collocated_mass_residual(self.mesh, uf),
            "pressure_correction": cell_l2_norm(self.mesh, p_corr),
            "pressure_update": relative_l2_update(self.mesh, p_update, p),
            "pressure_relax": pressure_relax,
            "pressure_relax_reduced": action == "reduce",
            "pressure_relax_action": action,
            "nonorthogonal_iterations": self.last_pressure_nonorthogonal_iterations,
            "momentum_nonorthogonal_iterations": self.last_nonorthogonal_iterations,
        }

    @staticmethod
    def _simple_iteration_log_message(
        *,
        simple_iteration: int,
        nonorthogonal_iterations: int,
        pressure_criterion: float,
        pressure_relax: float,
        mass_residual: float,
        pressure_correction: float,
    ) -> str:
        """Format one SIMPLE iteration diagnostic line."""
        return format_pressure_correction_log(
            iteration=simple_iteration,
            nonorthogonal_iterations=nonorthogonal_iterations,
            pressure_criterion=pressure_criterion,
            pressure_relax=pressure_relax,
            mass_residual=mass_residual,
            pressure_correction=pressure_correction,
            label="SIMPLE",
        )

    def _log_simple_iteration(self, iteration, residual):
        """Log one SIMPLE pressure-correction diagnostic record."""
        self.logger.info(
            self._simple_iteration_log_message(
                simple_iteration=iteration,
                nonorthogonal_iterations=residual["nonorthogonal_iterations"],
                pressure_criterion=residual["pressure_update"],
                pressure_relax=residual["pressure_relax"],
                mass_residual=residual["mass"],
                pressure_correction=residual["pressure_correction"],
            )
        )
        action = residual["pressure_relax_action"]
        if action in {"reduce", "increase"}:
            self.logger.info(
                f"[Iter {iteration}] pressure relaxation {action}d to "
                f"{residual['pressure_relax']:.2e}"
            )

    @staticmethod
    def _simple_tolerances(tol, tol_mass, tol_pressure_update):
        """Return mass and pressure-update stopping tolerances."""
        return (
            tol if tol_mass is None else tol_mass,
            10.0 * tol if tol_pressure_update is None else tol_pressure_update,
        )

    @staticmethod
    def _pressure_relaxation_controller(
        relax,
        adaptive_pressure_relax,
        relaxation_config,
    ):
        """Create the optional pressure-relaxation controller."""
        if relax <= 0:
            raise ValueError("relax must be positive.")
        if not adaptive_pressure_relax:
            return None
        if isinstance(relaxation_config, dict):
            relaxation_config = PressureRelaxationConfig(**relaxation_config)
        return PressureRelaxationController(
            initial=relax,
            config=relaxation_config,
        )

    def _initial_simple_fields(self):
        """Return zero initial pressure, face velocity, and cell velocity."""
        field_dtype = self.cm.dtype
        return (
            bm.zeros(self.NC, dtype=field_dtype),
            bm.zeros((self.mesh.number_of_faces(), 2), dtype=field_dtype),
            bm.zeros(2 * self.NC, dtype=field_dtype),
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

    def _pressure_update_step(
        self,
        uf,
        p_corr,
        p,
        pressure_relax,
        relaxation,
    ):
        """Apply pressure relaxation control and build the residual record."""
        p_update = pressure_relax * p_corr
        residual = self._simple_residual(
            uf, p_corr, p_update, p, pressure_relax, "keep"
        )
        self.residuals.append(residual)

        relax_action = "keep"
        if relaxation is not None:
            pressure_relax, relax_action = relaxation.update(self.residuals)
            if relax_action in {"reduce", "increase"}:
                p_update = pressure_relax * p_corr
                residual["pressure_update"] = relative_l2_update(
                    self.mesh, p_update, p
                )

        residual["pressure_relax"] = pressure_relax
        residual["pressure_relax_reduced"] = relax_action == "reduce"
        residual["pressure_relax_action"] = relax_action
        return pressure_relax, p_update, residual

    def _store_solution(self, u, p):
        """Store and return the separated velocity and pressure fields."""
        self.uh = u[:self.NC]
        self.vh = u[self.NC:]
        self.ph = p
        return self.uh, self.vh, self.ph

    def solve(
        self,
        max_iter: int = 100,
        tol: float = 1e-5,
        relax: float = 0.32,
        tol_mass=None,
        tol_pressure_update=None,
        adaptive_pressure_relax: bool = True,
        relaxation_config: Optional[PressureRelaxationConfig] = None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Main SIMPLE loop."""
        tol_mass, tol_pressure_update = self._simple_tolerances(
            tol, tol_mass, tol_pressure_update
        )
        relaxation = self._pressure_relaxation_controller(
            relax,
            adaptive_pressure_relax,
            relaxation_config,
        )
        pressure_relax = relax
        p, uf, u = self._initial_simple_fields()
        ap, u = self.temporary_velocity(p, uf, u)
        self.residuals = []
        bd_edge, bdedgeu = self._boundary_face_velocity()
        rhie_chow = RhieChowInterpolation(self.mesh)

        for iteration in range(1, max_iter + 1):
            response_coef = self._pressure_response_face_coefficient(ap)
            uf = self._face_velocity(
                rhie_chow, u, ap, p, response_coef, bd_edge, bdedgeu
            )
            p_corr = self.pressure_correct(ap, uf)
            pressure_relax, p_update, residual = self._pressure_update_step(
                uf, p_corr, p, pressure_relax, relaxation
            )
            self._log_simple_iteration(iteration, residual)

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

        return self._store_solution(u, p)

    def compute_error(self) -> Tuple[float, float]:
        """Compute errors for velocity and pressure."""
        cell_centers = self.mesh.entity_barycenter('cell')
        self.uI = self.pde.velocity(cell_centers)[:, 0]
        self.vI = self.pde.velocity(cell_centers)[:, 1]
        self.pI = self.pde.pressure(cell_centers)
        uerror = bm.sqrt(bm.sum(self.cm * (self.uh - self.uI)**2))
        verror = bm.sqrt(bm.sum(self.cm * (self.vh - self.vI)**2))
        perror = bm.sqrt(bm.sum(self.cm * (self.ph - self.pI)**2))
        return uerror, verror, perror

    def plot(self) -> None:
        """Plot numerical and exact solutions for u, v, and p."""
        import matplotlib.pyplot as plt
        cell_centers = self.mesh.entity_barycenter('cell')
        x, y = cell_centers[:, 0], cell_centers[:, 1]

        fig = plt.figure(figsize=(15, 10))
        titles = [
            ("Error u", self.uh - self.uI),
            ("Error v", self.vh - self.vI),
            ("Error p", self.ph - self.pI),
        ]
        for i, (title, data) in enumerate(titles):
            ax = fig.add_subplot(2, 3, i + 1, projection='3d')
            ax.plot_trisurf(x, y, data, cmap='viridis')
            ax.set_title(title)
        plt.tight_layout()
        plt.show()

    def plot_residual(self) -> None:
        """Plot residual decay curve."""
        import matplotlib.pyplot as plt

        mass = [residual["mass"] for residual in self.residuals]
        pressure_update = [
            residual["pressure_update"] for residual in self.residuals
        ]
        plt.figure(figsize=(8, 5))
        plt.semilogy(mass, marker="o", linestyle="-", color="b", label="mass")
        plt.semilogy(
            pressure_update,
            marker="s",
            linestyle="-",
            color="r",
            label="pressure update",
        )
        plt.legend()
        plt.title("SIMPLE Residuals vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
