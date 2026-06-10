from typing import Optional, Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d
from fealpy.fem import BilinearForm, LinearForm, BlockForm

from fealpy.solver import spsolve

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarSourceIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
)
from .staggered_mesh_manager import StaggeredMeshManager
from ..simple_residual import (
    cell_l2_norm,
    relative_l2_update,
    staggered_mass_residual,
)

class StokesFVMStaggeredSimpleModel(ComputationalModel):
    """
    A 2D Stokes solver using finite volume method on staggered mesh.
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "INFO"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.pmesh.number_of_cells()} pressure cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[str, object]) -> None:
        """Set the PDE model."""
        self.pde = PDEModelManager("stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self, nx: int = 10, ny: int = 10) -> None:
        """Set the computational staggered mesh."""
        self.staggered_mesh = StaggeredMeshManager(self.pde.domain(), nx, ny)
        self.umesh = self.staggered_mesh.umesh
        self.vmesh = self.staggered_mesh.vmesh
        self.pmesh = self.staggered_mesh.pmesh
        self.div = DivergenceReconstruct(self.pmesh)
        self.pcm = self.pmesh.entity_measure("cell")
        self.ucm = self.umesh.entity_measure("cell")
        self.vcm = self.vmesh.entity_measure("cell")
        self.ppoints = self.pmesh.entity_barycenter("cell")
        self.upoints = self.umesh.entity_barycenter("cell")
        self.vpoints = self.vmesh.entity_barycenter("cell")
        self.u_gradient = GradientReconstruct(self.umesh, method="green_gauss")
        self.v_gradient = GradientReconstruct(self.vmesh, method="green_gauss")

    def compute_velocity_u(self, p_u) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity u* using the momentum equation."""
        uspace = ScaledMonomialSpace2d(self.umesh, 0)
        A = BilinearForm(uspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(uspace).add_integrator(ScalarSourceIntegrator(self.pde.source_u, q=2)).assembly()
        grad_p = self.u_gradient.cell_gradient(p_u)
        f -= bm.einsum('i,i->i', grad_p[:, 0], self.ucm)
        dbc = DirichletBC(self.umesh, self.pde.dirichlet_velocity_u,
                          threshold=lambda x: (bm.abs(x) < 1e-10) | (bm.abs(x - 1) < 1e-10))
        A, f = dbc.DiffusionApply(A, f)
        A, f = dbc.ThresholdApply(A, f)
        uap = A.diags().values
        return spsolve(A, f,"mumps"), uap

    def compute_velocity_v(self, p_v) -> Tuple[TensorLike, TensorLike]:
        """Solve for temporary velocity v* using the momentum equation."""
        vspace = ScaledMonomialSpace2d(self.vmesh, 0)
        A = BilinearForm(vspace).add_integrator(ScalarDiffusionIntegrator(q=2)).assembly()
        f = LinearForm(vspace).add_integrator(ScalarSourceIntegrator(self.pde.source_v, q=2)).assembly()
        grad_p = self.v_gradient.cell_gradient(p_v)
        f -= bm.einsum('i,i->i', grad_p[:, 1], self.vcm)
        dbc = DirichletBC(self.vmesh, self.pde.dirichlet_velocity_v,
                          threshold=lambda y: (bm.abs(y) < 1e-10) | (bm.abs(y - 1) < 1e-10))
        A, f = dbc.DiffusionApply(A, f)
        A, f = dbc.ThresholdApply(A, f)
        vap = A.diags().values
        return spsolve(A, f,"mumps"), vap

    def correct_pressure_compute(self, f: TensorLike, a_p_edge: TensorLike) -> TensorLike:
        """
        Solve for pressure correction p' to enforce continuity.
        """
        LagA = self.pmesh.entity_measure('cell')
        pspace = ScaledMonomialSpace2d(self.pmesh, 0)
        # Mathematical risk:
        # This is a historical pressure-correction coefficient.  The
        # edge_length**2/a_p_edge scaling should be re-derived before this
        # Stokes SIMPLE model is used as a reference implementation.
        p_edge = self.pmesh.entity_measure('edge')
        p_edge2 = bm.einsum('i,i->i', p_edge,p_edge)
        A = BilinearForm(pspace).add_integrator(
            ScalarDiffusionIntegrator(q=2,coef=p_edge2 / a_p_edge)
        ).assembly()
        # LagA = self.pmesh.entity_measure('cell')
        gauge_index = bm.stack(
            [
                bm.zeros(len(LagA), dtype=bm.int32),
                bm.arange(len(LagA), dtype=bm.int32),
            ],
            axis=0,
        )
        A1 = COOTensor(gauge_index, LagA, spshape=(1, len(LagA)))
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format='csr')
        b0 = bm.array([0])
        b = bm.concatenate([f, b0], axis=0)
        sol = spsolve(A, b,"mumps")
        p_correct = sol[:-1]   
        return p_correct

    def _adapt_pressure_relaxation(
        self,
        residuals,
        current_relax: float,
        deterioration_count: int,
        *,
        small_update_count: int = 0,
        cooldown: int = 0,
        relax_min: float,
        relax_max: float,
        growth_factor: float,
        patience: int,
        severe_growth_factor: Optional[float] = None,
        reduction_factor: float = 0.5,
        increase_patience: int = 3,
        increase_factor: float = 1.25,
        small_update: Optional[float] = None,
        cooldown_steps: int = 3,
        mass_growth_factor: float = 1.05,
        residual_key: str = "pressure_correction",
        update_key: str = "pressure_update",
        mass_key: str = "mass",
    ) -> Tuple[float, int, int, int, str]:
        """Adapt pressure relaxation without changing the staggered discretization."""
        if len(residuals) < 2:
            return current_relax, 0, 0, 0, "keep"

        current = float(residuals[-1][residual_key])
        previous = float(residuals[-2][residual_key])

        def reduce_relaxation():
            new_relax = max(reduction_factor * current_relax, relax_min)
            if new_relax >= current_relax:
                return current_relax, 0, 0, 0, "keep"
            return new_relax, 0, 0, cooldown_steps, "reduce"

        if (
            severe_growth_factor is not None
            and current > severe_growth_factor * previous
        ):
            return reduce_relaxation()

        if cooldown > 0:
            return current_relax, 0, small_update_count, cooldown - 1, "cooldown"

        if current > growth_factor * previous:
            deterioration_count += 1
        else:
            deterioration_count = 0

        if deterioration_count >= patience:
            return reduce_relaxation()

        can_increase = (
            small_update is not None
            and current_relax < relax_max
            and update_key in residuals[-1]
            and residuals[-1][update_key] < small_update
            and current <= growth_factor * previous
        )
        if can_increase and mass_key in residuals[-1] and mass_key in residuals[-2]:
            mass = float(residuals[-1][mass_key])
            previous_mass = float(residuals[-2][mass_key])
            can_increase = mass <= mass_growth_factor * previous_mass

        if can_increase:
            small_update_count += 1
        else:
            small_update_count = 0

        if small_update_count < increase_patience:
            return current_relax, deterioration_count, small_update_count, 0, "keep"

        new_relax = min(increase_factor * current_relax, relax_max)
        if new_relax <= current_relax:
            return current_relax, deterioration_count, 0, 0, "keep"
        return new_relax, 0, 0, cooldown_steps, "increase"

    def _simple_iteration_log_message(
        self,
        *,
        simple_iteration: int,
        nonorthogonal_iterations: int,
        pressure_criterion: float,
        pressure_relax: float,
        mass_residual: float,
        pressure_correction: float,
    ) -> str:
        """Format the main SIMPLE iteration diagnostic line."""
        return (
            f"[SIMPLE {simple_iteration}] "
            f"nonorthogonal iterations: {nonorthogonal_iterations}, "
            f"pressure criterion: {pressure_criterion:.2e}, "
            f"pressure relax: {pressure_relax:.2e}, "
            f"mass residual: {mass_residual:.2e}, "
            f"pressure correction L2: {pressure_correction:.2e}"
        )

    def solve(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        relax: float = 0.02,
        tol_mass=None,
        tol_pressure_update=None,
        adaptive_pressure_relax: bool = True,
        relax_min: float = 1.0e-4,
        relax_growth_factor: float = 1.05,
        relax_severe_growth_factor: Optional[float] = 2.0,
        relax_patience: int = 3,
        relax_reduction_factor: float = 0.5,
        relax_increase_patience: int = 3,
        relax_increase_factor: float = 1.25,
        relax_small_update: Optional[float] = 1.0e-3,
        relax_cooldown_steps: int = 3,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Solve the Stokes equation using the SIMPLE algorithm."""
        if relax <= 0:
            raise ValueError("relax must be positive.")
        if relax_min <= 0:
            raise ValueError("relax_min must be positive.")
        if relax_growth_factor <= 1.0:
            raise ValueError("relax_growth_factor must be greater than 1.")
        if (
            relax_severe_growth_factor is not None
            and relax_severe_growth_factor <= 1.0
        ):
            raise ValueError("relax_severe_growth_factor must be greater than 1.")
        if relax_patience < 1:
            raise ValueError("relax_patience must be positive.")
        if not 0.0 < relax_reduction_factor < 1.0:
            raise ValueError("relax_reduction_factor must be in (0, 1).")
        if relax_increase_patience < 1:
            raise ValueError("relax_increase_patience must be positive.")
        if relax_increase_factor <= 1.0:
            raise ValueError("relax_increase_factor must be greater than 1.")
        if relax_small_update is not None and relax_small_update <= 0.0:
            raise ValueError("relax_small_update must be positive.")
        if relax_cooldown_steps < 0:
            raise ValueError("relax_cooldown_steps must be non-negative.")

        tol_mass = tol if tol_mass is None else tol_mass
        tol_pressure_update = 10.0 * tol if tol_pressure_update is None else tol_pressure_update
        pressure_relax = relax
        relax_deterioration_count = 0
        relax_small_update_count = 0
        relax_cooldown = 0
        p = bm.zeros(self.ppoints.shape[0], dtype=self.pcm.dtype)
        self.residuals = []
        for i in range(max_iter):
            
            p_u, p_v = self.staggered_mesh.map_pressure_pcell_to_uvedge(p)
            uh, a_p_u = self.compute_velocity_u(p_u)
            vh, a_p_v = self.compute_velocity_v(p_v)
            edge_vel, a_p_edge = self.staggered_mesh.map_velocity_uvcell_to_pedge(uh, vh, a_p_u, a_p_v)
            self.div_rhs = self.div.StagReconstruct(edge_vel)
            p_corr = self.correct_pressure_compute(-self.div_rhs, a_p_edge)
            p_update = pressure_relax * p_corr
            residual = {
                "mass": staggered_mass_residual(self.pmesh, edge_vel),
                "pressure_update": relative_l2_update(self.pmesh, p_update, p),
                "pressure_correction": cell_l2_norm(self.pmesh, p_corr),
            }
            self.residuals.append(residual)
            relax_action = "keep"
            if adaptive_pressure_relax:
                (
                    pressure_relax,
                    relax_deterioration_count,
                    relax_small_update_count,
                    relax_cooldown,
                    relax_action,
                ) = self._adapt_pressure_relaxation(
                    self.residuals,
                    current_relax=pressure_relax,
                    deterioration_count=relax_deterioration_count,
                    small_update_count=relax_small_update_count,
                    cooldown=relax_cooldown,
                    relax_min=relax_min,
                    relax_max=relax,
                    growth_factor=relax_growth_factor,
                    severe_growth_factor=relax_severe_growth_factor,
                    patience=relax_patience,
                    reduction_factor=relax_reduction_factor,
                    increase_patience=relax_increase_patience,
                    increase_factor=relax_increase_factor,
                    small_update=relax_small_update,
                    cooldown_steps=relax_cooldown_steps,
                )
                if relax_action in {"reduce", "increase"}:
                    p_update = pressure_relax * p_corr
                    residual["pressure_update"] = relative_l2_update(
                        self.pmesh, p_update, p
                    )
            residual["pressure_relax"] = pressure_relax
            residual["pressure_relax_reduced"] = relax_action == "reduce"
            residual["pressure_relax_action"] = relax_action
            residual["nonorthogonal_iterations"] = 0
            self.logger.info(
                self._simple_iteration_log_message(
                    simple_iteration=i + 1,
                    nonorthogonal_iterations=residual["nonorthogonal_iterations"],
                    pressure_criterion=residual["pressure_update"],
                    pressure_relax=pressure_relax,
                    mass_residual=residual["mass"],
                    pressure_correction=residual["pressure_correction"],
                )
            )
            if relax_action in {"reduce", "increase"}:
                self.logger.info(
                    f"[Iter {i+1}] pressure relaxation {relax_action}d to "
                    f"{pressure_relax:.2e}"
                )
            if (
                residual["mass"] < tol_mass
                and residual["pressure_update"] < tol_pressure_update
            ):
                self.logger.info("Converged.")
                break
            p += p_update
        self.uh, self.vh, self.ph = uh, vh, p
        return uh, vh, p

    def compute_error(self) -> Tuple[float, float, float]:
        """
        Compute errors for velocity and pressure.
        """
        self.uI = self.pde.velocity_u(self.upoints)
        self.vI = self.pde.velocity_v(self.vpoints)
        self.pI = self.pde.pressure(self.ppoints)
        
        uerror = bm.sqrt(bm.sum(self.ucm * (self.uh - self.uI)**2))
        verror = bm.sqrt(bm.sum(self.vcm * (self.vh - self.vI)**2))
        perror = bm.sqrt(bm.sum(self.pcm * (self.ph - self.pI)**2))
        return uerror, verror, perror

    def plot(self) -> None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(15, 5))

        px, py = self.pmesh.entity_barycenter("cell").T
        ux, uy = self.umesh.entity_barycenter("cell").T
        vx, vy = self.vmesh.entity_barycenter("cell").T

        ax1 = fig.add_subplot(1, 3, 1, projection="3d")
        ax1.plot_trisurf(ux, uy, self.uh-self.uI, cmap="viridis")
        ax1.set_title("Error u")

        ax2 = fig.add_subplot(1, 3, 2, projection="3d")
        ax2.plot_trisurf(vx, vy, self.vh-self.vI, cmap="viridis")
        ax2.set_title("Error v")

        ax3 = fig.add_subplot(1, 3, 3, projection="3d")
        ax3.plot_trisurf(px, py, self.ph-self.pI, cmap="viridis")
        ax3.set_title("Error p")

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
