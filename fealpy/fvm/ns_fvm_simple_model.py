from typing import Optional, Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarCrossDiffusionIntegrator,
    ScalarSourceIntegrator,
    ConvectionIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
    NonOrthogonalGeometry,
    RhieChowInterpolation,
    FVMLinearSolver,
    FVMLinearSolverConfig,
)
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


class NSFVMSimpleModel(ComputationalModel):
    """
    Finite Volume SIMPLE solver for 2D steady incompressible Navier–Stokes equations.
    """

    def __init__(self, options):
        super().__init__(
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        self.options = options
        self.pde = self._resolve_pde(options["pde"])
        self.mesh = self._init_mesh(options)
        self.cm = self.mesh.entity_measure("cell")
        self.NC = self.mesh.number_of_cells()
        self._init_discretization(degree=0)
        self.linear_solver = self._init_linear_solver(options)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    @staticmethod
    def _resolve_pde(pde: Union[int, object]):
        """Return a Navier-Stokes PDE model from an example id or object."""
        return (
            PDEModelManager("navier_stokes").get_example(pde)
            if isinstance(pde, int)
            else pde
        )

    def _init_mesh(self, options):
        """Build the PDE default mesh and apply optional uniform refinement."""
        mesh_type = options.get("mesh_type") or getattr(
            self.pde, "default_mesh_type", "uniform_tri"
        )
        mesh_refine = int(options.get("mesh_refine", 0))
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")

        if getattr(self.pde, "supports_geometric_refine", False):
            return self.pde.init_mesh[mesh_type](mesh_refine=mesh_refine)

        mesh = self.pde.init_mesh[mesh_type]()
        if mesh_refine == 0:
            return mesh

        if hasattr(mesh, "uniform_refine"):
            mesh.uniform_refine(mesh_refine)
            return mesh

        raise ValueError("mesh does not provide uniform_refine().")

    def _init_discretization(self, degree: int = 0) -> None:
        """Initialize spaces and reusable FVM operators."""
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, degree)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))

        self.pressure_gradient = GradientReconstruct(self.mesh)
        self.velocity_gradient = GradientReconstruct(
            self.mesh,
            gd=self.pde.dirichlet_velocity,
            bc_type="dirichlet",
        )
        self.divergence = DivergenceReconstruct(self.mesh)
        self.nonorthogonal_geometry = NonOrthogonalGeometry(self.mesh)
        self.velocity_dirichlet_bc = DirichletBC(
            self.mesh, self.pde.dirichlet_velocity
        )

        self.e2c = self.mesh.edge_to_cell()
        self.edge_measure = self.mesh.entity_measure("edge")
        self.last_nonorthogonal_iterations = 0

    def _init_linear_solver(self, options):
        """Return the linear-system solve boundary for this model."""
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

        cross = self.compute_cross_diffusion(u0)
        nonorthogonal_iterations = 0
        for nonorthogonal_iterations in range(1, 11):
            rhs = f + cross
            uh_new = self.linear_solver.solve(B, rhs)
            err = bm.max(bm.abs(uh_new - u))
            if err < 10e-5:
                break
            u = uh_new
            cross = self.compute_cross_diffusion(u)
        self.last_nonorthogonal_iterations = nonorthogonal_iterations
        
        return ap, u
    
    def compute_cross_diffusion(self, uh: TensorLike) -> TensorLike:
        """Compute cross-diffusion term based on current velocity uh."""
        lform = LinearForm(self.velocity_space)
        U = bm.stack((uh[:self.NC], uh[self.NC:]), axis=1)
        grad_u = self.velocity_gradient.cell_gradient(U)
        grad_f = self.velocity_gradient.face_gradient(grad_u)  # (NE, 2, 2)
        lform.add_integrator(
            ScalarCrossDiffusionIntegrator(
                uh,
                grad_f,
                geometry=self.nonorthogonal_geometry,
                boundary_policy="all",
            )
        )
        return lform.assembly()
    
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

    def pressure_correct(
        self,
        ap: TensorLike,
        uf: TensorLike,
        response_coef: Optional[TensorLike] = None,
    ) -> TensorLike:
        """Solve the SIMPLE pressure-correction equation."""
        dp_edge = (
            self._pressure_response_face_coefficient(ap)
            if response_coef is None
            else response_coef
        )
        div_u = self.divergence.Reconstruct(uf)  # (NC,)
        bform2 = BilinearForm(self.space)
        bform2.add_integrator(ScalarDiffusionIntegrator(q=2,coef=dp_edge))
        A = bform2.assembly()
        LagA = self.mesh.entity_measure("cell")
        gauge_index = bm.stack(
            [
                bm.zeros(len(LagA), dtype=bm.int32),
                bm.arange(len(LagA), dtype=bm.int32),
            ],
            axis=0,
        )
        A1 = COOTensor(gauge_index,LagA,
            spshape=(1, len(LagA)),
        )
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format="csr")
        b0 = bm.array([0])
        b = bm.concatenate([-div_u, b0], axis=0)
        sol = self.linear_solver.solve(A, b)
        p_c = sol[:-1]
        return p_c

    def _simple_residual(self, uf, p_corr, p_update, p, pressure_relax, action):
        """Build one pressure-correction residual record."""
        return {
            "mass": collocated_mass_residual(self.mesh, uf),
            "pressure_correction": cell_l2_norm(self.mesh, p_corr),
            "pressure_update": relative_l2_update(self.mesh, p_update, p),
            "pressure_relax": pressure_relax,
            "pressure_relax_reduced": action == "reduce",
            "pressure_relax_action": action,
            "nonorthogonal_iterations": self.last_nonorthogonal_iterations,
        }

    def _log_simple_iteration(self, iteration, residual):
        """Log one SIMPLE pressure-correction diagnostic record."""
        self.logger.info(
            format_pressure_correction_log(
                iteration=iteration,
                nonorthogonal_iterations=residual["nonorthogonal_iterations"],
                pressure_criterion=residual["pressure_update"],
                pressure_relax=residual["pressure_relax"],
                mass_residual=residual["mass"],
                pressure_correction=residual["pressure_correction"],
                label="SIMPLE",
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

    def _boundary_face_velocity(self):
        """Return boundary face indices and prescribed boundary velocities."""
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter("edge")
        bdedgepoint = edge_middle_point[bd_edge]
        return bd_edge, self.pde.dirichlet_velocity(bdedgepoint)

    @staticmethod
    def _face_velocity(rhie_chow, u, ap, p, response_coef, bd_edge, bdedgeu):
        """Construct Rhie-Chow face velocity and enforce velocity Dirichlet data."""
        uf = rhie_chow.Interpolation(
            u, ap, p, face_response_coefficient=response_coef
        )
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
            p_corr = self.pressure_correct(ap, uf, response_coef)
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
