from typing import Optional, Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm

from fealpy.solver import spsolve

from . import (
    ScalarDiffusionIntegrator,
    ScalarCrossDiffusionIntegrator,
    ScalarSourceIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
    NonOrthogonalGeometry,
    RhieChowInterpolation,
    VectorDecomposition,
)
from .simple_residual import (
    cell_l2_norm,
    collocated_mass_residual,
    relative_l2_update,
)

class StokesFVMSimpleModel(ComputationalModel):
    """
    The Stokes equation in two-dimensional cases is solved by the finite volume method
    using the SIMPLE algorithm. The velocity and pressure are iteratively corrected
    to satisfy the continuity equation.
    """
    def __init__(self, options):
        self.options = options
        self._validate_options()
        super().__init__(
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        self.set_pde(options["pde"])
        self.set_mesh()
        self.set_space(0)

    def _validate_options(self) -> None:
        """Reject legacy mesh sizing hooks for the collocated Stokes solver."""
        allowed = {"pde", "mesh_type", "mesh_refine", "pbar_log", "log_level"}
        unsupported = set(self.options).difference(allowed)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                "StokesFVMSimpleModel options are limited to pde, mesh_type "
                f"and mesh_refine; unsupported: {names}"
            )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[str, object]) -> None:
        """Set the PDE model."""
        self.pde = PDEModelManager("stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self) -> None:
        """Initialize and optionally refine the computational mesh."""
        mesh_type = (
            self.options.get("mesh_type")
            or getattr(self.pde, "default_mesh_type", "uniform_quad")
        )
        mesh_type = {
            "quad": "uniform_quad",
            "tri": "uniform_tri",
        }.get(mesh_type, mesh_type)
        mesh_refine = int(self.options.get("mesh_refine", 0))
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")

        if getattr(self.pde, "supports_geometric_refine", False):
            mesh = self.pde.init_mesh[mesh_type](mesh_refine=mesh_refine)
        else:
            mesh = self.pde.init_mesh[mesh_type]()
        if mesh_refine and not getattr(self.pde, "supports_geometric_refine", False):
            if not hasattr(mesh, "uniform_refine"):
                raise ValueError("mesh does not provide uniform_refine().")
            mesh.uniform_refine(mesh_refine)

        self.mesh = mesh
        self.cm = self.mesh.entity_measure('cell')
        self.NC = self.mesh.number_of_cells()

    def set_space(self, degree: int = 0) -> None:
        """Set the function spaces for velocity and pressure."""
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))
        self.pressure_gradient = GradientReconstruct(self.mesh)
        self.velocity_gradient = GradientReconstruct(
            self.mesh,
            method="green_gauss",
            gd=self.pde.dirichlet_velocity,
            bc_type="dirichlet",
        )
        self.divergence = DivergenceReconstruct(self.mesh)
        self.rhie_chow = RhieChowInterpolation(self.mesh)
        self.nonorthogonal_geometry = NonOrthogonalGeometry(self.mesh)
        self.edge_centroid_vector, self.edge_centroid_distance = (
            VectorDecomposition(self.mesh).centroid_vector_calculation()
        )
        self._momentum_system_cache = None
        self._pressure_correction_system_cache = None
        self.last_nonorthogonal_iterations = 0

    def momentum_system(self):
        """Assemble and cache the constant Stokes momentum system."""
        if self._momentum_system_cache is not None:
            return self._momentum_system_cache

        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        B = bform.assembly()
        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()

        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        B, f = dbc.DiffusionApply(B, f)
        # FEALPy sparse assembly can leave duplicate entries here.
        B = B.tocoo().coalesce().tocsr()
        ap = B.diags().values
        self._momentum_system_cache = (B, f, ap)
        return self._momentum_system_cache

    def temporary_velocity(self, p, uf, u0) -> Tuple[TensorLike, TensorLike]:
        """Solve the Stokes momentum equation for intermediate velocity u*."""
        B, f, ap = self.momentum_system()
        grad_p = self.pressure_gradient.cell_gradient(p)
        p1 = bm.einsum('i,i->i', grad_p[:,0], self.cm)
        p2 = bm.einsum('i,i->i', grad_p[:,1], self.cm)
        p_grad_integrator = bm.concatenate((p1,p2))
        f = f - p_grad_integrator
        u = spsolve(B, f,"mumps")

        cross = self.compute_cross_diffusion(u0)
        nonorthogonal_iterations = 0
        for nonorthogonal_iterations in range(1, 11):
            rhs = f + cross
            uh_new = spsolve(B, rhs)
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
        grad_f = self.velocity_gradient.face_gradient(grad_u)
        lform.add_integrator(
            ScalarCrossDiffusionIntegrator(
                uh,
                grad_f,
                geometry=self.nonorthogonal_geometry,
                boundary_policy="all",
            )
        )
        return lform.assembly()

    def pressure_response(self, ap: TensorLike) -> TensorLike:
        """Cell pressure response ``rAU = V/a_p`` for integrated momentum rows."""
        return self.cm / ap[:self.NC]

    def face_pressure_response(self, ap: TensorLike) -> TensorLike:
        """Arithmetic interpolation of ``V/a_p`` from cells to faces."""
        response = self.pressure_response(ap)
        e2c = self.mesh.edge_to_cell()
        return 0.5 * (response[e2c[:, 0]] + response[e2c[:, 1]])

    def face_flux(self, face_velocity: TensorLike) -> TensorLike:
        """Return scalar face flux ``phi_f = U_f · S_f``."""
        return bm.einsum("ij,ij->i", face_velocity, self.mesh.edge_normal())

    def divergence_from_flux(self, phi: TensorLike) -> TensorLike:
        """Scatter scalar face fluxes to cell-integrated continuity residuals."""
        e2c = self.mesh.edge_to_cell()[:, :2]
        div_phi = bm.zeros(self.NC, dtype=phi.dtype)
        is_internal = e2c[:, 0] != e2c[:, 1]
        div_phi = bm.index_add(div_phi, e2c[:, 0], phi, axis=0)
        div_phi = bm.index_add(
            div_phi, e2c[is_internal, 1], phi[is_internal], axis=0, alpha=-1
        )
        return div_phi

    def pressure_correction_flux(
        self, p_corr: TensorLike, ap: TensorLike
    ) -> TensorLike:
        """Face flux correction generated by the SIMPLE pressure equation."""
        coef = self.face_pressure_response(ap)
        e2c = self.mesh.edge_to_cell()
        Sf = self.mesh.edge_normal()
        e = self.edge_centroid_vector
        d = self.edge_centroid_distance
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        e_dot_Sf = bm.einsum("ij,ij->i", e, Sf)
        e_norm = bm.linalg.norm(e, axis=-1)
        kf = (Sf_dot_Sf / e_dot_Sf * e_norm) / d * coef
        return kf * (p_corr[e2c[:, 0]] - p_corr[e2c[:, 1]])

    def enforce_face_flux(
        self, face_velocity: TensorLike, target_flux: TensorLike
    ) -> TensorLike:
        """Adjust only the face-normal component so ``U_f · S_f`` matches target."""
        Sf = self.mesh.edge_normal()
        current_flux = self.face_flux(face_velocity)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        normal_delta = ((target_flux - current_flux) / Sf_dot_Sf)[:, None] * Sf
        return face_velocity + normal_delta

    def velocity_pressure_correction(
        self, u_flat: TensorLike, p_corr: TensorLike, ap: TensorLike
    ) -> TensorLike:
        """Apply the SIMPLE cell-velocity correction ``U <- U - rAU grad(p')``."""
        grad_p = self.pressure_gradient.cell_gradient(p_corr)
        u_cell = bm.stack([u_flat[:self.NC], u_flat[self.NC:]], axis=-1)
        u_cell = u_cell - self.pressure_response(ap)[:, None] * grad_p
        return bm.concatenate([u_cell[:, 0], u_cell[:, 1]], axis=0)

    def pressure_correct(self, ap: TensorLike, uf: TensorLike) -> TensorLike:
        """Solve the pressure-correction equation used by SIMPLE."""
        A = self.pressure_correction_system(ap)
        div_u = self.divergence.Reconstruct(uf)  # (NC,)
        b0 = bm.array([0])
        b = bm.concatenate([-div_u, b0], axis=0)
        sol = spsolve(A, b, "mumps")
        p_c = sol[:-1]
        return p_c

    def pressure_correction_system(self, ap: TensorLike):
        """Assemble and cache the constant Stokes pressure-correction matrix."""
        if self._pressure_correction_system_cache is not None:
            return self._pressure_correction_system_cache

        # ``ap`` is the integrated momentum diagonal.  OpenFOAM's
        # ``rAU = 1/UEqn.A()`` corresponds to ``V/a_p`` in FEALPy because
        # ``UEqn.A()`` is normalized by cell volume.
        dp_edge = self.face_pressure_response(ap)
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
        self._pressure_correction_system_cache = A
        return A

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
        """Adapt pressure relaxation without changing the Stokes discretization."""
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
        max_iter: int = 100,
        tol: float = 1e-5,
        relax: float = 0.32,
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
    ) -> Tuple[TensorLike, TensorLike]:
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
        tol_pressure_update = (
            10.0 * tol if tol_pressure_update is None else tol_pressure_update
        )
        pressure_relax = relax
        relax_deterioration_count = 0
        relax_small_update_count = 0
        relax_cooldown = 0
        field_dtype = self.cm.dtype
        p = bm.zeros(self.NC, dtype=field_dtype)
        uf = bm.zeros((self.mesh.number_of_faces(), 2), dtype=field_dtype)
        u = bm.zeros(2 * self.NC, dtype=field_dtype)
        ap, u = self.temporary_velocity(p, uf, u)
        self.residuals = []
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter('edge')
        bdedgepoint = edge_middle_point[bd_edge]
        bdedgeu = self.pde.dirichlet_velocity(bdedgepoint)
        for i in range(max_iter):
            uf = self.rhie_chow.Interpolation(u,ap,p)
            uf = bm.set_at(uf, bd_edge, bdedgeu)
            # uf = self.Ucell2edge(u, self.pde.dirichlet_velocity)
            predictor_mass = collocated_mass_residual(self.mesh, uf)
            p_corr = self.pressure_correct(ap, uf)
            p_update = pressure_relax * p_corr
            phi = self.face_flux(uf) + self.pressure_correction_flux(p_corr, ap)
            uf = self.enforce_face_flux(uf, phi)
            uf = bm.set_at(uf, bd_edge, bdedgeu)
            u = self.velocity_pressure_correction(u, p_corr, ap)
            pressure_update_residual = relative_l2_update(self.mesh, p_update, p)
            residual = {
                "mass": collocated_mass_residual(self.mesh, uf),
                "predictor_mass": predictor_mass,
                "pressure_update": pressure_update_residual,
                "pressure_correction": cell_l2_norm(self.mesh, p_corr),
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
                        self.mesh, p_update, p
                    )
            residual["pressure_relax"] = pressure_relax
            residual["pressure_relax_reduced"] = relax_action == "reduce"
            residual["pressure_relax_action"] = relax_action
            residual["nonorthogonal_iterations"] = self.last_nonorthogonal_iterations
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
            _, u = self.temporary_velocity(p,uf,u)

        self.uh = u[:self.NC]
        self.vh = u[self.NC:]
        self.ph = p
        return self.uh, self.vh, self.ph

    def compute_error(self) -> Tuple[float, float]:
        """Compute errors for velocity and pressure."""
        cell_centers = self.mesh.entity_barycenter('cell')
        self.uI = self.pde.velocity(cell_centers)[:, 0]
        self.vI = self.pde.velocity(cell_centers)[:, 1]
        self.pI = self.pde.pressure(cell_centers)
        uerror = bm.sqrt(bm.sum(self.cm * (self.uh - self.uI)**2))
        verror = bm.sqrt(bm.sum(self.cm * (self.vh - self.vI)**2))
        perror = bm.sqrt(bm.sum(self.cm * (self.ph - self.pI)**2))
        # uerror = bm.max(bm.abs(self.uh - self.uI))
        # verror = bm.max(bm.abs(self.vh - self.vI))
        # perror = bm.max(bm.abs(self.ph - self.pI))
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
