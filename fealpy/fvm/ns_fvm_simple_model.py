from typing import Union, Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.sparse import COOTensor

from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm
from fealpy.solver import spsolve

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ScalarCrossDiffusionIntegrator,
    ScalarSourceIntegrator,
    ConvectionIntegrator,
    GradientReconstruct,
    DivergenceReconstruct,
    DirichletBC,
    RhieChowInterpolation    
)
from .simple_residual import (
    cell_l2_norm,
    collocated_mass_residual,
    relative_l2_update,
)


class NSFVMSimpleModel(ComputationalModel):
    """
    Finite Volume SIMPLE solver for 2D steady incompressible Navier–Stokes equations.
    """

    def __init__(self, options):
        self.options = options
        super().__init__(pbar_log=options.get("pbar_log", False),
                         log_level=options.get("log_level", "WARNING"))
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space(options["space_degree"])

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def set_pde(self, pde: Union[int, object]) -> None:
        """Set PDE model (Navier–Stokes example)."""
        self.pde = PDEModelManager("navier_stokes").get_example(pde) if isinstance(pde, int) else pde

    def set_mesh(self, nx: int, ny: int) -> None:
        """Initialize computational mesh."""
        self.mesh = self.pde.init_mesh['uniform_tri'](nx=nx, ny=ny)
        self.cm = self.mesh.entity_measure('cell')
        self.NC = self.mesh.number_of_cells()

    def set_space(self, degree: int) -> None:
        """Define pressure/velocity function spaces."""
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))

    def temporary_velocity(self, p, uf, u0) -> Tuple[TensorLike, TensorLike]:
        """Solve momentum eqn for intermediate velocity u*."""
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        bform.add_integrator(ConvectionIntegrator(q=self.p + 2, coef=uf))
        B = bform.assembly()
        lform = LinearForm(self.velocity_space)
        lform.add_integrator(ScalarSourceIntegrator(self.pde.source, q=self.p + 2))
        f = lform.assembly()
        dbc = DirichletBC(self.mesh, self.pde.dirichlet_velocity)
        B, f = dbc.DiffusionApply(B, f)
        f = dbc.ConvectionApply(f, uf)
        # FEALPy sparse assembly can leave duplicate entries here.
        B = B.tocoo().coalesce().tocsr()
        ap = B.diags().values
        grad_p = GradientReconstruct(self.mesh).LSQ(p)  # (NC, 2)
        p1 = bm.einsum('i,i->i', grad_p[:,0], self.cm)
        p2 = bm.einsum('i,i->i', grad_p[:,1], self.cm)
        p_grad_integrator = bm.concatenate((p1,p2))
        f = f - p_grad_integrator
        u = spsolve(B, f, "mumps")

        cross = self.compute_cross_diffusion(u0)
        for _ in range(10):
            rhs = f + cross
            uh_new = spsolve(B, rhs)
            err = bm.max(bm.abs(uh_new - u))
            if err < 10e-5:
                break
            u = uh_new
            cross = self.compute_cross_diffusion(u)
        
        return ap, u
    
    def compute_cross_diffusion(self, uh: TensorLike) -> TensorLike:
        """Compute cross-diffusion term based on current velocity uh."""
        lform = LinearForm(self.velocity_space)
        U = bm.stack((uh[:self.NC], uh[self.NC:]), axis=1)
        grad_u = GradientReconstruct(self.mesh).AverageGradientreDirichlet(U,self.pde.dirichlet_velocity)
        grad_f = GradientReconstruct(self.mesh).reconstruct(grad_u)  # (NE, 2)
        lform.add_integrator(ScalarCrossDiffusionIntegrator(uh, grad_f))
        return lform.assembly()
    
    def pressure_correct(self, ap: TensorLike, uf: TensorLike) -> TensorLike:
        """Solve pressure correction equation.
        在这里实际上是有问题的,理论上计算dp_edge的程序应该是:
        dp = cm/ap[:len(cm)]
        dp_edge = (dp[e2c[:,0]]+dp[e2c[:,1]])/2
        但是随着网格加密,使用此dp_edge会导致求出的压力修正过大,必须使用非常小的松弛因子比如0.001才能使迭代收敛.
        所以进行了改动,使用dp = 1/ap[:len(cm)],然后计算
        dp_edge = (dp[e2c[:,0]]+dp[e2c[:,1]])/2
        dp_edge = dp_edge*em
        最终不需要过大的收敛因子,而且迭代次数也相对会减少一些.
        但这样做,在算法原理上是矛盾的,而且随着网格的加密,松弛因子仍然要不断减小,大概是网格加密一倍,松弛因子就要减小到原来的一半.
        这个问题需要进一步研究.
        """
        cm = self.mesh.entity_measure('cell')
        em = self.mesh.entity_measure('edge')
        # Mathematical risk:
        # The standard SIMPLE response would use V/a_p on cells before face
        # interpolation.  This historical branch uses 1/a_p multiplied by
        # face measure to maintain the current convergence behavior, so it
        # should be revisited before treating this model as a general SIMPLE
        # discretization.
        dp = 1/ap[:len(cm)]
        e2c = self.mesh.edge_to_cell()
        dp_edge = (dp[e2c[:,0]]+dp[e2c[:,1]])/2
        dp_edge = dp_edge*em
        div_u = DivergenceReconstruct(self.mesh).Reconstruct(uf)  # (NC,)
        bform2 = BilinearForm(self.space)
        bform2.add_integrator(ScalarDiffusionIntegrator(q=2,coef=dp_edge))
        A = bform2.assembly()
        LagA = self.mesh.entity_measure("cell")
        A1 = COOTensor(bm.array([bm.zeros(len(LagA), dtype=bm.int32),
                 bm.arange(len(LagA), dtype=bm.int32)]),LagA,
            spshape=(1, len(LagA)),
        )
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format="csr")
        b0 = bm.array([0])
        b = bm.concatenate([-div_u, b0], axis=0)
        sol = spsolve(A, b, "mumps")
        p_c = sol[:-1]
        return p_c

    def solve(
        self,
        max_iter: int = 100,
        tol: float = 1e-5,
        relax: float = 0.32,
        tol_mass=None,
        tol_pressure_update=None,
    ) -> Tuple[TensorLike, TensorLike]:
        """Main SIMPLE loop."""
        tol_mass = tol if tol_mass is None else tol_mass
        tol_pressure_update = (
            10.0 * tol if tol_pressure_update is None else tol_pressure_update
        )
        p = bm.zeros(self.NC)
        uf = bm.zeros((self.mesh.number_of_faces(), 2))
        u = bm.zeros(2 * self.NC)
        ap, u = self.temporary_velocity(p, uf, u)
        self.residuals = []
        bd_edge = self.mesh.boundary_face_index()
        edge_middle_point = self.mesh.entity_barycenter('edge')
        bdedgepoint = edge_middle_point[bd_edge]
        bdedgeu = self.pde.dirichlet_velocity(bdedgepoint)
        rhie_chow = RhieChowInterpolation(self.mesh)
        for i in range(max_iter):
            uf = rhie_chow.Interpolation(u,ap,p)
            uf[bd_edge, :] = bdedgeu
            p_corr = self.pressure_correct(ap, uf)
            p_update = relax * p_corr
            residual = {
                "mass": collocated_mass_residual(self.mesh, uf),
                "pressure_update": relative_l2_update(self.mesh, p_update, p),
                "pressure_correction": cell_l2_norm(self.mesh, p_corr),
            }
            self.residuals.append(residual)
            self.logger.info(
                f"[Iter {i+1}] mass residual: {residual['mass']:.2e}, "
                f"pressure update residual: {residual['pressure_update']:.2e}, "
                f"pressure correction L2: {residual['pressure_correction']:.2e}"
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
