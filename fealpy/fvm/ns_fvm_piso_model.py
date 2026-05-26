from typing import Tuple, Union

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import PDEModelManager, ComputationalModel
from fealpy.functionspace import ScaledMonomialSpace2d, TensorFunctionSpace
from fealpy.fem import BilinearForm, LinearForm, BlockForm
from fealpy.sparse import COOTensor, CSRTensor
from fealpy.solver import spsolve

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ConvectionIntegrator,
    ScalarSourceIntegrator,
    GradientReconstruct,
    DirichletBC,
    RhieChowInterpolation,
    VectorDecomposition,
)
from fealpy.decorator import cartesian


class NSFVMPISOModel(ComputationalModel):
    """Collocated finite-volume PISO solver with Rhie-Chow interpolation."""

    def __init__(self, options):
        self.options = options
        super().__init__(
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        self.duration = tuple(options.get("duration", (0, 1)))
        self.nt = options.get("nt", 20)
        self.tau = (self.duration[1] - self.duration[0]) / self.nt
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space(options.get("space_degree", 0))

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
            f"  Time steps: {self.nt}\n"
        )

    def set_pde(self, pde: Union[int, object]) -> None:
        self.pde = (
            PDEModelManager("navier_stokes").get_example(pde)
            if isinstance(pde, int)
            else pde
        )

    def set_mesh(self, nx: int, ny: int) -> None:
        self.mesh = self.pde.init_mesh["uniform_qrad"](nx=nx, ny=ny)
        self.cm = self.mesh.entity_measure("cell")
        self.points = self.mesh.entity_barycenter("cell")
        self.epoints = self.mesh.entity_barycenter("edge")
        self.NC = self.mesh.number_of_cells()

    def set_space(self, degree: int) -> None:
        self.p = degree
        self.space = ScaledMonomialSpace2d(self.mesh, self.p)
        self.velocity_space = TensorFunctionSpace(self.space, shape=(2, -1))
        self.pressure_gradient = GradientReconstruct(self.mesh)
        self.rhie_chow = RhieChowInterpolation(self.mesh)

    def initial_solution(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        t0 = self.duration[0]
        U0 = self.pde.velocity_0(self.points, t0)
        Uf0 = self.pde.velocity_0(self.epoints, t0)
        p0 = self.pde.pressure_0(self.points, t0)
        return U0, Uf0, p0

    def temporary_velocity(self, U0, Uf0, p0, t):
        bform = BilinearForm(self.velocity_space)
        bform.add_integrator(ScalarDiffusionIntegrator(q=self.p + 2))
        bform.add_integrator(ConvectionIntegrator(q=self.p + 2, coef=Uf0))
        A = bform.assembly()

        M = CSRTensor(
            crow=bm.arange(2*self.NC + 1),
            col=bm.arange(2*self.NC),
            values=bm.concatenate([self.cm / self.tau, self.cm / self.tau]),
            spshape=(2*self.NC, 2*self.NC),
        )

        @cartesian
        def src(p):
            return self.pde.source(p, t)

        f = LinearForm(self.velocity_space).add_integrator(
            ScalarSourceIntegrator(src, q=self.p + 2)
        ).assembly()

        grad_p = self.pressure_gradient.cell_gradient(p0)
        p1 = bm.einsum("i,i->i", grad_p[:, 0], self.cm)
        p2 = bm.einsum("i,i->i", grad_p[:, 1], self.cm)
        p_grad_integrator = bm.concatenate((p1, p2))
        A = A + M
        b = (
            f
            - p_grad_integrator
            + (U0 * (self.cm / self.tau)[:, None]).flatten(order="F")
        )
        dbc = DirichletBC(self.mesh, self.pde.velocity_dirichlet)
        A, b = dbc.DiffusionApply(A, b)
        b = dbc.ConvectionApply(b, Uf0)
        # FEALPy sparse assembly can leave duplicate entries here.
        A = A.tocoo().coalesce().tocsr()
        a_p = A.diags().values
        U = spsolve(A, b, "mumps")
        return U, a_p

    def solve_pressure_correction(self, rhs, a_p):
        dp = self.cm/a_p[:self.NC]
        e2c = self.mesh.edge_to_cell()
        coef = (dp[e2c[:, 0]] + dp[e2c[:, 1]])/2

        A = BilinearForm(self.space).add_integrator(
            ScalarDiffusionIntegrator(q=self.p + 2, coef=coef)
        ).assembly()

        A1 = COOTensor(
            bm.array([
                bm.zeros(self.NC, dtype=bm.int32),
                bm.arange(self.NC, dtype=bm.int32),
            ]),
            self.cm,
            spshape=(1, self.NC),
        )
        A = BlockForm([[A, A1.T], [A1, None]])
        A = A.assembly_sparse_matrix(format="csr")
        b = bm.concatenate([rhs, bm.array([0])], axis=0)
        sol = spsolve(A, b, "mumps")
        return sol[:-1]

    # Local algebra for the incremental PISO step below.
    #
    # These helpers are kept in this prototype solver on purpose.  They do not
    # form a public finite-volume API yet: the reusable mathematical object has
    # to be clearer than "what this PISO solver happens to need".  The generic
    # pieces here are the scalar surface flux phi_f = u_f * S_f and its cell
    # imbalance.  The remaining functions belong to this first-order
    # pressure-rate correction:
    #
    #     q = p' / dt
    #     U <- U - rAU grad(q),       rAU = V / a_p
    #     p <- p + dt q
    #     phi <- phi + phi_q
    #
    # If another solver needs the same operators, they should be promoted only
    # after the shared finite-volume algebra and boundary contract are explicit.

    def face_flux(self, face_velocity):
        """Return the signed surface flux ``phi_f = u_f dot S_f``."""
        return bm.einsum("ij,ij->i", face_velocity, self.mesh.edge_normal())

    def divergence_from_flux(self, phi):
        """Scatter signed face fluxes to the cell flux imbalance."""
        e2c = self.mesh.edge_to_cell()[:, :2]
        div_phi = bm.zeros(self.NC, dtype=phi.dtype)
        is_internal = e2c[:, 0] != e2c[:, 1]
        bm.add_at(div_phi, e2c[:, 0], phi)
        bm.add_at(div_phi, e2c[is_internal, 1], -phi[is_internal])
        return div_phi

    def pressure_correction_flux(self, p_corr, a_p):
        """Return the PISO scalar flux increment produced by pressure rate q."""
        dp = self.cm/a_p[:self.NC]
        e2c = self.mesh.edge_to_cell()
        coef = (dp[e2c[:, 0]] + dp[e2c[:, 1]])/2
        e, d = VectorDecomposition(self.mesh).centroid_vector_calculation()
        Sf = self.mesh.edge_normal()
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        e_dot_Sf = bm.einsum("ij,ij->i", e, Sf)
        e_norm = bm.linalg.norm(e, axis=-1)
        ef_abs = Sf_dot_Sf/e_dot_Sf*e_norm
        kf = ef_abs/d*coef
        return kf*(p_corr[e2c[:, 0]] - p_corr[e2c[:, 1]])

    def enforce_face_flux(self, face_velocity, target_flux):
        """Adjust only the normal component of a vector face velocity."""
        Sf = self.mesh.edge_normal()
        current_flux = self.face_flux(face_velocity)
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        return face_velocity + ((target_flux - current_flux)/Sf_dot_Sf)[:, None]*Sf

    def apply_face_velocity_dirichlet(self, face_velocity, boundary_velocity=None):
        """Apply externally supplied velocity Dirichlet data on boundary faces."""
        if boundary_velocity is None:
            return face_velocity

        face_velocity = bm.array(face_velocity)
        bd_edge = self.mesh.boundary_face_index()
        boundary_velocity = bm.array(boundary_velocity)
        if boundary_velocity.shape[0] == self.mesh.number_of_faces():
            boundary_velocity = boundary_velocity[bd_edge]
        face_velocity[bd_edge] = boundary_velocity
        return face_velocity

    def rhie_chow_face_velocity(
        self, u_flat, a_p, pressure, target_flux=None, boundary_velocity=None
    ):
        """Build a collocated face velocity for the current PISO substep."""
        face_velocity = self.rhie_chow.Interpolation(u_flat, a_p, pressure)
        if target_flux is not None:
            face_velocity = self.enforce_face_flux(face_velocity, target_flux)
        return self.apply_face_velocity_dirichlet(face_velocity, boundary_velocity)

    def velocity_pressure_correction(self, u_flat, pressure_rate, a_p):
        """Apply the cell velocity correction ``U <- U - rAU grad(q)``."""
        grad_p = self.pressure_gradient.cell_gradient(pressure_rate)
        u_cell = bm.stack([u_flat[:self.NC], u_flat[self.NC:]], axis=-1)
        u_cell = u_cell - (self.cm/a_p[:self.NC])[:, None] * grad_p
        return u_cell.flatten(order="F")

    def apply_pressure_rate_correction(self, u_flat, pressure, pressure_rate, a_p):
        """Apply incremental PISO correction with q = p' / dt."""
        u_new = self.velocity_pressure_correction(u_flat, pressure_rate, a_p)
        p_new = pressure + self.tau * pressure_rate
        return u_new, p_new

    def solve(
        self,
        U0=None,
        Uf0=None,
        p0=None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        if U0 is None or Uf0 is None or p0 is None:
            U0, Uf0, p0 = self.initial_solution()

        u_tem3 = None
        p2 = p0
        for n in range(self.nt):
            t = self.duration[0] + n * self.tau
            u_tem, a_p = self.temporary_velocity(U0, Uf0, p0, t + self.tau)
            bd_edge = self.mesh.boundary_face_index()
            boundary_velocity = self.pde.velocity_dirichlet(self.epoints[bd_edge])

            uf = self.rhie_chow_face_velocity(
                u_tem, a_p, p0, boundary_velocity=boundary_velocity
            )
            phi = self.face_flux(uf)
            p_rate1 = self.solve_pressure_correction(
                -self.divergence_from_flux(phi), a_p
            )
            u_tem2, p1 = self.apply_pressure_rate_correction(
                u_tem, p0, p_rate1, a_p
            )
            phi1 = phi + self.pressure_correction_flux(p_rate1, a_p)

            uf2 = self.rhie_chow_face_velocity(
                u_tem2, a_p, p1, phi1, boundary_velocity=boundary_velocity
            )
            p_rate2 = self.solve_pressure_correction(
                -self.divergence_from_flux(self.face_flux(uf2)), a_p
            )
            u_tem3, p2 = self.apply_pressure_rate_correction(
                u_tem2, p1, p_rate2, a_p
            )
            phi2 = self.face_flux(uf2) + self.pressure_correction_flux(p_rate2, a_p)
            Uf0 = self.rhie_chow_face_velocity(
                u_tem3, a_p, p2, phi2, boundary_velocity=boundary_velocity
            )

            U0 = bm.stack([u_tem3[:self.NC], u_tem3[self.NC:]], axis=-1)
            p0 = p2

        self.uh = u_tem3[:self.NC]
        self.vh = u_tem3[self.NC:]
        self.ph = p2
        return self.uh, self.vh, self.ph

    def compute_error(self) -> Tuple[float, float, float]:
        t = self.duration[1]
        self.uI = self.pde.velocity_u(self.points, t)
        self.vI = self.pde.velocity_v(self.points, t)
        self.pI = self.pde.pressure(self.points, t)
        uerror = bm.sqrt(bm.sum(self.cm * (self.uh - self.uI) ** 2))
        verror = bm.sqrt(bm.sum(self.cm * (self.vh - self.vI) ** 2))
        perror = bm.sqrt(bm.sum(self.cm * (self.ph - self.pI) ** 2))
        return uerror, verror, perror

    def plot(self) -> None:
        import matplotlib.pyplot as plt

        cell_centers = self.mesh.entity_barycenter("cell")
        x, y = cell_centers[:, 0], cell_centers[:, 1]

        fig = plt.figure(figsize=(15, 10))
        titles = [
            ("Error u", self.uh - self.uI),
            ("Error v", self.vh - self.vI),
            ("Error p", self.ph - self.pI),
        ]
        for i, (title, data) in enumerate(titles):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")
            ax.plot_trisurf(x, y, data, cmap="viridis")
            ax.set_title(title)
        plt.tight_layout()
        plt.show()
