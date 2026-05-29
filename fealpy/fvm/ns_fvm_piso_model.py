from typing import Optional, Tuple, Union

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel
from fealpy.fem import BilinearForm, LinearForm
from fealpy.sparse import CSRTensor

from fealpy.fvm import (
    ScalarDiffusionIntegrator,
    ConvectionIntegrator,
    ScalarSourceIntegrator,
    RhieChowInterpolation,
    VectorDecomposition,
)
from .collocated_ns_fvm_utils import CollocatedNSFVMOperators
from fealpy.decorator import cartesian


class NSFVMPISOModel(ComputationalModel, CollocatedNSFVMOperators):
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
        self.n_correctors = int(options.get("n_correctors", 2))
        mesh_type = self._normalized_mesh_type(
            options.get("mesh_type", "uniform_quad")
        )
        self.use_transient_flux_correction = bool(
            options.get("use_transient_flux_correction", True)
        )
        default_nonorthogonal_iter = (
            0 if mesh_type == "uniform_quad" else 10
        )
        self.momentum_nonorthogonal_max_iter = self._option_int(
            options,
            "momentum_nonorthogonal_max_iter",
            default_nonorthogonal_iter,
        )
        self.momentum_nonorthogonal_tol = float(
            options.get("momentum_nonorthogonal_tol", 1.0e-5)
        )
        self.pressure_nonorthogonal_max_iter = self._option_int(
            options,
            "pressure_nonorthogonal_max_iter",
            default_nonorthogonal_iter,
        )
        self.pressure_nonorthogonal_tol = float(
            options.get("pressure_nonorthogonal_tol", 1.0e-5)
        )
        self.validate_piso_controls(self.n_correctors)
        self.validate_nonorthogonal_controls(
            self.momentum_nonorthogonal_max_iter,
            self.momentum_nonorthogonal_tol,
            self.pressure_nonorthogonal_max_iter,
            self.pressure_nonorthogonal_tol,
        )
        self.set_pde(options["pde"])
        self.set_mesh(options["nx"], options["ny"])
        self.set_space(options.get("space_degree", 0))
        self.linear_solver = self._init_linear_solver(options)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
            f"  Time steps: {self.nt}\n"
            f"  PISO correctors: {self.n_correctors}\n"
            f"  Momentum nonorthogonal corrections: "
            f"{self.momentum_nonorthogonal_max_iter}\n"
            f"  Pressure nonorthogonal corrections: "
            f"{self.pressure_nonorthogonal_max_iter}\n"
        )

    @staticmethod
    def validate_piso_controls(n_correctors: int) -> None:
        if n_correctors < 1:
            raise ValueError("n_correctors must be positive.")

    @staticmethod
    def validate_nonorthogonal_controls(
        momentum_max_iter: int,
        momentum_tol: float,
        pressure_max_iter: int,
        pressure_tol: float,
    ) -> None:
        if momentum_max_iter < 0:
            raise ValueError("momentum_nonorthogonal_max_iter must be non-negative.")
        if pressure_max_iter < 0:
            raise ValueError("pressure_nonorthogonal_max_iter must be non-negative.")
        if momentum_tol <= 0.0:
            raise ValueError("momentum_nonorthogonal_tol must be positive.")
        if pressure_tol <= 0.0:
            raise ValueError("pressure_nonorthogonal_tol must be positive.")

    def set_pde(self, pde: Union[int, object]) -> None:
        self.pde = self._resolve_navier_stokes_pde(pde)

    def set_mesh(self, nx: int, ny: int) -> None:
        mesh_type = self._normalized_mesh_type(
            self.options.get("mesh_type", "uniform_quad")
        )
        self.mesh = self._init_navier_stokes_mesh(
            {"mesh_type": mesh_type, "nx": nx, "ny": ny},
            default_mesh_type="uniform_quad",
            normalize_mesh_type=True,
        )
        self.cm = self.mesh.entity_measure("cell")
        self.points = self.mesh.entity_barycenter("cell")
        self.epoints = self.mesh.entity_barycenter("edge")
        self.NC = self.mesh.number_of_cells()

    def set_space(self, degree: int) -> None:
        self._init_collocated_discretization(
            degree,
            self.pde.velocity_dirichlet,
            with_velocity_dirichlet_bc=True,
        )
        self.rhie_chow = RhieChowInterpolation(self.mesh)

    def initial_solution(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        t0 = self.duration[0]
        U0 = self.pde.velocity_0(self.points, t0)
        Uf0 = self.pde.velocity_0(self.epoints, t0)
        p0 = self.pde.pressure_0(self.points, t0)
        return U0, Uf0, p0

    def temporary_velocity(self, U0, Uf0, p0, t, return_matrix=False):
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
        A, b = self.velocity_dirichlet_bc.DiffusionApply(A, b)
        b = self.velocity_dirichlet_bc.ConvectionApply(b, Uf0)
        # FEALPy sparse assembly can leave duplicate entries here.
        A = A.tocoo().coalesce().tocsr()
        a_p = A.diags().values
        U = self.linear_solver.solve(A, b)
        U = self.correct_momentum_nonorthogonal_diffusion(A, b, U, U0)
        if return_matrix:
            return U, a_p, A
        return U, a_p

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
        """Return face pressure response ``D_f`` used by pressure correction."""
        dp = self.cm/a_p[:self.NC]
        return self.face_interpolate_cell_scalar(dp)

    def solve_pressure_correction(
        self,
        rhs,
        a_p,
        nonorthogonal_max_iter: Optional[int] = None,
        nonorthogonal_tol: Optional[float] = None,
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
        return self._solve_pressure_correction_with_cross_rhs(
            rhs,
            coef,
            q=self.p + 2,
            nonorthogonal_max_iter=nonorthogonal_max_iter,
            nonorthogonal_tol=nonorthogonal_tol,
            cross_flux=self.pressure_correction_cross_flux,
        )

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
    def pressure_correction_flux(self, pressure_increment, a_p):
        """Return the scalar flux contribution produced by a pressure field."""
        coef = self.pressure_response_face_coefficient(a_p)
        return self.pressure_correction_flux_from_coefficient(
            pressure_increment, coef
        )

    def pressure_correction_flux_from_coefficient(self, pressure_increment, coef):
        """Return pressure-correction flux from a face response coefficient."""
        return (
            self.pressure_correction_orthogonal_flux(pressure_increment, coef)
            - self.pressure_correction_cross_flux(pressure_increment, coef)
        )

    def pressure_correction_orthogonal_flux(self, pressure_increment, coef):
        """Return the implicit orthogonal pressure-correction face flux."""
        e2c = self.e2c
        e, d = VectorDecomposition(self.mesh).centroid_vector_calculation()
        Sf = self.mesh.edge_normal()
        Sf_dot_Sf = bm.einsum("ij,ij->i", Sf, Sf)
        e_dot_Sf = bm.einsum("ij,ij->i", e, Sf)
        e_norm = bm.linalg.norm(e, axis=-1)
        ef_abs = Sf_dot_Sf/e_dot_Sf*e_norm
        kf = ef_abs/d*coef
        return kf*(pressure_increment[e2c[:, 0]] - pressure_increment[e2c[:, 1]])

    def pressure_correction_cross_flux(self, pressure_increment, coef):
        """Return the explicit non-orthogonal pressure-correction face flux."""
        return self._pressure_correction_cross_flux(pressure_increment, coef)

    def rhie_chow_face_velocity(
        self,
        u_flat,
        a_p,
        pressure,
        target_flux=None,
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
        return self.apply_face_velocity_dirichlet(face_velocity, boundary_velocity)

    def cell_velocity_face_flux(self, cell_velocity):
        """Return the face flux from interpolated cell-centred velocity."""
        return self.face_flux(self.face_interpolate_cell_vector(cell_velocity))

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
        response_coef = self.pressure_response_face_coefficient(a_p)
        return response_coef * (previous_face_flux - previous_cell_flux) / self.tau

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
        boundary_velocity,
        previous_cell_velocity=None,
        previous_face_velocity=None,
    ):
        """Perform one PISO pressure-correction step.

        The pressure equation is solved for the pressure state associated with
        the current pressure-free velocity estimate, matching the OpenFOAM PISO
        pressure-corrector semantics without exposing an HbyA-style abstraction.
        """
        pressure_free_velocity = self.pressure_free_velocity(
            intermediate_velocity, pressure, a_p
        )
        face_velocity = self.face_interpolate_cell_vector(pressure_free_velocity)
        flux = self.face_flux(face_velocity)
        flux = flux + self.transient_face_flux_correction(
            previous_cell_velocity,
            previous_face_velocity,
            a_p,
        )
        flux = self.apply_boundary_flux_constraint(flux, boundary_velocity)
        pressure_state = self.solve_pressure_correction(
            -self.divergence_from_flux(flux), a_p
        )
        corrected_velocity = self.velocity_pressure_correction(
            pressure_free_velocity,
            pressure_state,
            a_p,
        )
        corrected_flux = flux + self.pressure_correction_flux(
            pressure_state, a_p
        )
        corrected_flux = self.apply_boundary_flux_constraint(
            corrected_flux, boundary_velocity
        )
        return corrected_velocity, pressure_state, corrected_flux

    def solve(
        self,
        U0=None,
        Uf0=None,
        p0=None,
        n_correctors=None,
    ) -> Tuple[TensorLike, TensorLike, TensorLike]:
        if U0 is None or Uf0 is None or p0 is None:
            U0, Uf0, p0 = self.initial_solution()

        n_correctors = (
            self.n_correctors if n_correctors is None else int(n_correctors)
        )
        self.validate_piso_controls(n_correctors)

        current_velocity = None
        current_pressure = p0
        for n in range(self.nt):
            t = self.duration[0] + n * self.tau
            previous_cell_velocity = U0
            previous_face_velocity = Uf0
            u_tem, a_p, momentum_matrix = self.temporary_velocity(
                U0, Uf0, p0, t + self.tau, return_matrix=True
            )
            _, boundary_velocity = self._boundary_face_velocity()

            previous_velocity = u_tem
            current_velocity = u_tem
            current_pressure = p0
            phi = None
            correction = 0
            while True:
                correction += 1
                if correction == 1:
                    intermediate_velocity = current_velocity
                else:
                    intermediate_velocity = self.operator_splitting_velocity_correction(
                        current_velocity, previous_velocity, momentum_matrix, a_p
                    )

                next_velocity, next_pressure, phi = self.pressure_correction_step(
                    intermediate_velocity,
                    current_pressure,
                    a_p,
                    boundary_velocity,
                    previous_cell_velocity,
                    previous_face_velocity,
                )
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
                boundary_velocity=boundary_velocity,
                face_response_coefficient=self.pressure_response_face_coefficient(a_p),
            )

            U0 = bm.stack(
                [current_velocity[:self.NC], current_velocity[self.NC:]], axis=-1
            )
            p0 = current_pressure

        self.uh = current_velocity[:self.NC]
        self.vh = current_velocity[self.NC:]
        self.ph = current_pressure
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
