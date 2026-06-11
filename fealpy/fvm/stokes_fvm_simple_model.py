"""Manufactured-case adapter for the collocated SIMPLE Stokes solve."""

from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel, PDEModelManager

from .collocated_simple_solver import CollocatedSimpleSolver
from .cell_average_error import cell_average_l2_error
from .engineering_boundary_conditions import BoundaryConditionData
from .solver_controls import SimpleSolverControls


class StokesFVMSimpleModel(ComputationalModel, CollocatedSimpleSolver):
    """Finite-volume SIMPLE model for Stokes PDE examples."""

    def __init__(self, options):
        self.options = options
        self._validate_options()
        ComputationalModel.__init__(
            self,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        self.pde = self._resolve_stokes_pde(options["pde"])
        self.error_quadrature_order = int(options.get("error_quadrature_order", 4))
        mesh = self._init_mesh(options)
        self.mu = self._init_diffusion_coef(options)
        CollocatedSimpleSolver.__init__(
            self,
            mesh=mesh,
            diffusion_coef=self.mu,
            convection_coef=0.0,
            source=self.pde.source,
            boundary_conditions=BoundaryConditionData(
                self.pde.dirichlet_velocity
            ).to_pde_boundary(mesh),
            controls=self._simple_controls_from_options(options),
            linear_solver=options.get("linear_solver"),
            linear_solver_config=options.get("linear_solver_config"),
            logger=self.logger,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )

    def _validate_options(self) -> None:
        allowed = {
            "pde",
            "mesh_type",
            "mesh_refine",
            "nx",
            "ny",
            "mu",
            "space_degree",
            "pressure_gradient_method",
            "velocity_gradient_method",
            "rhie_chow_pressure_gradient_method",
            "face_interpolation_method",
            "momentum_face_interpolation",
            "pressure_response_interpolation",
            "rhie_chow_velocity_interpolation",
            "momentum_nonorthogonal_max_iter",
            "momentum_nonorthogonal_tol",
            "pressure_nonorthogonal_max_iter",
            "pressure_nonorthogonal_tol",
            "error_quadrature_order",
            "linear_solver",
            "linear_solver_config",
            "pbar_log",
            "log_level",
        }
        unsupported = set(self.options).difference(allowed)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported StokesFVMSimpleModel options: {names}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    @staticmethod
    def _resolve_stokes_pde(pde):
        return PDEModelManager("stokes").get_example(pde) if isinstance(pde, int) else pde

    def _init_mesh(self, options):
        mesh_type = (
            options.get("mesh_type")
            or getattr(self.pde, "default_mesh_type", "uniform_quad")
        )
        mesh_type = {"quad": "uniform_quad", "tri": "uniform_tri"}.get(
            mesh_type, mesh_type
        )
        mesh_refine = int(options.get("mesh_refine", 0))
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")

        if getattr(self.pde, "supports_geometric_refine", False):
            return self.pde.init_mesh[mesh_type](mesh_refine=mesh_refine)

        mesh_options = {}
        if "nx" in options:
            mesh_options["nx"] = int(options["nx"])
        if "ny" in options:
            mesh_options["ny"] = int(options["ny"])
        mesh = self.pde.init_mesh[mesh_type](**mesh_options)
        if mesh_refine == 0:
            return mesh
        if hasattr(mesh, "uniform_refine"):
            mesh.uniform_refine(mesh_refine)
            return mesh
        raise ValueError("mesh does not provide uniform_refine().")

    def _init_diffusion_coef(self, options):
        if options.get("mu") is not None:
            return self._as_positive_scalar(options["mu"], "mu")
        for name in ("viscosity", "mu"):
            if hasattr(self.pde, name):
                return self._as_positive_scalar(getattr(self.pde, name), "mu")
        return 1.0

    @staticmethod
    def _simple_controls_from_options(options):
        return SimpleSolverControls(
            space_degree=options.get("space_degree", 0),
            pressure_gradient_method=options.get(
                "pressure_gradient_method", "extended_lsq"
            ),
            velocity_gradient_method=options.get(
                "velocity_gradient_method", "extended_lsq"
            ),
            rhie_chow_pressure_gradient_method=options.get(
                "rhie_chow_pressure_gradient_method", "extended_lsq"
            ),
            face_interpolation_method=options.get("face_interpolation_method", "average"),
            momentum_face_interpolation=options.get("momentum_face_interpolation"),
            pressure_response_interpolation=options.get(
                "pressure_response_interpolation"
            ),
            rhie_chow_velocity_interpolation=options.get(
                "rhie_chow_velocity_interpolation"
            ),
            momentum_equation_relaxation=options.get(
                "momentum_equation_relaxation", 1.0
            ),
            momentum_nonorthogonal_max_iter=options.get(
                "momentum_nonorthogonal_max_iter", 10
            ),
            momentum_nonorthogonal_tol=options.get(
                "momentum_nonorthogonal_tol", 1.0e-4
            ),
            pressure_nonorthogonal_max_iter=options.get(
                "pressure_nonorthogonal_max_iter", 10
            ),
            pressure_nonorthogonal_tol=options.get(
                "pressure_nonorthogonal_tol", 1.0e-5
            ),
        )

    def compute_error(self) -> Tuple[float, float, float]:
        """Compute errors against exact control-volume averages."""
        velocity_error, velocity_average = cell_average_l2_error(
            self.mesh,
            self.pde.velocity,
            self.velocity,
            q=self.error_quadrature_order,
        )
        perror, self.pI = cell_average_l2_error(
            self.mesh,
            self.pde.pressure,
            self.ph,
            q=self.error_quadrature_order,
        )
        self.uI = velocity_average[:, 0]
        self.vI = velocity_average[:, 1] if self.GD > 1 else bm.zeros_like(self.uI)
        if self.GD > 2:
            self.wI = velocity_average[:, 2]
        return tuple(velocity_error[i] for i in range(self.GD)) + (perror,)

    def plot(self) -> None:
        """Plot numerical and exact solution errors for u, v, and p."""
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
