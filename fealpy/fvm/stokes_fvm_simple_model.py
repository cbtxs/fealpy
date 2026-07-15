"""Manufactured-case adapter for the collocated SIMPLE Stokes solve."""

from fealpy.model import ComputationalModel, PDEModelManager

from .collocated_simple_solver import CollocatedSimpleSolver
from .cell_average_error import cell_average_l2_error
from .engineering_boundary_conditions import PDEBoundaryConditions
from .solver_controls import SimpleSolverControls, positive_scalar


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
            boundary_conditions=PDEBoundaryConditions(
                mesh,
                dirichlet_velocity=self.pde.dirichlet_velocity,
            ),
            controls=SimpleSolverControls.from_mapping({
                "rhie_chow_velocity_scheme": "second_order_reconstructed",
                **options,
            }),
            linear_solver=options.get("linear_solver"),
            linear_solver_config=options.get("linear_solver_config"),
            logger=self.logger,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )

    def _validate_options(self) -> None:
        allowed = set(SimpleSolverControls.option_names()) | {
            "pde",
            "mesh_type",
            "mesh_refine",
            "nx",
            "ny",
            "mu",
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
            return positive_scalar(options["mu"], "mu")
        for name in ("viscosity", "mu"):
            if hasattr(self.pde, name):
                return positive_scalar(getattr(self.pde, name), "mu")
        return 1.0

    def compute_error(self) -> tuple[float, ...]:
        """Compute errors against exact control-volume averages."""
        velocity_error, self.exact_velocity = cell_average_l2_error(
            self.mesh,
            self.pde.velocity,
            self.velocity,
            q=self.error_quadrature_order,
        )
        pressure_error, self.exact_pressure = cell_average_l2_error(
            self.mesh,
            self.pde.pressure,
            self.pressure,
            q=self.error_quadrature_order,
        )
        return tuple(velocity_error[i] for i in range(self.GD)) + (pressure_error,)

    def plot(self) -> None:
        """Plot numerical and exact solution errors for u, v, and p."""
        import matplotlib.pyplot as plt

        cell_centers = self.mesh.entity_barycenter("cell")
        x, y = cell_centers[:, 0], cell_centers[:, 1]

        fig = plt.figure(figsize=(15, 10))
        titles = [
            ("Error u", self.velocity[:, 0] - self.exact_velocity[:, 0]),
            ("Error v", self.velocity[:, 1] - self.exact_velocity[:, 1]),
            ("Error p", self.pressure - self.exact_pressure),
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
        pressure_correction = [
            residual["pressure_correction"] for residual in self.residuals
        ]
        plt.figure(figsize=(8, 5))
        plt.semilogy(mass, marker="o", linestyle="-", color="b", label="mass")
        plt.semilogy(
            pressure_correction,
            marker="s",
            linestyle="-",
            color="r",
            label="pressure correction",
        )
        plt.legend()
        plt.title("SIMPLE Residuals vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Residual (log scale)")
        plt.grid(True, which="both", ls="--")
        plt.tight_layout()
        plt.show()
