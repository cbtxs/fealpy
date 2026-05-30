"""Manufactured-case adapter for the collocated SIMPLE solver."""

from inspect import signature
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel

from .collocated_simple_solver import CollocatedSimpleSolver
from .simple_solver_data import SimpleBoundaryConditions, SimpleSolverControls


class NSFVMSimpleModel(ComputationalModel, CollocatedSimpleSolver):
    """Finite Volume SIMPLE model for PDE examples with exact solutions."""

    def __init__(self, options):
        ComputationalModel.__init__(
            self,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        pde = self._resolve_navier_stokes_pde(options["pde"])
        self.pde = pde
        self._init_momentum_coefficients(options)
        mesh = self._init_mesh(options)
        boundary_conditions = self._init_simple_boundary_conditions(options, mesh, pde)
        self.engineering_bc = (
            boundary_conditions
            if options.get("boundary_conditions") is not None
            else None
        )
        self._init_simple_solver(
            mesh=mesh,
            diffusion_coef=self.mu,
            convection_coef=self.rho,
            source=pde.source,
            boundary_conditions=boundary_conditions,
            controls=SimpleSolverControls.from_options(options),
            linear_solver=options.get("linear_solver"),
            linear_solver_config=options.get("linear_solver_config"),
            logger=self.logger,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )

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

    def _init_simple_boundary_conditions(self, options, mesh, pde):
        """Translate model or engineering boundary input to solver boundary data."""
        boundary_conditions = options.get("boundary_conditions")
        if boundary_conditions is None:
            return SimpleBoundaryConditions(pde.dirichlet_velocity)

        if callable(boundary_conditions) and not hasattr(
            boundary_conditions, "dirichlet_threshold"
        ):
            return self._call_boundary_condition_factory(
                boundary_conditions,
                mesh,
                pde,
            )
        return boundary_conditions

    @staticmethod
    def _call_boundary_condition_factory(factory, mesh, pde):
        """Call a boundary-condition factory with mesh or mesh plus PDE."""
        try:
            parameters = list(signature(factory).parameters.values())
        except (TypeError, ValueError):
            return factory(mesh, pde)

        accepts_varargs = any(
            parameter.kind == parameter.VAR_POSITIONAL
            for parameter in parameters
        )
        positional = [
            parameter
            for parameter in parameters
            if parameter.kind
            in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
        ]
        if accepts_varargs or len(positional) >= 2:
            return factory(mesh, pde)
        return factory(mesh)

    def compute_error(self) -> Tuple[float, float, float]:
        """Compute errors for velocity and pressure."""
        cell_centers = self.mesh.entity_barycenter("cell")
        self.uI = self.pde.velocity(cell_centers)[:, 0]
        self.vI = self.pde.velocity(cell_centers)[:, 1]
        self.pI = self.pde.pressure(cell_centers)
        uerror = bm.sqrt(bm.sum(self.cm * (self.uh - self.uI) ** 2))
        verror = bm.sqrt(bm.sum(self.cm * (self.vh - self.vI) ** 2))
        perror = bm.sqrt(bm.sum(self.cm * (self.ph - self.pI) ** 2))
        return uerror, verror, perror

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
        """Plot SIMPLE residual decay for manufactured-case examples."""
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
