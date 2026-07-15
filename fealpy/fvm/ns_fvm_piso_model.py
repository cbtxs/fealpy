"""Manufactured-case adapter for the collocated PISO solver."""

from fealpy.typing import TensorLike
from fealpy.model import ComputationalModel
from fealpy.model import PDEModelManager

from .collocated_piso_solver import CollocatedPisoSolver
from .cell_average_error import cell_average_l2_error
from .engineering_boundary_conditions import (
    EngineeringBoundaryConditions,
    PDEBoundaryConditions,
)
from .solver_controls import PisoSolverControls, positive_scalar


class NSFVMPISOModel(ComputationalModel, CollocatedPisoSolver):
    """Finite-volume PISO model for PDE examples with exact solutions."""

    def __init__(self, options):
        self.options = options
        self._validate_options()
        ComputationalModel.__init__(
            self,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        pde_input = options["pde"]
        if isinstance(pde_input, int):
            pde = PDEModelManager("navier_stokes").get_example(pde_input)
        else:
            pde = pde_input
        self.pde = pde
        self.error_quadrature_order = int(options.get("error_quadrature_order", 4))

        rho_value = options.get("rho", None)
        if rho_value is None:
            rho_value = getattr(pde, "rho", 1.0)
        mu_value = options.get("mu", None)
        if mu_value is None:
            for name in ("mu", "viscosity", "nu"):
                if hasattr(pde, name):
                    mu_value = getattr(pde, name)
                    break
            else:
                mu_value = 1.0
        self.rho = positive_scalar(rho_value, "rho")
        self.mu = positive_scalar(mu_value, "mu")

        mesh_type = options.get("mesh_type", "uniform_quad")
        mesh_type = mesh_type or getattr(pde, "default_mesh_type", "uniform_quad")
        mesh_refine = int(options.get("mesh_refine", 0) or 0)
        if mesh_refine < 0:
            raise ValueError("mesh_refine must be non-negative.")
        if getattr(pde, "supports_geometric_refine", False):
            mesh = pde.init_mesh[mesh_type](mesh_refine=mesh_refine)
        else:
            mesh_options = {}
            if options.get("nx") is not None:
                mesh_options["nx"] = int(options["nx"])
            if options.get("ny") is not None:
                mesh_options["ny"] = int(options["ny"])
            if options.get("nz") is not None:
                mesh_options["nz"] = int(options["nz"])
            mesh = pde.init_mesh[mesh_type](**mesh_options)
            if mesh_refine > 0:
                if not hasattr(mesh, "uniform_refine"):
                    raise ValueError("mesh does not provide uniform_refine().")
                mesh.uniform_refine(mesh_refine)

        boundary_input = options.get("boundary_conditions")
        if boundary_input is None:
            boundary_input = PDEBoundaryConditions(
                mesh,
                dirichlet_velocity=pde.dirichlet_velocity,
            )
        elif callable(boundary_input):
            boundary_input = boundary_input(mesh, pde)

        if isinstance(boundary_input, EngineeringBoundaryConditions):
            self.engineering_bc = boundary_input
            boundary_conditions = boundary_input.to_pde_boundary()
        elif isinstance(boundary_input, PDEBoundaryConditions):
            self.engineering_bc = None
            boundary_conditions = boundary_input
        else:
            raise TypeError(
                "boundary_conditions must be PDEBoundaryConditions, "
                "EngineeringBoundaryConditions, or a factory(mesh, pde) "
                "returning one of these types."
            )
        controls = PisoSolverControls.from_mapping(options)
        CollocatedPisoSolver.__init__(
            self,
            mesh=mesh,
            diffusion_coef=self.mu,
            convection_coef=self.rho,
            source=pde.source,
            boundary_conditions=boundary_conditions,
            controls=controls,
            linear_solver=options.get("linear_solver"),
            linear_solver_config=options.get("linear_solver_config"),
            logger=self.logger,
        )

    def _validate_options(self) -> None:
        allowed = set(PisoSolverControls.option_names()) | {
            "pde",
            "mesh_type",
            "mesh_refine",
            "nx",
            "ny",
            "nz",
            "rho",
            "mu",
            "error_quadrature_order",
            "boundary_conditions",
            "linear_solver",
            "linear_solver_config",
            "pbar_log",
            "log_level",
        }
        unsupported = set(self.options).difference(allowed)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(f"unsupported NSFVMPISOModel options: {names}")

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
            f"  Time steps: {self.controls.nt}\n"
            f"  PISO correctors: {self.controls.n_correctors}\n"
            f"  Momentum nonorthogonal corrections: "
            f"{self.controls.momentum_nonorthogonal_max_iter}\n"
            f"  Pressure nonorthogonal corrections: "
            f"{self.controls.pressure_nonorthogonal_max_iter}\n"
        )

    def initial_solution(self) -> tuple[TensorLike, TensorLike, TensorLike]:
        """Return the initial velocity, face velocity, and pressure fields."""
        t0 = self.controls.duration[0]
        U0 = self.pde.velocity_0(self.cell_center, t0)
        Uf0 = self.pde.velocity_0(self.face_center, t0)
        p0 = self.pde.pressure_0(self.cell_center, t0)
        return U0, Uf0, p0

    def compute_error(self) -> tuple[float, ...]:
        """Compute final-time errors against exact control-volume averages."""
        t = self.controls.duration[1]

        def exact_velocity(points):
            return self.pde.velocity(points, t)

        def exact_pressure(points):
            return self.pde.pressure(points, t)

        velocity_error, self.exact_velocity = cell_average_l2_error(
            self.mesh,
            exact_velocity,
            self.velocity,
            q=self.error_quadrature_order,
        )
        pressure_error, self.exact_pressure = cell_average_l2_error(
            self.mesh,
            exact_pressure,
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
