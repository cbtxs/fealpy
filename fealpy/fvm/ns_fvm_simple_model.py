"""Manufactured-case adapter for the collocated SIMPLE solver."""

from inspect import signature
from typing import Tuple

from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel
from fealpy.model import PDEModelManager

from .collocated_simple_solver import CollocatedSimpleSolver
from .cell_average_error import cell_average_l2_error
from .engineering_boundary_conditions import BoundaryConditionData
from .solver_controls import SimpleSolverControls


def _call_boundary_condition_factory(factory, mesh, pde):
    try:
        parameters = list(signature(factory).parameters.values())
    except (TypeError, ValueError):
        return factory(mesh, pde)

    accepts_varargs = any(
        parameter.kind == parameter.VAR_POSITIONAL for parameter in parameters
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


class NSFVMSimpleModel(ComputationalModel, CollocatedSimpleSolver):
    """Finite Volume SIMPLE model for PDE examples with exact solutions."""

    def __init__(self, options):
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
        self.rho = self._as_positive_scalar(rho_value, "rho")
        self.mu = self._as_positive_scalar(mu_value, "mu")

        mesh_type = options.get("mesh_type") or getattr(pde, "default_mesh_type", "uniform_tri")
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
            boundary_input = BoundaryConditionData(pde.dirichlet_velocity)
        elif callable(boundary_input) and not hasattr(boundary_input, "dirichlet_threshold"):
            boundary_input = _call_boundary_condition_factory(boundary_input, mesh, pde)
        self.engineering_bc = (
            boundary_input
            if options.get("boundary_conditions") is not None
            and hasattr(boundary_input, "to_pde_boundary")
            else None
        )
        if isinstance(boundary_input, BoundaryConditionData):
            boundary_conditions = boundary_input.to_pde_boundary(mesh)
        elif hasattr(boundary_input, "to_pde_boundary"):
            boundary_conditions = boundary_input.to_pde_boundary()
        else:
            boundary_conditions = boundary_input
        controls = SimpleSolverControls(
            space_degree=options.get("space_degree", 0),
            pressure_gradient_method=options.get("pressure_gradient_method", "extended_lsq"),
            velocity_gradient_method=options.get("velocity_gradient_method", "extended_lsq"),
            rhie_chow_pressure_gradient_method=options.get("rhie_chow_pressure_gradient_method", "extended_lsq"),
            face_interpolation_method=options.get("face_interpolation_method", "average"),
            momentum_face_interpolation=options.get("momentum_face_interpolation"),
            pressure_response_interpolation=options.get("pressure_response_interpolation"),
            rhie_chow_velocity_interpolation=options.get("rhie_chow_velocity_interpolation"),
            momentum_equation_relaxation=options.get("momentum_equation_relaxation", 0.7),
            momentum_nonorthogonal_max_iter=options.get("momentum_nonorthogonal_max_iter", 10),
            momentum_nonorthogonal_tol=options.get("momentum_nonorthogonal_tol", 1.0e-4),
            pressure_nonorthogonal_max_iter=options.get("pressure_nonorthogonal_max_iter", 10),
            pressure_nonorthogonal_tol=options.get("pressure_nonorthogonal_tol", 1.0e-5),
        )
        CollocatedSimpleSolver.__init__(
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
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}:\n"
            f"  Mesh shape: {self.mesh.number_of_cells()} cells\n"
            f"  PDE type: {type(self.pde).__name__}\n"
        )

    def compute_error(self) -> Tuple[float, ...]:
        """Compute errors against exact control-volume averages."""
        velocity = getattr(self, "velocity", bm.stack([self.uh, self.vh], axis=-1))
        velocity_error, velocity_average = cell_average_l2_error(
            self.mesh,
            self.pde.velocity,
            velocity,
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
