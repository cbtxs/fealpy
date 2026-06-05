"""Manufactured-case adapter for the collocated PISO solver."""

from inspect import signature
from typing import Tuple

from fealpy.typing import TensorLike
from fealpy.backend import backend_manager as bm
from fealpy.model import ComputationalModel

from .collocated_piso_solver import CollocatedPisoSolver
from .engineering_boundary_conditions import BoundaryConditionData
from .navier_stokes_model_adapter import NavierStokesModelAdapter
from .solver_controls import PisoSolverControls


class NSFVMPISOModel(ComputationalModel, NavierStokesModelAdapter, CollocatedPisoSolver):
    """Finite-volume PISO model for PDE examples with exact solutions."""

    def __init__(self, options):
        self.options = options
        ComputationalModel.__init__(
            self,
            pbar_log=options.get("pbar_log", False),
            log_level=options.get("log_level", "WARNING"),
        )
        pde = self._resolve_navier_stokes_pde(options["pde"])
        self.pde = pde
        self._init_momentum_coefficients(options)
        mesh_type = self._piso_mesh_type(options)
        mesh = self._init_mesh(options, mesh_type)
        boundary_input = self._init_piso_boundary_conditions(options, mesh, pde)
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
        CollocatedPisoSolver.__init__(
            self,
            mesh=mesh,
            diffusion_coef=self.mu,
            convection_coef=self.rho,
            source=pde.source,
            boundary_conditions=boundary_conditions,
            controls=self._piso_controls_from_options(options, mesh_type),
            linear_solver=options.get("linear_solver"),
            linear_solver_config=options.get("linear_solver_config"),
            logger=self.logger,
        )

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

    def _piso_mesh_type(self, options):
        """Return the normalized mesh type for the PISO manufactured adapter."""
        return self._normalized_mesh_type(options.get("mesh_type", "uniform_quad"))

    def _init_mesh(self, options, mesh_type):
        """Build the PDE mesh used by the PISO manufactured adapter."""
        mesh_options = {"mesh_type": mesh_type}
        if options.get("nx") is not None:
            mesh_options["nx"] = options.get("nx")
        if options.get("ny") is not None:
            mesh_options["ny"] = options.get("ny")
        if options.get("mesh_refine") is not None:
            mesh_options["mesh_refine"] = options.get("mesh_refine")
        return self._init_navier_stokes_mesh(
            mesh_options,
            default_mesh_type="uniform_quad",
            normalize_mesh_type=True,
        )

    def _init_piso_boundary_conditions(self, options, mesh, pde):
        """Translate model or engineering boundary input to solver boundary data."""
        boundary_conditions = options.get("boundary_conditions")
        if boundary_conditions is None:
            velocity_dirichlet = getattr(pde, "velocity_dirichlet", None)
            if velocity_dirichlet is None:
                velocity_dirichlet = pde.dirichlet_velocity
            return BoundaryConditionData(velocity_dirichlet)

        if callable(boundary_conditions) and not hasattr(boundary_conditions, "dirichlet_threshold"):
            return self._call_boundary_condition_factory(boundary_conditions, mesh, pde)
        return boundary_conditions

    @staticmethod
    def _piso_controls_from_options(options, mesh_type):
        """Translate historical manufactured-model options to PISO controls."""
        default_pressure_nonorthogonal_iter = 1 if mesh_type == "uniform_quad" else 3
        legacy_momentum_route = options.get("momentum_explicit_correction", "openfoam")
        if legacy_momentum_route not in (None, "openfoam"):
            raise ValueError(
                "momentum_explicit_correction='current' has been removed from "
                "PISO; the boundary-corrected explicit momentum source is now "
                "the only supported route."
            )
        if "transient_flux_correction_limiter" in options:
            raise ValueError(
                "transient_flux_correction_limiter has been removed from PISO; "
                "the limited ddtCorr route is now always used when "
                "use_transient_flux_correction is enabled."
            )

        def option_int(name: str, default: int) -> int:
            value = options.get(name, default)
            return default if value is None else int(value)

        return PisoSolverControls(
            space_degree=options.get("space_degree", 0),
            duration=tuple(options.get("duration", (0, 1))),
            nt=option_int("nt", 20),
            n_correctors=option_int("n_correctors", 2),
            snapshot_interval=option_int("snapshot_interval", 1),
            snapshot_start_step=option_int("snapshot_start_step", 1),
            pressure_gradient_method=options.get("pressure_gradient_method", "extended_lsq"),
            velocity_gradient_method=options.get("velocity_gradient_method", "extended_lsq"),
            rhie_chow_pressure_gradient_method=options.get("rhie_chow_pressure_gradient_method", "extended_lsq"),
            face_interpolation_method=options.get("face_interpolation_method", "average"),
            rhie_chow_velocity_interpolation=options.get("rhie_chow_velocity_interpolation"),
            use_transient_flux_correction=bool(options.get("use_transient_flux_correction", True)),
            momentum_nonorthogonal_max_iter=option_int("momentum_nonorthogonal_max_iter", 1),
            momentum_nonorthogonal_tol=float(options.get("momentum_nonorthogonal_tol", 1.0e-5)),
            pressure_nonorthogonal_max_iter=option_int("pressure_nonorthogonal_max_iter", default_pressure_nonorthogonal_iter),
            pressure_nonorthogonal_tol=float(options.get("pressure_nonorthogonal_tol", 1.0e-5)),
            diagnostics_enabled=bool(options.get("diagnostics_enabled", False)),
        )

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

    def initial_solution(self) -> Tuple[TensorLike, TensorLike, TensorLike]:
        """Return the initial velocity, face velocity, and pressure fields."""
        t0 = self.controls.duration[0]
        U0 = self.pde.velocity_0(self.points, t0)
        Uf0 = self.pde.velocity_0(self.epoints, t0)
        p0 = self.pde.pressure_0(self.points, t0)
        return U0, Uf0, p0

    def compute_error(self) -> Tuple[float, float, float]:
        """Compute manufactured-solution errors at the final time."""
        t = self.controls.duration[1]
        self.uI = self.pde.velocity_u(self.points, t)
        self.vI = self.pde.velocity_v(self.points, t)
        self.pI = self.pde.pressure(self.points, t)
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
