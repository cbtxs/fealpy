"""Input data containers for the collocated PISO solver."""

from __future__ import annotations

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry


@dataclass(frozen=True)
class PisoSolverControls:
    """Discretization and iteration controls for ``CollocatedPisoSolver``."""

    space_degree: int = 0
    duration: tuple[float, float] = (0.0, 1.0)
    nt: int = 20
    n_correctors: int = 2
    snapshot_interval: int = 1
    snapshot_start_step: int = 1
    pressure_gradient_method: str = "extended_lsq"
    velocity_gradient_method: str = "extended_lsq"
    rhie_chow_pressure_gradient_method: str = "extended_lsq"
    face_interpolation_method: str = "average"
    rhie_chow_velocity_interpolation: str | None = None
    use_transient_flux_correction: bool = True
    transient_flux_correction_limiter: str = "none"
    momentum_explicit_correction: str = "current"
    momentum_nonorthogonal_max_iter: int = 10
    momentum_nonorthogonal_tol: float = 1.0e-5
    pressure_nonorthogonal_max_iter: int = 10
    pressure_nonorthogonal_tol: float = 1.0e-5
    diagnostics_enabled: bool = False

    def __post_init__(self):
        self.validate_piso_controls(self.n_correctors)
        self.validate_snapshot_controls(self.snapshot_interval, self.snapshot_start_step)
        self.validate_face_interpolation_method(self.face_interpolation_method)
        if self.rhie_chow_velocity_interpolation is not None:
            self.validate_face_interpolation_method(self.rhie_chow_velocity_interpolation)
            if self.rhie_chow_velocity_interpolation != self.face_interpolation_method:
                raise ValueError(
                    "rhie_chow_velocity_interpolation must match "
                    "face_interpolation_method."
                )
        self.validate_transient_flux_correction_limiter(self.transient_flux_correction_limiter)
        self.validate_momentum_explicit_correction(self.momentum_explicit_correction)
        self.validate_nonorthogonal_controls(
            self.momentum_nonorthogonal_max_iter,
            self.momentum_nonorthogonal_tol,
            self.pressure_nonorthogonal_max_iter,
            self.pressure_nonorthogonal_tol,
        )

    @classmethod
    def from_options(cls, options, *, mesh_type: str = "uniform_quad"):
        """Build controls from the historical PISO model option dictionary."""
        default_nonorthogonal_iter = 0 if mesh_type == "uniform_quad" else 10

        def option_int(name: str, default: int) -> int:
            value = options.get(name, default)
            return default if value is None else int(value)

        return cls(
            space_degree=options.get("space_degree", 0),
            duration=tuple(options.get("duration", (0, 1))),
            nt=option_int("nt", 20),
            n_correctors=option_int("n_correctors", 2),
            snapshot_interval=option_int("snapshot_interval", 1),
            snapshot_start_step=option_int("snapshot_start_step", 1),
            pressure_gradient_method=options.get(
                "pressure_gradient_method", "extended_lsq"
            ),
            velocity_gradient_method=options.get(
                "velocity_gradient_method", "extended_lsq"
            ),
            rhie_chow_pressure_gradient_method=options.get(
                "rhie_chow_pressure_gradient_method", "extended_lsq"
            ),
            face_interpolation_method=options.get(
                "face_interpolation_method", "average"
            ),
            rhie_chow_velocity_interpolation=options.get(
                "rhie_chow_velocity_interpolation"
            ),
            use_transient_flux_correction=bool(
                options.get("use_transient_flux_correction", True)
            ),
            transient_flux_correction_limiter=options.get(
                "transient_flux_correction_limiter", "none"
            ),
            momentum_explicit_correction=options.get(
                "momentum_explicit_correction", "current"
            ),
            momentum_nonorthogonal_max_iter=option_int(
                "momentum_nonorthogonal_max_iter",
                default_nonorthogonal_iter,
            ),
            momentum_nonorthogonal_tol=float(
                options.get("momentum_nonorthogonal_tol", 1.0e-5)
            ),
            pressure_nonorthogonal_max_iter=option_int(
                "pressure_nonorthogonal_max_iter",
                default_nonorthogonal_iter,
            ),
            pressure_nonorthogonal_tol=float(
                options.get("pressure_nonorthogonal_tol", 1.0e-5)
            ),
            diagnostics_enabled=bool(options.get("diagnostics_enabled", False)),
        )

    @property
    def tau(self) -> float:
        """Return the uniform time-step size."""
        return (self.duration[1] - self.duration[0]) / self.nt

    @staticmethod
    def validate_piso_controls(n_correctors: int) -> None:
        if n_correctors < 1:
            raise ValueError("n_correctors must be positive.")

    @staticmethod
    def validate_snapshot_controls(interval: int, start_step: int) -> None:
        if interval < 1:
            raise ValueError("snapshot_interval must be positive.")
        if start_step < 1:
            raise ValueError("snapshot_start_step must be positive.")

    @staticmethod
    def validate_face_interpolation_method(method: str) -> str:
        if method not in {"average", "linear"}:
            raise ValueError("face_interpolation_method must be 'average' or 'linear'.")
        return method

    @staticmethod
    def validate_transient_flux_correction_limiter(limiter: str) -> str:
        if limiter not in {"openfoam", "none"}:
            raise ValueError(
                "transient_flux_correction_limiter must be 'openfoam' or 'none'."
            )
        return limiter

    @staticmethod
    def validate_momentum_explicit_correction(correction: str) -> str:
        if correction not in {"current", "openfoam"}:
            raise ValueError(
                "momentum_explicit_correction must be 'current' or 'openfoam'."
            )
        return correction

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


@dataclass(frozen=True)
class PisoBoundaryConditions:
    """Minimal boundary data adapter for PISO manufactured examples."""

    velocity_dirichlet: object
    velocity_dirichlet_threshold: object = None
    velocity_natural_threshold: object = None
    pressure_dirichlet: object = None
    pressure_dirichlet_threshold_value: object = None

    def conditions_for(self, variable: str, kind: str | None = None):
        """Return non-empty markers matching the available conditions."""
        if variable == "velocity" and kind in (None, "dirichlet"):
            return (self,) if self.velocity_dirichlet is not None else ()
        if variable == "velocity" and kind == "natural":
            return (self,) if self.velocity_natural_threshold is not None else ()
        if variable == "pressure" and kind in (None, "dirichlet"):
            return (self,) if self.has_pressure_dirichlet() else ()
        return ()

    def dirichlet_value(self, variable: str):
        """Return the Dirichlet value callable for one variable."""
        if variable == "velocity" and self.velocity_dirichlet is not None:
            return self.velocity_dirichlet
        if variable == "pressure" and self.has_pressure_dirichlet():
            return self.pressure_dirichlet_value()
        raise ValueError(f"{variable!r} has no Dirichlet boundary condition.")

    def dirichlet_threshold(self, variable: str):
        """Return the Dirichlet threshold for one variable."""
        if variable == "velocity":
            return self.velocity_dirichlet_threshold
        if variable == "pressure":
            return self.pressure_dirichlet_threshold()
        raise ValueError(f"unsupported boundary variable: {variable!r}.")

    def natural_threshold(self, variable: str):
        """Return the natural threshold for one variable."""
        if variable != "velocity":
            raise ValueError("only velocity natural boundary is supported.")
        return self.velocity_natural_threshold

    def boundary_face_velocity(self, variable: str = "velocity", *, mesh):
        """Return selected boundary faces and prescribed velocities."""
        if variable != "velocity":
            raise ValueError("boundary_face_velocity only supports 'velocity'.")
        geometry = FVMGeometry(mesh)
        boundary_faces = bm.nonzero(geometry.is_boundary)[0]
        points = geometry.face_center[boundary_faces]
        threshold = self.velocity_dirichlet_threshold
        if threshold is None:
            flag = bm.ones(boundary_faces.shape[0], dtype=bm.bool)
        else:
            flag = threshold(points)
        return boundary_faces[flag], self.velocity_dirichlet(points)[flag]

    def has_pressure_dirichlet(self) -> bool:
        """Return whether pressure Dirichlet data are available."""
        return self.pressure_dirichlet is not None and self.pressure_dirichlet_threshold_value is not None

    def pressure_dirichlet_threshold(self):
        """Return pressure Dirichlet threshold."""
        return self.pressure_dirichlet_threshold_value

    def pressure_dirichlet_value(self):
        """Return pressure Dirichlet values as a callable."""
        value = self.pressure_dirichlet
        if callable(value):
            return value

        def constant(points):
            return bm.broadcast_to(bm.array(value, dtype=points.dtype), (points.shape[0],))

        return constant
