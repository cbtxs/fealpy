"""Control parameter containers for collocated FVM solvers."""

from __future__ import annotations

from dataclasses import dataclass

from fealpy.backend import backend_manager as bm


def positive_scalar(value, name: str) -> float:
    """Return ``value`` as a positive Python scalar."""
    if callable(value):
        value = value()
    try:
        scalar = float(value)
    except TypeError:
        scalar = float(bm.to_numpy(value))
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return scalar


def nonnegative_scalar(value, name: str) -> float:
    """Return ``value`` as a non-negative Python scalar."""
    if callable(value):
        value = value()
    try:
        scalar = float(value)
    except TypeError:
        scalar = float(bm.to_numpy(value))
    if scalar < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return scalar


@dataclass(frozen=True)
class SimpleSolverControls:
    """Discretization controls for ``CollocatedSimpleSolver``."""

    space_degree: int = 0
    pressure_gradient_method: str = "layered_lsq"
    velocity_gradient_method: str = "layered_lsq"
    rhie_chow_pressure_gradient_method: str = "layered_lsq"
    face_interpolation_method: str = "average"
    momentum_face_interpolation: str | None = None
    pressure_response_interpolation: str | None = None
    rhie_chow_velocity_interpolation: str | None = None
    momentum_equation_relaxation: float = 0.7
    momentum_nonorthogonal_max_iter: int = 10
    momentum_nonorthogonal_tol: float = 1.0e-4
    pressure_nonorthogonal_max_iter: int = 10
    pressure_nonorthogonal_tol: float = 1.0e-5

    def __post_init__(self):
        self._validate_face_interpolation("face_interpolation_method", self.face_interpolation_method)
        for name in (
            "momentum_face_interpolation",
            "pressure_response_interpolation",
            "rhie_chow_velocity_interpolation",
        ):
            value = getattr(self, name)
            if value is not None:
                self._validate_face_interpolation(name, value)

        if self.momentum_nonorthogonal_max_iter < 0:
            raise ValueError("momentum_nonorthogonal_max_iter must be non-negative.")
        if self.pressure_nonorthogonal_max_iter < 0:
            raise ValueError("pressure_nonorthogonal_max_iter must be non-negative.")
        if not 0.0 < self.momentum_equation_relaxation <= 1.0:
            raise ValueError("momentum_equation_relaxation must be in (0, 1].")
        if self.momentum_nonorthogonal_tol <= 0.0:
            raise ValueError("momentum_nonorthogonal_tol must be positive.")
        if self.pressure_nonorthogonal_tol <= 0.0:
            raise ValueError("pressure_nonorthogonal_tol must be positive.")

    @staticmethod
    def _validate_face_interpolation(name: str, method: str) -> None:
        if method not in {"average", "linear"}:
            raise ValueError(f"{name} must be 'average' or 'linear'.")

    def face_interpolation(self, key: str) -> str:
        """Return a face interpolation choice with shared fallback."""
        value = getattr(self, key)
        return self.face_interpolation_method if value is None else value


@dataclass(frozen=True)
class PisoSolverControls:
    """Discretization and iteration controls for ``CollocatedPisoSolver``.

    The momentum and pressure non-orthogonal counters control explicit
    non-orthogonal correction solves.  A value of zero disables the explicit
    cross correction while still solving the base equation once.
    """

    space_degree: int = 0
    duration: tuple[float, float] = (0.0, 1.0)
    nt: int = 20
    n_correctors: int = 2
    snapshot_interval: int = 1
    snapshot_start_step: int = 1
    pressure_gradient_method: str = "layered_lsq"
    velocity_gradient_method: str = "layered_lsq"
    rhie_chow_pressure_gradient_method: str = "layered_lsq"
    face_interpolation_method: str = "average"
    rhie_chow_velocity_interpolation: str | None = None
    use_transient_flux_correction: bool = True
    momentum_nonorthogonal_max_iter: int = 1
    momentum_nonorthogonal_tol: float = 1.0e-5
    pressure_nonorthogonal_max_iter: int = 3
    pressure_nonorthogonal_tol: float = 1.0e-5
    diagnostics_enabled: bool = False

    def __post_init__(self):
        self.validate_time_controls(self.duration, self.nt)
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
        self.validate_nonorthogonal_controls(
            self.momentum_nonorthogonal_max_iter,
            self.momentum_nonorthogonal_tol,
            self.pressure_nonorthogonal_max_iter,
            self.pressure_nonorthogonal_tol,
        )

    @property
    def tau(self) -> float:
        """Return the uniform time-step size."""
        return (self.duration[1] - self.duration[0]) / self.nt

    @staticmethod
    def validate_time_controls(duration, nt: int) -> None:
        if nt < 1:
            raise ValueError("nt must be positive.")
        if len(duration) != 2 or duration[1] <= duration[0]:
            raise ValueError("duration must be an increasing pair.")

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
