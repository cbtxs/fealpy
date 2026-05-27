"""Iteration-control helpers for pressure-correction FVM solvers."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PressureRelaxationConfig:
    """Policy parameters for adaptive pressure-correction relaxation."""

    min_value: float = 1.0e-4
    max_value: Optional[float] = None
    growth_factor: float = 1.05
    severe_growth_factor: Optional[float] = 2.0
    patience: int = 3
    reduction_factor: float = 0.5
    increase_patience: int = 3
    increase_factor: float = 1.25
    small_update: Optional[float] = 1.0e-3
    cooldown_steps: int = 3
    mass_growth_factor: float = 1.05
    residual_key: str = "pressure_correction"
    update_key: str = "pressure_update"
    mass_key: str = "mass"

    def __post_init__(self):
        if self.min_value <= 0.0:
            raise ValueError("min_value must be positive.")
        if self.max_value is not None and self.max_value <= 0.0:
            raise ValueError("max_value must be positive.")
        if self.growth_factor <= 1.0:
            raise ValueError("growth_factor must be greater than 1.")
        if (
            self.severe_growth_factor is not None
            and self.severe_growth_factor <= 1.0
        ):
            raise ValueError("severe_growth_factor must be greater than 1.")
        if self.patience < 1:
            raise ValueError("patience must be positive.")
        if not 0.0 < self.reduction_factor < 1.0:
            raise ValueError("reduction_factor must be in (0, 1).")
        if self.increase_patience < 1:
            raise ValueError("increase_patience must be positive.")
        if self.increase_factor <= 1.0:
            raise ValueError("increase_factor must be greater than 1.")
        if self.small_update is not None and self.small_update <= 0.0:
            raise ValueError("small_update must be positive.")
        if self.cooldown_steps < 0:
            raise ValueError("cooldown_steps must be non-negative.")


class PressureRelaxationController:
    """Adaptive scalar relaxation controller for pressure corrections.

    The controller only changes the nonlinear iteration step length.  It does
    not alter the pressure-correction equation, Rhie-Chow interpolation, or
    velocity correction formulas.
    """

    def __init__(
        self,
        initial: float,
        config: Optional[PressureRelaxationConfig] = None,
    ):
        if initial <= 0.0:
            raise ValueError("initial pressure relaxation must be positive.")
        self.config = config or PressureRelaxationConfig(max_value=initial)
        self.value = initial
        self.max_value = (
            initial if self.config.max_value is None else self.config.max_value
        )
        self.deterioration_count = 0
        self.small_update_count = 0
        self.cooldown = 0

    def update(self, residuals):
        """Update relaxation from residual history and return ``(value, action)``."""
        if len(residuals) < 2:
            self._reset_deterioration()
            return self.value, "keep"

        current = float(residuals[-1][self.config.residual_key])
        previous = float(residuals[-2][self.config.residual_key])

        if (
            self.config.severe_growth_factor is not None
            and current > self.config.severe_growth_factor * previous
        ):
            return self._reduce()

        if self.cooldown > 0:
            self.deterioration_count = 0
            self.cooldown -= 1
            return self.value, "cooldown"

        if current > self.config.growth_factor * previous:
            self.deterioration_count += 1
        else:
            self.deterioration_count = 0

        if self.deterioration_count >= self.config.patience:
            return self._reduce()

        return self._maybe_increase(residuals, current, previous)

    def _reduce(self):
        new_value = max(self.config.reduction_factor * self.value, self.config.min_value)
        if new_value >= self.value:
            self._reset_deterioration()
            return self.value, "keep"
        self.value = new_value
        self.deterioration_count = 0
        self.small_update_count = 0
        self.cooldown = self.config.cooldown_steps
        return self.value, "reduce"

    def _maybe_increase(self, residuals, current, previous):
        can_increase = (
            self.config.small_update is not None
            and self.value < self.max_value
            and self.config.update_key in residuals[-1]
            and residuals[-1][self.config.update_key] < self.config.small_update
            and current <= self.config.growth_factor * previous
        )
        if can_increase and self._has_mass_history(residuals):
            mass = float(residuals[-1][self.config.mass_key])
            previous_mass = float(residuals[-2][self.config.mass_key])
            can_increase = mass <= self.config.mass_growth_factor * previous_mass

        if can_increase:
            self.small_update_count += 1
        else:
            self.small_update_count = 0

        if self.small_update_count < self.config.increase_patience:
            return self.value, "keep"

        new_value = min(self.config.increase_factor * self.value, self.max_value)
        if new_value <= self.value:
            self.small_update_count = 0
            return self.value, "keep"
        self.value = new_value
        self.deterioration_count = 0
        self.small_update_count = 0
        self.cooldown = self.config.cooldown_steps
        return self.value, "increase"

    def _has_mass_history(self, residuals):
        return (
            self.config.mass_key in residuals[-1]
            and self.config.mass_key in residuals[-2]
        )

    def _reset_deterioration(self):
        self.deterioration_count = 0
        self.small_update_count = 0
        self.cooldown = 0


def pressure_correction_converged(residual, tol_mass, tol_pressure_update):
    """Return whether pressure-correction residuals satisfy stopping criteria."""
    return (
        residual["mass"] < tol_mass
        and residual["pressure_update"] < tol_pressure_update
    )


def format_pressure_correction_log(
    *,
    iteration: int,
    nonorthogonal_iterations: int,
    pressure_criterion: float,
    pressure_relax: float,
    mass_residual: float,
    pressure_correction: float,
    label: str = "SIMPLE",
) -> str:
    """Format one pressure-correction iteration diagnostic line."""
    return (
        f"[{label} {iteration}] "
        f"nonorthogonal iterations: {nonorthogonal_iterations}, "
        f"pressure criterion: {pressure_criterion:.2e}, "
        f"pressure relax: {pressure_relax:.2e}, "
        f"mass residual: {mass_residual:.2e}, "
        f"pressure correction L2: {pressure_correction:.2e}"
    )
