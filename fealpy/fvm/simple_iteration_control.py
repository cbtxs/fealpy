"""Outer-iteration coordination for the collocated SIMPLE solver."""

from .pressure_correction_control import (
    PressureRelaxationConfig,
    PressureRelaxationController,
)
from .simple_residual import (
    cell_l2_norm,
    collocated_mass_residual,
    relative_l2_update,
)
from .solver_diagnostics import log_simple_iteration, simple_iteration_log_message


class SimpleIterationControl:
    """Coordinate SIMPLE residual records, pressure relaxation, and logging."""

    def __init__(self, mesh, logger):
        self.mesh = mesh
        self.logger = logger

    def residual(
        self,
        uf,
        p_corr,
        p_update,
        p,
        pressure_relax,
        action,
        *,
        nonorthogonal_iterations,
        momentum_nonorthogonal_iterations,
    ):
        """Build one SIMPLE residual record."""
        return {
            "mass": collocated_mass_residual(self.mesh, uf),
            "pressure_correction": cell_l2_norm(self.mesh, p_corr),
            "pressure_update": relative_l2_update(self.mesh, p_update, p),
            "pressure_relax": pressure_relax,
            "pressure_relax_reduced": action == "reduce",
            "pressure_relax_action": action,
            "nonorthogonal_iterations": nonorthogonal_iterations,
            "momentum_nonorthogonal_iterations": momentum_nonorthogonal_iterations,
        }

    @staticmethod
    def iteration_log_message(
        *,
        simple_iteration: int,
        nonorthogonal_iterations: int,
        pressure_criterion: float,
        pressure_relax: float,
        mass_residual: float,
        pressure_correction: float,
    ) -> str:
        """Format one SIMPLE iteration diagnostic line."""
        return simple_iteration_log_message(
            simple_iteration=simple_iteration,
            nonorthogonal_iterations=nonorthogonal_iterations,
            pressure_criterion=pressure_criterion,
            pressure_relax=pressure_relax,
            mass_residual=mass_residual,
            pressure_correction=pressure_correction,
        )

    def log_iteration(self, iteration, residual):
        """Log one SIMPLE pressure-correction diagnostic record."""
        log_simple_iteration(self.logger, iteration, residual)

    @staticmethod
    def tolerances(tol, tol_mass, tol_pressure_update):
        """Return mass and pressure-update stopping tolerances."""
        return (
            tol if tol_mass is None else tol_mass,
            10.0 * tol if tol_pressure_update is None else tol_pressure_update,
        )

    @staticmethod
    def pressure_relaxation_controller(relax, adaptive_pressure_relax, relaxation_config):
        """Create the optional pressure-relaxation controller."""
        if relax <= 0:
            raise ValueError("relax must be positive.")
        if not adaptive_pressure_relax:
            return None
        if isinstance(relaxation_config, dict):
            relaxation_config = PressureRelaxationConfig(**relaxation_config)
        return PressureRelaxationController(initial=relax, config=relaxation_config)

    def pressure_update_step(
        self,
        residuals,
        uf,
        p_corr,
        p,
        pressure_relax,
        relaxation,
        *,
        nonorthogonal_iterations,
        momentum_nonorthogonal_iterations,
    ):
        """Apply pressure relaxation control and build the residual record."""
        p_update = pressure_relax * p_corr
        residual = self.residual(
            uf,
            p_corr,
            p_update,
            p,
            pressure_relax,
            "keep",
            nonorthogonal_iterations=nonorthogonal_iterations,
            momentum_nonorthogonal_iterations=momentum_nonorthogonal_iterations,
        )
        residuals.append(residual)

        relax_action = "keep"
        if relaxation is not None:
            pressure_relax, relax_action = relaxation.update(residuals)
            if relax_action in {"reduce", "increase"}:
                p_update = pressure_relax * p_corr
                residual["pressure_update"] = relative_l2_update(self.mesh, p_update, p)

        residual["pressure_relax"] = pressure_relax
        residual["pressure_relax_reduced"] = relax_action == "reduce"
        residual["pressure_relax_action"] = relax_action
        return pressure_relax, p_update, residual
