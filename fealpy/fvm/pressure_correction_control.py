"""Stopping criterion for pressure-correction iterations."""


def pressure_correction_converged(
    residual,
    tol_mass,
    tol_pressure_correction,
):
    """Return whether pressure-correction residuals satisfy stopping criteria."""
    if residual["mass"] >= tol_mass:
        return False
    return residual["pressure_correction"] < tol_pressure_correction
