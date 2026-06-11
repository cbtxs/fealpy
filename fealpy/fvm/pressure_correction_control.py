"""Stopping criterion for pressure-correction iterations."""


def pressure_correction_converged(
    residual,
    tol_mass,
    tol_pressure_update,
    tol_pressure_correction=None,
):
    """Return whether pressure-correction residuals satisfy stopping criteria."""
    if residual["mass"] >= tol_mass:
        return False
    if tol_pressure_correction is not None:
        return residual["pressure_correction"] < tol_pressure_correction
    return residual["pressure_update"] < tol_pressure_update
