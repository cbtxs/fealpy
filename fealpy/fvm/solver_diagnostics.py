"""Reusable scalar diagnostics for FVM solver loops."""

from fealpy.backend import backend_manager as bm


def linf_norm(value):
    """Return the infinity norm as a Python float."""
    return float(bm.to_numpy(bm.max(bm.abs(value))))


def pressure_correction_diagnostics(
    *,
    free_divergence,
    corrected_divergence,
    free_flux,
    corrected_flux,
    pressure_flux,
    pressure_flux_parts,
    transient_flux,
    pressure_nonorthogonal_iterations,
):
    """Return scalar diagnostics for one pressure-correction solve."""
    orthogonal_flux = pressure_flux_parts["orthogonal_flux"]
    cross_flux = pressure_flux_parts["cross_flux"]
    boundary_pressure_flux = pressure_flux_parts["boundary_pressure_flux"]
    return {
        "pressure_free_divergence_linf": linf_norm(free_divergence),
        "pressure_corrected_divergence_linf": linf_norm(corrected_divergence),
        "pressure_free_flux_linf": linf_norm(free_flux),
        "pressure_corrected_flux_linf": linf_norm(corrected_flux),
        "pressure_flux_linf": linf_norm(pressure_flux),
        "pressure_orthogonal_flux_linf": linf_norm(orthogonal_flux),
        "pressure_cross_flux_linf": linf_norm(cross_flux),
        "pressure_boundary_flux_linf": linf_norm(boundary_pressure_flux),
        "transient_flux_correction_linf": linf_norm(transient_flux),
        "pressure_nonorthogonal_iterations": int(pressure_nonorthogonal_iterations),
    }


def attach_corrector_diagnostics(
    diagnostics,
    *,
    step,
    time,
    corrector,
    n_correctors,
    splitting_linf,
    current_velocity,
    next_velocity,
    current_pressure,
    next_pressure,
    target_flux_error,
):
    """Attach outer-loop update diagnostics to a pressure-corrector row."""
    row = dict(diagnostics)
    row.update(
        {
            "step": int(step),
            "time": float(time),
            "corrector": int(corrector),
            "n_correctors": int(n_correctors),
            "operator_splitting_compensation_linf": float(splitting_linf),
            "velocity_update_linf": linf_norm(next_velocity - current_velocity),
            "pressure_update_linf": linf_norm(next_pressure - current_pressure),
            "rhie_chow_flux_error_linf": linf_norm(target_flux_error),
        }
    )
    return row
