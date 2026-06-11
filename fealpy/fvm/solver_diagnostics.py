"""Reusable scalar diagnostics for FVM solver loops."""

from fealpy.backend import backend_manager as bm


def linf_norm(value):
    """Return the infinity norm as a Python float."""
    return float(bm.to_numpy(bm.max(bm.abs(value))))


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


def simple_iteration_log_message(
    *,
    simple_iteration: int,
    nonorthogonal_iterations: int,
    pressure_criterion: float,
    pressure_relax: float,
    mass_residual: float,
    pressure_correction: float,
) -> str:
    """Format one SIMPLE iteration diagnostic line."""
    return format_pressure_correction_log(
        iteration=simple_iteration,
        nonorthogonal_iterations=nonorthogonal_iterations,
        pressure_criterion=pressure_criterion,
        pressure_relax=pressure_relax,
        mass_residual=mass_residual,
        pressure_correction=pressure_correction,
        label="SIMPLE",
    )


def log_simple_iteration(logger, iteration, residual):
    """Log one SIMPLE pressure-correction diagnostic record."""
    pressure_criterion = residual.get("pressure_criterion", residual["pressure_update"])
    logger.info(
        simple_iteration_log_message(
            simple_iteration=iteration,
            nonorthogonal_iterations=residual["nonorthogonal_iterations"],
            pressure_criterion=pressure_criterion,
            pressure_relax=residual["pressure_relax"],
            mass_residual=residual["mass"],
            pressure_correction=residual["pressure_correction"],
        )
    )
    action = residual["pressure_relax_action"]
    if action in {"reduce", "increase"}:
        logger.info(
            f"[Iter {iteration}] pressure relaxation {action}d to "
            f"{residual['pressure_relax']:.2e}"
        )


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


def record_piso_corrector_diagnostics(
    storage,
    callback,
    diagnostics,
    *,
    step,
    time,
    correction,
    n_correctors,
    splitting_linf,
    current_velocity,
    next_velocity,
    current_pressure,
    next_pressure,
    a_p,
    phi,
    boundary_faces,
    boundary_velocity,
    face_response_coefficient,
    rhie_chow_face_velocity,
    face_flux,
):
    """Store one PISO corrector diagnostics row and call the optional callback."""
    target_face_velocity = rhie_chow_face_velocity(
        next_velocity,
        a_p,
        next_pressure,
        phi,
        boundary_faces=boundary_faces,
        boundary_velocity=boundary_velocity,
        face_response_coefficient=face_response_coefficient,
    )
    target_flux_error = face_flux(target_face_velocity) - phi
    row = attach_corrector_diagnostics(
        diagnostics,
        step=step,
        time=time,
        corrector=correction,
        n_correctors=n_correctors,
        splitting_linf=splitting_linf,
        current_velocity=current_velocity,
        next_velocity=next_velocity,
        current_pressure=current_pressure,
        next_pressure=next_pressure,
        target_flux_error=target_flux_error,
    )
    storage.append(row)
    if callback is not None:
        callback(**row)
