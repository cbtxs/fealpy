"""Reusable scalar diagnostics for FVM solver loops."""

from fealpy.backend import backend_manager as bm


def normalized_equation_residual(
    lhs,
    rhs,
    eps=1.0e-30,
    *,
    normalization_lhs=None,
    normalization_rhs=None,
    norm_weights=None,
    scale_mode="sum",
):
    """Return norms of the complete discrete balance ``lhs - rhs``.

    The residual vector always uses ``lhs - rhs``.  By default its relative
    value is normalized by ``||lhs|| + ||rhs||``.  Callers solving an
    algebraically relaxed equation may instead provide the corresponding
    unrelaxed physical balance through ``normalization_lhs`` and
    ``normalization_rhs``.  ``scale_mode="max"`` matches the physical SIMPLE
    momentum residual convention without changing pressure/Poisson semantics.
    """
    if (normalization_lhs is None) != (normalization_rhs is None):
        raise ValueError(
            "normalization_lhs and normalization_rhs must be provided together."
        )
    def residual_norm(value):
        if norm_weights is None:
            return bm.linalg.norm(value)
        return bm.sqrt(bm.sum(norm_weights * value**2))

    residual = lhs - rhs
    absolute = float(bm.to_numpy(residual_norm(residual)))
    scale_lhs = lhs if normalization_lhs is None else normalization_lhs
    scale_rhs = rhs if normalization_rhs is None else normalization_rhs
    lhs_norm = float(bm.to_numpy(residual_norm(scale_lhs)))
    rhs_norm = float(bm.to_numpy(residual_norm(scale_rhs)))
    if scale_mode == "sum":
        scale = lhs_norm + rhs_norm
    elif scale_mode == "max":
        scale = max(lhs_norm, rhs_norm)
    else:
        raise ValueError("scale_mode must be 'sum' or 'max'.")
    return {
        "absolute": absolute,
        "relative": absolute / max(scale, float(eps)),
        "scale": scale,
    }


def normalized_relaxed_equation_residual(
    lhs,
    rhs,
    solution,
    previous_solution,
    relaxation_diagonal,
    eps=1.0e-30,
    *,
    norm_weights=None,
):
    """Normalize a relaxed-system defect by its unrelaxed physical balance.

    For ``A_alpha = A + Delta`` and
    ``b_alpha = b + Delta * x_old``, ``lhs`` and ``rhs`` are the relaxed
    balance ``A_alpha x`` and ``b_alpha + c(x)``.  The residual vector remains
    their exact difference, while the normalization references ``A x`` and
    ``b + c(x)``.
    """
    physical_lhs = lhs - relaxation_diagonal * solution
    physical_rhs = rhs - relaxation_diagonal * previous_solution
    return normalized_equation_residual(
        lhs,
        rhs,
        eps,
        normalization_lhs=physical_lhs,
        normalization_rhs=physical_rhs,
        norm_weights=norm_weights,
        scale_mode="max",
    )


def inexact_inner_tolerance(
    configured_tolerance,
    target_tolerance,
    outer_residual=None,
    *,
    forcing_factor=0.1,
):
    """Return a residual-driven tolerance for an inexact inner solve.

    ``configured_tolerance`` is the loose early-iteration limit and
    ``target_tolerance`` is the accuracy required at the outer fixed point.
    Once an outer residual is available, the inner tolerance follows a simple
    forcing sequence and is kept between those two limits.
    """
    configured_tolerance = float(configured_tolerance)
    target_tolerance = float(target_tolerance)
    forcing_factor = float(forcing_factor)
    if configured_tolerance <= 0.0:
        raise ValueError("configured_tolerance must be positive.")
    if target_tolerance <= 0.0:
        raise ValueError("target_tolerance must be positive.")
    if forcing_factor <= 0.0:
        raise ValueError("forcing_factor must be positive.")
    if outer_residual is None:
        return configured_tolerance
    outer_residual = float(outer_residual)
    if outer_residual < 0.0:
        raise ValueError("outer_residual must be non-negative.")

    strict_tolerance = min(configured_tolerance, target_tolerance)
    forced_tolerance = forcing_factor * outer_residual
    return max(
        strict_tolerance,
        min(configured_tolerance, forced_tolerance),
    )


def equation_residual_converged(metrics, *, rtol, atol):
    """Return whether a complete equation residual meets mixed tolerances."""
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative.")
    if atol < 0.0:
        raise ValueError("atol must be non-negative.")
    if rtol == 0.0 and atol == 0.0:
        raise ValueError("at least one of rtol or atol must be positive.")
    return metrics["absolute"] <= atol + rtol * metrics["scale"]


def linf_norm(value):
    """Return the infinity norm as a Python float."""
    return float(bm.to_numpy(bm.max(bm.abs(value))))


def format_pressure_correction_log(
    *,
    iteration: int,
    nonorthogonal_iterations: int,
    pressure_criterion: float,
    momentum_residual: float,
    mass_residual: float,
    pressure_correction: float,
    label: str = "SIMPLE",
) -> str:
    """Format one pressure-correction iteration diagnostic line."""
    return (
        f"[{label} {iteration}] "
        f"nonorthogonal iterations: {nonorthogonal_iterations}, "
        f"pressure criterion: {pressure_criterion:.2e}, "
        f"momentum residual: {momentum_residual:.2e}, "
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
    momentum_residual: float,
    mass_residual: float,
    pressure_correction: float,
) -> str:
    """Format one SIMPLE iteration diagnostic line."""
    return format_pressure_correction_log(
        iteration=simple_iteration,
        nonorthogonal_iterations=nonorthogonal_iterations,
        pressure_criterion=pressure_criterion,
        momentum_residual=momentum_residual,
        mass_residual=mass_residual,
        pressure_correction=pressure_correction,
        label="SIMPLE",
    )


def log_simple_iteration(logger, iteration, residual):
    """Log one SIMPLE pressure-correction diagnostic record."""
    logger.info(
        simple_iteration_log_message(
            simple_iteration=iteration,
            nonorthogonal_iterations=residual["nonorthogonal_iterations"],
            pressure_criterion=residual["pressure_criterion"],
            momentum_residual=residual["momentum_residual_relative"],
            mass_residual=residual["mass"],
            pressure_correction=residual["pressure_correction"],
        )
    )


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
    current_face_flux,
    boundary_faces,
    boundary_velocity,
    face_response_coefficient,
    rhie_chow_face_velocity,
    face_flux_operator,
):
    """Store one PISO corrector diagnostics row and call the optional callback."""
    target_face_velocity = rhie_chow_face_velocity(
        next_velocity,
        a_p,
        next_pressure,
        current_face_flux,
        boundary_faces=boundary_faces,
        boundary_velocity=boundary_velocity,
        face_response_coefficient=face_response_coefficient,
    )
    target_flux_error = face_flux_operator(target_face_velocity) - current_face_flux
    row = dict(diagnostics)
    row.update(
        {
            "step": int(step),
            "time": float(time),
            "corrector": int(correction),
            "n_correctors": int(n_correctors),
            "operator_splitting_compensation_linf": float(splitting_linf),
            "velocity_update_linf": linf_norm(next_velocity - current_velocity),
            "pressure_update_linf": linf_norm(next_pressure - current_pressure),
            "rhie_chow_flux_error_linf": linf_norm(target_flux_error),
        }
    )
    storage.append(row)
    if callback is not None:
        callback(**row)
