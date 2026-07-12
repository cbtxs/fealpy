"""Residual measures shared by finite-volume SIMPLE-like solvers.

The functions in this module are intentionally small algebraic utilities, not
solver logic.  They define the residual quantities that several pressure-
velocity coupling schemes report:

* cell L2 norms for cell-centred correction fields;
* raw pressure-correction sizes for SIMPLE stopping criteria;
* normalized finite-volume mass imbalance from signed face fluxes.

Keeping these definitions in one place is useful because collocated SIMPLE-like
solvers should stop on the same mathematical residuals.

PISO solvers can reuse the same normalized face-flux residual definition, but
they often already hold scalar face fluxes instead of vector face velocities.
That route should be added when ``collocated_piso_solver.py`` is refactored, so
PISO residual field names can stay separate from SIMPLE pressure-update
residuals.
"""

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry
from .solver_diagnostics import log_simple_iteration, simple_iteration_log_message


def _as_float(value):
    """Convert backend scalar tensors to Python floats for logging/tests."""
    try:
        return float(bm.to_numpy(value))
    except AttributeError:
        return float(value)


def cell_l2_norm(mesh, value):
    """Return the cell-volume weighted L2 norm of a cell-centred field.

    For cell values ``q_K`` this computes

        ||q||_0 = sqrt(sum_K |K| q_K^2).

    The result is a dimensional norm; no domain-volume normalization is applied.
    """
    cell_measure = mesh.entity_measure("cell")
    return _as_float(bm.sqrt(bm.sum(cell_measure * value**2)))


def relative_l2_update(mesh, update, reference, floor=1.0):
    """Return ``||update|| / (||reference|| + floor)`` for iteration control.

    The ``floor`` prevents a zero initial pressure or velocity field from
    turning the first relative update into a singular diagnostic.  It is part
    of the stopping criterion scale, not a numerical correction to the solve.
    """
    update_norm = cell_l2_norm(mesh, update)
    reference_norm = cell_l2_norm(mesh, reference)
    return update_norm / (reference_norm + floor)


def normalized_flux_residual(
    mesh,
    cell_flux_imbalance,
    face_flux,
    eps=1.0e-30,
    geometry=None,
):
    """Return a dimensionless finite-volume continuity residual.

    ``cell_flux_imbalance`` is the cell residual

        r_K = sum_{f in dK} phi_f,

    with signed outward flux convention.  The numerator is ``sum_K |r_K|``.
    The denominator is the absolute face-flux scale with internal faces counted
    twice and boundary faces once, matching how each face contributes to cell
    balances.  A value near zero means that the supplied face fluxes satisfy
    discrete incompressibility relative to their own flux scale.
    """
    geometry = geometry if geometry is not None else FVMGeometry(mesh)
    is_internal = geometry.is_internal
    numerator = _as_float(bm.sum(bm.abs(cell_flux_imbalance)))
    denominator = bm.sum(bm.abs(face_flux))
    denominator = denominator + bm.sum(bm.abs(face_flux[is_internal]))
    denominator = _as_float(denominator)

    if denominator <= eps:
        return 0.0 if numerator <= eps else numerator
    return numerator / denominator


def collocated_mass_residual(mesh, face_velocity, geometry=None):
    """Mass residual for collocated vector face velocities.

    The scalar flux is ``phi_f = dot(u_f, S_f)`` where ``S_f`` is the oriented
    face area vector.  The cell imbalance is obtained by scattering this same
    face flux with the owner-oriented geometry convention.
    """
    geometry = geometry if geometry is not None else FVMGeometry(mesh)
    face_flux = bm.einsum("ij,ij->i", face_velocity, geometry.S_f)
    cell_flux_imbalance = geometry.scatter_face_flux_to_cells(face_flux)
    return normalized_flux_residual(
        mesh,
        cell_flux_imbalance,
        face_flux,
        geometry=geometry,
    )


def simple_iteration_residual(
    mesh,
    face_velocity,
    pressure_correction,
    pressure,
    *,
    nonorthogonal_iterations,
    momentum_nonorthogonal_iterations,
    stopping_face_velocity=None,
    record_mass_before_pressure_correction=False,
    record_pressure_correction_relative=False,
    geometry=None,
):
    """Return one SIMPLE residual record.

    SIMPLE solves a pressure-correction equation and then updates
    ``p <- p + alpha_p p'`` with a fixed scalar pressure relaxation.  The
    residual record intentionally reports the raw pressure-correction norm,
    because the current stopping criterion should not be scaled by the chosen
    pressure relaxation factor.  The mass residual is evaluated after the
    face-velocity pressure correction when ``stopping_face_velocity`` is
    supplied, which is the residual used for stopping the current collocated
    SIMPLE loop.
    """
    pressure_correction_l2 = cell_l2_norm(mesh, pressure_correction)
    residual = {}
    if record_pressure_correction_relative:
        residual["pressure_correction_relative"] = relative_l2_update(
            mesh,
            pressure_correction,
            pressure,
        )
    if stopping_face_velocity is None:
        mass = collocated_mass_residual(mesh, face_velocity, geometry=geometry)
    else:
        mass = collocated_mass_residual(
            mesh,
            stopping_face_velocity,
            geometry=geometry,
        )
        if record_mass_before_pressure_correction:
            residual["mass_before_pressure_correction"] = collocated_mass_residual(
                mesh,
                face_velocity,
                geometry=geometry,
            )
    residual.update({
        "mass": mass,
        "pressure_correction": pressure_correction_l2,
        "pressure_criterion": pressure_correction_l2,
        "nonorthogonal_iterations": nonorthogonal_iterations,
        "momentum_nonorthogonal_iterations": momentum_nonorthogonal_iterations,
    })
    return residual


def simple_pressure_update_step(
    residuals,
    mesh,
    face_velocity,
    pressure_correction,
    pressure,
    *,
    pressure_relax,
    nonorthogonal_iterations,
    momentum_nonorthogonal_iterations,
    stopping_face_velocity=None,
    record_mass_before_pressure_correction=False,
    record_pressure_correction_relative=False,
    geometry=None,
):
    """Return fixed-relaxation pressure update and append its residual row."""
    pressure_update = pressure_relax * pressure_correction
    residual = simple_iteration_residual(
        mesh,
        face_velocity,
        pressure_correction,
        pressure,
        nonorthogonal_iterations=nonorthogonal_iterations,
        momentum_nonorthogonal_iterations=momentum_nonorthogonal_iterations,
        stopping_face_velocity=stopping_face_velocity,
        record_mass_before_pressure_correction=record_mass_before_pressure_correction,
        record_pressure_correction_relative=record_pressure_correction_relative,
        geometry=geometry,
    )
    residuals.append(residual)
    return pressure_update, residual


def simple_tolerances(tol, tol_mass, tol_pressure_correction):
    """Return mass and pressure-correction stopping tolerances for SIMPLE."""
    return (
        tol if tol_mass is None else tol_mass,
        10.0 * tol if tol_pressure_correction is None else tol_pressure_correction,
    )


def pressure_correction_converged(residual, tol_mass, tol_pressure_correction):
    """Return whether SIMPLE pressure-correction residuals satisfy stopping criteria."""
    if residual["mass"] >= tol_mass:
        return False
    return residual["pressure_correction"] < tol_pressure_correction


def log_simple_residual(logger, iteration, residual):
    """Log one SIMPLE pressure-correction residual record."""
    log_simple_iteration(logger, iteration, residual)
