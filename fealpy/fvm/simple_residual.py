"""Residual measures shared by finite-volume SIMPLE-like solvers.

The functions in this module are intentionally small algebraic utilities, not
solver logic.  They define the residual quantities that several pressure-
velocity coupling schemes report:

* cell L2 norms for cell-centred correction fields;
* relative update sizes for relaxed fixed-point iterations;
* normalized finite-volume mass imbalance from signed face fluxes.

Keeping these definitions in one place is useful because collocated and
staggered SIMPLE solvers should stop on the same mathematical residuals even
though their face-velocity representations differ.

PISO solvers can reuse the same normalized face-flux residual definition, but
they often already hold scalar face fluxes instead of vector face velocities.
That route should be added when ``collocated_piso_solver.py`` is refactored, so
PISO residual field names can stay separate from SIMPLE pressure-update
residuals.
"""

from fealpy.backend import backend_manager as bm

from .div_reconstruct import DivergenceReconstruct
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


def normalized_flux_residual(mesh, cell_flux_imbalance, face_flux, eps=1.0e-30):
    """Return a dimensionless finite-volume continuity residual.

    ``cell_flux_imbalance`` is the cell residual

        r_K = sum_{f in dK} phi_f,

    with signed outward flux convention.  The numerator is ``sum_K |r_K|``.
    The denominator is the absolute face-flux scale with internal faces counted
    twice and boundary faces once, matching how each face contributes to cell
    balances.  A value near zero means that the supplied face fluxes satisfy
    discrete incompressibility relative to their own flux scale.
    """
    is_internal = FVMGeometry(mesh).is_internal
    numerator = _as_float(bm.sum(bm.abs(cell_flux_imbalance)))
    denominator = bm.sum(bm.abs(face_flux))
    denominator = denominator + bm.sum(bm.abs(face_flux[is_internal]))
    denominator = _as_float(denominator)

    if denominator <= eps:
        return 0.0 if numerator <= eps else numerator
    return numerator / denominator


def collocated_mass_residual(mesh, face_velocity):
    """Mass residual for collocated vector face velocities.

    The scalar flux is ``phi_f = dot(u_f, S_f)`` where ``S_f`` is the oriented
    face area vector.  The cell imbalance is obtained by scattering this same
    face flux with the owner-oriented geometry convention.
    """
    geometry = FVMGeometry(mesh)
    face_flux = bm.einsum("ij,ij->i", face_velocity, geometry.S_f)
    cell_flux_imbalance = geometry.scatter_face_flux_to_cells(face_flux)
    return normalized_flux_residual(mesh, cell_flux_imbalance, face_flux)


def simple_iteration_residual(
    mesh,
    face_velocity,
    pressure_correction,
    pressure,
    *,
    pressure_relax,
    nonorthogonal_iterations,
    momentum_nonorthogonal_iterations,
    stopping_face_velocity=None,
):
    """Return one SIMPLE residual record.

    SIMPLE solves a pressure-correction equation and then updates
    ``p <- p + alpha_p p'`` with a fixed scalar pressure relaxation
    ``alpha_p``.  This record reports both the raw correction norm and the
    relaxed pressure-update size.  The mass residual is evaluated after the
    face-velocity pressure correction when ``stopping_face_velocity`` is
    supplied, which is the residual used for stopping the current collocated
    SIMPLE loop.
    """
    pressure_update = pressure_relax * pressure_correction
    pressure_update_relative = relative_l2_update(mesh, pressure_update, pressure)
    pressure_correction_relative = relative_l2_update(
        mesh, pressure_correction, pressure
    )
    mass_before_pressure_correction = collocated_mass_residual(mesh, face_velocity)
    if stopping_face_velocity is None:
        mass = mass_before_pressure_correction
    else:
        mass = collocated_mass_residual(mesh, stopping_face_velocity)
    return {
        "mass": mass,
        "mass_before_pressure_correction": mass_before_pressure_correction,
        "pressure_correction": cell_l2_norm(mesh, pressure_correction),
        "pressure_correction_relative": pressure_correction_relative,
        "pressure_update": pressure_update_relative,
        "pressure_criterion": pressure_update_relative,
        "pressure_relax": pressure_relax,
        "pressure_relax_reduced": False,
        "pressure_relax_action": "fixed",
        "nonorthogonal_iterations": nonorthogonal_iterations,
        "momentum_nonorthogonal_iterations": momentum_nonorthogonal_iterations,
    }


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
):
    """Return fixed-relaxation pressure update and append its residual row."""
    pressure_update = pressure_relax * pressure_correction
    residual = simple_iteration_residual(
        mesh,
        face_velocity,
        pressure_correction,
        pressure,
        pressure_relax=pressure_relax,
        nonorthogonal_iterations=nonorthogonal_iterations,
        momentum_nonorthogonal_iterations=momentum_nonorthogonal_iterations,
        stopping_face_velocity=stopping_face_velocity,
    )
    residuals.append(residual)
    return pressure_update, residual


def simple_tolerances(tol, tol_mass, tol_pressure_update):
    """Return mass and pressure-update stopping tolerances for SIMPLE."""
    return (
        tol if tol_mass is None else tol_mass,
        10.0 * tol if tol_pressure_update is None else tol_pressure_update,
    )


def log_simple_residual(logger, iteration, residual):
    """Log one SIMPLE pressure-correction residual record."""
    log_simple_iteration(logger, iteration, residual)


def staggered_mass_residual(mesh, edge_velocity):
    """Mass residual for scalar staggered velocities on pressure faces.

    ``edge_velocity`` is interpreted as the normal velocity component stored on
    the pressure-mesh face.  On the current structured staggered meshes the
    signed scalar face measure is recovered from the oriented area vector by
    summing its components, because each pressure face is axis-aligned.  This
    helper is therefore a residual definition for the current staggered mesh
    contract, not a generic unstructured normal-flux conversion.
    """
    signed_face_measure = bm.sum(mesh.edge_normal(), axis=1)
    face_flux = edge_velocity * signed_face_measure
    cell_flux_imbalance = DivergenceReconstruct(mesh).StagReconstruct(edge_velocity)
    return normalized_flux_residual(mesh, cell_flux_imbalance, face_flux)
