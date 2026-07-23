"""Residual measures shared by finite-volume SIMPLE-like solvers.

The functions in this module are intentionally small algebraic utilities, not
solver logic.  They define the residual quantities that several pressure-
velocity coupling schemes report:

* cell L2 norms for cell-centred correction fields;
* raw pressure-correction sizes for iteration diagnostics;
* normalized and maximum finite-volume mass imbalance from signed face fluxes;
* scheme-independent momentum/continuity fixed-point stopping criteria.

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


def _as_float(value):
    """Convert backend scalar tensors to Python floats for logging/tests."""
    try:
        return float(bm.to_numpy(value))
    except AttributeError:
        return float(value)


def cell_l2_norm(mesh, value, geometry=None):
    """Return the cell-volume weighted L2 norm of a cell-centred field.

    For cell values ``q_K`` this computes

        ||q||_0 = sqrt(sum_K |K| q_K^2).

    The result is a dimensional norm; no domain-volume normalization is applied.
    """
    geometry = FVMGeometry(mesh) if geometry is None else geometry
    cell_measure = geometry.cell_measure
    return _as_float(bm.sqrt(bm.sum(cell_measure * value**2)))


def relative_l2_update(mesh, update, reference, floor=1.0, geometry=None):
    """Return ``||update|| / (||reference|| + floor)`` for iteration control.

    The ``floor`` prevents a zero initial pressure or velocity field from
    turning the first relative update into a singular diagnostic.  It is part
    of the stopping criterion scale, not a numerical correction to the solve.
    """
    update_norm = cell_l2_norm(mesh, update, geometry=geometry)
    reference_norm = cell_l2_norm(mesh, reference, geometry=geometry)
    return update_norm / (reference_norm + floor)


def normalized_cell_integral_residual(
    lhs,
    rhs,
    cell_measure,
    eps=1.0e-30,
):
    r"""Return a mesh-scaled norm for cell-integrated residual vectors.

    If ``r_K`` is an integrated finite-volume residual, the discrete strong
    residual norm is

    .. math::

        \|r\|_{0,h}^2 = \sum_K \frac{|r_K|^2}{|K|}.

    Component-major vectors repeat the same cell-volume weights for every
    physical component.
    """
    if lhs.shape != rhs.shape:
        raise ValueError("lhs and rhs must have the same shape.")
    cell_measure = bm.array(cell_measure, dtype=lhs.dtype)
    if lhs.ndim != 1 or cell_measure.ndim != 1:
        raise ValueError("cell-integral residual inputs must be one-dimensional.")
    if lhs.shape[0] % cell_measure.shape[0] != 0:
        raise ValueError("residual size must be an integer multiple of cell count.")
    components = lhs.shape[0] // cell_measure.shape[0]
    inverse_measure = 1.0 / bm.tile(cell_measure, (components,))

    def weighted_norm(value):
        return _as_float(bm.sqrt(bm.sum(value**2 * inverse_measure)))

    residual = lhs - rhs
    absolute = weighted_norm(residual)
    lhs_norm = weighted_norm(lhs)
    rhs_norm = weighted_norm(rhs)
    scale = max(lhs_norm, rhs_norm, float(eps))
    return {
        "absolute_l2": absolute,
        "relative_l2": absolute / scale,
    }


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
    return collocated_mass_metrics(
        mesh,
        face_velocity,
        geometry=geometry,
    )["relative_l2"]


def collocated_mass_metrics(mesh, face_velocity, geometry=None):
    """Return mesh-scaled continuity metrics for face velocities."""
    geometry = geometry if geometry is not None else FVMGeometry(mesh)
    face_flux = bm.einsum("ij,ij->i", face_velocity, geometry.S_f)
    cell_flux_imbalance = geometry.scatter_face_flux_to_cells(face_flux)
    cell_measure = geometry.cell_measure
    divergence_l2 = _as_float(
        bm.sqrt(bm.sum(cell_flux_imbalance**2 / cell_measure))
    )

    absolute_face_flux = bm.abs(face_flux)
    cell_throughput = bm.zeros_like(cell_flux_imbalance)
    cell_throughput = bm.index_add(
        cell_throughput,
        geometry.owner,
        absolute_face_flux,
        axis=0,
    )
    cell_throughput = bm.index_add(
        cell_throughput,
        geometry.neighbour[geometry.is_internal],
        absolute_face_flux[geometry.is_internal],
        axis=0,
    )
    throughput_l2 = _as_float(
        bm.sqrt(bm.sum(cell_throughput**2 / cell_measure))
    )
    if throughput_l2 <= 1.0e-30:
        relative_l2 = 0.0 if divergence_l2 <= 1.0e-30 else divergence_l2
    else:
        relative_l2 = divergence_l2 / throughput_l2
    return {
        "relative_l1": normalized_flux_residual(
            mesh,
            cell_flux_imbalance,
            face_flux,
            geometry=geometry,
        ),
        "relative_l2": relative_l2,
        "divergence_l2": divergence_l2,
        "absolute_linf": _as_float(bm.max(bm.abs(cell_flux_imbalance))),
    }


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
    pressure_correction_l2 = cell_l2_norm(
        mesh, pressure_correction, geometry=geometry
    )
    residual = {}
    if record_pressure_correction_relative:
        residual["pressure_correction_relative"] = relative_l2_update(
            mesh,
            pressure_correction,
            pressure,
            geometry=geometry,
        )
    if stopping_face_velocity is None:
        mass_metrics = collocated_mass_metrics(
            mesh, face_velocity, geometry=geometry
        )
    else:
        mass_metrics = collocated_mass_metrics(
            mesh,
            stopping_face_velocity,
            geometry=geometry,
        )
        if record_mass_before_pressure_correction:
            residual["mass_before_pressure_correction"] = collocated_mass_metrics(
                mesh,
                face_velocity,
                geometry=geometry,
            )["relative_l1"]
    residual.update({
        "mass": mass_metrics["relative_l2"],
        "mass_relative_l1": mass_metrics["relative_l1"],
        "mass_divergence_l2": mass_metrics["divergence_l2"],
        "mass_imbalance_linf": mass_metrics["absolute_linf"],
        "pressure_correction": pressure_correction_l2,
        "pressure_criterion": pressure_correction_l2,
        "nonorthogonal_iterations": nonorthogonal_iterations,
        "momentum_nonorthogonal_iterations": momentum_nonorthogonal_iterations,
    })
    return residual


def simple_tolerances(tol, tol_momentum, tol_mass):
    """Return scheme-independent momentum and mass stopping tolerances."""
    return (
        tol if tol_momentum is None else tol_momentum,
        tol if tol_mass is None else tol_mass,
    )


def simple_fixed_point_converged(residual, tol_momentum, tol_mass):
    """Return whether physical momentum and continuity residuals converge."""
    return (
        residual["momentum_residual_relative"] < tol_momentum
        and residual["mass"] < tol_mass
    )
