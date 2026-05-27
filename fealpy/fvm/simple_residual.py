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
"""

from fealpy.backend import backend_manager as bm

from .div_reconstruct import DivergenceReconstruct


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
    e2c = mesh.edge_to_cell()[:, :2]
    is_internal = e2c[:, 0] != e2c[:, 1]
    numerator = _as_float(bm.sum(bm.abs(cell_flux_imbalance)))
    denominator = bm.sum(bm.abs(face_flux))
    denominator = denominator + bm.sum(bm.abs(face_flux[is_internal]))
    denominator = _as_float(denominator)

    if denominator <= eps:
        return 0.0 if numerator <= eps else numerator
    return numerator / denominator


def collocated_mass_residual(mesh, face_velocity):
    """Mass residual for collocated vector face velocities.

    The scalar flux is ``phi_f = u_f · S_f`` where ``S_f`` is the oriented face
    area vector.  The same vector face velocity is passed to the collocated
    divergence reconstruction, so the diagnostic measures exactly the flux
    imbalance represented by the supplied face velocity field.
    """
    face_flux = bm.einsum("ij,ij->i", face_velocity, mesh.edge_normal())
    cell_flux_imbalance = DivergenceReconstruct(mesh).Reconstruct(face_velocity)
    return normalized_flux_residual(mesh, cell_flux_imbalance, face_flux)


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
