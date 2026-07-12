"""Cell-average exact values and errors for finite-volume solutions."""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


def cell_average(mesh, func, *, q: int = 4) -> TensorLike:
    r"""Return the control-volume average of an exact function.

    Finite-volume unknowns represent cell averages.  For an exact function
    ``u`` this routine computes

    .. math::
        \bar u_K = \frac{1}{|K|}\int_K u(x)\,dx

    on every cell ``K``.
    """
    integral = mesh.integral(func, q=q, celltype=True)
    cell_measure = mesh.entity_measure("cell")
    if integral.ndim == 1:
        return integral / cell_measure
    return integral / cell_measure.reshape((cell_measure.shape[0],) + (1,) * (integral.ndim - 1))


def cell_average_l2_error(mesh, func, numerical: TensorLike, *, q: int = 4):
    """Return the L2 error between cell averages and a finite-volume field."""
    average = cell_average(mesh, func, q=q)
    cell_measure = mesh.entity_measure("cell")
    diff = numerical - average
    if diff.ndim == 1:
        error = bm.sqrt(bm.sum(cell_measure * diff**2))
    else:
        error = bm.sqrt(bm.sum(cell_measure[:, None] * diff**2, axis=0))
    return error, average
