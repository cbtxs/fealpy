"""Internal backend-array helpers for FVM implementation code."""

from fealpy.backend import backend_manager as bm


def as_backend_array(value, dtype=None):
    """Return ``value`` as a backend array, optionally cast to ``dtype``.

    This helper is intentionally small and internal.  It keeps boundary,
    reconstruction, and coupling code from repeating the same NumPy/PyTorch/JAX
    dtype checks while leaving mathematical operators free of backend-specific
    branching.
    """
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is None:
        return bm.array(value, dtype=dtype) if dtype is not None else bm.array(value)
    if dtype is not None and value_dtype != dtype:
        return bm.astype(value, dtype)
    return value


def cast_like(value, reference):
    """Cast ``value`` to the dtype of ``reference`` when a dtype is available."""
    dtype = getattr(reference, "dtype", None)
    if dtype is None:
        return value
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is None:
        return bm.array(value, dtype=dtype)
    if value_dtype == dtype:
        return value
    return bm.astype(value, dtype)
