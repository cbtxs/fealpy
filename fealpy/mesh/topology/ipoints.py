
from itertools import combinations_with_replacement

from ...backend import bm
from ...backend import Tensor, dtype


class InterpolationPoints:
    @classmethod
    def multi_index_matrix(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index matrix for interpolation points of
        degree p with n vertices. The multi-index matrix is of shape
        (C(p+n-1, n-1), n) and each row corresponds to the multi-index of
        an interpolation point.

        Parameters:
            p (int): Degree of interpolation.
            n (int): Number of vertices in the Simplex.
            dtype (dtype, optional): Data type of the output tensor. If None, it will
                default to int32.

        Returns:
            Tensor: A tensor of shape (C(p+n-1, n-1), n) containing the multi-indices.
        """
        if dtype is None:
            dtype = bm.int32

        sep = bm.flip(bm.asarray(
            tuple(combinations_with_replacement(range(p+1), n-1)),
            dtype=dtype
        ), axis=0)
        raw = bm.zeros((sep.shape[0], n+1), dtype=dtype)
        raw[:, -1] = p
        raw[:, 1:-1] = sep
        return (raw[:, 1:] - raw[:, :-1])

    @classmethod
    def multi_index_inner(cls, p: int, n: int, *, dtype: dtype | None = None) -> Tensor:
        """Generate the multi-index corresponding to the inner interpolation
        points of degree p with n vertices.

        See also: `multi_index_matrix`."""
        if p < n:
            if dtype is None:
                dtype = bm.int32
            return bm.zeros((0, n), dtype=dtype)
        return cls.multi_index_matrix(p - n, n, dtype=dtype) + 1
