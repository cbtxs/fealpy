
from  dataclasses import dataclass

from ...backend import bm, Tensor

__all__ = ["Relation"]


@dataclass(slots=True)
class Relation:
    src_name: str
    tgt_name: str
    tgt_indices: Tensor
    """
    The indices to the target entities. This can be a 2D array for
    homogeneous relations (e.g., edges) or a 1D array for heterogeneous
    relations (e.g., node-to-face).
    """
    src_indices: Tensor | None = None
    """
    The indices to the source entities. This is `None` for homogeneous relations
    (e.g., edges) and a 1D array for heterogeneous relations (e.g., node-to-face).
    """
    local_index: Tensor | None = None

    def as_array(self) -> Tensor:
        if self.src_indices is None:
            return self.tgt_indices
        raise ValueError("Cannot convert heterogeneous relation to array. "
                         "Use `as_coo` or `as_csr` instead.")

    def as_coo(self):
        from ...sparse import coo_matrix
        if self.src_indices is None:
            src = bm.arange(self.tgt_indices.shape[0], dtype=bm.int32)
            tgt = bm.copy(self.tgt_indices).reshape(-1)
        else:
            src = bm.copy(self.src_indices)
            tgt = bm.copy(self.tgt_indices)
        data = bm.ones_like(src, dtype=bm.bool)
        return coo_matrix(
            (data, (src, tgt)),
            shape=(bm.max(src) + 1, bm.max(tgt) + 1) # type: ignore
        )

    def as_csr(self):
        return self.as_coo().tocsr()

    def inverse(self) -> "Relation":
        """Return the inverse relation."""

        if self.src_indices is None:
            src = bm.arange(self.tgt_indices.shape[0], dtype=bm.int32)
            tgt = bm.copy(self.tgt_indices).reshape(-1)
        else:
            src = bm.copy(self.src_indices)
            tgt = bm.copy(self.tgt_indices)

        return Relation(
            src_name=self.tgt_name,
            tgt_name=self.src_name,
            src_indices=tgt,
            tgt_indices=src,
            local_index=self.local_index
        )
