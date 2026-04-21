
from  dataclasses import dataclass, field

from ...backend import Tensor

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
