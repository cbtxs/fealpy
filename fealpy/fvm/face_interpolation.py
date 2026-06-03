"""Face interpolation weights for collocated finite-volume schemes."""

from fealpy.backend import backend_manager as bm
from fealpy.mesh import HomogeneousMesh
from fealpy.typing import Index, TensorLike, _S

from .fvm_geometry import FVMGeometry


def face_interpolation_owner_weight(
    mesh: HomogeneousMesh,
    *,
    method: str = "linear",
    index: Index = _S,
) -> TensorLike:
    """Return owner-side interpolation weights on selected faces."""
    if not isinstance(mesh, HomogeneousMesh):
        raise RuntimeError(
            "face_interpolation_owner_weight only supports homogeneous meshes, "
            f"but got {type(mesh).__name__}."
        )
    if method not in {"average", "linear", "distance"}:
        raise ValueError("method must be 'average', 'linear', or 'distance'.")
    geometry = FVMGeometry(mesh, index=index)
    if method == "linear":
        return geometry.linear_owner_weight()

    owner = geometry.owner
    neighbour = geometry.neighbour
    is_internal = geometry.is_internal

    if method == "average":
        weight = 0.5 * bm.ones_like(geometry.mag_S_f)
        return bm.where(is_internal, weight, 1.0)

    face_centers = geometry.face_center
    cell_centers = geometry.cell_center
    owner_dist = bm.linalg.norm(face_centers - cell_centers[owner], axis=-1)
    neighbour_dist = bm.linalg.norm(cell_centers[neighbour] - face_centers,axis=-1)

    total_dist = owner_dist + neighbour_dist
    weight = bm.where(total_dist > 0.0, neighbour_dist / total_dist, 0.5)
    return bm.where(is_internal, weight, 1.0)
