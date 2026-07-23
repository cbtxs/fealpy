"""Face-gradient construction for finite-volume solvers."""

from __future__ import annotations

from typing import Optional

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .fvm_geometry import FVMGeometry, interpolate_cell_to_face


def reconstruct_face_gradient(
    mesh,
    cell_gradient: TensorLike,
    *,
    geometry: Optional[FVMGeometry] = None,
    cell_values: Optional[TensorLike] = None,
    interpolation_method: str = "average",
    dirichlet_faces: Optional[TensorLike] = None,
    dirichlet_values: Optional[TensorLike] = None,
    neumann_faces: Optional[TensorLike] = None,
    neumann_sn_grad: Optional[TensorLike] = None,
    boundary_faces: Optional[TensorLike] = None,
    boundary_sn_grad: Optional[TensorLike] = None,
) -> TensorLike:
    """Build face gradients from cell gradients and optional patch data.

    Internal faces use the requested owner-neighbour interpolation.  Boundary
    faces are corrected by the supplied normal derivative for Dirichlet,
    Neumann, or generic boundary patch data.
    """
    geometry = geometry if geometry is not None else FVMGeometry(mesh)
    cell_gradient = bm.array(cell_gradient)
    face_gradient = interpolate_cell_to_face(
        cell_gradient,
        geometry=geometry,
        method=interpolation_method,
    )

    patch_sn_grads = []
    if dirichlet_faces is not None:
        if cell_values is None:
            raise ValueError("cell_values must be given with dirichlet_faces.")
        if dirichlet_values is None:
            raise ValueError("dirichlet_values must be given with dirichlet_faces.")

        cell_values = bm.array(cell_values)
        faces = bm.array(dirichlet_faces, dtype=bm.int64)
        boundary_values = bm.array(dirichlet_values, dtype=cell_values.dtype)
        expected_shape = (faces.shape[0],) + cell_values.shape[1:]
        if boundary_values.shape != expected_shape:
            raise ValueError(
                f"dirichlet_values must have shape {expected_shape} matching "
                "dirichlet_faces."
            )
        owner = geometry.owner[faces]
        distance = geometry.normal_distance(faces)
        if bm.any(distance <= 0.0):
            raise ValueError("boundary face has zero owner-normal distance.")

        distance_shape = (distance.shape[0],) + (1,) * (cell_values.ndim - 1)
        sn_grad = (boundary_values - cell_values[owner]) / distance.reshape(distance_shape)
        patch_sn_grads.append((faces, sn_grad, "dirichlet_sn_grad"))

    if neumann_faces is not None:
        if neumann_sn_grad is None:
            raise ValueError("neumann_sn_grad must be given with neumann_faces.")
        faces = bm.array(neumann_faces, dtype=bm.int64)
        patch_sn_grads.append((faces, neumann_sn_grad, "neumann_sn_grad"))

    if boundary_faces is not None:
        if boundary_sn_grad is None:
            raise ValueError("boundary_sn_grad must be given with boundary_faces.")
        faces = bm.array(boundary_faces, dtype=bm.int64)
        patch_sn_grads.append((faces, boundary_sn_grad, "boundary_sn_grad"))

    for faces, sn_grad, name in patch_sn_grads:
        sn_grad = bm.array(sn_grad, dtype=cell_gradient.dtype)
        expected_shape = (faces.shape[0],) + cell_gradient.shape[1:-1]
        if sn_grad.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape} matching its faces."
            )
        owner_gradient = cell_gradient[geometry.owner[faces]]
        unit_normal = geometry.n_f[faces]

        if owner_gradient.ndim == 2:
            current_sn_grad = bm.einsum("fi,fi->f", owner_gradient, unit_normal)
            correction = sn_grad - current_sn_grad
            corrected_boundary = owner_gradient + correction[:, None] * unit_normal
        elif owner_gradient.ndim == 3:
            current_sn_grad = bm.einsum("fij,fj->fi", owner_gradient, unit_normal)
            correction = sn_grad - current_sn_grad
            corrected_boundary = owner_gradient + correction[:, :, None] * unit_normal[:, None, :]
        else:
            raise ValueError("cell_gradient must have shape (NC, GD) or (NC, n_component, GD).")
        face_gradient = bm.set_at(face_gradient, faces, corrected_boundary)

    return face_gradient
