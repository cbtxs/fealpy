"""Face-centre vector decompositions used by FVM diffusion operators."""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike


class VectorDecomposition:
    """Compute owner-to-neighbour and tangential face geometry vectors.

    This class is a low-level geometric helper.  It does not know about PDE
    coefficients, boundary data, or solver iterations.  Diffusion and
    non-orthogonal correction integrators use it to recover the vector from the
    owner cell centre to the neighbour cell centre, or to the boundary face
    centre on boundary faces.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.Sf = mesh.edge_normal()  # (NE, self.mesh.GD)

    def centroid_vector_calculation(self) -> TensorLike:
        """Return ``(e_f, |e_f|)`` for every face.

        For an internal face, ``e_f`` points from owner cell centre to
        neighbour cell centre.  For a boundary face, it points from owner cell
        centre to face centre.  This owner-oriented convention is shared by
        diffusion, non-orthogonal correction, and Rhie-Chow geometry code.
        """
        cell_centers = self.mesh.entity_barycenter('cell')  # (NC, 2)
        edge_middle_point = self.mesh.entity_barycenter('face')
        e2c = self.mesh.edge_to_cell()
        NE = self.mesh.number_of_edges()
        is_interior = e2c[:, 0] != e2c[:, 1]
        is_boundary = ~is_interior

        e = bm.zeros((NE, self.mesh.GD), dtype=cell_centers.dtype)
        e = bm.set_at(
            e,
            is_interior,
            cell_centers[e2c[:, 1][is_interior]]
            - cell_centers[e2c[:, 0][is_interior]],
        )
        e = bm.set_at(
            e,
            is_boundary,
            edge_middle_point[is_boundary] - cell_centers[e2c[:, 0][is_boundary]],
        )

        d = bm.linalg.norm(e, axis=-1, keepdims=True).reshape(-1)
        return e, d

    def Sor(self) -> TensorLike:
        """Return the orthogonal face length scale used by diffusion assembly."""
        e, _ = self.centroid_vector_calculation()
        Sf_dot_Sf = bm.einsum('ij,ij->i', self.Sf, self.Sf).reshape(-1, 1)  
        e_dot_Sf = bm.einsum('ij,ij->i', e, self.Sf).reshape(-1, 1)    
        e_norm = bm.linalg.norm(e, axis=-1, keepdims=True)  
        Ef_abs = (Sf_dot_Sf / e_dot_Sf) * e_norm
        return Ef_abs.reshape(-1)   # shape: (NE, 1)

    def tangential_vector_calculation(self) -> TensorLike:
        """Return the residual tangential vector ``T_f = S_f - E_f``."""
        e, _ = self.centroid_vector_calculation()    # (NE, 2)
        dot_e_Sf = bm.sum(e * self.Sf, axis=1, keepdims=True)  # (NE, 1)
        norm_Sf_sq = bm.sum(self.Sf * self.Sf, axis=1, keepdims=True)
        Ef = (norm_Sf_sq / (dot_e_Sf + 1e-13)) * e
        Tf = self.Sf - Ef
        return Tf  # shape: (NE, 2)
