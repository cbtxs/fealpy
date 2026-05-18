"""Geometric decompositions for non-orthogonal finite-volume fluxes."""

from fealpy.backend import backend_manager as bm
from fealpy.typing import TensorLike

from .vector_decomposition import VectorDecomposition


class NonOrthogonalGeometry:
    """Geometry for finite-volume non-orthogonal diffusion corrections.

    For a face ``f`` between owner ``P`` and neighbour ``N``, finite-volume
    diffusion uses the oriented face area vector ``S_f`` and the cell-centre
    vector ``d_f = x_N - x_P``.  On non-orthogonal meshes ``S_f`` is not
    parallel to ``d_f``.  Diffusion fluxes are therefore split into an implicit
    orthogonal part, based on the cell jump, plus an explicit correction part:

        S_f · grad(phi)_f
        = |S_f| delta_f (phi_N - phi_P)
          + C_f · grad(phi)_f.

    This class stores only the geometric terms in that split.  It does not
    assemble equations and it does not define boundary data; those contracts
    remain in the integrator and boundary-condition layers.
    """

    def __init__(self, mesh, eps=0.05):
        """Create geometry helpers for ``mesh``.

        ``eps`` is the lower-bound fraction used when computing
        ``delta_f = 1 / max(n_f · d_f, eps |d_f|)``.  The bound prevents a
        degenerate or severely skewed face from producing an unbounded
        orthogonal coefficient.
        """
        self.mesh = mesh
        self.eps = eps

    def owner_neighbour(self) -> TensorLike:
        """Return owner-neighbour cell indices for each face."""
        return self.mesh.edge_to_cell()[:, :2]

    def cell_center_vector(self) -> TensorLike:
        """Return ``d_f``, the owner-to-neighbour/boundary centre vector."""
        return VectorDecomposition(self.mesh).centroid_vector_calculation()[0]

    def face_area_vector(self) -> TensorLike:
        """Return the oriented face area vector ``S_f``."""
        return self.mesh.edge_normal()

    def face_area_norm(self) -> TensorLike:
        """Return ``|S_f|`` for each face."""
        return bm.linalg.norm(self.face_area_vector(), axis=1)

    def unit_normal(self) -> TensorLike:
        """Return the unit face normal ``n_f = S_f / |S_f|``."""
        return self.face_area_vector() / self.face_area_norm()[:, None]

    def legacy_tangential_vector(self) -> TensorLike:
        """Return the historical FEALPy tangential correction vector.

        This preserves the old cross-diffusion behavior for regression and
        comparison.  New non-orthogonal correction experiments should prefer a
        named decomposition such as ``openfoam_correction_vector`` so its
        geometric meaning is explicit.
        """
        return VectorDecomposition(self.mesh).tangential_vector_calculation()

    def openfoam_delta_coeff(self) -> TensorLike:
        """Return the bounded orthogonal coefficient ``delta_f``.

        This follows the OpenFOAM-style stabilized projection

            delta_f = 1 / max(n_f · d_f, eps |d_f|),

        where ``n_f`` is the face unit normal.  The coefficient is used for the
        implicit orthogonal diffusion contribution based on ``phi_N - phi_P``.
        """
        delta = self.cell_center_vector()
        projected = bm.einsum("ij,ij->i", self.unit_normal(), delta)
        lower_bound = self.eps * bm.linalg.norm(delta, axis=1)
        return 1.0 / bm.maximum(projected, lower_bound)

    def openfoam_correction_vector(self) -> TensorLike:
        """Return the explicit non-orthogonal correction vector ``C_f``.

        With ``delta_f`` from :meth:`openfoam_delta_coeff`, this returns

            C_f = |S_f| (n_f - delta_f d_f).

        The cross-diffusion integrator uses ``C_f · grad(phi)_f`` as an explicit
        face-flux correction.  This keeps the matrix contribution tied to the
        orthogonal cell jump while moving the skew/non-orthogonal part to the
        right-hand side.
        """
        delta = self.cell_center_vector()
        correction = self.unit_normal() - delta * self.openfoam_delta_coeff()[:, None]
        return self.face_area_norm()[:, None] * correction

    def zero_correction_vector(self) -> TensorLike:
        """Return zero explicit correction for purely orthogonal treatment."""
        return bm.zeros_like(self.face_area_vector())
