"""Finite-volume cell divergence reconstruction from face fluxes."""

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry


class DivergenceReconstruct:
    """Reconstruct cell-integrated flux imbalance from face velocities.

    The returned value is the integrated cell flux imbalance, not divided by
    cell measure.  This convention matches the SIMPLE pressure-correction RHS
    and the residual diagnostics in :mod:`simple_residual`.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.fvm_geometry = FVMGeometry(mesh)

    def Reconstruct(self, face_velocity):
        """Divergence from vector collocated face velocities.

        ``face_velocity`` is defined on mesh faces.  The owner-oriented scalar
        flux is ``phi_f = dot(u_f, S_f)`` and is scattered to the owner cell
        with positive sign and to the neighbour cell with negative sign.
        """
        Sf = self.fvm_geometry.S_f
        face_flux = bm.einsum("ij,ij->i", face_velocity, Sf)
        return self.fvm_geometry.scatter_face_flux_to_cells(face_flux)
