"""Finite-volume cell divergence reconstruction from face fluxes."""

from fealpy.backend import backend_manager as bm

from .fvm_geometry import FVMGeometry


class DivergenceReconstruct:
    """Scatter signed face fluxes to owner and neighbour cells.

    The returned value is the integrated cell flux imbalance, not divided by
    cell measure.  This convention matches the SIMPLE pressure-correction RHS
    and the residual diagnostics in :mod:`simple_residual`.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.fvm_geometry = FVMGeometry(mesh)

    def StagReconstruct(self, edge_velocity):
        """Divergence from scalar staggered normal velocities."""
        signed_face_measure = bm.sum(self.mesh.edge_normal(), axis=1)
        flux = edge_velocity * signed_face_measure
        PNC = self.mesh.number_of_cells()
        pe2c = self.mesh.edge_to_cell()[:,:2]
        div_u = bm.zeros(PNC, dtype=flux.dtype)
        mask = pe2c[:, 1] != pe2c[:, 0]  # 非边界边
        div_u = bm.index_add(div_u, pe2c[:, 0], flux, axis=0)
        div_u = bm.index_add(div_u, pe2c[mask, 1], flux[mask], axis=0, alpha=-1)
        return div_u
    
    def Reconstruct(self, edge_velocity):
        """Divergence from vector collocated face velocities."""
        Sf = self.fvm_geometry.S_f
        integrator = bm.einsum('ij,ij->i', edge_velocity, Sf)
        return self.fvm_geometry.scatter_face_flux_to_cells(integrator)
