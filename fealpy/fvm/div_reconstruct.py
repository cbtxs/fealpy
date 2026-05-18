from fealpy.backend import backend_manager as bm

class DivergenceReconstruct:
    """
    Divergence reconstruction for finite volume method.
    This class provides methods to compute the divergence of a velocity field
    defined on edges of a mesh.
    """
    def __init__(self, mesh):
        self.mesh = mesh

    def StagReconstruct(self, edge_velocity):
        signed_face_measure = bm.sum(self.mesh.edge_normal(), axis=1)
        flux = edge_velocity * signed_face_measure
        PNC = self.mesh.number_of_cells()
        pe2c = self.mesh.edge_to_cell()[:,:2]
        div_u = bm.zeros(PNC)
        mask = pe2c[:, 1] != pe2c[:, 0]  # 非边界边
        bm.add_at(div_u, pe2c[:, 0], flux)
        bm.add_at(div_u, pe2c[mask, 1], -flux[mask])
        return div_u
    
    def Reconstruct(self, edge_velocity):
        Sf = self.mesh.edge_normal()
        integrator = bm.einsum('ij,ij->i', edge_velocity, Sf)
        e2c = self.mesh.edge_to_cell()[:,:2]
        div_u = bm.zeros(self.mesh.number_of_cells())
        mask = e2c[:, 1] != e2c[:, 0]
        bm.add.at(div_u, e2c[:, 0], integrator)
        bm.add.at(div_u, e2c[mask, 1], -integrator[mask])
        # bm.add.at(div_u, e2c[:, 1], -integrator)
        return div_u
